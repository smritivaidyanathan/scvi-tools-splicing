from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Literal, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli, kl_divergence
from torch.nn.functional import one_hot

from scvi import REGISTRY_KEYS, settings
from scvi.module._constants import MODULE_KEYS
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data


if TYPE_CHECKING:
    from collections.abc import Callable
    from torch.distributions import Distribution

logger = logging.getLogger(__name__)


def binomial_loss_function(
    logits: torch.Tensor,
    junction_counts: torch.Tensor,
    cluster_counts: torch.Tensor,
    n,
    k,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Reconstruction Binomial loss function for VAE.
    
    Parameters:
      logits: Reconstructed logits from decoder.
      junction_counts: Junction counts from data.
      cluster_counts: Cluster counts from data.
      mask: Mask to exclude missing values when running partialVAE.
      
    Returns:
      total_loss: Reconstruction loss according to binomial likelihood.
    """

    if mask is not None:
        mask = mask.to(torch.bool)  # ← convert floats/ints to bool
        logits = logits[mask]
        junction_counts = junction_counts[mask]
        cluster_counts = cluster_counts[mask]

    probs = torch.sigmoid(logits)
    log_p = torch.log(probs + 1e-8)
    log_1_p = torch.log(1 - probs + 1e-8)
    log_lik = (
        junction_counts * log_p + (cluster_counts - junction_counts) * log_1_p
    )
    return -log_lik.mean()


def beta_binomial_log_pmf(k, n, alpha, beta):
    """
    Calculate the log probability mass function for the beta-binomial distribution.
    
    Parameters:
      k: Number of successes.
      n: Number of trials.
      alpha: Alpha parameter of beta distribution.
      beta: Beta parameter of beta distribution.
      
    Returns:
      log_pmf: Log of probability mass function.
    """
    return (
        torch.lgamma(n + 1)
        - torch.lgamma(k + 1)
        - torch.lgamma(n - k + 1)
        + torch.lgamma(k + alpha)
        + torch.lgamma(n - k + beta)
        - torch.lgamma(alpha)
        - torch.lgamma(beta)
        + torch.lgamma(alpha + beta)
        - torch.lgamma(n + alpha + beta)
    )


def beta_binomial_loss_function(
    logits: torch.Tensor,
    junction_counts: torch.Tensor,
    cluster_counts: torch.Tensor,
    n: int,
    k: int,
    concentration: float = 1.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Beta-binomial loss function for VAE.
    
    Parameters:
      logits: Reconstructed logits from decoder.
      junction_counts: Junction counts from data.
      cluster_counts: Cluster counts from data.
      n: Number of samples in dataset.
      k: Number of batches in dataloader.
      concentration: Concentration parameter.
      mask: Optional mask to exclude missing values.
      
    Returns:
      total_loss: Combined reconstruction and KL loss.
    """
    if mask is not None:
        mask = mask.to(torch.bool)  # ← convert floats/ints to bool
        logits = logits[mask]
        junction_counts = junction_counts[mask]
        cluster_counts = cluster_counts[mask]


    probs = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    alpha = probs * concentration
    beta = (1.0 - probs) * concentration
    log_lik = beta_binomial_log_pmf(
        k=junction_counts, n=cluster_counts, alpha=alpha, beta=beta
    )
    log_lik = log_lik #* (float(n) / float(k))
    return -log_lik.mean()


class PartialEncoder(nn.Module):
    def __init__(self, input_dim: int, h_hidden_dim: int, encoder_hidden_dim: int, 
                 latent_dim: int, code_dim: int, dropout_rate: float = 0.0):
        """
        Encoder network inspired by PointNet for partially observed data.

        Processes each observed feature individually using a shared network ('h_layer')
        combined with learnable feature embeddings and biases, then aggregates
        the results before mapping to the latent space.

        Parameters:
          input_dim (int): Dimension of input features (D). Number of junctions/features.
          h_hidden_dim (int): Hidden dimension for the shared 'h_layer'.
                           (Replaces the misuse of num_hidden_layers in the original h_layer definition).
          encoder_hidden_dim (int): Hidden dimension for the final 'encoder_mlp'.
                                 (Replaces the hardcoded 256 in the original encoder_mlp).
          latent_dim (int): Dimension of latent space (Z).
          code_dim (int): Dimension of feature embeddings and intermediate representations (K).
          dropout_rate (float): Dropout rate for regularization applied within h_layer and encoder_mlp.
        """
        super().__init__()
        self.input_dim = input_dim
        self.code_dim = code_dim
        self.latent_dim = latent_dim

        # Learnable feature embedding (F_d in paper notation)
        self.feature_embedding = nn.Parameter(torch.randn(input_dim, code_dim)) #D by K initialized via PCA 

        # Shared function h(.) applied to each feature representation s_d = [x_d, F_d]
        # Input dim: 1 (feature value) + K (embedding) = K + 1
        # Output dim: K (code_dim)
        self.h_layer = nn.Sequential(
            nn.Linear(1 + code_dim, h_hidden_dim),
            nn.LayerNorm(h_hidden_dim),           
            nn.ReLU(),
            nn.Dropout(dropout_rate), 
            nn.Linear(h_hidden_dim, code_dim),
            nn.LayerNorm(code_dim),   
            nn.ReLU()
        )

        # MLP to map aggregated representation 'c' to latent distribution parameters
        # Input dim: K (code_dim)
        # Output dim: 2 * Z (for mu and logvar)
        self.encoder_mlp = nn.Sequential(
            nn.Linear(code_dim, encoder_hidden_dim),
            nn.LayerNorm(encoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate), 
            nn.Linear(encoder_hidden_dim, 2 * latent_dim) # outputs both mu and logvar
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the encoder.

        Args:
            x (torch.Tensor): Input data (batch_size, input_dim). Missing values can be anything (e.g., 0, NaN),
                              as they will be masked out based on the 'mask' tensor.
                              It's crucial that the *observed* values in x are the actual measurements.
            mask (torch.Tensor): Binary mask (batch_size, input_dim). 1 indicates observed, 0 indicates missing.
                               Must be float or long/int and compatible with multiplication.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - mu (torch.Tensor): Mean of latent distribution (batch_size, latent_dim).
                - logvar (torch.Tensor): Log variance of latent distribution (batch_size, latent_dim).
        """
        batch_size = x.size(0)

        # --- Input Validation ---
        if x.shape[1] != self.input_dim or mask.shape[1] != self.input_dim:
             raise ValueError(f"Input tensor feature dimension ({x.shape[1]}) or mask dimension ({mask.shape[1]}) "
                              f"does not match encoder input_dim ({self.input_dim})")
        if x.shape != mask.shape:
             raise ValueError(f"Input tensor shape ({x.shape}) and mask shape ({mask.shape}) must match.")
        if x.ndim != 2 or mask.ndim != 2:
             raise ValueError(f"Input tensor and mask must be 2D (batch_size, input_dim). Got shapes {x.shape} and {mask.shape}")

        # Step 1: Reshape inputs for processing each feature independently
        # Flatten batch and feature dimensions: (B, D) -> (B*D, 1)
        x_flat = x.reshape(-1, 1)                                # Shape: (B*D, 1)

        # Step 2: Prepare feature embeddings and biases for each item in the flattened batch
        # Feature embeddings F_d: (D, K) -> (B*D, K) by repeating for each batch item

        # Efficient expansion using broadcasting
        F_embed = self.feature_embedding.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, self.code_dim) # Shape: (B*D, K)

        # Step 3: Construct input for the shared 'h' function for each feature instance
        # Input s_d = [x_d, F_d]
        h_input = torch.cat([x_flat, F_embed], dim=1)  # Shape: (B*D, 1 + K + 1)

        # Step 4: Apply the shared h network to each feature representation s_d
        h_out_flat = self.h_layer(h_input)                      # Shape: (B*D, K)

        # Step 5: Reshape back to (batch_size, num_features, code_dim)
        h_out = h_out_flat.view(batch_size, self.input_dim, self.code_dim)  # Shape: (B, D, K)

        # Step 6: Apply the mask. Zero out representations of missing features.
        mask_float = mask.float() 
        # Expand mask: (B, D) -> (B, D, 1) for broadcasting
        mask_exp = mask_float.unsqueeze(-1)                           # Shape: (B, D, 1)
        h_masked = h_out * mask_exp                             # Shape: (B, D, K)

        # Step 7: Aggregate over observed features (permutation-invariant function g)
        # Sum along the feature dimension (dim=1) --> combining Features Per Cell 
        c = h_masked.sum(dim=1)                                 # Shape: (B, K)

        # Step 8: Pass the aggregated representation 'c' through the final MLP 
        enc_out = self.encoder_mlp(c)                           # Shape: (B, 2*Z)

        # Step 9: Split the output into mean (mu) and log variance (logvar)
        mu, logvar = enc_out.chunk(2, dim=-1)                   # Shapes: (B, Z), (B, Z)

        return mu, logvar

class LinearDecoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int):
        """
        Simple linear decoder that directly maps from latent space to output space.
        
        Parameters:
          latent_dim (int): Dimension of latent space (Z).
          output_dim (int): Dimension of output space (D).
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Simple linear layer from latent space to output space
        self.linear = nn.Linear(latent_dim, output_dim)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder.
        
        Args:
            z (torch.Tensor): Latent vector (batch_size, latent_dim).
            
        Returns:
            torch.Tensor: Reconstructed data (batch_size, output_dim).
        """
        # Direct linear mapping from latent to output
        return self.linear(z)

class PARTIALVAE(BaseModuleClass):
    """
    Partial Variational autoencoder module for splicing data in scvi-tools.

    Parameters
    ----------
    n_input
        Number of splicing features (junctions).
    n_batch
        Number of batches; 0 = no batch correction.
    n_labels
        Number of labels; 0 = no label correction.
    n_hidden
        Hidden size for Encoder/Decoder combination (unused here).
    n_latent
        Dimension of latent space.
    n_continuous_cov
        Number of continuous covariates.
    n_cats_per_cov
        Categories per covariate list.
    dropout_rate
        Dropout rate for all layers.
    splice_likelihood
        One of:
        * "binomial": use binomial reconstruction loss.
        * "beta_binomial": use beta-binomial reconstruction loss.
    latent_distribution
        "normal" or "ln" latent prior.
    encode_covariates
        Whether to concatenate covariates to inputs.
    deeply_inject_covariates
        Whether to inject covariates in hidden layers.
    batch_representation
        "one-hot" or "embedding" batch encoding.
    use_batch_norm
        Where to apply batch norm: "none", "encoder", "decoder", "both".
    use_layer_norm
        Where to apply layer norm.
    extra_payload_autotune
        Return extra payload for autotune.
    code_dim
        Dimensionality of feature embeddings.
    h_hidden_dim
        Hidden size for shared h-layer.
    encoder_hidden_dim
        Hidden size for encoder final MLP.
    decoder_hidden_dim
        Hidden size for decoder processors.
    learn_concentration
        If True, learn beta-binomial concentration.
    """
    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_continuous_cov: int = 0,
        n_cats_per_cov: list[int] | None = None,
        dropout_rate: float = 0.1,
        splice_likelihood: Literal["binomial", "beta_binomial"] = "beta_binomial",
        latent_distribution: Literal["normal", "ln"] = "normal",
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = True,
        batch_representation: Literal["one-hot", "embedding"] = "one-hot",
        use_batch_norm: Literal["none", "encoder", "decoder", "both"] = "both",
        use_layer_norm: Literal["none", "encoder", "decoder", "both"] = "none",
        extra_payload_autotune: bool = False,
        code_dim: int = 32,
        h_hidden_dim: int = 64,
        encoder_hidden_dim: int = 128,
        learn_concentration: bool = True,
    ):
        super().__init__()

        # Store parameters
        self.splice_likelihood = splice_likelihood
        self.latent_distribution = latent_distribution
        self.extra_payload_autotune = extra_payload_autotune # what's this? 
        self.learn_concentration = learn_concentration
        self.input_dim = n_input

        # Concentration parameter for Beta-Binomial
        if learn_concentration:
            self.log_concentration = nn.Parameter(torch.tensor(0.0))
        else:
            self.log_concentration = None

        # Instantiate PartialEncoder/Decoder
        self.encoder = PartialEncoder(
            input_dim=n_input,
            code_dim=code_dim,
            h_hidden_dim=h_hidden_dim,
            encoder_hidden_dim=encoder_hidden_dim,
            latent_dim=n_latent,
            dropout_rate=dropout_rate,
        )
        self.decoder = LinearDecoder(
            latent_dim=n_latent,
            output_dim=n_input,
        )

    def initialize_feature_embedding_from_pca(self, pca_components: np.ndarray):

        """
        Inject PCA components into feature embedding of PartialEncoder.
        """

        if not isinstance(pca_components, np.ndarray):
            raise TypeError("pca_components must be a numpy array.")
    
        assert pca_components.shape == (self.input_dim, self.encoder.code_dim), \
            f"PCA shape {pca_components.shape} does not match model ({self.input_dim}, {self.encoder.code_dim})"

        with torch.no_grad():
            self.encoder.feature_embedding.copy_(
                torch.tensor(pca_components, dtype=self.encoder.feature_embedding.dtype)
            )

    def _get_inference_input(
        self,
        tensors: dict[str, torch.Tensor],
        full_forward_pass: bool = False,
    ) -> dict[str, torch.Tensor]:
        # only x and mask
        return {
            MODULE_KEYS.X_KEY: tensors[REGISTRY_KEYS.X_KEY],      # your input ratios
            "mask": tensors.get(REGISTRY_KEYS.PSI_MASK_KEY, None),
        }

    def _get_generative_input(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        # only z needed
        return {MODULE_KEYS.Z_KEY: inference_outputs["z"]}

    @auto_move_data
    def inference(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        n_samples: int = 1,
    ) -> dict[str, torch.Tensor]:
        # 1) encode to mu, raw_logvar
        mu, raw_logvar = self.encoder(x, mask)
        # 2) clamp logvar then build Normal
        logvar = torch.clamp(raw_logvar, min=-10.0, max=10.0)
        qz = Normal(mu, torch.exp(0.5 * logvar))
        # 3) sample / rsample
        if n_samples == 1:
            z = qz.rsample()
        else:
            z = qz.sample((n_samples,))
        # 4) return z, posterior params
        return {"z": z, "qz_m": mu, "qz_v": torch.exp(logvar)}
    
    @auto_move_data
    def generative(
        self,
        z: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # decode to logits → p
        reconstruction = self.decoder(z)
        return {"reconstruction": reconstruction}


    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor],
        generative_outputs: dict[str, torch.Tensor],
        kl_weight: float | torch.Tensor = 1.0,
    ) -> LossOutput:
        # 1) unpack data
        x = tensors[REGISTRY_KEYS.X_KEY]                      # (B, D) input ratios (unused here)
        mask = tensors.get(REGISTRY_KEYS.PSI_MASK_KEY, None)  # (B, D) binary
        junc = tensors["junction_counts"]                     # (B, D) successes
        clus = tensors["cluster_counts"]                      # (B, D) trials

        # 2) reconstruction
        p = generative_outputs["reconstruction"]  # (B, D) Bernoulli probabilities
        if self.splice_likelihood == "binomial":
            reconst = binomial_loss_function(
                logits=p,
                junction_counts=junc,
                cluster_counts=clus,
                n=x.numel(),
                k=x.shape[0],
                mask=mask,
            )
        else:
            concentration = torch.exp(self.log_concentration)
            reconst = beta_binomial_loss_function(
                logits=p,
                junction_counts=junc,
                cluster_counts=clus,
                n=x.numel(),
                k=x.shape[0],
                concentration=concentration,
                mask=mask,
            )
        # reconst is a scalar: -E_q [ log p(x|z) ]

        # 3) KL divergence
        mu = inference_outputs["qz_m"]   # (B, Z)
        var = inference_outputs["qz_v"]  # (B, Z) = σ²
        sigma = torch.sqrt(var)          # (B, Z)
        qz = Normal(mu, sigma)
        pz = Normal(torch.zeros_like(mu), torch.ones_like(sigma))
        kl_local = kl_divergence(qz, pz).sum(dim=1)  # (B,) sum over latent dims

        # 4) total ELBO
        #    average KL over batch, add to reconst
        total = reconst + kl_weight * kl_local.mean()

        # 5) wrap in LossOutput
        return LossOutput(
            loss=total,
            reconstruction_loss=reconst,
            kl_local=kl_local,
            n_obs_minibatch=x.shape[0],
        )

    @torch.inference_mode()
    def sample(self, tensors: Dict[str, torch.Tensor], n_samples: int = 1) -> torch.Tensor:
        _, gen_out = self.forward(tensors, compute_loss=False)
        return gen_out["reconstruction"].cpu() 