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
    n: int,
    k: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Reconstruction Binomial loss function for VAE.
    
    Parameters:
      logits: Reconstructed logits from decoder.
      junction_counts: Junction counts from data.
      cluster_counts: Cluster counts from data.
      n: Number of samples in dataset.
      k: Number of batches in dataloader.
      mask: Optional mask to exclude missing values.
      
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
    log_lik = log_lik * (float(n) / float(k))
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
    log_lik = log_lik * (float(n) / float(k))
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
        # Shape: (D, K)
        self.feature_embedding = nn.Parameter(torch.randn(input_dim, code_dim))

        # Learnable bias term per feature (b_d in paper notation)
        # Shape: (D, 1)
        self.feature_bias = nn.Parameter(torch.zeros(input_dim, 1))

        # Shared function h(.) applied to each feature representation s_d = [x_d, F_d, b_d]
        # Input dim: 1 (feature value) + K (embedding) + 1 (bias) = K + 2
        # Output dim: K (code_dim)
        self.h_layer = nn.Sequential(
            nn.Linear(1 + code_dim + 1, h_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate), # Added dropout
            nn.Linear(h_hidden_dim, code_dim),
            nn.ReLU() # ReLU after last linear is common in intermediate feature extractors
        )

        # MLP to map aggregated representation 'c' to latent distribution parameters
        # Input dim: K (code_dim)
        # Output dim: 2 * Z (for mu and logvar)
        self.encoder_mlp = nn.Sequential(
            nn.Linear(code_dim, encoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate), # Added dropout
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
        F_embed = self.feature_embedding.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, self.code_dim)
        b_embed = self.feature_bias.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 1)

        # Step 3: Construct input for the shared 'h' function for each feature instance
        h_input = torch.cat([x_flat, F_embed, b_embed], dim=1)

        # Step 4: Apply the shared h network to each feature representation s_d
        h_out_flat = self.h_layer(h_input)

        # Step 5: Reshape back to (batch_size, num_features, code_dim)
        h_out = h_out_flat.view(batch_size, self.input_dim, self.code_dim)

        # Step 6: Apply the mask. Zero out representations of missing features.
        mask_exp = mask.float().unsqueeze(-1)
        h_masked = h_out * mask_exp

        # Step 7: Aggregate over observed features
        c = h_masked.sum(dim=1)

        # Step 8: Pass the aggregated representation 'c' through the final MLP
        enc_out = self.encoder_mlp(c)

        # Step 9: Split the output into mean (mu) and log variance (logvar)
        mu, logvar = enc_out.chunk(2, dim=-1)

        return mu, logvar


class PartialDecoder(nn.Module):
    def __init__(self, latent_dim: int, decoder_hidden_dim: int, output_dim: int, code_dim: int, dropout_rate: float = 0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.code_dim = code_dim

        self.z_processor = nn.Sequential(
            nn.Linear(latent_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # Input: processed_z + F_d + b_d
        self.j_layer = nn.Sequential(
            nn.Linear(decoder_hidden_dim + code_dim + 1, decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(decoder_hidden_dim, 1) # Predict 1 value per feature
        )

    def forward(self, z: torch.Tensor, feature_embedding: nn.Parameter, feature_bias: nn.Parameter) -> torch.Tensor:
        batch_size = z.size(0)
        processed_z = self.z_processor(z)
        processed_z_expanded = processed_z.unsqueeze(1).expand(-1, self.output_dim, -1)
        F = feature_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        b = feature_bias.unsqueeze(0).expand(batch_size, -1, -1)

        j_input = torch.cat([processed_z_expanded, F, b], dim=2)
        j_out = self.j_layer(j_input.view(-1, j_input.shape[-1]))
        reconstruction = j_out.view(batch_size, self.output_dim)
        return reconstruction


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
        code_dim: int = 16,
        h_hidden_dim: int = 64,
        encoder_hidden_dim: int = 128,
        decoder_hidden_dim: int = 64,
        learn_concentration: bool = True,
    ):
        super().__init__()

        # Store parameters
        self.splice_likelihood = splice_likelihood
        self.latent_distribution = latent_distribution
        self.extra_payload_autotune = extra_payload_autotune
        self.learn_concentration = learn_concentration

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
        self.decoder = PartialDecoder(
            latent_dim=n_latent,
            decoder_hidden_dim=decoder_hidden_dim,
            output_dim=n_input,
            code_dim=code_dim,
            dropout_rate=dropout_rate,
        )

    def _get_inference_input(
        self,
        tensors: Dict[str, torch.Tensor | None],
        full_forward_pass: bool = False,
    ) -> Dict[str, torch.Tensor | None]:
        return {
            MODULE_KEYS.X_KEY: tensors[REGISTRY_KEYS.X_KEY],          # “x”
            MODULE_KEYS.BATCH_INDEX_KEY: tensors[REGISTRY_KEYS.BATCH_KEY],  # “batch_index”
            "mask": tensors.get(REGISTRY_KEYS.PSI_MASK_KEY, None),     # your optional mask
        }

    def _get_generative_input(
        self,
        tensors: Dict[str, torch.Tensor | None],
        inference_outputs: Dict[str, torch.Tensor | Distribution | None],
    ) -> Dict[str, torch.Tensor | None]:
        return {
            MODULE_KEYS.Z_KEY: inference_outputs[MODULE_KEYS.Z_KEY],   # “z”
            MODULE_KEYS.BATCH_INDEX_KEY: tensors[REGISTRY_KEYS.BATCH_KEY],  # “batch_index”
        }


    @auto_move_data
    def inference(
        self,
        x: torch.Tensor,
        batch_index: torch.Tensor,
        mask: torch.Tensor | None = None,
        n_samples: int = 1,
    ) -> dict[str, torch.Tensor | Normal]:
        mu, raw_logvar = self.encoder(x, mask)
        logvar = torch.clamp(raw_logvar, min=-10.0, max=20.0)
        qz = Normal(mu, torch.exp(0.5 * logvar))
        z = qz.rsample() if n_samples == 1 else qz.sample((n_samples,))
        return {MODULE_KEYS.Z_KEY: z, MODULE_KEYS.QZ_KEY: qz}
    
    @auto_move_data
    def generative(
        self,
        z: torch.Tensor,
        batch_index: torch.Tensor | None = None,
    ) -> dict[str, Distribution]:
        logits = self.decoder(z, self.encoder.feature_embedding, self.encoder.feature_bias)
        px = Bernoulli(logits=logits)
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))
        return {MODULE_KEYS.PX_KEY: px, MODULE_KEYS.PZ_KEY: pz}

    def loss(
        self,
        tensors: Dict[str, torch.Tensor],
        inference_outputs: Dict[str, torch.Tensor | Distribution],
        generative_outputs: Dict[str, Distribution],
        kl_weight: float | torch.Tensor = 1.0,
    ) -> LossOutput:
        x = tensors[REGISTRY_KEYS.X_KEY]
        mask = tensors.get(REGISTRY_KEYS.PSI_MASK_KEY, None)
        junc = tensors["junction_counts"]
        clus = tensors["cluster_counts"]

        qz = inference_outputs[MODULE_KEYS.QZ_KEY]
        px = generative_outputs[MODULE_KEYS.PX_KEY]

        # reconstruction
        n, k = x.numel(), x.shape[0]
        reconst = (
            binomial_loss_function(px.logits, junc, clus, n, k, mask)
            if self.splice_likelihood == "binomial"
            else beta_binomial_loss_function(px.logits, junc, clus, n, k, torch.exp(self.log_concentration), mask)
        )
        # KL
        mu, logvar = qz.loc, 2 * torch.log(qz.scale)
        kl_z = kl_divergence(Normal(mu, torch.exp(0.5 * logvar)), Normal(0, 1)).sum(dim=1)

        total = (reconst + kl_weight * kl_z).mean()
        return LossOutput(loss=total, reconstruction_loss=reconst, kl_local=kl_z, n_obs_minibatch=x.shape[0])

    @torch.inference_mode()
    def sample(self, tensors: Dict[str, torch.Tensor], n_samples: int = 1) -> torch.Tensor:
        _, gen_out = self.forward(tensors, compute_loss=False)
        return gen_out[MODULE_KEYS.PX_KEY].sample().cpu()