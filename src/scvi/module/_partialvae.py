from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli, kl_divergence
from torch.nn.functional import one_hot
import torch.nn.functional as F

from scvi import REGISTRY_KEYS, settings
from scvi.module._constants import MODULE_KEYS
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.nn import FCLayers


if TYPE_CHECKING:
    from collections.abc import Callable
    from torch.distributions import Distribution

logger = logging.getLogger(__name__)

from torch.special import gammaln

def group_logsumexp(A, B):
    """
    Computes the log-sum-exp of B, grouped by A.
    A is a sparse matrix (must be coalesced already!) with shape (P, G), where P is the number of variables (junctions)
    and G is the number of groups (ATSEs). B is a dense matrix with shape (N, P), where
    N is the number of samples (cells). The result is a dense matrix with shape (N, G).
    """
    idx_p, idx_g = A.indices()              # (2, P), since each var belongs to one group
    idx_p = idx_p.to(B.device)
    idx_g = idx_g.to(B.device)

    N, P = B.shape
    G = A.shape[1]

    # (N, P) → gather relevant group index for each variable
    group_idx = torch.empty(P, dtype=torch.long, device=B.device)
    group_idx[idx_p] = idx_g

    # Get max within each group
    max_B = torch.full((N, G), float('-inf'), device=B.device)
    max_B.scatter_reduce_(1, group_idx.expand(N, -1), B, reduce='amax')

    # Center B by group max
    B_centered = B - max_B.gather(1, group_idx.expand(N, -1))

    exp_B_centered = torch.exp(B_centered)

    # Sum exp over each group
    sum_exp = torch.zeros((N, G), device=B.device)
    sum_exp.scatter_add_(1, group_idx.expand(N, -1), exp_B_centered)

    return torch.log(sum_exp + 1e-8) + max_B

def subtract_group_logsumexp(A, B, group_logsumexp):
    """
    Subtracts the group log-sum-exp from each element in B, grouped by A.
    A is a sparse matrix (must be coalesced already!) with shape (P, G), where P is the number of variables (junctions)
    and G is the number of groups (ATSEs). B is a dense matrix with shape (N, P), where
    N is the number of samples (cells). The result is a dense matrix with shape (N, P).
    """
    idx_p, idx_g = A.indices()
    idx_g = idx_g.to(B.device)
    N = B.size(0)
    return B - group_logsumexp[:, idx_g]

def nbetaln(count, alpha):  
    """
    Computes log(count * beta(count, alpha)) == gammaln(count + 1) + gammaln(alpha) - gammaln(count + alpha). 
    Note this is zero when count == 0, which could potentially be used to improve efficiency.
    """
    return gammaln(alpha) + gammaln(count+1.) - gammaln(alpha + count)

def dirichlet_multinomial_likelihood(counts, atse_counts, junc2atse, alpha): 
    """
    Computes the log likelihood of the Dirichlet multinomial model.
    counts: (N, P) matrix of counts
    atse_counts: (N, G) matrix of counts for each ATSE (G is number of ATSEs)
    junc2atse: (P, G) sparse matrix of ATSE assignments
    alpha: (N, P) vector of Dirichlet parameters
    """

    idx_p, idx_g = junc2atse.indices()      # maps each junction p → group g
    idx_p = idx_p.to(counts.device)
    idx_g = idx_g.to(counts.device)
    device = counts.device
    junc2atse = junc2atse.to(device)
    alpha = alpha.to(device)

    N, P = counts.shape
    G = junc2atse.shape[1]

    true_atse = torch.zeros((N, G), 
                           dtype=atse_counts.dtype, 
                        device=counts.device)
    # for each junction p we scatter its repeated atse_counts into column g = idx_g[p]
    # since they're all identical within a group, amax just picks that constant
    true_atse.scatter_reduce_(
        dim=1,
        index=idx_g.expand(N, -1),    # shape (N, P)
        src=atse_counts,               # shape (N, P)
        reduce="amax"
    )

    atse_counts = true_atse
    alpha_sums = (alpha @ junc2atse)
    per_atse = nbetaln(atse_counts, alpha_sums).sum(dim=1).mean()  # sum over ATSEs, mean over cells
    per_junc = nbetaln(counts,alpha).sum(dim=1).mean()  # sum over ATSEs, mean over cells
    return per_atse - per_junc


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


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable


class PartialEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        code_dim: int,
        h_hidden_dim: int,
        encoder_hidden_dim: int,
        latent_dim: int,
        dropout_rate: float = 0.0,
        n_cat_list: Iterable[int] | None = None,
        n_cont: int = 0,
        inject_covariates: bool = True,
        n_heads: int = 4,              # how many attention heads
        ff_dim: int | None = None,     # inner dim of the FFN
    ):
        super().__init__()
        # keep track of how many one-hot cats and cont features
        self.n_cat_list = [n for n in (n_cat_list or []) if n > 1]
        self.n_cont = n_cont
        self.inject_covariates = inject_covariates
        total_cov = sum(self.n_cat_list) + self.n_cont
        self.code_dim = code_dim

        ff_dim = ff_dim or (code_dim * 4)

        # replace masked‐mean with a transformer encoder layer
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=code_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout_rate,
            activation="relu",
            batch_first=True,  # so it accepts (B, D, C)
        )


        # per-feature embeddings
        self.feature_embedding = nn.Parameter(torch.randn(input_dim, code_dim))

        # shared h-network only sees [x_d, F_d]
        in_dim = 1 + code_dim
        self.h_layer = nn.Sequential(
            nn.Linear(in_dim, h_hidden_dim),
            nn.LayerNorm(h_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h_hidden_dim, code_dim),
            nn.LayerNorm(code_dim),
            nn.ReLU(),
        )

        # ——— replace manual MLP with FCLayers ———
        # note: n_in = code_dim; FCLayers will handle covariates internally
        self.encoder_mlp = FCLayers(
            n_in=code_dim,
            n_out=2 * latent_dim,
            n_cat_list=n_cat_list or [],
            n_cont=n_cont,
            n_layers=2,                    # [code_dim] -> [encoder_hidden_dim] -> [2*latent_dim]
            n_hidden=encoder_hidden_dim,
            dropout_rate=dropout_rate,
            use_batch_norm=False,          # your original used LayerNorm not BatchNorm
            use_layer_norm=True,
            inject_covariates=inject_covariates,
        )

    def forward(
        self,
        x: torch.Tensor,                # (B, D)
        mask: torch.Tensor,             # (B, D)
        *cat_list: torch.Tensor,        # each (B,1)
        cont: torch.Tensor | None = None,  # (B, n_cont)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, D = x.shape

        # --- step 1: flatten features for per-feature h_layer ---
        x_flat = x.reshape(-1, 1)  # (B*D,1)
        F_embed = (
            self.feature_embedding
                .unsqueeze(0)       # (1,D,code_dim)
                .expand(B, D, -1)    # (B,D,code_dim)
                .reshape(-1, self.feature_embedding.size(1))  # (B*D,code_dim)
        )

        # --- step 2: shared h_layer on [x_flat, F_embed] only ---
        h_in = torch.cat([x_flat, F_embed], dim=1)  # (B*D, 1+code_dim)
        h_out = self.h_layer(h_in).view(B, D, -1)    # (B, D, code_dim)

        # --- step 3: transformer‐based aggregation ---
        # build padding mask: True=positions to ignore
        # nn.TransformerEncoderLayer with batch_first=True wants a (B, D, C) input
        # and src_key_padding_mask of shape (B, D) with True where we want to mask
        padding_mask = (mask == 0)

        # apply the transformer layer
        # output shape is (B, D, code_dim)
        #tr_out = self.transformer_layer(h_out, src_key_padding_mask=padding_mask)
        tr_out = h_out
        # now pool across the D dimension (e.g. simple masked mean again)
        mask_exp = mask.unsqueeze(-1).float()                 # (B, D, 1)
        c = (tr_out * mask_exp).sum(dim=1) / (mask_exp.sum(dim=1) + 1e-8)

        # --- step 4: now pass the output into our mlp (FC layers handles cont and cat covariates automatically) ---
        mu_logvar = self.encoder_mlp(c, *cat_list, cont=cont)  # -> (B, 2*latent_dim)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        return mu, logvar


class LinearDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        n_cat_list: Iterable[int] | None = None,
        n_cont: int = 0,
    ):
        super().__init__()
        # track covariates exactly as in encoder
        self.n_cat_list = [n for n in (n_cat_list or []) if n > 1]
        self.n_cont     = n_cont
        total_cov       = sum(self.n_cat_list) + self.n_cont

        # single linear layer: input = [z, covariates]
        self.linear = nn.Linear(latent_dim + total_cov, output_dim)

    def forward(
        self,
        z: torch.Tensor,              # (B, latent_dim)
        *cat_list: torch.Tensor,      # each (B,1)
        cont: torch.Tensor | None = None,  # (B, n_cont)
    ) -> torch.Tensor:
        
        # if self.n_cat_list:
        #     batch_idx = cat_list[0]
        #     print("▶ LinearDecoder: batch_index unique →", torch.unique(batch_idx).tolist())
        # if cont is not None:
        #     print("▶ LinearDecoder: cont_covs shape →", cont.shape)

        # rebuild the same covariate concatenation
        covs: list[torch.Tensor] = []
        if cont is not None:
            covs.append(cont)
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            # one-hot encode each categorical and append
            oh = F.one_hot(cat.squeeze(-1), n_cat).float()
            covs.append(oh)
        # final input to linear: (B, latent_dim + total_cov)
        inp = torch.cat([z, *covs], dim=1) if covs else z
        return self.linear(inp)




# class PartialEncoder(nn.Module):
#     def __init__(self, input_dim: int, h_hidden_dim: int, encoder_hidden_dim: int, 
#                  latent_dim: int, code_dim: int, dropout_rate: float = 0.0):
#         """
#         Encoder network inspired by PointNet for partially observed data.

#         Processes each observed feature individually using a shared network ('h_layer')
#         combined with learnable feature embeddings and biases, then aggregates
#         the results before mapping to the latent space.

#         Parameters:
#           input_dim (int): Dimension of input features (D). Number of junctions/features.
#           h_hidden_dim (int): Hidden dimension for the shared 'h_layer'.
#                            (Replaces the misuse of num_hidden_layers in the original h_layer definition).
#           encoder_hidden_dim (int): Hidden dimension for the final 'encoder_mlp'.
#                                  (Replaces the hardcoded 256 in the original encoder_mlp).
#           latent_dim (int): Dimension of latent space (Z).
#           code_dim (int): Dimension of feature embeddings and intermediate representations (K).
#           dropout_rate (float): Dropout rate for regularization applied within h_layer and encoder_mlp.
#         """
#         super().__init__()
#         self.input_dim = input_dim
#         self.code_dim = code_dim
#         self.latent_dim = latent_dim

#         # Learnable feature embedding (F_d in paper notation)
#         self.feature_embedding = nn.Parameter(torch.randn(input_dim, code_dim)) #D by K initialized via PCA 

#         # Shared function h(.) applied to each feature representation s_d = [x_d, F_d]
#         # Input dim: 1 (feature value) + K (embedding) = K + 1
#         # Output dim: K (code_dim)
#         self.h_layer = nn.Sequential(
#             nn.Linear(1 + code_dim, h_hidden_dim),
#             nn.LayerNorm(h_hidden_dim),           
#             nn.ReLU(),
#             nn.Dropout(dropout_rate), 
#             nn.Linear(h_hidden_dim, code_dim),
#             nn.LayerNorm(code_dim),   
#             nn.ReLU()
#         )

#         # MLP to map aggregated representation 'c' to latent distribution parameters
#         # Input dim: K (code_dim)
#         # Output dim: 2 * Z (for mu and logvar)
#         self.encoder_mlp = nn.Sequential(
#             nn.Linear(code_dim, encoder_hidden_dim),
#             nn.LayerNorm(encoder_hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate), 
#             nn.Linear(encoder_hidden_dim, 2 * latent_dim) # outputs both mu and logvar
#         )

#     def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#         """
#         Forward pass of the encoder.

#         Args:
#             x (torch.Tensor): Input data (batch_size, input_dim). Missing values can be anything (e.g., 0, NaN),
#                               as they will be masked out based on the 'mask' tensor.
#                               It's crucial that the *observed* values in x are the actual measurements.
#             mask (torch.Tensor): Binary mask (batch_size, input_dim). 1 indicates observed, 0 indicates missing.
#                                Must be float or long/int and compatible with multiplication.

#         Returns:
#             tuple[torch.Tensor, torch.Tensor]:
#                 - mu (torch.Tensor): Mean of latent distribution (batch_size, latent_dim).
#                 - logvar (torch.Tensor): Log variance of latent distribution (batch_size, latent_dim).
#         """
#         batch_size = x.size(0)

#         # --- Input Validation ---
#         if x.shape[1] != self.input_dim or mask.shape[1] != self.input_dim:
#              raise ValueError(f"Input tensor feature dimension ({x.shape[1]}) or mask dimension ({mask.shape[1]}) "
#                               f"does not match encoder input_dim ({self.input_dim})")
#         if x.shape != mask.shape:
#              raise ValueError(f"Input tensor shape ({x.shape}) and mask shape ({mask.shape}) must match.")
#         if x.ndim != 2 or mask.ndim != 2:
#              raise ValueError(f"Input tensor and mask must be 2D (batch_size, input_dim). Got shapes {x.shape} and {mask.shape}")

#         # Step 1: Reshape inputs for processing each feature independently
#         # Flatten batch and feature dimensions: (B, D) -> (B*D, 1)
#         x_flat = x.reshape(-1, 1)                                # Shape: (B*D, 1)

#         # Step 2: Prepare feature embeddings and biases for each item in the flattened batch
#         # Feature embeddings F_d: (D, K) -> (B*D, K) by repeating for each batch item

#         # Efficient expansion using broadcasting
#         F_embed = self.feature_embedding.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, self.code_dim) # Shape: (B*D, K)

#         # Step 3: Construct input for the shared 'h' function for each feature instance
#         # Input s_d = [x_d, F_d]
#         h_input = torch.cat([x_flat, F_embed], dim=1)  # Shape: (B*D, 1 + K + 1)

#         # Step 4: Apply the shared h network to each feature representation s_d
#         h_out_flat = self.h_layer(h_input)                      # Shape: (B*D, K)

#         # Step 5: Reshape back to (batch_size, num_features, code_dim)
#         h_out = h_out_flat.view(batch_size, self.input_dim, self.code_dim)  # Shape: (B, D, K)

#         # Step 6: Apply the mask. Zero out representations of missing features.
#         mask_float = mask.float() 
#         # Expand mask: (B, D) -> (B, D, 1) for broadcasting
#         mask_exp = mask_float.unsqueeze(-1)                           # Shape: (B, D, 1)
#         h_masked = h_out * mask_exp                             # Shape: (B, D, K)

#         # Step 7: Aggregate over observed features (permutation-invariant function g)
#         # Sum along the feature dimension (dim=1) --> combining Features Per Cell 
#         c = h_masked.sum(dim=1)                                 # Shape: (B, K)

#         # Step 8: Pass the aggregated representation 'c' through the final MLP 
#         enc_out = self.encoder_mlp(c)                           # Shape: (B, 2*Z)

#         # Step 9: Split the output into mean (mu) and log variance (logvar)
#         mu, logvar = enc_out.chunk(2, dim=-1)                   # Shapes: (B, Z), (B, Z)

#         return mu, logvar

# class LinearDecoder(nn.Module):
#     def __init__(self, latent_dim: int, output_dim: int):
#         """
#         Simple linear decoder that directly maps from latent space to output space.
        
#         Parameters:
#           latent_dim (int): Dimension of latent space (Z).
#           output_dim (int): Dimension of output space (D).
#         """
#         super().__init__()
#         self.latent_dim = latent_dim
#         self.output_dim = output_dim
        
#         # Simple linear layer from latent space to output space
#         self.linear = nn.Linear(latent_dim, output_dim)
    
#     def forward(self, z: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass of the decoder.
        
#         Args:
#             z (torch.Tensor): Latent vector (batch_size, latent_dim).
            
#         Returns:
#             torch.Tensor: Reconstructed data (batch_size, output_dim).
#         """
#         # Direct linear mapping from latent to output
#         return self.linear(z)

class PARTIALVAE(BaseModuleClass):
    """
    Partial Variational autoencoder module for splicing data in scvi-tools.

    Parameters
    ----------
    n_input : int
        Number of splicing features (junctions).
    n_batch : int
        Number of batches; 0 = no batch correction.
    n_labels : int
        Number of labels; 0 = no label correction.
    n_hidden : int
        Hidden size for internal representations (unused directly here).
    n_latent : int
        Dimension of the latent space Z.
    n_continuous_cov : int
        Number of continuous covariates to include.
    n_cats_per_cov : list[int] | None
        Number of categories for each categorical covariate 
        (e.g. `[n_batch] + other_cats`).
    dropout_rate : float
        Dropout rate applied in encoder/decoder.
    splice_likelihood : {"binomial","beta_binomial", "dirichlet_multinomial"}
        Which reconstruction loss to use.
    latent_distribution : {"normal","ln"}
        Latent prior type.
    encode_covariates : bool
        Whether to concatenate covariates to **encoder** inputs.
    deeply_inject_covariates : bool
        Whether to concatenate covariates to **decoder** inputs.
    batch_representation : {"one-hot","embedding"}
        How to treat the batch covariate internally.
    use_batch_norm : {"none","encoder","decoder","both"}
        Where to apply batch normalization.
    use_layer_norm : {"none","encoder","decoder","both"}
        Where to apply layer normalization.
    extra_payload_autotune : bool
        Return extra payload for autotune (advanced).
    code_dim : int
        Dimensionality of per-feature embeddings in the encoder.
    h_hidden_dim : int
        Hidden size for the shared “h” layer in the encoder.
    encoder_hidden_dim : int
        Hidden size for the final encoder MLP.
    learn_concentration : bool
        If True, learn Beta‐Binomial concentration parameter.
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
        learn_concentration: bool = True,
    ):
        super().__init__()

        # Store high-level flags
        self.encode_covariates = encode_covariates
        self.deeply_inject_covariates = deeply_inject_covariates

        # Reconstruction / prior settings
        self.splice_likelihood = splice_likelihood
        self.latent_distribution = latent_distribution


        self.learn_concentration = learn_concentration
        self.extra_payload_autotune = extra_payload_autotune

        # AnnData dimensions
        self.input_dim = n_input
        self.n_continuous_cov = n_continuous_cov
        self.n_cats_per_cov = n_cats_per_cov or []

        cat_list = [n_batch] + list(n_cats_per_cov) if n_cats_per_cov is not None else []
        encoder_cat_list = cat_list if encode_covariates else None

        # ——— shared concentration parameter ϕ ———
        # initialize to log(100) ≈ 4.6
        if learn_concentration:
            self.log_concentration = nn.Parameter(torch.tensor(4.6))
        else:
            self.log_concentration = None

        # Instantiate encoder + decoder, passing covariate specs
        self.encoder = PartialEncoder(
            input_dim=n_input,
            code_dim=code_dim,
            h_hidden_dim=h_hidden_dim,
            encoder_hidden_dim=encoder_hidden_dim,
            latent_dim=n_latent,
            dropout_rate=dropout_rate,
            n_cat_list=encoder_cat_list,
            n_cont=n_continuous_cov,
            inject_covariates=encode_covariates,
        )
        if latent_distribution == "ln":
            self.encoder.z_transformation = nn.Softmax(dim=-1)
        else:
            self.encoder.z_transformation = lambda x: x

        self.decoder = LinearDecoder(
            latent_dim=n_latent,
            output_dim=n_input,
            n_cat_list=cat_list,
            n_cont=n_continuous_cov,
        )

    @auto_move_data
    def _get_inference_input(self, tensors):
        return {
            MODULE_KEYS.X_KEY: tensors[REGISTRY_KEYS.X_KEY],        # your junction ratios
            "mask":           tensors.get(REGISTRY_KEYS.PSI_MASK_KEY),
            "batch_index":    tensors[REGISTRY_KEYS.BATCH_KEY],     # integer batch code
            "cat_covs":       tensors.get(REGISTRY_KEYS.CAT_COVS_KEY),  # stacked one‐hot columns
            "cont_covs":      tensors.get(REGISTRY_KEYS.CONT_COVS_KEY), # numeric covariates
        }

    @auto_move_data
    def inference(
        self,
        x,
        mask,
        batch_index,
        cat_covs=None,
        cont_covs=None,
        n_samples=1,
    ) -> dict[str, torch.Tensor]:
        # 1) prepare encoder inputs (concatenate continuous covariates if requested)
        if cont_covs is not None and self.encode_covariates:
            x_input = torch.cat([x, cont_covs], dim=-1)
        else:
            x_input = x

        # 2) split out any one-hot categorical covariates
        if cat_covs is not None and self.encode_covariates:
            cat_list = torch.split(cat_covs, 1, dim=1)
        else:
            cat_list = ()

        # 3) run encoder → returns (mu, raw_logvar)
        #    we still pass batch_index as the first “category”
        mu, raw_logvar = self.encoder(
            x_input, mask, batch_index, *cat_list, cont=cont_covs
        )

        # --- ensure logvar is in a safe range ---
        # prevents under/overflow in exp()
        logvar = torch.clamp(raw_logvar, min=-10.0, max=10.0)

        # 4) build the posterior Normal
        var   = torch.exp(logvar)         # variance = exp(logvar) > 0
        sigma = torch.sqrt(var)           # standard deviation
        qz    = Normal(mu, sigma)

        # 5) sample (rsample so that gradients flow)
        if n_samples == 1:
            z = qz.rsample()
        else:
            z = qz.sample((n_samples,))

        # 6) apply any “z_transformation” (e.g. identity or softmax)
        z = self.encoder.z_transformation(z)

        return {
            "z":    z,      # (B, Z) or (n_samples, B, Z)
            "qz_m": mu,     # posterior means
            "qz_v": var,    # **variance**, not logvar
            "x":    x,
        }


    def _get_generative_input(self, tensors, inference_outputs):
        return {
            MODULE_KEYS.Z_KEY:   inference_outputs["z"],
            "qz_m":              inference_outputs["qz_m"],      # <<< add this
            "batch_index":       tensors[REGISTRY_KEYS.BATCH_KEY],
            "cat_covs":          tensors.get(REGISTRY_KEYS.CAT_COVS_KEY),
            "cont_covs":         tensors.get(REGISTRY_KEYS.CONT_COVS_KEY),
        }

    @auto_move_data
    def generative(
        self,
        z,
        qz_m,
        batch_index,
        cat_covs=None,
        cont_covs=None,
        use_z_mean=False,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        # if you want to use the posterior mean:
        if use_z_mean:
            z = qz_m

        # again split cat_covs → tuple
        if cat_covs is not None:
            cat_list = torch.split(cat_covs, 1, dim=1)
        else:
            cat_list = ()

        # prepare decoder input by concatenating cont_covs if desired
        decoder_input = (
            torch.cat([z, cont_covs], dim=-1)
            if (cont_covs is not None and self.deeply_inject_covariates)
            else z
        )

        # finally call your decoder, passing batch_index first:
        reconstruction = self.decoder(decoder_input, batch_index, *cat_list, cont=cont_covs)

        return {"reconstruction": reconstruction}   # now shape (J,)



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

        elif self.splice_likelihood == "beta_binomial":
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

        elif self.splice_likelihood == "dirichlet_multinomial":
            # --- prepare inputs ---
            # junc:        (B,D)  = counts per junction per cell
            # clus:        (B,D)  = repeated ATSE totals per junction per cell
            # self.junc2atse: sparse (D, G)

            # decoder output p is currently a “logit” tensor of shape (B,D)
            logits = p

            # we need atse_counts of shape (B, D), which you already have in `clus`.
            atse_counts = clus

            # concentration scalar fixed at 10.0 for now
            
            concentration = torch.exp(self.log_concentration)
            print(f"[DM ϕ] concentration = {concentration.item():.4f}")

            # --- turn logits into per‐group softmax‐logits ---
            # group_logsumexp and subtract_group_logsumexp expect (N,P)→(N,G)
            lse = group_logsumexp(self.junc2atse, logits)            # → (N,G)
            softmax_logits = subtract_group_logsumexp(
                self.junc2atse, logits, lse
            )                                                           # → (N,P)

            # build alpha = conc * exp(softmax_logits)
            alpha = concentration * torch.exp(softmax_logits)

            # compute the DM log‐likelihood
            ll = dirichlet_multinomial_likelihood(
                counts=junc,
                atse_counts=atse_counts,
                junc2atse=self.junc2atse,
                alpha=alpha,
            )
            # negative mean‐log‐lik
            reconst = -ll

        else:
            raise ValueError(f"Unknown splice_likelihood={self.splice_likelihood}")

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