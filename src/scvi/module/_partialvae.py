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

def dirichlet_multinomial_likelihood(
    counts: torch.Tensor,        # (N, P)
    atse_counts: torch.Tensor,   # (N, G)
    junc2atse: torch.sparse.Tensor,  # (P, G)
    alpha: torch.Tensor,         # (N, P)
    mask: torch.Tensor | None = None,  # optional (N, P)
) -> torch.Tensor:
    """
    Computes the batch-averaged Dirichlet–multinomial log-likelihood,
    **masking out** any junctions where mask==0 and any ATSEs where atse_counts==0.

    Returns:
        scalar = mean_over_cells( sum_per_ATSE(LL_atse * ATSE_mask)
                                - sum_per_junction(LL_junc * junc_mask) )
    """
    device = counts.device
    atse_counts = atse_counts.to(device)
    alpha       = alpha.to(device)
    mask        = mask.to(device) if mask is not None else None
    junc2atse   = junc2atse.to(device)
    N, P = counts.shape
    G     = junc2atse.shape[1]

    idx_p, idx_g = junc2atse.indices()
    idx_p = idx_p.to(counts.device)
    idx_g = idx_g.to(counts.device)
    idx_p, idx_g = junc2atse.indices()
    idx_p = idx_p.to(device)
    idx_g = idx_g.to(device)             # each length ≈ P

    true_atse = torch.zeros((N, G), device=device, dtype=atse_counts.dtype)
    true_atse.scatter_reduce_(
        dim=1,
        index=idx_g.expand(N, -1),                    # (N, P)
        src=atse_counts,                              # (N, P)
        reduce="amax",
    )
    atse_counts = true_atse                           # (N, G)

    # 2) compute group-sums of alpha → (N, G)
    alpha_sums = alpha @ junc2atse                    # (N, G)

    # 3) per-ATSE log-likelihoods
    ll_atse = nbetaln(atse_counts, alpha_sums)        # (N, G)
    # mask out any ATSEs with zero count
    atse_mask = (atse_counts > 0).to(ll_atse.dtype)   # (N, G)
    ll_atse = ll_atse * atse_mask                     # zeros where atse_count==0
    # sum per ATSE and then average over cells
    per_atse = ll_atse.sum(dim=1).mean()              # scalar

    # 4) per-junction log-likelihoods
    ll_junc = nbetaln(counts, alpha)                  # (N, P)
    if mask is not None:
        mask_bool = mask.to(torch.bool)               # (N, P)
        ll_junc = ll_junc * mask_bool.to(ll_junc.dtype)
    # sum per junction and then average over cells
    per_junc = ll_junc.sum(dim=1).mean()              # scalar

    # 5) return batch-averaged difference
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
from torch import nn
from collections.abc import Iterable
from scvi.nn import FCLayers

#proper EDDI partial encoder. only passes observed junctions and their embeddings into the h layer. 
class PartialEncoderEDDI(nn.Module):
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
        pool_mode: Literal["mean","sum"] = "mean",
    ):
        super().__init__()
        # keep track of how many one-hot cats and cont features
        self.n_cat_list = [n for n in (n_cat_list or []) if n > 1]
        self.n_cont = n_cont
        self.inject_covariates = inject_covariates
        total_cov = sum(self.n_cat_list) + self.n_cont
        self.code_dim = code_dim
        self.pool_mode = pool_mode

        print(f"pool_mode: {pool_mode}")

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
        x: torch.Tensor,       # (B, D)
        mask: torch.Tensor,    # (B, D)
        *cat_list: torch.Tensor,
        cont: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, D = x.shape
        outputs = []

        for b in range(B):
            obs_idx = mask[b].bool()  # (D,)
            x_obs = x[b, obs_idx]     # (n_obs,)
            F_obs = self.feature_embedding[obs_idx]  # (n_obs, code_dim)

            h_in = torch.cat([x_obs.unsqueeze(-1), F_obs], dim=1)  # (n_obs, 1 + code_dim)
            h_out = self.h_layer(h_in)  # (n_obs, code_dim)

            summed = h_out.sum(dim=0)

            if self.pool_mode == "mean":
                num_obs = obs_idx.sum().clamp(min=1)
                c = summed / num_obs
            else:
                c = summed   # (code_dim,)

            outputs.append(c)

        c = torch.stack(outputs, dim=0)  # (B, code_dim)

        mu_logvar = self.encoder_mlp(c, *cat_list, cont=cont)  # (B, 2*latent_dim)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        return mu, logvar


import torch
from torch import nn
import torch.nn.functional as F
from scvi.nn import FCLayers
from typing import Iterable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Literal
from scvi.nn import FCLayers

class PartialEncoderEDDIATSE(nn.Module):
    """
    EDDI Partial Encoder with learnable ATSE embeddings (no one-hot).

    - Per-junction embedding: (J, D)
    - Per-ATSE embedding table: (A, Ae) created in `register_junc2atse`
    - h_layer sees [psi (1) | junction_embed (D) | atse_embed (Ae)]
    - Per-cell Python loop preserved.
    """
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
        pool_mode: Literal["mean","sum"] = "mean",
        atse_embedding_dimension: int = 16,
    ):
        super().__init__()
        self.code_dim = code_dim
        self.h_hidden_dim = h_hidden_dim
        self.dropout_rate = dropout_rate
        self.pool_mode = pool_mode
        self.n_cat_list = [n for n in (n_cat_list or []) if n > 1]
        self.n_cont = n_cont
        self.inject_covariates = inject_covariates
        self.atse_embedding_dimension = int(atse_embedding_dimension)

        # Per-junction embeddings
        self.feature_embedding = nn.Parameter(torch.randn(input_dim, code_dim))

        # Placeholder h_layer; rebuilt after register_junc2atse() to include Ae
        self.h_layer = self._make_h_layer(n_atse_embed=0)

        # Encoder MLP (handles covariates internally)
        self.encoder_mlp = FCLayers(
            n_in=code_dim,
            n_out=2 * latent_dim,
            n_cat_list=n_cat_list or [],
            n_cont=n_cont,
            n_layers=2,
            n_hidden=encoder_hidden_dim,
            dropout_rate=dropout_rate,
            use_batch_norm=False,
            use_layer_norm=True,
            inject_covariates=inject_covariates,
        )

        # Sparse mapping rows=junctions (J), cols=ATSEs (A)
        self.register_buffer("junc2atse", torch.sparse_coo_tensor(size=(input_dim, 0)))
        self.register_buffer("atse_index_per_j", torch.empty(0, dtype=torch.long))
        self.n_atse = 0

        # Will be created in register_junc2atse
        self.atse_embedding: nn.Parameter | None = None

    def _make_h_layer(self, n_atse_embed: int) -> nn.Sequential:
        in_dim = 1 + self.code_dim + n_atse_embed
        return nn.Sequential(
            nn.Linear(in_dim, self.h_hidden_dim),
            nn.LayerNorm(self.h_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.h_hidden_dim, self.code_dim),
            nn.LayerNorm(self.code_dim),
            nn.ReLU(),
        )

    @torch.no_grad()
    def register_junc2atse(self, junc2atse: torch.sparse.Tensor):
        """
        Provide the (J x A) sparse mapping (one nonzero per junction).
        Builds ATSE embedding table (A, Ae) and rebuilds h_layer to accept Ae dims.
        Also precomputes a dense index vector mapping each junction→ATSE id.
        """
        j2a = junc2atse.coalesce().to(self.feature_embedding.device)
        self.register_buffer("junc2atse", j2a)

        # Count ATSEs
        _, idx_g = j2a.indices()
        self.n_atse = int(idx_g.max().item()) + 1 if idx_g.numel() > 0 else 0

        # Create (or clear) ATSE embeddings
        if self.n_atse == 0:
            self.atse_embedding = None
            self.register_buffer("atse_index_per_j", torch.zeros(self.feature_embedding.shape[0], dtype=torch.long, device=j2a.device))
            self.h_layer = self._make_h_layer(n_atse_embed=0)
        else:
            self.atse_embedding = nn.Parameter(
                torch.randn(self.n_atse, self.atse_embedding_dimension, device=j2a.device)
            )
            # Build a dense junction→ATSE lookup (size J)
            J = self.feature_embedding.shape[0]
            idx_p, idx_g = j2a.indices()
            atse_idx = torch.zeros(J, dtype=torch.long, device=j2a.device)
            atse_idx[idx_p] = idx_g
            self.register_buffer("atse_index_per_j", atse_idx)
            # Now h_layer expects Ae inputs
            self.h_layer = self._make_h_layer(n_atse_embed=self.atse_embedding_dimension)

    def forward(
        self,
        x: torch.Tensor,       # (B, J)
        mask: torch.Tensor,    # (B, J)
        *cat_list: torch.Tensor,
        cont: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.atse_embedding is None:
            raise RuntimeError("ATSE embeddings not initialized. Call `register_junc2atse(...)` first.")

        B, J = x.shape
        D = self.code_dim
        device = x.device
        outputs = []

        for b in range(B):
            obs_idx = torch.where(mask[b].bool())[0]  # (n_obs,)
            if obs_idx.numel() == 0:
                outputs.append(torch.zeros(D, device=device))
                continue

            x_obs = x[b, obs_idx]                           # (n_obs,)
            F_obs = self.feature_embedding[obs_idx]         # (n_obs, D)

            # ATSE embedding for each observed junction
            atse_ids = self.atse_index_per_j[obs_idx]       # (n_obs,)
            Ae_obs   = self.atse_embedding[atse_ids]        # (n_obs, Ae)

            h_in  = torch.cat([x_obs.unsqueeze(-1), F_obs, Ae_obs], dim=1)  # (n_obs, 1+D+Ae)
            h_out = self.h_layer(h_in)                       # (n_obs, D)

            summed = h_out.sum(dim=0)
            if self.pool_mode == "mean":
                c = summed / obs_idx.numel()
            else:
                c = summed
            outputs.append(c)

        c = torch.stack(outputs, dim=0)  # (B, D)

        mu_logvar = self.encoder_mlp(c, *cat_list, cont=cont)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        return mu, logvar


import torch
import torch.nn.functional as F
from torch import nn
from typing import Iterable, Literal
from scvi.nn import FCLayers


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable
from scvi.nn import FCLayers

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable
from scvi.nn import FCLayers

class PartialEncoderWeightedSumEDDIMultiWeightATSE(nn.Module):
    """
    EDDI-style multi-head weighted-sum encoder with learnable ATSE embeddings.

    - Per-junction embedding: (J, D)
    - Per-ATSE embedding table: (A, Ae) created in `register_junc2atse`
    - h_layer computes per-junction codes from [psi | junction_embed]
    - Gate sees [h_out (D) | atse_embed (Ae)]  (Ae known at init → no gate rebuild)
    - Per-cell Python loop preserved.
    """
    def __init__(
        self,
        input_dim: int,           # J
        code_dim: int,            # D
        h_hidden_dim: int,
        encoder_hidden_dim: int,
        latent_dim: int,
        num_weight_vectors: int = 4,  # W
        dropout_rate: float = 0.0,
        n_cat_list: Iterable[int] | None = None,
        n_cont: int = 0,
        inject_covariates: bool = True,
        temperature_value: float = 1.0,
        temperature_fixed: bool = True,
        atse_embedding_dimension: int = 16,
    ):
        super().__init__()
        self.code_dim = code_dim
        self.num_weight_vectors = num_weight_vectors
        self.temperature_value = float(temperature_value)
        self.temperature_fixed = bool(temperature_fixed)
        self.n_cat_list = [n for n in (n_cat_list or []) if n > 1]
        self.n_cont = n_cont
        self.inject_covariates = inject_covariates
        self.atse_embedding_dimension = int(atse_embedding_dimension)

        # Per-junction embeddings (J, D)
        self.feature_embedding = nn.Parameter(torch.randn(input_dim, code_dim))

        # h-layer: [psi (1) + junction-embedding (D)] -> D
        in_dim_h = 1 + code_dim
        self.h_layer = nn.Sequential(
            nn.Linear(in_dim_h, h_hidden_dim),
            nn.LayerNorm(h_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h_hidden_dim, code_dim),
            nn.LayerNorm(code_dim),
            nn.ReLU(),
        )

        # Gate net: input is [h_out (D) + ATSE emb (Ae)] — Ae is fixed by ctor, so no rebuild
        in_dim_gate = code_dim + self.atse_embedding_dimension
        hidden_gate = max(in_dim_gate // 2, 1)
        self.gate_net = nn.Sequential(
            nn.Linear(in_dim_gate, hidden_gate),
            nn.ReLU(),
            nn.Linear(hidden_gate, self.num_weight_vectors),
        )

        # Combiner: (W * D) -> D
        if self.num_weight_vectors == 1:
            self.combiner = nn.Identity()
        else:
            self.combiner = nn.Sequential(
                nn.Linear(self.num_weight_vectors * self.code_dim, self.code_dim),
                nn.LayerNorm(self.code_dim),
                nn.ReLU(),
            )

        # Final encoder MLP
        self.encoder_mlp = FCLayers(
            n_in=code_dim,
            n_out=2 * latent_dim,
            n_cat_list=n_cat_list or [],
            n_cont=n_cont,
            n_layers=2,
            n_hidden=encoder_hidden_dim,
            dropout_rate=dropout_rate,
            use_batch_norm=False,
            use_layer_norm=True,
            inject_covariates=inject_covariates,
        )

        # Sparse mapping J×A and dense junction→ATSE index
        self.register_buffer("junc2atse", torch.sparse_coo_tensor(size=(input_dim, 0)))
        self.register_buffer("atse_index_per_j", torch.empty(0, dtype=torch.long))
        self.n_atse = 0

        # ATSE embeddings created in register_junc2atse
        self.atse_embedding: nn.Parameter | None = None

    @torch.no_grad()
    def register_junc2atse(self, junc2atse: torch.sparse.Tensor):
        """
        Provide (J x A) sparse mapping (one nonzero per junction).
        Creates ATSE embedding table (A, Ae) and a dense junction→ATSE index.
        """
        j2a = junc2atse.coalesce().to(self.feature_embedding.device)
        self.register_buffer("junc2atse", j2a)

        _, idx_g = j2a.indices()
        self.n_atse = int(idx_g.max().item()) + 1 if idx_g.numel() > 0 else 0

        J = self.feature_embedding.shape[0]
        if self.n_atse == 0:
            self.atse_embedding = None
            self.register_buffer("atse_index_per_j", torch.zeros(J, dtype=torch.long, device=j2a.device))
        else:
            # ATSE embedding table (A, Ae)
            self.atse_embedding = nn.Parameter(
                torch.randn(self.n_atse, self.atse_embedding_dimension, device=j2a.device)
            )
            # Dense junction→ATSE lookup
            idx_p, idx_g = j2a.indices()
            atse_idx = torch.zeros(J, dtype=torch.long, device=j2a.device)
            atse_idx[idx_p] = idx_g
            self.register_buffer("atse_index_per_j", atse_idx)

    def forward(
        self,
        x: torch.Tensor,            # (B, J)
        mask: torch.Tensor,         # (B, J) 1=observed, 0=missing
        *cat_list: torch.Tensor,
        cont: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.atse_embedding is None:
            raise RuntimeError("ATSE embeddings not initialized. Call `register_junc2atse(...)` first.")

        B, J = x.shape
        D = self.code_dim
        W = self.num_weight_vectors
        device = x.device

        outputs = []

        # Optional warning for empty cells
        if (mask.sum(dim=1) == 0).any():
            bad = (mask.sum(dim=1) == 0).nonzero(as_tuple=True)[0]
            print(f"[Warning] Found {bad.numel()} cells in batch with 0 observed junctions: {bad.tolist()}")

        for b in range(B):
            obs_idx = mask[b].bool().nonzero(as_tuple=True)[0]  # (n_obs,)
            n_obs = obs_idx.numel()
            if n_obs == 0:
                outputs.append(torch.zeros(D, device=device))
                continue

            # Per-junction code
            x_obs = x[b, obs_idx]                    # (n_obs,)
            F_obs = self.feature_embedding[obs_idx]  # (n_obs, D)
            h_in  = torch.cat([x_obs.unsqueeze(-1), F_obs], dim=1)  # (n_obs, 1+D)
            h_out = self.h_layer(h_in)               # (n_obs, D)

            # Gate input with ATSE embedding
            atse_ids = self.atse_index_per_j[obs_idx]           # (n_obs,)
            Ae_obs   = self.atse_embedding[atse_ids]            # (n_obs, Ae)
            gate_in  = torch.cat([h_out, Ae_obs], dim=1)        # (n_obs, D+Ae)
            raw_gates = self.gate_net(gate_in)                   # (n_obs, W)

            # Temperature / neighbor scaling
            if self.temperature_fixed:
                logits = raw_gates * self.temperature_value
            else:
                scale = (float(n_obs)) ** -0.5
                logits = raw_gates * scale

            logits = logits.clamp(min=-10.0, max=10.0)          # numerical hygiene

            # Normalize over junctions (softmax along dim=0)
            weights = torch.softmax(logits, dim=0)              # (n_obs, W)

            # Weighted sums per head: (W, D)
            head_sums = (weights.T.unsqueeze(-1) * h_out.unsqueeze(0)).sum(dim=1)

            # Combine heads → (D,)
            combined = self.combiner(head_sums.reshape(-1))
            outputs.append(combined)

        c = torch.stack(outputs, dim=0)  # (B, D)

        mu_logvar = self.encoder_mlp(c, *cat_list, cont=cont)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        return mu, logvar





class PartialEncoderWeightedSumEDDIMultiWeight(nn.Module):
    """
    EDDI-style encoder that for each junction produces W weight vectors, yielding W separate
    weighted sums of per-junction codes. Those W head-sums are then combined via a learned
    linear combiner into a final code.

    Dimensions:
        B: batch size (cells)
        J: number of junctions
        D: code_dim (size of per-junction code vectors)
        W: number of weight vectors / heads
    """
    def __init__(
        self,
        input_dim: int,  # J
        code_dim: int,   # D
        h_hidden_dim: int,
        encoder_hidden_dim: int,
        latent_dim: int,
        num_weight_vectors: int = 4,  # W
        dropout_rate: float = 0.0,
        n_cat_list: Iterable[int] | None = None,
        n_cont: int = 0,
        inject_covariates: bool = True,
        temperature_value: float = 1.0,
        temperature_fixed: bool = True, #if temperature is fixed, it is fixed to the value of temperature_value
    ):
        super().__init__()
        self.n_cat_list = [n for n in (n_cat_list or []) if n > 1]
        self.n_cont = n_cont
        self.inject_covariates = inject_covariates
        self.code_dim = code_dim  # code dimension
        self.num_weight_vectors = num_weight_vectors
        self.temperature_value = temperature_value
        self.temperature_fixed = temperature_fixed


        # per-junction embeddings (J, D)
        self.feature_embedding = nn.Parameter(torch.randn(input_dim, self.code_dim))

        # h-layer: [psi, embedding] -> D
        in_dim = 1 + self.code_dim
        self.h_layer = nn.Sequential(
            nn.Linear(in_dim, h_hidden_dim),
            nn.LayerNorm(h_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h_hidden_dim, self.code_dim),
            nn.LayerNorm(self.code_dim),
            nn.ReLU(),
        )

        # gate outputs W scores per junction: (n_obs, W)
        self.gate_net = nn.Sequential(
            nn.Linear(self.code_dim, self.code_dim // 2),
            nn.ReLU(),
            nn.Linear(self.code_dim // 2, self.num_weight_vectors),
        )

        # combiner: (W * D) -> D
        if self.num_weight_vectors == 1:
            self.combiner = nn.Identity()
        else:
            self.combiner = nn.Sequential(
                nn.Linear(self.num_weight_vectors * self.code_dim, self.code_dim),
                nn.LayerNorm(self.code_dim),
                nn.ReLU(),
            )

        # final encoder MLP
        self.encoder_mlp = FCLayers(
            n_in=self.code_dim,
            n_out=2 * latent_dim,
            n_cat_list=n_cat_list or [],
            n_cont=n_cont,
            n_layers=2,
            n_hidden=encoder_hidden_dim,
            dropout_rate=dropout_rate,
            use_batch_norm=False,
            use_layer_norm=True,
            inject_covariates=inject_covariates,
        )

    def forward(
        self,
        x: torch.Tensor,                # (B, J)
        mask: torch.Tensor,             # (B, J)
        *cat_list: torch.Tensor,
        cont: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, J = x.shape
        device = x.device
        D = self.code_dim
        W = self.num_weight_vectors

        # inside forward(), right after you unpack (B, J)
        if (mask.sum(dim=1) == 0).any():
            bad_idx = (mask.sum(dim=1) == 0).nonzero(as_tuple=True)[0]
            print(f"[Warning] Found {bad_idx.numel()} cells in batch with 0 observed junctions: {bad_idx.tolist()}")

        outputs = []  # will collect (D,) per cell
        for b in range(B):          
            obs_idx = mask[b].bool()  # (J,)

            n_obs = obs_idx.sum()
            if n_obs == 0:
                outputs.append(torch.zeros(D, device=device))
                continue

            x_obs = x[b, obs_idx]  # (n_obs,)
            F_obs = self.feature_embedding[obs_idx]  # (n_obs, D)

            h_in = torch.cat([x_obs.unsqueeze(-1), F_obs], dim=1)  # (n_obs, 1 + D)
            h_out = self.h_layer(h_in)  # (n_obs, D)

            # temperature from mean observed h_out
            mean_h = h_out.mean(dim=0)  # (D,)

            # raw gate scores: (n_obs, W)
            raw_gates = self.gate_net(h_out)  # (n_obs, W)

            # neighbor scaling
            if self.temperature_fixed:
                neighbor_scale = self.temperature_value
            else:
                neighbor_scale = 1.0 / torch.sqrt(torch.tensor(n_obs, device=device).float().clamp(min=1.0))
            logits = (raw_gates * neighbor_scale) 
            logits = logits.clamp(min=-10.0, max=10.0)

            # softmax over junctions for each head: dim=0 → (n_obs, W)
            weights = torch.softmax(logits, dim=0)  # (n_obs, W)

            # head sums: for each of W heads, weighted sum over junction codes -> (W, D)
            # weights.T: (W, n_obs), h_out: (n_obs, D)
            head_sums = (weights.T.unsqueeze(-1) * h_out.unsqueeze(0)).sum(dim=1)  # (W, D)

            # flatten and combine to code vector: (W * D,) -> (D,)
            combined = self.combiner(head_sums.reshape(-1))  # (D,)
            outputs.append(combined)

        c = torch.stack(outputs, dim=0)  # (B, D)
        mu_logvar = self.encoder_mlp(c, *cat_list, cont=cont)  # (B, 2*latent)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        return mu, logvar


# ===============================
# FAST VARIANTS (batched, masked)
# ===============================

class PartialEncoderEDDIFast(nn.Module):
    """
    Fast EDDI-style partial encoder (batched).
    Changes vs. PartialEncoderEDDI:
      - Processes all junctions (no per-cell Python loop).
      - Uses psi_mask to exclude unobserved junctions from pooling.
    """
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
        pool_mode: Literal["mean","sum"] = "mean",
    ):
        super().__init__()
        self.n_cat_list = [n for n in (n_cat_list or []) if n > 1]
        self.n_cont = n_cont
        self.inject_covariates = inject_covariates
        self.code_dim = code_dim
        self.pool_mode = pool_mode

        # per-feature embeddings (J, D)
        self.feature_embedding = nn.Parameter(torch.randn(input_dim, code_dim))

        # shared h-network sees [psi, embedding]
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

        # cell-level encoder MLP
        self.encoder_mlp = FCLayers(
            n_in=code_dim,
            n_out=2 * latent_dim,
            n_cat_list=n_cat_list or [],
            n_cont=n_cont,
            n_layers=2,
            n_hidden=encoder_hidden_dim,
            dropout_rate=dropout_rate,
            use_batch_norm=False,
            use_layer_norm=True,
            inject_covariates=inject_covariates,
        )

    def forward(
        self,
        x: torch.Tensor,       # (B, J)
        mask: torch.Tensor,    # (B, J) 1=observed, 0=missing
        *cat_list: torch.Tensor,
        cont: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, J = x.shape
        D = self.code_dim
        device = x.device
        # inside forward(), right after you unpack (B, J)
        if (mask.sum(dim=1) == 0).any():
            bad_idx = (mask.sum(dim=1) == 0).nonzero(as_tuple=True)[0]
            print(f"[Warning] Found {bad_idx.numel()} cells in batch with 0 observed junctions: {bad_idx.tolist()}")

        mask_bool = mask.bool()                     # (B, J)
        F = self.feature_embedding                  # (J, D)
        x_exp = x.unsqueeze(-1)                     # (B, J, 1)
        F_exp = F.unsqueeze(0).expand(B, J, D)      # (B, J, D)

        h_in = torch.cat([x_exp, F_exp], dim=-1)    # (B, J, 1+D)
        # Flatten batch+junction for MLP, then reshape back
        h_out = self.h_layer(h_in.view(B * J, -1)).view(B, J, D)  # (B, J, D)

        # Mask out unobserved junctions from pooling
        masked_h = h_out * mask_bool.unsqueeze(-1)  # (B, J, D)
        pooled = masked_h.sum(dim=1)                # (B, D)

        if self.pool_mode == "mean":
            denom = mask_bool.sum(dim=1, keepdim=True).clamp(min=1)  # (B,1)
            c = pooled / denom                                       # (B, D)
        else:
            c = pooled

        mu_logvar = self.encoder_mlp(c, *cat_list, cont=cont)        # (B, 2*latent)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        return mu, logvar


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Literal
from scvi.nn import FCLayers

class PartialEncoderEDDIATSEFast(nn.Module):
    """
    Fast EDDI + ATSE-embedding encoder (batched).

    - Per-junction embeddings: (J, D)
    - Per-ATSE embeddings: (A, Ae) as a trainable nn.Parameter
    - ATSE indices for each junction derived on the fly from self.junc2atse
    """

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
        pool_mode: Literal["mean","sum"] = "mean",
        atse_embedding_dimension: int = 16,
    ):
        super().__init__()
        self.code_dim = code_dim
        self.h_hidden_dim = h_hidden_dim
        self.dropout_rate = dropout_rate
        self.pool_mode = pool_mode
        self.n_cat_list = [n for n in (n_cat_list or []) if n > 1]
        self.n_cont = n_cont
        self.inject_covariates = inject_covariates
        self.atse_embedding_dimension = int(atse_embedding_dimension)

        # Per-junction embeddings
        self.feature_embedding = nn.Parameter(torch.randn(input_dim, code_dim))

        # Placeholder h-layer; rebuilt after register_junc2atse()
        self.h_layer = self._make_h_layer(n_atse_embed=0)

        # Encoder MLP
        self.encoder_mlp = FCLayers(
            n_in=code_dim,
            n_out=2 * latent_dim,
            n_cat_list=n_cat_list or [],
            n_cont=n_cont,
            n_layers=2,
            n_hidden=encoder_hidden_dim,
            dropout_rate=dropout_rate,
            use_batch_norm=False,
            use_layer_norm=True,
            inject_covariates=inject_covariates,
        )

        # Sparse mapping rows=junctions (J), cols=ATSEs (A)
        self.register_buffer("junc2atse", torch.sparse_coo_tensor(size=(input_dim, 0)))
        self.n_atse = 0

        # ATSE embeddings (Parameter) will be created in register_junc2atse
        self.atse_embedding: torch.nn.Parameter | None = None

    def _make_h_layer(self, n_atse_embed: int) -> nn.Sequential:
        in_dim = 1 + self.code_dim + n_atse_embed
        return nn.Sequential(
            nn.Linear(in_dim, self.h_hidden_dim),
            nn.LayerNorm(self.h_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.h_hidden_dim, self.code_dim),
            nn.LayerNorm(self.code_dim),
            nn.ReLU(),
        )

    @torch.no_grad()
    def register_junc2atse(self, junc2atse: torch.sparse.Tensor):
        """
        Provide the (J x A) sparse mapping (one nonzero per junction).
        Builds ATSE embeddings (Parameter) and rebuilds h_layer to accept Ae dims.
        """
        j2a = junc2atse.coalesce().to(self.feature_embedding.device)
        self.register_buffer("junc2atse", j2a)

        # Number of ATSEs
        _, idx_g = j2a.indices()
        self.n_atse = int(idx_g.max().item()) + 1 if idx_g.numel() > 0 else 0

        if self.n_atse == 0:
            self.atse_embedding = None
            self.h_layer = self._make_h_layer(n_atse_embed=0)
        else:
            # Create ATSE Embeddings Parameter 
            self.atse_embedding = nn.Parameter(
                torch.randn(self.n_atse, self.atse_embedding_dimension, device=j2a.device)
            )
            self.h_layer = self._make_h_layer(n_atse_embed=self.atse_embedding_dimension)

    def forward(
        self,
        x: torch.Tensor,       # (B, J) psi or similar scalar per junction
        mask: torch.Tensor,    # (B, J) 1=observed, 0=missing
        *cat_list: torch.Tensor,
        cont: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, J = x.shape
        D = self.code_dim
        device = x.device

        # Warn if a cell has zero observed junctions
        if (mask.sum(dim=1) == 0).any():
            bad_idx = (mask.sum(dim=1) == 0).nonzero(as_tuple=True)[0]
            print(f"[Warning] Found {bad_idx.numel()} cells in batch with 0 observed junctions: {bad_idx.tolist()}")

        mask_bool = mask.bool()

        # Junction embeddings + psi
        F = self.feature_embedding                                 # (J, D)
        x_exp = x.unsqueeze(-1)                                    # (B, J, 1)
        F_exp = F.unsqueeze(0).expand(B, J, D)                     # (B, J, D)

        # --- ATSE embeddings per junction ---
        if self.atse_embedding is None:
            raise RuntimeError(
                "ATSE embeddings not initialized. "
                "Call `register_junc2atse(...)` before using this encoder."
            )

        idx_p, idx_g = self.junc2atse.indices()
        idx_p = idx_p.to(device)
        idx_g = idx_g.to(device)

        # Map each junction → its ATSE index
        atse_idx = torch.full((J,), 0, dtype=torch.long, device=device)
        atse_idx[idx_p] = idx_g

        # Lookup ATSE embeddings and prepare it to attach to the cells
        atse_emb = self.atse_embedding[atse_idx]               # (J, Ae)
        atse_emb_exp = atse_emb.unsqueeze(0).expand(B, J, -1)  # (B, J, Ae)

        # h-layer input
        h_in = torch.cat([x_exp, F_exp, atse_emb_exp], dim=-1)     # (B, J, 1 + D + Ae)
        h_out = self.h_layer(h_in.view(B * J, -1)).view(B, J, D)

        # Mask + pool
        masked_h = h_out * mask_bool.unsqueeze(-1)
        pooled = masked_h.sum(dim=1)

        if self.pool_mode == "mean":
            denom = mask_bool.sum(dim=1, keepdim=True).clamp(min=1)
            c = pooled / denom
        else:
            c = pooled

        # Encoder MLP
        mu_logvar = self.encoder_mlp(c, *cat_list, cont=cont)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        return mu, logvar



class PartialEncoderWeightedSumEDDIMultiWeightFast(nn.Module):
    """
    Fast multi-head weighted-sum EDDI encoder (batched).
    Changes vs. PartialEncoderWeightedSumEDDIMultiWeight:
      - Processes all junctions at once (no per-cell loop).
      - Mask unobserved junctions by:
         (i) setting gate logits to -inf before softmax (so weights=0),
         (ii) masking h_out in the weighted sum.
      - Softmax is taken along the junction axis dim=1 (J).
    """
    def __init__(
        self,
        input_dim: int,
        code_dim: int,
        h_hidden_dim: int,
        encoder_hidden_dim: int,
        latent_dim: int,
        num_weight_vectors: int = 4,
        dropout_rate: float = 0.0,
        n_cat_list: Iterable[int] | None = None,
        n_cont: int = 0,
        inject_covariates: bool = True,
        temperature_value: float = 1.0,
        temperature_fixed: bool = True,
    ):
        super().__init__()
        self.n_cat_list = [n for n in (n_cat_list or []) if n > 1]
        self.n_cont = n_cont
        self.inject_covariates = inject_covariates
        self.code_dim = code_dim
        self.num_weight_vectors = num_weight_vectors
        self.temperature_value = float(temperature_value)
        self.temperature_fixed = temperature_fixed

        self.feature_embedding = nn.Parameter(torch.randn(input_dim, self.code_dim))

        in_dim = 1 + self.code_dim
        self.h_layer = nn.Sequential(
            nn.Linear(in_dim, h_hidden_dim),
            nn.LayerNorm(h_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h_hidden_dim, self.code_dim),
            nn.LayerNorm(self.code_dim),
            nn.ReLU(),
        )

        self.gate_net = nn.Sequential(
            nn.Linear(self.code_dim, self.code_dim // 2),
            nn.ReLU(),
            nn.Linear(self.code_dim // 2, self.num_weight_vectors),
        )

        if self.num_weight_vectors == 1:
            self.combiner = nn.Identity()
        else:
            self.combiner = nn.Sequential(
                nn.Linear(self.num_weight_vectors * self.code_dim, self.code_dim),
                nn.LayerNorm(self.code_dim),
                nn.ReLU(),
            )

        self.encoder_mlp = FCLayers(
            n_in=self.code_dim,
            n_out=2 * latent_dim,
            n_cat_list=n_cat_list or [],
            n_cont=n_cont,
            n_layers=2,
            n_hidden=encoder_hidden_dim,
            dropout_rate=dropout_rate,
            use_batch_norm=False,
            use_layer_norm=True,
            inject_covariates=inject_covariates,
        )

    def forward(
        self,
        x: torch.Tensor,            # (B, J)
        mask: torch.Tensor,         # (B, J)
        *cat_list: torch.Tensor,
        cont: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, J = x.shape
        D = self.code_dim
        W = self.num_weight_vectors
        device = x.device

        # inside forward(), right after you unpack (B, J)
        if (mask.sum(dim=1) == 0).any():
            bad_idx = (mask.sum(dim=1) == 0).nonzero(as_tuple=True)[0]
            print(f"[Warning] Found {bad_idx.numel()} cells in batch with 0 observed junctions: {bad_idx.tolist()}")

        mask_bool = mask.bool()                             # (B, J)
        F = self.feature_embedding                          # (J, D)
        x_exp = x.unsqueeze(-1)                             # (B, J, 1)
        F_exp = F.unsqueeze(0).expand(B, J, D)              # (B, J, D)

        h_in = torch.cat([x_exp, F_exp], dim=-1)            # (B, J, 1+D)
        h_out = self.h_layer(h_in.view(B * J, -1)).view(B, J, D)  # (B, J, D)

        # Gate scores per junction
        raw_gates = self.gate_net(h_out)                    # (B, J, W)

        # Neighbor scaling (per cell if not fixed)
        if self.temperature_fixed:
            scale = self.temperature_value
            logits = raw_gates * scale                      # (B, J, W)
        else:
            n_obs = mask_bool.sum(dim=1).clamp(min=1)       # (B,)
            scale = (n_obs.float()).rsqrt().view(B, 1, 1)   # (B,1,1)
            logits = raw_gates * scale                      # (B, J, W)

        # Mask unobserved junctions in logits → -inf so softmax gives 0
        logits = logits.masked_fill(~mask_bool.unsqueeze(-1), float(-1e9))  # (B, J, W)

        # Softmax over junction axis
        weights = torch.softmax(logits, dim=1)              # (B, J, W)

        # Also mask h_out so unobserved contribute nothing
        h_masked = h_out * mask_bool.unsqueeze(-1)          # (B, J, D)

        # Weighted sums per head: (B, W, D)
        head_sums = (weights.transpose(1, 2).unsqueeze(-1) * h_masked.unsqueeze(1)).sum(dim=2)

        if W == 1:
            combined = head_sums.squeeze(1)                 # (B, D)
        else:
            combined = self.combiner(head_sums.reshape(B, W * D))  # (B, D)

        mu_logvar = self.encoder_mlp(combined, *cat_list, cont=cont)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        return mu, logvar


import torch
import torch.nn as nn
from typing import Iterable, Literal
from scvi.nn import FCLayers

class PartialEncoderWeightedSumEDDIMultiWeightATSEFast(nn.Module):
    """
    Fast multi-head weighted-sum EDDI + ATSE encoder (batched), using per-ATSE embeddings.

    - Per-junction embeddings: (J, D)
    - Per-ATSE embeddings: (A, Ae) as a trainable nn.Parameter, created in register_junc2atse
    - Gate input uses concatenation [h_out_d, atse_emb_d] with fixed Ae (no gate rebuild).
    - ATSE indices are derived from self.junc2atse (sparse J x A) on the fly.
    """
    def __init__(
        self,
        input_dim: int,           # J
        code_dim: int,            # D
        h_hidden_dim: int,
        encoder_hidden_dim: int,
        latent_dim: int,
        num_weight_vectors: int = 4,
        dropout_rate: float = 0.0,
        n_cat_list: Iterable[int] | None = None,
        n_cont: int = 0,
        inject_covariates: bool = True,
        temperature_value: float = 1.0,
        temperature_fixed: bool = True,
        atse_embedding_dimension: int = 16,   # known at init → no gate rebuild later
    ):
        super().__init__()
        self.code_dim = code_dim
        self.num_weight_vectors = num_weight_vectors
        self.temperature_value = float(temperature_value)
        self.temperature_fixed = temperature_fixed
        self.n_cat_list = [n for n in (n_cat_list or []) if n > 1]
        self.n_cont = n_cont
        self.inject_covariates = inject_covariates
        self.atse_embedding_dimension = int(atse_embedding_dimension)

        # (J, D) per-junction embeddings
        self.feature_embedding = nn.Parameter(torch.randn(input_dim, code_dim))

        # h-layer: input [psi (1) + junction-embedding (D)] -> D
        in_dim_h = 1 + code_dim
        self.h_layer = nn.Sequential(
            nn.Linear(in_dim_h, h_hidden_dim),
            nn.LayerNorm(h_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h_hidden_dim, code_dim),
            nn.LayerNorm(code_dim),
            nn.ReLU(),
        )

        # Gate network: input is [h_out (D) + ATSE emb (Ae)] — Ae known at init.
        in_dim_gate = code_dim + self.atse_embedding_dimension
        self.gate_net = nn.Sequential(
            nn.Linear(in_dim_gate, max(in_dim_gate // 2, 1)),
            nn.ReLU(),
            nn.Linear(max(in_dim_gate // 2, 1), self.num_weight_vectors),
        )

        # Combine W head vectors → (B, D)
        if self.num_weight_vectors == 1:
            self.combiner = nn.Identity()
        else:
            self.combiner = nn.Sequential(
                nn.Linear(self.num_weight_vectors * self.code_dim, self.code_dim),
                nn.LayerNorm(self.code_dim),
                nn.ReLU(),
            )

        # Final encoder MLP → (mu, logvar)
        self.encoder_mlp = FCLayers(
            n_in=code_dim,
            n_out=2 * latent_dim,
            n_cat_list=n_cat_list or [],
            n_cont=n_cont,
            n_layers=2,
            n_hidden=encoder_hidden_dim,
            dropout_rate=dropout_rate,
            use_batch_norm=False,
            use_layer_norm=True,
            inject_covariates=inject_covariates,
        )

        # Sparse mapping rows=junctions (J), cols=ATSEs (A)
        self.register_buffer("junc2atse", torch.sparse_coo_tensor(size=(input_dim, 0)))
        self.n_atse = 0

        # Per-ATSE embedding table (A, Ae), created when we learn A via register_junc2atse()
        self.atse_embedding: nn.Parameter | None = None

    @torch.no_grad()
    def register_junc2atse(self, junc2atse: torch.sparse.Tensor):
        """
        Provide the (J x A) sparse mapping (exactly one nonzero per junction).
        Creates the ATSE embedding Parameter (A, Ae) on the same device as feature_embedding.
        """
        j2a = junc2atse.coalesce().to(self.feature_embedding.device)
        self.register_buffer("junc2atse", j2a)

        # Count #ATSEs
        _, atse_idx = j2a.indices()
        self.n_atse = int(atse_idx.max().item()) + 1 if atse_idx.numel() > 0 else 0

        if self.n_atse == 0:
            self.atse_embedding = None
        else:
            # Create ATSE embeddings (A, Ae) on same device
            self.atse_embedding = nn.Parameter(
                torch.randn(self.n_atse, self.atse_embedding_dimension, device=j2a.device)
            )

    def forward(
        self,
        x: torch.Tensor,        # (B, J)
        mask: torch.Tensor,     # (B, J)
        *cat_list: torch.Tensor,
        cont: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, J = x.shape
        D = self.code_dim
        W = self.num_weight_vectors
        device = x.device

        if (mask.sum(dim=1) == 0).any():
            bad = (mask.sum(dim=1) == 0).nonzero(as_tuple=True)[0]
            print(f"[Warning] Found {bad.numel()} cells in batch with 0 observed junctions: {bad.tolist()}")

        if self.atse_embedding is None or self.n_atse == 0:
            raise RuntimeError(
                "ATSE embeddings not initialized. Call `register_junc2atse(...)` before using this encoder."
            )

        mask_bool = mask.bool()                          # (B, J)
        F = self.feature_embedding                       # (J, D)

        # ensure ATSE table is on the same device as F (matches your other models' pattern)
        atse_table = self.atse_embedding
        if atse_table.device != F.device:
            atse_table = atse_table.to(F.device)

        # h-layer inputs
        x_exp = x.unsqueeze(-1)                          # (B, J, 1)
        F_exp = F.unsqueeze(0).expand(B, J, D)           # (B, J, D)
        h_in  = torch.cat([x_exp, F_exp], dim=-1)        # (B, J, 1+D)
        h_out = self.h_layer(h_in.view(B * J, -1)).view(B, J, D)

        # Build per-junction ATSE indices from sparse mapping
        idx_p, idx_g = self.junc2atse.indices()
        idx_p = idx_p.to(F.device)
        idx_g = idx_g.to(F.device)
        atse_idx = torch.full((J,), 0, dtype=torch.long, device=F.device)
        atse_idx[idx_p] = idx_g                           # (J,)

        # Gather ATSE embeddings and broadcast to batch
        atse_emb = atse_table[atse_idx]                   # (J, Ae)
        atse_emb_exp = atse_emb.unsqueeze(0).expand(B, J, -1)  # (B, J, Ae)

        # Gate input: concat per-junction code with ATSE embedding
        gate_in = torch.cat([h_out, atse_emb_exp], dim=-1)      # (B, J, D+Ae)
        raw_gates = self.gate_net(gate_in)                      # (B, J, W)

        # Temperature / neighbor scaling
        if self.temperature_fixed:
            logits = raw_gates * self.temperature_value
        else:
            n_obs = mask_bool.sum(dim=1).clamp(min=1)           # (B,)
            scale = (n_obs.float()).rsqrt().view(B, 1, 1)
            logits = raw_gates * scale

        # Mask unobserved junctions in logits → softmax zero weight
        logits = logits.masked_fill(~mask_bool.unsqueeze(-1), float(-1e9))  # (B, J, W)
        weights = torch.softmax(logits, dim=1)                               # (B, J, W)

        # Also mask h_out
        h_masked = h_out * mask_bool.unsqueeze(-1)            # (B, J, D)

        # Weighted sums per head: (B, W, D)
        head_sums = (weights.transpose(1, 2).unsqueeze(-1) * h_masked.unsqueeze(1)).sum(dim=2)

        # Combine heads
        if W == 1:
            combined = head_sums.squeeze(1)                   # (B, D)
        else:
            combined = self.combiner(head_sums.reshape(B, W * D))  # (B, D)

        # Final projection → mu, logvar
        mu_logvar = self.encoder_mlp(combined, *cat_list, cont=cont)  # (B, 2Z)
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
        n_heads: int = 4,
        ff_dim: int | None = None,
        num_transformer_layers: int = 2,
        encoder_type: Literal[
            "PartialEncoderEDDI",
            "PartialEncoderEDDIATSE",
            "PartialEncoderWeightedSumEDDIMultiWeight",
            "PartialEncoderWeightedSumEDDIMultiWeightATSE",
        ] = "PartialEncoderEDDI",
        pool_mode: Literal["mean","sum"] = "mean",
        num_weight_vectors: int = 4,
        temperature_value: float = -1.0, #if temperature_value is set to -1, then the median number of observations is used as the fixed temperature value.
        temperature_fixed: bool = True, #if temperature is fixed, it is fixed to the value of temperature_value
    ):
        super().__init__()
        self.encoder_type = encoder_type

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

        print(f"Using code_dim={code_dim}!")
        print(f"Using latent_dim={n_latent}")

        # instantiate the requested encoder
        if encoder_type == "PartialEncoderEDDI":
            print(f"Using EDDI Partial Encoder")
            self.encoder = PartialEncoderEDDI(
                input_dim=n_input,
                code_dim=code_dim,
                h_hidden_dim=h_hidden_dim,
                encoder_hidden_dim=encoder_hidden_dim,
                latent_dim=n_latent,
                dropout_rate=dropout_rate,
                n_cat_list=encoder_cat_list,
                n_cont=n_continuous_cov,
                inject_covariates=encode_covariates,
                pool_mode=pool_mode,
            )

        elif encoder_type == "PartialEncoderEDDIATSE":
            print("Using EDDI + ATSE Partial Encoder")
            self.encoder = PartialEncoderEDDIATSE(
                input_dim=n_input,
                code_dim=code_dim,
                h_hidden_dim=h_hidden_dim,
                encoder_hidden_dim=encoder_hidden_dim,
                latent_dim=n_latent,
                dropout_rate=dropout_rate,
                n_cat_list=encoder_cat_list,
                n_cont=n_continuous_cov,
                inject_covariates=encode_covariates,
                pool_mode=pool_mode,
            )
        
        elif encoder_type == "PartialEncoderWeightedSumEDDIMultiWeight":
            print("Using PartialEncoderWeightedSumEDDIMultiWeight")
            self.encoder = PartialEncoderWeightedSumEDDIMultiWeight(
                input_dim=n_input,
                code_dim=code_dim,
                h_hidden_dim=h_hidden_dim,
                encoder_hidden_dim=encoder_hidden_dim,
                latent_dim=n_latent,
                dropout_rate=dropout_rate,
                n_cat_list=encoder_cat_list,
                n_cont=n_continuous_cov,
                inject_covariates=encode_covariates,
                num_weight_vectors = num_weight_vectors,
                temperature_value=temperature_value,
                temperature_fixed=temperature_fixed,
            )

        elif encoder_type == "PartialEncoderWeightedSumEDDIMultiWeightATSE":
            print("Using PartialEncoderWeightedSumEDDIMultiWeightATSE")
            self.encoder = PartialEncoderWeightedSumEDDIMultiWeightATSE(
                input_dim=n_input,
                code_dim=code_dim,
                h_hidden_dim=h_hidden_dim,
                encoder_hidden_dim=encoder_hidden_dim,
                latent_dim=n_latent,
                dropout_rate=dropout_rate,
                n_cat_list=encoder_cat_list,
                n_cont=n_continuous_cov,
                inject_covariates=encode_covariates,
                num_weight_vectors = num_weight_vectors,
                temperature_value=temperature_value,
                temperature_fixed=temperature_fixed,
            )
        
        else:
            raise ValueError(f"Unknown encoder_type={encoder_type!r}")
            
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
            if torch.rand(1).item() < 1e-6:
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
                mask = mask,
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