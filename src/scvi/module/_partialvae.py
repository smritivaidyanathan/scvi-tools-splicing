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
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

class LinformerSelfAttention(nn.Module):
    """
    A lean version of self-attention: we squish the keys and values
    down to a smaller dimension k before doing dot-product attention.
    This is the core trick from the Linformer paper (Wang et al. 2020):
    "We show that self-attention can be projected onto a low-rank
    subspace, reducing the complexity from O(n^2) to O(nk)."
    Eq. (5) in the paper: Approx softmax(Q K^T / sqrt(d)) V
    but with K, V replaced by E X and F X.
    """
    def __init__(self, embed_dim: int, num_heads: int, k: int, max_seq_len: int):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # scaling factor 1/sqrt(d)
        self.scale = self.head_dim ** -0.5

        # standard linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # These are the low-rank projection matrices (E for keys, F for values).
        # We learn E, F of shape (k, embed_dim), so seq_len D -> k.
        self.E_proj = nn.Linear(max_seq_len, k, bias=False)
        self.F_proj = nn.Linear(max_seq_len, k, bias=False)
        self.max_seq_len = max_seq_len

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        x: (B, D, C) where D = sequence length, C = embed_dim
        mask: (B, D) with True for positions to keep
        """
        B, D, C = x.shape

        if D != self.max_seq_len:
            raise RuntimeError(f"Expected sequence length {self.max_seq_len}, got {D}")

        # 1) project inputs to queries, keys, values
        q = self.q_proj(x)  # (B, D, C)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2) split into heads: shape (B, h, D, head_dim)
        def split_heads(t):
            return t.view(B, D, self.num_heads, self.head_dim)\
                    .transpose(1, 2)
        q, k, v = map(split_heads, (q, k, v))

        # 3) Linformer trick: project seq_len D down to k for k & v
        #    flatten batch & heads -> (B*h, D, head_dim)
        Bh = B * self.num_heads
        k = k.reshape(Bh, D, self.head_dim)
        v = v.reshape(Bh, D, self.head_dim)
        # project seq-axis D→k via E_proj, F_proj
        # k,v are (B*h, D, head_dim) → transpose to (B*h, head_dim, D)
        k = self.E_proj(k.transpose(1,2)).transpose(1,2)  # → (B*h, k, head_dim)
        v = self.F_proj(v.transpose(1,2)).transpose(1,2)

        # 4) reshape q similarly for matmul: (B*h, D, head_dim)
        q = q.reshape(Bh, D, self.head_dim)

        # 5) compute scaled dot-product attention
        #    eq: A = softmax( Q K^T / sqrt(d) )
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (Bh, D, k)
        if mask is not None:
            # mask: True means keep, False means pad
            # expand mask to (Bh, D, k)
            m = mask.unsqueeze(1).repeat(1, self.num_heads, 1)  # (B, h, D)
            m = m.reshape(Bh, D).unsqueeze(-1).expand(-1, -1, k.size(1))
            attn_scores = attn_scores.masked_fill(~m, float('-inf'))
        attn = F.softmax(attn_scores, dim=-1)
        # replace any NaNs (from all-masked rows) with zeros
        attn = torch.nan_to_num(attn, nan=0.0)

        # 6) weighted sum: (Bh, D, k) x (Bh, k, head_dim) -> (Bh, D, head_dim)
        out = torch.matmul(attn, v)
        # 7) reassemble heads -> (B, D, C)
        out = out.view(B, self.num_heads, D, self.head_dim)\
                  .transpose(1, 2)\
                  .reshape(B, D, C)

        # 8) final linear layer
        return self.out_proj(out)


class LinformerEncoderLayer(nn.Module):
    """
    A single transformer encoder block using LinformerSelfAttention.
    Pretty much just like nn.TransformerEncoderLayer, but with our lean attention.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        k: int,
        dim_feedforward: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        # our custom self-attention
        self.self_attn = LinformerSelfAttention(embed_dim, num_heads, k, max_seq_len)
        # a small feed-forward network
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        # layer norms & dropouts as usual
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # use ReLU like the original
        self.activation = F.relu

    def forward(self, src: Tensor, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        src: (B, D, C)
        src_key_padding_mask: (B, D), True=keep, False=pad
        """
        x = src
        # 1) self-attention + add & norm
        attn_out = self.self_attn(
            x,
            mask=(~src_key_padding_mask) if src_key_padding_mask is not None else None
        )
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)

        # 2) feed-forward + add & norm
        ff = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(ff)
        x = self.norm2(x)
        return x



import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable


import torch
from torch import nn
from collections.abc import Iterable
from scvi.nn import FCLayers

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

        tr_out = h_out

        # now pool across the D dimension
        mask_exp = mask.unsqueeze(-1).float()  # (B, D, 1)
        summed = (tr_out * mask_exp).sum(dim=1)
        if self.pool_mode == "mean":
            # divide by # observed junctions
            counts = mask_exp.sum(dim=1).clamp(min=1e-8)
            c = summed / counts
        else:
            # just sum
            c = summed

        # --- step 4: now pass the output into our mlp (FC layers handles cont and cat covariates automatically) ---
        mu_logvar = self.encoder_mlp(c, *cat_list, cont=cont)  # -> (B, 2*latent_dim)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        return mu, logvar




class PartialEncoderWeightedSum(nn.Module):
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
        junction_inclusion: str = "all_junctions",
    ):
        super().__init__()
        # keep track of how many one-hot cats and cont features
        self.n_cat_list = [n for n in (n_cat_list or []) if n > 1]
        self.n_cont = n_cont
        self.inject_covariates = inject_covariates
        total_cov = sum(self.n_cat_list) + self.n_cont
        self.code_dim = code_dim
        self.junction_inclusion = junction_inclusion
        if self.junction_inclusion == "all_junctions":
            print("Including all junctions into weighted sum")
        elif self.junction_inclusion == "observed_junctions":
            print("Only Including observed junctions into weighted sum")

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

        self.gate_net = nn.Sequential(
            nn.Linear(code_dim, code_dim // 2),
            nn.ReLU(),
            nn.Linear(code_dim // 2, 1),
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

        # ─── step 3: compute per-junction gate scores ───────────────────
        # raw_gates: (B, D, 1)
        raw_gates = self.gate_net(h_out)  

        if self.junction_inclusion == "observed_junctions":
            #mask out missing junctions so they get zero weight
            raw_gates = raw_gates.masked_fill((mask == 0).unsqueeze(-1), float('-1e9'))

        # normalize to [0,1] sum to 1 across D
        weights = torch.softmax(raw_gates.squeeze(-1), dim=1)  # (B, D)

        # ─── step 4: weighted sum instead of mean ──────────────────────
        # bring weights to (B, D, 1) to match h_out
        w_exp = weights.unsqueeze(-1)  
        # c = Σ_j w_{b,j} * h_out[b,j,:]
        c = (h_out * w_exp).sum(dim=1)  

        # ─── step 5: your encoder MLP as before ────────────────────────
        mu_logvar = self.encoder_mlp(c, *cat_list, cont=cont)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        return mu, logvar

class PartialEncoderImpute(nn.Module):
    def __init__(
        self,
        input_dim: int,           
        code_dim: int,
        h_hidden_dim: int,
        latent_dim: int,
        encoder_hidden_dim: int,      # ← re-introduced
        dropout_rate: float = 0.0,
        n_cat_list: Iterable[int] | None = None,
        n_cont: int = 0,
        inject_covariates: bool = True,
        junction_inclusion: str = "all_junctions",
    ):
        super().__init__()
        # per-junction embedding
        self.feature_embedding = nn.Parameter(torch.randn(input_dim, code_dim))
        self.code_dim = code_dim
        self.finished_training = False
        self.junction_inclusion = junction_inclusion
        if self.junction_inclusion == "observed_junctions":
            "Only Imputing for Non Observed Junctions"
        elif self.junction_inclusion == "all_junctions":
            "Imputing for All Junctions"
        # combine [psi_value, embedding] → code_dim
        self.h_layer = nn.Sequential(
            nn.Linear(1 + code_dim, h_hidden_dim),
            nn.LayerNorm(h_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h_hidden_dim, code_dim),
            nn.LayerNorm(code_dim),
            nn.ReLU(),
        )
        # collapse code_dim → 1 scalar ∈ [0,1]
        self.impute_net = nn.Sequential(
            nn.Linear(code_dim, code_dim // 2),
            nn.ReLU(),
            nn.Linear(code_dim // 2, 1),
            nn.Sigmoid(),
        )
        # now our “new x” has shape (B, D) → feed into encoder_mlp
        self.encoder_mlp = FCLayers(
            n_in       = input_dim,
            n_out      = 2 * latent_dim,
            n_cat_list = list(n_cat_list or []),
            n_cont     = n_cont,
            n_layers   = 2,
            n_hidden   = encoder_hidden_dim,  # use your passed-in hidden size    
            dropout_rate=dropout_rate,
            use_batch_norm=False,
            use_layer_norm=True,
            inject_covariates=inject_covariates,
        )

    def forward(
        self,
        x: torch.Tensor,    # (B, D)
        mask: torch.Tensor, # (B, D)
        *cat_list,
        cont: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, D = x.shape
        # 1) build per-junction features
        x_flat  = x.reshape(-1,1)                              # (B*D,1)
        F_flat  = self.feature_embedding.unsqueeze(0)          # (1,D,code_dim)
        F_flat  = F_flat.expand(B, D, -1).reshape(-1, F_flat.size(-1))  # (B*D,code_dim)
        h_in    = torch.cat([x_flat, F_flat], dim=1)           # (B*D,1+code_dim)

        # 2) per-junction embedding → (B, D, code_dim)
        h_out   = self.h_layer(h_in).view(B, D, -1)

        # 3) collapse each code_dim vector to a scalar ∈ [0,1]
        x_imp   = self.impute_net(h_out).squeeze(-1)           # (B, D)


        # only compute & print Pearson‐R when in eval mode, once
        if self.finished_training and not hasattr(self, "_corr_printed"):
            orig = x.reshape(-1)
            pred = x_imp.reshape(-1)
            vx, vy = orig - orig.mean(), pred - pred.mean()
            corr = (vx * vy).sum() / torch.sqrt((vx**2).sum() * (vy**2).sum() + 1e-8)
            print(f"[Eval] impute_net PearsonR = {corr.item():.4f}")
            self._corr_printed = True

        # 4) optional: keep original where observed
        if self.junction_inclusion == "observed_junctions":
            x_imp = x_imp * (1-mask) + x * mask
            

        # 5) feed your original encoder MLP on shape (B, D)
        mu_logvar = self.encoder_mlp(x_imp, *cat_list, cont=cont)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        return mu, logvar



class PartialEncoderTransformer(nn.Module):
    """
    Transformer-based encoder that embeds every possible junction,
    attends over the subset observed in each cell, then maps the
    aggregated representation to VAE latent parameters (μ, log σ²).

    Args
    ----
    input_dim : int
        Number of distinct junctions J.
    code_dim : int
        Size D of each learned junction embedding.
    latent_dim : int
        Size Z of the downstream latent space.
    encoder_hidden_dim : int
        Hidden width inside the two-layer MLP that produces [μ‖log σ²].
    dropout_rate : float
        Dropout used inside Transformer feed-forward blocks.
    n_cat_list : Iterable[int] | None
        Sizes of one-hot categorical covariates (e.g. batch).
    n_cont : int
        Number of continuous covariates.
    inject_covariates : bool
        Whether FCLayers should append covariates to the first layer.
    n_heads : int
        Number of self-attention heads.
    ff_dim : int | None
        Inner width of Transformer feed-forward network; defaults to 4×D.
    num_transformer_layers : int
        How many encoder layers to stack.
    """

    def __init__(
        self,
        input_dim: int,
        code_dim: int,
        latent_dim: int,
        encoder_hidden_dim: int,
        dropout_rate: float = 0.0,
        n_cat_list: Iterable[int] | None = None,
        n_cont: int = 0,
        inject_covariates: bool = True,
        n_heads: int = 2,
        ff_dim: int | None = None,
        num_transformer_layers: int = 1,
    ):
        super().__init__()

        # ------------------------------------------------------------------ #
        # 1. Junction “identity” embeddings                                  #
        # ------------------------------------------------------------------ #
        self.feature_embedding = nn.Parameter(
            torch.empty(input_dim, code_dim)
        )
        self.code_dim = code_dim
        nn.init.xavier_uniform_(self.feature_embedding)  # swap in PCA init if available

        # ------------------------------------------------------------------ #
        # 2. Transformer over observed-junction tokens                       #
        # ------------------------------------------------------------------ #
        ff_dim = ff_dim or (code_dim * 4)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=code_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )

        # ------------------------------------------------------------------ #
        # 3. MLP → latent μ, logσ²                                           #
        # ------------------------------------------------------------------ #
        self.encoder_mlp = FCLayers(
            n_in=code_dim,
            n_out=2 * latent_dim,
            n_cat_list=list(n_cat_list or []),
            n_cont=n_cont,
            n_layers=2,
            n_hidden=encoder_hidden_dim,
            dropout_rate=dropout_rate,
            use_batch_norm=False,
            use_layer_norm=True,
            inject_covariates=inject_covariates,
        )

    # ---------------------------------------------------------------------- #
    # Forward                                                                #
    # ---------------------------------------------------------------------- #
    # def forward(
    #     self,
    #     x: torch.Tensor,          # (B, J)   – PSI / expression values ∈ [0, 1]
    #     mask: torch.Tensor,       # (B, J)   – 1 where observed
    #     *cat_list: torch.Tensor,  # optional categorical covariates
    #     cont: torch.Tensor | None = None,  # optional continuous covariates
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Returns
    #     -------
    #     mu, logvar : (B, Z) tensors
    #         Mean and log-variance of q(z | x) for each cell.
    #     """
    #     B, J = x.shape
    #     device = x.device
    #     cell_reprs = []

    #     for b in range(B):
    #         obs_idx = mask[b].nonzero(as_tuple=False).squeeze(1)  # (n_obs,)
    #         if obs_idx.numel() == 0:
    #             # If a cell somehow has no observed junctions, fall back to zeros
    #             cell_reprs.append(torch.zeros(self.feature_embedding.size(1),
    #                                           device=device))
    #             continue

    #         # (n_obs, D) embeddings × scalar PSI
    #         tokens = (
    #             self.feature_embedding[obs_idx] *
    #             x[b, obs_idx].unsqueeze(-1)
    #         )  # shape (n_obs, D)

    #         # Transformer expects (batch, seq, dim)
    #         attended = self.transformer(tokens.unsqueeze(0))  # (1, n_obs, D)

    #         # Mean-pool over the variable-length sequence → (D,)
    #         cell_repr = attended.mean(dim=1).squeeze(0)
    #         cell_reprs.append(cell_repr)

    #     # Stack back into (B, D)
    #     c = torch.stack(cell_reprs, dim=0)

    #     # Project to latent parameters
    #     mu_logvar = self.encoder_mlp(c, *cat_list, cont=cont)
    #     mu, logvar = mu_logvar.chunk(2, dim=-1)
    #     return mu, logvar

    def forward(
        self,
        x: torch.Tensor,              # (B, J)
        mask: torch.Tensor,           # (B, J)
        *cat_list: torch.Tensor,
        cont: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # --- call counter to limit prints ---
        if not hasattr(self, "_fw_calls"):
            self._fw_calls = 0
        self._fw_calls += 1
        do_print = self._fw_calls <= 10

        import time
        t0 = time.time()

        B, J = x.shape
        device = x.device

        # 1) compute lengths and update max_obs
        t1 = time.time()
        mask_long = mask.to(torch.long)           # (B, J)
        lengths   = mask_long.sum(dim=1)          # (B,)
        batch_max = int(lengths.max().item())
        if not hasattr(self, "max_obs"):
            self.max_obs = batch_max
        else:
            self.max_obs = max(self.max_obs, batch_max)
        L = self.max_obs
        t2 = time.time()

        # 2) topk indices
        _, idxs = mask_long.topk(L, dim=1)        # (B, L)
        t3 = time.time()

        # 3) gather embeddings & weights
        emb_tokens = self.feature_embedding[idxs]            # (B, L, D)
        x_weights  = x.gather(1, idxs).unsqueeze(-1)         # (B, L, 1)
        tokens     = emb_tokens * x_weights                  # (B, L, D)
        t4 = time.time()

        # 4) build pad mask
        pad_mask = (
            torch.arange(L, device=device)[None, :].expand(B, L)
            >= lengths[:, None]
        )                                                    # (B, L)
        t5 = time.time()

        # 5) transformer
        out = self.transformer(tokens, src_key_padding_mask=pad_mask)
        t6 = time.time()

        # 6) masked mean‐pool
        keep   = (~pad_mask).unsqueeze(-1).float()           # (B, L, 1)
        summed = (out * keep).sum(dim=1)                     # (B, D)
        c      = summed / (lengths.unsqueeze(-1).float() + 1e-8)
        t7 = time.time()

        # 7) MLP → [μ‖log σ²]
        mu_logvar = self.encoder_mlp(c, *cat_list, cont=cont)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        t8 = time.time()

        # print timing breakdown for first few forwards
        # ── print timing + max_obs for first few forwards ─────────────────
        if do_print:
            print(
                f"[Forward #{self._fw_calls}] total={t8-t0:.4f}s  "
                f"max_obs={self.max_obs}  "
                f"len={t2-t1:.4f}s  topk={t3-t2:.4f}s  "
                f"gather={t4-t3:.4f}s  mask={t5-t4:.4f}s  "
                f"transform={t6-t5:.4f}s  pool={t7-t6:.4f}s  "
                f"mlp={t8-t7:.4f}s"
            )

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
            "PartialEncoder",
            "PartialEncoderImpute",
            "PartialEncoderWeightedSum",
            "PartialEncoderTransformer",
        ] = "PartialEncoder",
        junction_inclusion: Literal["all_junctions", "observed_junctions"] = "all_junctions",
        pool_mode: Literal["mean","sum"] = "mean",
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.junction_inclusion = junction_inclusion

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

        # instantiate the requested encoder
        if encoder_type == "PartialEncoder":
            print(f"Using Regular Partial Encoder")
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
                pool_mode=pool_mode,
            )
        elif encoder_type == "PartialEncoderImpute":
            print("Using Impute encoder")
            self.encoder = PartialEncoderImpute(
                input_dim=n_input,
                code_dim=code_dim,
                h_hidden_dim=h_hidden_dim,
                encoder_hidden_dim=encoder_hidden_dim,
                latent_dim=n_latent,
                dropout_rate=dropout_rate,
                n_cat_list=encoder_cat_list,
                n_cont=n_continuous_cov,
                inject_covariates=encode_covariates,
                junction_inclusion=junction_inclusion,
            )
        elif encoder_type == "PartialEncoderWeightedSum":
            print("Using WeightedSum encoder")
            self.encoder = PartialEncoderWeightedSum(
                input_dim=n_input,
                code_dim=code_dim,
                h_hidden_dim=h_hidden_dim,
                encoder_hidden_dim=encoder_hidden_dim,
                latent_dim=n_latent,
                dropout_rate=dropout_rate,
                n_cat_list=encoder_cat_list,
                n_cont=n_continuous_cov,
                inject_covariates=encode_covariates,
                junction_inclusion=junction_inclusion,
            )
        elif encoder_type == "PartialEncoderTransformer":
            print("Using Transformer encoder")
            self.encoder = PartialEncoderTransformer(
                input_dim=n_input,
                code_dim=code_dim,
                encoder_hidden_dim=encoder_hidden_dim,
                latent_dim=n_latent,
                dropout_rate=dropout_rate,
                n_cat_list=encoder_cat_list,
                n_cont=n_continuous_cov,
                inject_covariates=encode_covariates,
                n_heads=n_heads,
                ff_dim=ff_dim,
                num_transformer_layers=num_transformer_layers,
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