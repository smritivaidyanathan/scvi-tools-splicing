from collections.abc import Iterable
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal, Poisson
from torch.distributions import kl_divergence as kld
from torch.nn import functional as F

from scvi import REGISTRY_KEYS
from scvi.distributions import NegativeBinomial, NegativeBinomialMixture, ZeroInflatedNegativeBinomial
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.nn import DecoderSCVI, Encoder, FCLayers, LinearDecoderSCVI
from scvi.module._partialvae import PartialEncoderEDDI, PartialEncoderEDDIATSE,PartialEncoderWeightedSumEDDIMultiWeight, PartialEncoderWeightedSumEDDIMultiWeightATSE, PartialEncoderEDDIFast, PartialEncoderEDDIATSEFast,PartialEncoderWeightedSumEDDIMultiWeightFast, PartialEncoderWeightedSumEDDIMultiWeightATSEFast, PartialEncoderEDDIFaster, PartialEncoderEDDIATSEFaster,PartialEncoderWeightedSumEDDIMultiWeightFaster, PartialEncoderWeightedSumEDDIMultiWeightATSEFaster, LinearDecoder, group_logsumexp, subtract_group_logsumexp, nbetaln

from ._utils import masked_softmax


class LibrarySizeEncoder(torch.nn.Module):
    """Library size encoder for gene expression.

    Encodes library size information from gene expression input.
    """
    def __init__(
        self,
        n_input: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 2,
        n_hidden: int = 128,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        deep_inject_covariates: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            activation_fn=torch.nn.LeakyReLU,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            inject_covariates=deep_inject_covariates,
            **kwargs,
        )
        self.output = torch.nn.Sequential(torch.nn.Linear(n_hidden, 1), torch.nn.LeakyReLU())

    def forward(self, x: torch.Tensor, *cat_list: int):
        return self.output(self.px_decoder(x, *cat_list))


class DecoderSplice(torch.nn.Module):
    """Decoder for alternative splicing junction usage.

    Decodes a latent representation into splicing usage probabilities.
    """
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 2,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        deep_inject_covariates: bool = False,
    ):
        super().__init__()
        self.ps_decoder = FCLayers(
            n_in=n_input,
            n_out=n_output,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )

    def forward(self, z: torch.Tensor, *cat_list: int):
        ps = self.ps_decoder(z, *cat_list)
        return torch.sigmoid(ps)




class MULTIVAESPLICE(BaseModuleClass):
    """
    Variational auto-encoder for joint (paired or unpaired) RNA-seq gene expression
    and alternative splicing (junction usage). Two encoder–decoder branches (expression,
    splicing) produce/posterior latents that are mixed into a shared latent space z.

    Reconstruction is performed per modality; optional penalties align the two posteriors.

    Parameters
    ----------
    # --- Data & bookkeeping ---
    n_input_genes : int
        Number of gene-expression features (G).
    n_input_junctions : int
        Number of splicing junction features (J).
    n_batch : int, default 0
        Number of batches for batch-correction (categorical covariate).
    n_obs : int, default 0
        Number of observations (cells); needed when modality_weights="cell".
    n_labels : int, default 0
        Number of labels (for gene_dispersion="gene-label").
    n_cats_per_cov : Iterable[int] or None, default None
        Category counts for each categorical covariate provided.
    n_continuous_cov : int, default 0
        Number of continuous covariates.

    # --- Likelihoods & dispersion (expression) ---
    gene_likelihood : {"zinb","nb","poisson"}, default "zinb"
        Expression likelihood.
    gene_dispersion : {"gene","gene-batch","gene-label","gene-cell"}, default "gene"
        Dispersion layout for expression.
    use_size_factor_key : bool, default False
        If True, use provided library-size factors for expression (softplus scale);
        else learn a library-size encoder (see LibrarySizeEncoder).

    # --- Architecture toggles ---
    splicing_architecture : {"vanilla","partial"}, default "vanilla"
        • "vanilla": SCVI Encoder + (nonlinear) DecoderSplice (FCLayers).  
        • "partial": PartialEncoder* variants + Linear splicing decoder.
    expression_architecture : {"vanilla","linear"}, default "vanilla"
        Decoder for expression:
        • "vanilla": non-linear DecoderSCVI (FCLayers).  
        • "linear" : LinearDecoderSCVI.

    # --- Shared SCVI-style encoder/decoder hyperparameters (used by expression
    #     encoder/decoder, LibrarySizeEncoder, and splicing *when* splicing_architecture="vanilla") ---
    n_hidden : int or None, default None
        Hidden width for SCVI-style MLPs. If None, defaults to ~sqrt(J) (cap at 128) or sqrt(G).
    n_latent : int or None, default None
        Latent dimensionality. If None, defaults to ~sqrt(n_hidden). When
        modality_weights="concatenate", the *mixed* latent is doubled internally.
    n_layers_encoder : int, default 2
        Depth of SCVI encoders (expression, and splicing if "vanilla").
    n_layers_decoder : int, default 2
        Depth of non-linear decoders (expression "vanilla", splicing "vanilla").
        Not used by linear decoders.

    dropout_rate : float, default 0.1
        Dropout for SCVI-style MLPs.
    use_batch_norm : {"encoder","decoder","none","both"}, default "none"
        Apply BatchNorm to encoder/decoder stacks.
    use_layer_norm : {"encoder","decoder","none","both"}, default "both"
        Apply LayerNorm to encoder/decoder stacks.
    latent_distribution : {"normal","ln"}, default "normal"
        Posterior family for encoders. If "ln" (logistic-normal), the latent sample is
        softmax-transformed before decoding.

    deeply_inject_covariates : bool, default False
        Deeply inject (cat/cont) covariates into decoder layers.
    encode_covariates : bool, default False
        Concatenate continuous covariates to encoder inputs and pass categorical
        covariates via `n_cat_list`.

    # --- Splicing likelihood ---
    splicing_loss_type : {"binomial","beta_binomial","dirichlet_multinomial"}, default "beta_binomial"
        Reconstruction loss for splicing.
        • "binomial": needs (junc_counts, atse_counts).  
        • "beta_binomial": same as binomial but with concentration φ_j (per junction).  
        • "dirichlet_multinomial": uses grouped softmax within ATSEs.
    splicing_concentration : float or None, default None
        Optional scalar concentration (used by beta-binomial if provided).
    dm_concentration : {"atse","scalar"}, default "atse"
        For Dirichlet–multinomial: whether φ is per-ATSE or a global scalar
        (internally mapped to per-junction as needed).

    # --- PartialEncoder (splicing_architecture="partial") knobs ONLY ---
    encoder_type : {"PartialEncoderEDDI","PartialEncoderEDDIATSE",
                    "PartialEncoderWeightedSumEDDIMultiWeight",
                    "PartialEncoderWeightedSumEDDIMultiWeightATSE"}, default "PartialEncoderEDDI"
        Choice of PartialEncoder family/variant (ATSE-aware and/or weighted-sum gating).
    forward_style : {"per-cell","batched","scatter"}, default "batched"
        Implementation variant for speed/memory trade-offs (Fast/Faster classes under the hood).
    code_dim : int, default 16
        Dimensionality of per-junction codes before pooling.
    h_hidden_dim : int, default 64
        Hidden width of the per-junction “h” subnetwork (feature MLP) that builds codes.
    encoder_hidden_dim : int, default 128
        Hidden width of the post-pooling MLP that maps pooled features → (μ, logσ²) of z.
    pool_mode : {"mean","sum"}, default "mean"
        Aggregation of per-junction codes within each cell (and within ATSEs when applicable).
    atse_embedding_dimension : int, default 16
        ATSE-level embedding size (ATSE-aware variants only).
    num_weight_vectors : int, default 4
        Number of mixture weight vectors for MultiWeight variants.
    temperature_value : float, default -1.0
        Initial temperature for gating (if <0, use internal default).
    temperature_fixed : bool, default True
        If True, keep gating temperature fixed; else learn it.
    max_nobs : int, default -1
        (scatter mode) Optional cap on observed entries processed per step.

    # --- Modality mixing ---
    modality_weights : {"equal","cell","universal","concatenate"}, default "equal"
        How to combine expression & splicing posteriors:
        • "equal"     : uniform weights.  
        • "universal" : learn one weight per modality.  
        • "cell"      : learn weights per cell.  
        • "concatenate": do not mix; concat latents (z_expr || z_spl).
    modality_penalty : {"Jeffreys","MMD","None"}, default "Jeffreys"
        Alignment penalty between the two posteriors on paired cells.

    **model_kwargs :
        Forwarded to underlying components (encoders/decoders).

    Notes
    -----
    • SCVI-style hyperparameters (`n_hidden`, `n_layers_encoder/decoder`, `dropout_rate`,
      normalization flags) control the expression branch and the splicing branch only when
      `splicing_architecture="vanilla"`. They do **not** affect PartialEncoder internals.

    • PartialEncoder-specific widths:
        - `h_hidden_dim`   : per-junction feature network width.
        - `encoder_hidden_dim` : post-pooling MLP width (to z stats).
        - `code_dim`       : per-junction code dimensionality before pooling.

    • If `latent_distribution="ln"`, both encoders output logits that are softmaxed into
      simplex-valued latents prior to decoding.

    • Splicing losses:
        - Binomial/Beta-binomial expect `junc_counts` (successes) and `atse_counts`
          (trials). Beta-binomial uses φ_j = softplus(log_phi_j); an L2 prior on log_phi_j
          is applied during training.
        - Dirichlet–multinomial performs ATSE-wise softmax and supports φ as scalar or
          per-ATSE; values are mapped to per-junction when needed.

    • When `modality_weights="concatenate"`, the *mixed* latent doubles its size
      (z = [z_expr, z_spl]); penalties are skipped in this mode.

    • Protein modality is not supported. AnnData/MuData handling occurs in the model
      wrapper (AnnDataManager in setup).

    """

    def __init__(
        self,
        # --- Data & bookkeeping ---
        n_input_genes: int = 0,
        n_input_junctions: int = 0,
        n_batch: int = 0,
        n_obs: int = 0,
        n_labels: int = 0,
        n_cats_per_cov: Iterable[int] | None = None,
        n_continuous_cov: int = 0,

        # --- Likelihoods & dispersion (expression) ---
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        gene_dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        use_size_factor_key: bool = False,

        # --- Architecture toggles ---
        splicing_architecture: Literal["vanilla", "partial"] = "vanilla",
        expression_architecture: Literal["vanilla", "linear"] = "vanilla",

        # --- Shared SCVI-style encoder/decoder hyperparameters ---
        n_hidden: int = None,  # width for SCVI encoders/decoders
        n_latent: int = None,  # latent dim for SCVI encoders/decoders
        n_layers_encoder: int = 2,
        n_layers_decoder: int = 2,  # not used by linear decoder
        dropout_rate: float = 0.1,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        latent_distribution: Literal["normal", "ln"] = "normal",
        deeply_inject_covariates: bool = False,
        encode_covariates: bool = False,

        # --- Splicing likelihood ---
        splicing_loss_type: Literal["binomial", "beta_binomial", "dirichlet_multinomial"] = "beta_binomial",
        splicing_concentration: float | None = None,
        dm_concentration: Literal["atse", "scalar"] = "atse",

        # --- PartialEncoder (splicing_architecture="partial") knobs ---
        encoder_type: Literal[
            "PartialEncoderWeightedSumEDDIMultiWeight",
            "PartialEncoderWeightedSumEDDIMultiWeightATSE",
            "PartialEncoderEDDI",
            "PartialEncoderEDDIATSE"
        ] = "PartialEncoderEDDI",
        forward_style: Literal["per-cell", "batched", "scatter"] = "batched",
        code_dim: int = 16,
        h_hidden_dim: int = 64,
        encoder_hidden_dim: int = 128,
        pool_mode: Literal["mean", "sum"] = "mean",
        atse_embedding_dimension: int = 16,
        num_weight_vectors: int = 4,
        temperature_value: float = -1.0,
        temperature_fixed: bool = True,
        max_nobs: int = -1,

        # --- Modality mixing ---
        modality_weights: Literal["equal", "cell", "universal", "concatenate"] = "equal",
        modality_penalty: Literal["Jeffreys", "MMD", "None"] = "Jeffreys",

        # --- Misc ---
        **model_kwargs,
    ):

        super().__init__()
        self.n_input_genes = n_input_genes
        self.n_input_junctions = n_input_junctions

        if n_hidden is None:
            self.n_hidden = np.min([128, int(np.sqrt(n_input_junctions))]) if n_input_junctions > 0 else int(np.sqrt(n_input_genes))
        else:
            self.n_hidden = n_hidden

        self.n_batch = n_batch
        self.gene_likelihood = gene_likelihood
        self.latent_distribution = latent_distribution

        base_latent = int(np.sqrt(self.n_hidden)) if n_latent is None else n_latent
        self.encoder_latent_dim = base_latent
        self.n_latent = base_latent * (2 if modality_weights == "concatenate" else 1)

        self.n_layers_encoder = n_layers_encoder
        self.n_layers_decoder = n_layers_decoder
        self.n_cats_per_cov = n_cats_per_cov
        self.n_continuous_cov = n_continuous_cov
        self.dropout_rate = dropout_rate


        self.use_batch_norm_encoder = use_batch_norm in ("encoder", "both")
        self.use_batch_norm_decoder = use_batch_norm in ("decoder", "both")
        self.use_layer_norm_encoder = use_layer_norm in ("encoder", "both")
        self.use_layer_norm_decoder = use_layer_norm in ("decoder", "both")
        self.encode_covariates = encode_covariates
        self.deeply_inject_covariates = deeply_inject_covariates
        self.use_size_factor_key = use_size_factor_key

        # New splicing parameters
        self.splicing_loss_type = splicing_loss_type
        self.splicing_concentration = splicing_concentration
        self.splicing_architecture = splicing_architecture
        self.code_dim = code_dim
        self.h_hidden_dim = h_hidden_dim
        self.dm_concentration = dm_concentration

        cat_list = [n_batch] + list(n_cats_per_cov) if n_cats_per_cov is not None else []
        encoder_cat_list = cat_list if encode_covariates else None

        # ---------------- Expression Branch ----------------
        self.gene_dispersion = gene_dispersion
        if self.gene_dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_genes))
        elif self.gene_dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_genes, n_batch))
        elif self.gene_dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_genes, n_labels))
        elif self.gene_dispersion == "gene-cell":
            pass
        else:
            raise ValueError("gene_dispersion must be one of ['gene', 'gene-batch', 'gene-label', 'gene-cell']")
        input_exp = n_input_genes if n_input_genes > 0 else 1
        n_input_encoder_exp = input_exp + n_continuous_cov * int(encode_covariates)
        self.z_encoder_expression = Encoder(
            n_input=n_input_encoder_exp,
            n_output=self.encoder_latent_dim,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers_encoder,
            n_hidden=self.n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=self.use_batch_norm_encoder,
            use_layer_norm=self.use_layer_norm_encoder,
            activation_fn=torch.nn.LeakyReLU,
            var_eps=0,
            return_dist=False,
        )
        self.l_encoder_expression = LibrarySizeEncoder(
            n_input_encoder_exp,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers_encoder,
            n_hidden=self.n_hidden,
            use_batch_norm=self.use_batch_norm_encoder,
            use_layer_norm=self.use_layer_norm_encoder,
            deep_inject_covariates=deeply_inject_covariates,
        )

        n_input_decoder = self.n_latent + self.n_continuous_cov

        if expression_architecture == "vanilla":
            self.z_decoder_expression = DecoderSCVI(
                n_input_decoder,
                n_input_genes,
                n_cat_list=cat_list,
                n_layers=n_layers_decoder,
                n_hidden=self.n_hidden,
                inject_covariates=deeply_inject_covariates,
                use_batch_norm=self.use_batch_norm_decoder,
                use_layer_norm=self.use_layer_norm_decoder,
                scale_activation="softplus" if use_size_factor_key else "softmax",
            )
        else:
            self.z_decoder_expression = LinearDecoderSCVI(
                n_input_decoder,
                n_input_genes,
                n_cat_list=cat_list,
                use_batch_norm=self.use_batch_norm_decoder,
                use_layer_norm=self.use_layer_norm_decoder,
            )

        # ---------------- Splicing Branch ----------------
        input_spl = n_input_junctions if n_input_junctions > 0 else 1
        n_input_encoder_spl = input_spl + n_continuous_cov * int(encode_covariates)
        
        print("Initializing Log Phi Concentration parameter (if relevant)")
        # Initialize log_phi_j with a value of 100.0
        if self.splicing_loss_type == "beta_binomial":
            self.log_phi_j = nn.Parameter(torch.randn(n_input_junctions) * 0.5 + np.log(100.0))
            self.log_phi_j.requires_grad_(True)
        elif self.splicing_loss_type == "dirichlet_multinomial": 
            self.log_phi_j = nn.Parameter(torch.tensor(4.6))
            #if dm, once the anndata is set we will set the per atse concentration there in multivisplice.
            self.log_phi_j.requires_grad_(True)
            #later change to per atse?
        else:
            self.log_phi_j = nn.Parameter(torch.tensor(0.0))
            self.log_phi_j.requires_grad_(False)

        

        if (splicing_architecture=="vanilla"):
            self.z_encoder_splicing = Encoder(
                n_input=n_input_encoder_spl,
                n_layers=n_layers_encoder,
                n_output=self.encoder_latent_dim,
                n_hidden=self.n_hidden,
                n_cat_list=encoder_cat_list,
                dropout_rate=dropout_rate,
                activation_fn=torch.nn.LeakyReLU,
                distribution=latent_distribution,
                var_eps=0,
                use_batch_norm=self.use_batch_norm_encoder,
                use_layer_norm=self.use_layer_norm_encoder,
                return_dist=False,
            )
            self.z_decoder_splicing = DecoderSplice(
                n_input=n_input_decoder,
                n_output=n_input_junctions,
                n_cat_list=cat_list,
                n_layers=n_layers_decoder,
                n_hidden=self.n_hidden,
                dropout_rate=dropout_rate,
                use_batch_norm=self.use_batch_norm_decoder,
                use_layer_norm=self.use_layer_norm_decoder,
                deep_inject_covariates=deeply_inject_covariates,
            )
        else:

            # instantiate the requested encoder
            if forward_style == "per-cell":
                if encoder_type == "PartialEncoderEDDI":
                    print(f"Using EDDI Partial Encoder")
                    self.z_encoder_splicing = PartialEncoderEDDI(
                        input_dim=input_spl,
                        code_dim=code_dim,
                        h_hidden_dim=h_hidden_dim,
                        encoder_hidden_dim=encoder_hidden_dim,
                        latent_dim=self.encoder_latent_dim,
                        dropout_rate=dropout_rate,
                        n_cat_list=encoder_cat_list,
                        n_cont=n_continuous_cov,
                        inject_covariates=encode_covariates,
                        pool_mode=pool_mode,
                    )

                elif encoder_type == "PartialEncoderEDDIATSE":
                    print("Using EDDI + ATSE Partial Encoder")
                    self.z_encoder_splicing = PartialEncoderEDDIATSE(
                        input_dim=input_spl,
                        code_dim=code_dim,
                        h_hidden_dim=h_hidden_dim,
                        encoder_hidden_dim=encoder_hidden_dim,
                        latent_dim=self.encoder_latent_dim,
                        dropout_rate=dropout_rate,
                        n_cat_list=encoder_cat_list,
                        n_cont=n_continuous_cov,
                        inject_covariates=encode_covariates,
                        pool_mode=pool_mode,
                        atse_embedding_dimension=atse_embedding_dimension
                    )
                
                elif encoder_type == "PartialEncoderWeightedSumEDDIMultiWeight":
                    print("Using PartialEncoderWeightedSumEDDIMultiWeight")
                    self.z_encoder_splicing = PartialEncoderWeightedSumEDDIMultiWeight(
                        input_dim=input_spl,
                        code_dim=code_dim,
                        h_hidden_dim=h_hidden_dim,
                        encoder_hidden_dim=encoder_hidden_dim,
                        latent_dim=self.encoder_latent_dim,
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
                    self.z_encoder_splicing = PartialEncoderWeightedSumEDDIMultiWeightATSE(
                        input_dim=input_spl,
                        code_dim=code_dim,
                        h_hidden_dim=h_hidden_dim,
                        encoder_hidden_dim=encoder_hidden_dim,
                        latent_dim=self.encoder_latent_dim,
                        dropout_rate=dropout_rate,
                        n_cat_list=encoder_cat_list,
                        n_cont=n_continuous_cov,
                        inject_covariates=encode_covariates,
                        num_weight_vectors = num_weight_vectors,
                        temperature_value=temperature_value,
                        temperature_fixed=temperature_fixed,
                        atse_embedding_dimension=atse_embedding_dimension
                    )
            elif forward_style == "batched":
                if encoder_type == "PartialEncoderEDDI":
                    print(f"Using EDDI Partial Encoder Fast")
                    self.z_encoder_splicing = PartialEncoderEDDIFast(
                        input_dim=input_spl,
                        code_dim=code_dim,
                        h_hidden_dim=h_hidden_dim,
                        encoder_hidden_dim=encoder_hidden_dim,
                        latent_dim=self.encoder_latent_dim,
                        dropout_rate=dropout_rate,
                        n_cat_list=encoder_cat_list,
                        n_cont=n_continuous_cov,
                        inject_covariates=encode_covariates,
                        pool_mode=pool_mode,
                    )

                elif encoder_type == "PartialEncoderEDDIATSE":
                    print("Using EDDI + ATSE Partial Encoder Fast")
                    self.z_encoder_splicing = PartialEncoderEDDIATSEFast(
                        input_dim=input_spl,
                        code_dim=code_dim,
                        h_hidden_dim=h_hidden_dim,
                        encoder_hidden_dim=encoder_hidden_dim,
                        latent_dim=self.encoder_latent_dim,
                        dropout_rate=dropout_rate,
                        n_cat_list=encoder_cat_list,
                        n_cont=n_continuous_cov,
                        inject_covariates=encode_covariates,
                        pool_mode=pool_mode,
                        atse_embedding_dimension=atse_embedding_dimension
                    )
                
                elif encoder_type == "PartialEncoderWeightedSumEDDIMultiWeight":
                    print("Using PartialEncoderWeightedSumEDDIMultiWeight Fast")
                    self.z_encoder_splicing = PartialEncoderWeightedSumEDDIMultiWeightFast(
                        input_dim=input_spl,
                        code_dim=code_dim,
                        h_hidden_dim=h_hidden_dim,
                        encoder_hidden_dim=encoder_hidden_dim,
                        latent_dim=self.encoder_latent_dim,
                        dropout_rate=dropout_rate,
                        n_cat_list=encoder_cat_list,
                        n_cont=n_continuous_cov,
                        inject_covariates=encode_covariates,
                        num_weight_vectors = num_weight_vectors,
                        temperature_value=temperature_value,
                        temperature_fixed=temperature_fixed,
                    )

                elif encoder_type == "PartialEncoderWeightedSumEDDIMultiWeightATSE":
                    print("Using PartialEncoderWeightedSumEDDIMultiWeightATSE Fast")
                    self.z_encoder_splicing = PartialEncoderWeightedSumEDDIMultiWeightATSEFast(
                        input_dim=input_spl,
                        code_dim=code_dim,
                        h_hidden_dim=h_hidden_dim,
                        encoder_hidden_dim=encoder_hidden_dim,
                        latent_dim=self.encoder_latent_dim,
                        dropout_rate=dropout_rate,
                        n_cat_list=encoder_cat_list,
                        n_cont=n_continuous_cov,
                        inject_covariates=encode_covariates,
                        num_weight_vectors = num_weight_vectors,
                        temperature_value=temperature_value,
                        temperature_fixed=temperature_fixed,
                        atse_embedding_dimension=atse_embedding_dimension
                    )
            elif forward_style == "scatter":
                if encoder_type == "PartialEncoderEDDI":
                    print(f"Using EDDI Partial Encoder Faster")
                    self.z_encoder_splicing = PartialEncoderEDDIFaster(
                        input_dim=input_spl,
                        code_dim=code_dim,
                        h_hidden_dim=h_hidden_dim,
                        encoder_hidden_dim=encoder_hidden_dim,
                        latent_dim=self.encoder_latent_dim,
                        dropout_rate=dropout_rate,
                        n_cat_list=encoder_cat_list,
                        n_cont=n_continuous_cov,
                        inject_covariates=encode_covariates,
                        pool_mode=pool_mode,
                        max_nobs = max_nobs,
                    )

                elif encoder_type == "PartialEncoderEDDIATSE":
                    print("Using EDDI + ATSE Partial Encoder Faster")
                    self.z_encoder_splicing = PartialEncoderEDDIATSEFaster(
                        input_dim=input_spl,
                        code_dim=code_dim,
                        h_hidden_dim=h_hidden_dim,
                        encoder_hidden_dim=encoder_hidden_dim,
                        latent_dim=self.encoder_latent_dim,
                        dropout_rate=dropout_rate,
                        n_cat_list=encoder_cat_list,
                        n_cont=n_continuous_cov,
                        inject_covariates=encode_covariates,
                        pool_mode=pool_mode,
                        atse_embedding_dimension=atse_embedding_dimension,
                    )

                elif encoder_type == "PartialEncoderWeightedSumEDDIMultiWeight":
                    print("Using PartialEncoderWeightedSumEDDIMultiWeight Faster")
                    self.z_encoder_splicing = PartialEncoderWeightedSumEDDIMultiWeightFaster(
                        input_dim=input_spl,
                        code_dim=code_dim,
                        h_hidden_dim=h_hidden_dim,
                        encoder_hidden_dim=encoder_hidden_dim,
                        latent_dim=self.encoder_latent_dim,
                        dropout_rate=dropout_rate,
                        n_cat_list=encoder_cat_list,
                        n_cont=n_continuous_cov,
                        inject_covariates=encode_covariates,
                        num_weight_vectors=num_weight_vectors,
                        temperature_value=temperature_value,
                        temperature_fixed=temperature_fixed,
                    )

                elif encoder_type == "PartialEncoderWeightedSumEDDIMultiWeightATSE":
                    print("Using PartialEncoderWeightedSumEDDIMultiWeightATSE Faster")
                    self.z_encoder_splicing = PartialEncoderWeightedSumEDDIMultiWeightATSEFaster(
                        input_dim=input_spl,
                        code_dim=code_dim,
                        h_hidden_dim=h_hidden_dim,
                        encoder_hidden_dim=encoder_hidden_dim,
                        latent_dim=self.encoder_latent_dim,
                        dropout_rate=dropout_rate,
                        n_cat_list=encoder_cat_list,
                        n_cont=n_continuous_cov,
                        inject_covariates=encode_covariates,
                        num_weight_vectors=num_weight_vectors,
                        temperature_value=temperature_value,
                        temperature_fixed=temperature_fixed,
                        atse_embedding_dimension=atse_embedding_dimension,
                    )

                else:
                    raise ValueError(f"Unknown encoder_type={encoder_type!r}")
                

            if latent_distribution == "ln":
                self.z_encoder_splicing.z_transformation = nn.Softmax(dim=-1)
            else:
                self.z_encoder_splicing.z_transformation = lambda x: x

            input_linear_splicing_decoder = self.n_latent

            self.z_decoder_splicing = LinearDecoder(
                latent_dim=input_linear_splicing_decoder,
                output_dim=n_input_junctions,
                n_cat_list=cat_list,
                n_cont=n_continuous_cov,
            )

        # ---------------- Modality Alignment ----------------
        self.n_obs = n_obs
        self.modality_weights = modality_weights
        self.modality_penalty = modality_penalty
        self.n_modalities = int(n_input_genes > 0) + int(n_input_junctions > 0)
        max_n_modalities = 2
        if modality_weights == "equal":
            self.register_buffer("mod_weights", torch.ones(max_n_modalities))
        elif modality_weights == "universal":
            self.mod_weights = torch.nn.Parameter(torch.ones(max_n_modalities))
        else:
            self.mod_weights = torch.nn.Parameter(torch.ones(n_obs, max_n_modalities))
        
        # gate that controls how much of the "other" half a decoder can see (0=off, 1=on)
        self.register_buffer("cross_gate", torch.tensor(0.0))  # start closed during warmup

    def set_cross_gate(self, value: float):
        # value in [0,1]; keep as buffer so it's not optimized
        self.cross_gate.fill_(float(value))



    def _get_inference_input(self, tensors):
        """Assemble inputs for the inference network from registered fields."""
        x = tensors.get(REGISTRY_KEYS.X_KEY, None)
        x_junc = tensors.get(REGISTRY_KEYS.JUNC_RATIO_X_KEY, None)
        if x is not None and x_junc is not None:
            x = torch.cat((x, x_junc), dim=-1)
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        cell_idx = tensors.get(REGISTRY_KEYS.INDICES_KEY).long().ravel()
        cont_covs = tensors.get(REGISTRY_KEYS.CONT_COVS_KEY)
        cat_covs = tensors.get(REGISTRY_KEYS.CAT_COVS_KEY)
        label = tensors[REGISTRY_KEYS.LABELS_KEY]
        psi_mask = tensors.get(REGISTRY_KEYS.PSI_MASK_KEY)
        size_factor = tensors.get(REGISTRY_KEYS.SIZE_FACTOR_KEY, None)
        return {
            "x": x,
            "mask": psi_mask,
            "batch_index": batch_index,
            "cont_covs": cont_covs,
            "cat_covs": cat_covs,
            "label": label,
            "cell_idx": cell_idx,
            "size_factor": size_factor,
        }

    @auto_move_data
    def inference(self, x, mask, batch_index, cont_covs, cat_covs, label, cell_idx, size_factor, n_samples=1) -> dict[str, torch.Tensor]:
        """Run the inference network.

        Splits input x into gene expression and splicing parts, encodes each branch, and mixes their latent representations.
        """
        # Get Data and Additional Covs
        x_expr = x[:, : self.n_input_genes]
        x_spl = x[:, self.n_input_genes : (self.n_input_genes + self.n_input_junctions)]
        mask_expr = x_expr.sum(dim=1) > 0
        mask_spl = x_spl.sum(dim=1) > -10000000000000

        if cont_covs is not None and self.encode_covariates:
            encoder_input_expr = torch.cat((x_expr, cont_covs), dim=-1)
            encoder_input_spl = torch.cat((x_spl, cont_covs), dim=-1)
        else:
            encoder_input_expr = x_expr
            encoder_input_spl = x_spl

        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        # Z Encoders

        qzm_expr, qzv_expr, z_expr = self.z_encoder_expression(encoder_input_expr, batch_index, *categorical_input)

        # print(
        #     "ExprEncoder:",
        #     f"qzm_expr NaNs={torch.isnan(qzm_expr).sum().item()},",
        #     f"qzv_expr NaNs={torch.isnan(qzv_expr).sum().item()}"
        # )

        if self.splicing_architecture == "vanilla":
            # your existing SCVI encoder
            qzm_spl, qzv_spl, z_spl = self.z_encoder_splicing(
                encoder_input_spl, batch_index, *categorical_input
            )
        else:
            # PartialEncoder gives (mu, raw_logvar)
            #print(f"x_spl min/max={x_spl.min().item():.3e}/{x_spl.max().item():.3e}, mask sum={mask.sum().item()}")

            mu, raw_logvar = self.z_encoder_splicing(
                x_spl, mask, batch_index, *categorical_input, cont=cont_covs
            )

            # print(
            #     "PartialEncoder output:",
            #     f"mu shape={tuple(mu.shape)},  raw_logvar shape={tuple(raw_logvar.shape)}",
            #     f"  mu NaNs={torch.isnan(mu).sum().item()},",
            #     f" raw_logvar NaNs={torch.isnan(raw_logvar).sum().item()}",
            # )
            # clamp + exponentiate to get variance
            # logvar = torch.clamp(raw_logvar, min=-5.0, max=5.0)
            # var = torch.exp(logvar)

            var = F.softplus(raw_logvar) + 1e-6

            # print(
            #     f"  var min/max = {var.min().item():.3e}/{var.max().item():.3e},",
            #     f" var NaNs ={torch.isnan(var).sum().item()}",
            # )
            # build Normal dist
            qz_dist = Normal(mu, torch.sqrt(var))
            # sample / rsample
            if n_samples > 1:
                z_spl = qz_dist.sample((n_samples,))
            else:
                z_spl = qz_dist.rsample()

            # print(
            #     f"  z_spl shape={tuple(z_spl.shape)}, NaNs={torch.isnan(z_spl).sum().item()}"
            # )
            # apply any transformation (softmax for "ln")
            z_spl = self.z_encoder_splicing.z_transformation(z_spl)
            # now set the “posterior‐stats” to exactly the same names as vanilla
            qzm_spl = mu
            qzv_spl = var



        # L encoder
        if self.use_size_factor_key:
            libsize_expr = torch.log(size_factor[:, [0]] + 1e-6)
        else:
            libsize_expr = self.l_encoder_expression(
                encoder_input_expr, batch_index, *categorical_input
            )


        # mix representations

        if self.modality_weights == "concatenate":
            # just glue the two posterior stats end-to-end
            qz_m = torch.cat((qzm_expr, qzm_spl), dim=1)
            qz_v = torch.cat((qzv_expr, qzv_spl), dim=1)
        else:
            if self.modality_weights == "cell":
                weights = self.mod_weights[cell_idx, :]
            else:
                weights = self.mod_weights.unsqueeze(0).expand(len(cell_idx), -1)

            qz_m = mix_modalities((qzm_expr, qzm_spl), (mask_expr, mask_spl), weights)
            qz_v = mix_modalities((qzv_expr, qzv_spl), (mask_expr, mask_spl), weights, torch.sqrt)
            qz_v = torch.clamp(qz_v, min=1e-6) #please double check this variance logic. i think in multivae they treat it like std sometimes and variance other times. need to make it consistet at least in our code

        # print(
        #     "After mix:",
        #     f"qz_m NaNs={torch.isnan(qz_m).sum().item()},",
        #     f"qz_v NaNs={torch.isnan(qz_v).sum().item()}"
        # )


        # sample
        if n_samples > 1:
            def unsqz(zt, n_s):
                return zt.unsqueeze(0).expand((n_s, *zt.shape))
            untran_z_expr = Normal(qzm_expr, qzv_expr.sqrt()).sample((n_samples,))
            z_expr = self.z_encoder_expression.z_transformation(untran_z_expr)
            untran_z_spl = Normal(qzm_spl, qzv_spl.sqrt()).sample((n_samples,))
            z_spl = self.z_encoder_splicing.z_transformation(untran_z_spl)
            libsize_expr = unsqz(libsize_expr, n_samples)


        # sample from the mixed representation
        untran_z = Normal(qz_m, qz_v.sqrt()).rsample()
        z = self.z_encoder_expression.z_transformation(untran_z)

        return {
            "z": z,
            "qz_m": qz_m,
            "qz_v": qz_v,
            "z_expr": z_expr,
            "qzm_expr": qzm_expr,
            "qzv_expr": qzv_expr,
            "z_spl": z_spl,
            "qzm_spl": qzm_spl,
            "qzv_spl": qzv_spl,
            "libsize_expr": libsize_expr,
            "x": x,
        }

    def _get_generative_input(self, tensors, inference_outputs, transform_batch=None):
        """Get the input for the generative model."""
        z = inference_outputs["z"]
        qz_m = inference_outputs["qz_m"]
        libsize_expr = inference_outputs["libsize_expr"]

        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        label = tensors[REGISTRY_KEYS.LABELS_KEY]

        input_dict = {
            "z": z,
            "qz_m": qz_m,
            "batch_index": batch_index,
            "cont_covs": cont_covs,
            "cat_covs": cat_covs,
            "libsize_expr": libsize_expr,
            "label": label,
        }
        return input_dict

    @auto_move_data
    def generative(self, z, qz_m, batch_index, cont_covs=None, cat_covs=None, libsize_expr=None, use_z_mean=False, label: torch.Tensor = None):
        """Run the generative model to decode gene expression and splicing.

        Decodes the latent representation into parameters for gene expression reconstruction
        and splicing probabilities.

        Returns
        -------
        dict
            A dictionary with keys:
              - "p": decoded splicing probabilities,
              - "px_scale", "px_rate", "px_dropout": gene expression decoder outputs.
        """
        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        latent = z if not use_z_mean else qz_m

        # split halves only if you’re concatenating
        def _attach_cont(rep):
            if cont_covs is None:
                return rep
            elif rep.dim() != cont_covs.dim():
                return torch.cat([rep, cont_covs.unsqueeze(0).expand(rep.size(0), -1, -1)], dim=-1)
            else:
                return torch.cat([rep, cont_covs], dim=-1)

        if self.modality_weights == "concatenate":
            d = self.encoder_latent_dim
            z_e, z_s = latent.split(d, dim=-1)

            gate = self.cross_gate  # 0 during warmup, 1 after
            # block cross-gradients *and* scale by gate
            e_to_s = (z_e * gate) # expression half going into splicing decoder
            s_to_e = (z_s * gate)  # splicing   half going into expression decoder

            dec_in_expr = torch.cat([z_e, s_to_e], dim=-1)
            dec_in_spl  = torch.cat([e_to_s, z_s], dim=-1)
        else:
            # not concatenating → same latent for both
            dec_in_expr = latent
            dec_in_spl  = latent

        decoder_input_expr = _attach_cont(dec_in_expr)
        # NOTE: partial splicing decoder expects cont covs via arg, not concatenated
        decoder_input_spl  = _attach_cont(dec_in_spl) if self.splicing_architecture=="vanilla" else dec_in_spl

        # Splicing
        if self.splicing_architecture == "vanilla":
            p_s = self.z_decoder_splicing(decoder_input_spl, batch_index, *categorical_input)
        else:
            p_s_logits = self.z_decoder_splicing(dec_in_spl, batch_index, *categorical_input, cont=cont_covs)
            p_s = torch.sigmoid(p_s_logits)

        # Expression
        px_scale, _, px_rate, px_dropout = self.z_decoder_expression(
            self.gene_dispersion,
            decoder_input_expr,   # << use expr-specific input
            libsize_expr,
            batch_index,
            *categorical_input,
            label,
        )
        
        # Expression Dispersion
        if self.gene_dispersion == "gene-label":
            px_r = F.linear(
                F.one_hot(label.squeeze(-1), self.n_labels).float(), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
        elif self.gene_dispersion == "gene-batch":
            px_r = F.linear(F.one_hot(batch_index.squeeze(-1), self.n_batch).float(), self.px_r)
        elif self.gene_dispersion == "gene":
            px_r = self.px_r
        px_r = torch.exp(px_r)
        return {
            "p": p_s, # mean psi 
            "phi": F.softplus(self.log_phi_j), # φ ≈ 100, with overflow protection
            "px_scale": px_scale,
            "px_r": px_r,
            "px_rate": px_rate,
            "px_dropout": px_dropout,
        }

    def loss(self, tensors, inference_outputs, generative_outputs, kl_weight: float = 1.0):
        """
        Compute the total loss combining gene expression and splicing reconstruction losses,
        latent KL divergence, and the modality alignment penalty.

        For splicing, if count data is provided (via the keys "atse_counts_key" and "junc_counts_key"),
        the loss is computed using the specified binomial, DM, or beta-binomial likelihood; otherwise, binary
        cross-entropy is used.

        Returns
        -------
        LossOutput
            A container with total loss, reconstruction losses, and KL divergence details.
        """

        for name, val in inference_outputs.items():
            if isinstance(val, torch.Tensor):
                mask_finite = torch.isfinite(val)
                if not mask_finite.all():
                    # compute range over the finite entries
                    finite = val[mask_finite]
                    vmin = finite.min().item() if finite.numel() else float("nan")
                    vmax = finite.max().item() if finite.numel() else float("nan")
                    print(f"[loss][ERROR] inference_outputs['{name}'] has non-finite values!"
                            f"  finite range = {vmin:.4e} → {vmax:.4e}")
                    assert False, f"inference_outputs['{name}'] contains non-finite entries"

        # --- only-on-error checks & stats for generative_outputs ---
        for name, val in generative_outputs.items():
            if isinstance(val, torch.Tensor):
                mask_finite = torch.isfinite(val)
                if not mask_finite.all():
                    finite = val[mask_finite]
                    vmin = finite.min().item() if finite.numel() else float("nan")
                    vmax = finite.max().item() if finite.numel() else float("nan")
                    print(f"[loss][ERROR] generative_outputs['{name}'] has non-finite values!"
                            f"  finite range = {vmin:.4e} → {vmax:.4e}")
                    assert False, f"generative_outputs['{name}'] contains non-finite entries"

        # Get the data
        x = inference_outputs["x"]

        # Split x into gene expression and splicing components
        x_expr = x[:, :self.n_input_genes]
        x_spl = x[:, self.n_input_genes:(self.n_input_genes + self.n_input_junctions)]

        # Retrieve splicing count data if available
        total_counts = tensors.get("atse_counts_key", None)
        junction_counts = tensors.get("junc_counts_key", None)
        psi_mask = tensors.get(REGISTRY_KEYS.PSI_MASK_KEY, None)
        if psi_mask is not None:
            psi_mask = psi_mask.to(torch.bool)

        # Create masks for the modality alignment penalty
        mask_expr = x_expr.sum(dim=1) > 0
        mask_spl = x_spl.sum(dim=1) > -10000000
    
        # Compute Expression loss
        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        px_dropout = generative_outputs["px_dropout"]
        x_expression = x[:, : self.n_input_genes]
        rl_expression = self.get_reconstruction_loss_expression(
            x_expression, px_rate, px_r, px_dropout
        )
        
        # Compute splicing reconstruction loss
        # removed else statement
        rl_splicing = self.get_reconstruction_loss_splicing(
            x_spl,
            total_counts,
            junction_counts,
            psi_mask,
            generative_outputs["p"],
            generative_outputs["phi"]
        )
        
        # Combine both reconstruction losses
        recon_loss_expression = (rl_expression * mask_expr)
        recon_loss_splicing = rl_splicing
        recon_loss = (recon_loss_expression) + (recon_loss_splicing)

        # Compute KL divergence between approximate posterior and prior
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        kl_div_z = kld(Normal(qz_m, torch.sqrt(qz_v)), Normal(0, 1)).sum(dim=1)

        # Compute the KL divergence for paired data, passing in the precomputed masks
        kl_div_paired = self._compute_mod_penalty(
            (inference_outputs["qzm_expr"], inference_outputs["qzv_expr"]),
            (inference_outputs["qzm_spl"], inference_outputs["qzv_spl"]),
            mask_expr,
            mask_spl
        )

        # print(
        #     "[Loss] rl_expr: ",
        #     float(rl_expression.mean()), "nan_count:", int(torch.isnan(rl_expression).sum()),
        #     " | rl_spl:", float(rl_splicing.mean()), "nan_count:", int(torch.isnan(rl_splicing).sum()),
        #     " | KL_z:", float(kl_div_z.mean()), "nan_count:", int(torch.isnan(kl_div_z).sum()),
        #     " | KL_paired:", float(kl_div_paired.mean()), "nan_count:", int(torch.isnan(kl_div_paired).sum())
        # )

        # KL WARMUP
        # ───── KL warm-up (per-cell) ─────────────────────────────────────
        kl_local_for_warmup = kl_div_z
        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_div_paired

        # ───── L2 prior on log-concentrations ϕ_j (global → per-cell) ────
        lambda_prior = 1e-2                       # can tune 

        if self.splicing_loss_type == "beta_binomial" or self.splicing_loss_type == "dirichlet_multinomial": #add dirichlet multinomial atse concentration stuff
            prior_loss = lambda_prior * torch.square(self.log_phi_j).sum() / x.size(0)  # divide by batch_size so strength is constant
        else:
            prior_loss = 0.0 #do not compute prior loss if we're not using beta_binomial distribution
        

        # ───── total negative ELBO ───────────────────────────────────────
        loss = torch.mean(recon_loss + weighted_kl_local) + prior_loss

        # per cell tensors 
        recon_losses = {
            "reconstruction_loss_expression": recon_loss_expression,
            "reconstruction_loss_splicing": recon_loss_splicing
        }

        # scalar diagnostics (no grads)
        # print scalar diagnostics (no grad tracking)
        if self.training and torch.rand(1).item() < 0.01:  # 1% of iterations
            phi_values = generative_outputs["phi"]
            gene_dispersion = generative_outputs["px_r"]  # Gene concentration parameters
            print(f"φ stats: min={phi_values.min().item():.2f}, "
              f"max={phi_values.max().item():.2f}, "
              f"median={phi_values.median().item():.2f}")

            print(f"θ_g (gene disp):     min={gene_dispersion.min().item():.2f}, "
              f"max={gene_dispersion.max().item():.2f}, "
              f"median={gene_dispersion.median().item():.2f}")
    
            # Compare them
            phi_median = phi_values.median().item()
            gene_median = gene_dispersion.median().item()
            ratio = phi_median / gene_median if gene_median > 0 else float('inf')
            print(f"φ_j/θ_g ratio: {ratio:.2f}")

        kl_local = {
            "kl_divergence_z": kl_div_z,
            "kl_divergence_paired": kl_div_paired,
        }
        return LossOutput(loss=loss, reconstruction_loss=recon_losses, kl_local=kl_local)

    def get_reconstruction_loss_expression(self, x, px_rate, px_r, px_dropout):
        """Compute the reconstruction loss for gene expression data."""
        if self.gene_likelihood == "zinb":
            loss_val = -ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout).log_prob(x).sum(dim=-1)
        elif self.gene_likelihood == "nb":
            loss_val = -NegativeBinomial(mu=px_rate, theta=px_r).log_prob(x).sum(dim=-1)
        elif self.gene_likelihood == "poisson":
            loss_val = -Poisson(px_rate).log_prob(x).sum(dim=-1)
        else:
            loss_val = 0.0
        return loss_val
        

    def dirichlet_multinomial_likelihood(
        self,
        counts: torch.Tensor,           # (N, P)
        atse_counts: torch.Tensor,      # (N, G)
        junc2atse: torch.sparse.Tensor, # (P, G)
        alpha: torch.Tensor,            # (N, P)
        mask: torch.Tensor | None = None,  # optional (N, P)
    ) -> torch.Tensor:
        """
        Computes each cell’s Dirichlet–multinomial log-likelihood,
        **masking out** any junctions where mask==0 and any ATSE-groups where atse_counts==0.

        Returns:
            Tensor of shape (N,) giving per-cell log-likelihood:
                ll[i] = sum_over_groups LL_group(i,g)  –  sum_over_junctions LL_junc(i,p)
        """
        # 0) ensure everything on the same device
        device      = counts.device
        counts      = counts.to(device)
        atse_counts = atse_counts.to(device)
        alpha       = alpha.to(device)
        mask        = mask.to(device) if mask is not None else None
        junc2atse   = junc2atse.to(device)

        N, P = counts.shape
        G     = junc2atse.shape[1]

        # 1) build true per-cell ATSE_counts → (N, G)
        idx_p, idx_g = junc2atse.indices()  # each ≈ P long
        idx_g = idx_g.to(device)
        true_atse = torch.zeros((N, G), device=device, dtype=atse_counts.dtype)
        true_atse.scatter_reduce_(
            dim=1,
            index=idx_g.expand(N, -1),  # (N, P)
            src=atse_counts,            # (N, P)
            reduce="amax",
        )
        atse_counts = true_atse       # now (N, G)

        # 2) αₛ = sum_p α_p within each group → (N, G)
        alpha_sums = alpha @ junc2atse  # (N, G)

        # 3) per-group log-likelihood → (N, G)
        ll_atse = nbetaln(atse_counts, alpha_sums)
        # mask out any groups with zero denominator
        ll_atse = ll_atse * (atse_counts > 0).to(ll_atse.dtype)
        per_cell_atse = ll_atse.sum(dim=1)  # (N,)

        # 4) per-junction log-likelihood → (N, P)
        ll_junc = nbetaln(counts, alpha)
        if mask is not None:
            ll_junc = ll_junc * mask.to(ll_junc.dtype)
        per_cell_junc = ll_junc.sum(dim=1)  # (N,)

        # 5) return per-cell difference (no batch average)
        return per_cell_atse - per_cell_junc  # (N,)




    def get_reconstruction_loss_splicing(
            self, x, atse_counts, junc_counts, mask, p, phi):
        """
        x               – (N × J) binary matrix  (often unused)
        atse_counts     – (N × J) denominator counts
        junc_counts     – (N × J) numerator counts
        phi             – (J)     concentration parameter *per junction*
        """
        eps = 1e-8
        p   = p.clamp(eps, 1 - eps)

        if self.splicing_loss_type == "binomial":
            log_prob      = torch.log(p)
            log_prob_comp = torch.log(1 - p)
            log_likelihood = (
                junc_counts * log_prob
            + (atse_counts - junc_counts) * log_prob_comp
            )
            if mask is not None:
                log_likelihood = log_likelihood * mask.to(log_likelihood.dtype)
            return -log_likelihood.sum(dim=1)

        elif self.splicing_loss_type == "beta_binomial":
            # broadcast phi: (J) -> (N × J)
            phi = phi.unsqueeze(0)          # shape (1 × J)
            alpha = p *  phi
            beta  = (1 - p) * phi

            log_likelihood = (
                torch.lgamma(atse_counts + 1)
            - torch.lgamma(junc_counts + 1)
            - torch.lgamma(atse_counts - junc_counts + 1)
            + torch.lgamma(junc_counts + alpha)
            + torch.lgamma(atse_counts - junc_counts + beta)
            - torch.lgamma(atse_counts + alpha + beta)
            - torch.lgamma(alpha)
            - torch.lgamma(beta)
            + torch.lgamma(alpha + beta)
            )
            if mask is not None:
                log_likelihood = log_likelihood * mask.to(log_likelihood.dtype)
            return -log_likelihood.sum(dim=1)
        
        elif self.splicing_loss_type == "dirichlet_multinomial":
            # 1) invert sigmoid to get raw logits
            logits = torch.log(p) - torch.log1p(-p)

            # 2) group‐softmax on each ATSE
            lse = group_logsumexp(self.junc2atse, logits)            # → (N, G)
            sm_logits = subtract_group_logsumexp(
                self.junc2atse, logits, lse
            )                                                       # → (N, J)

            # 3) build your concentration per junction from phi
            #    phi may be:
            #      • a scalar tensor → same phi for every junction
            #      • a 1-D tensor of length G → one phi per ATSE
            #      • a 1-D tensor of length J → one phi per junction
            if phi.ndim == 0:
                # scalar
                conc_junc = phi
                #print("Phi is a Scalar Value")
            elif phi.ndim == 1 and phi.shape[0] == self.junc2atse.size(1):
                # per-ATSE → map to per-junction by sparse‐matrix multiply
                #    self.junc2atse : (J × G), phi.unsqueeze(1) : (G × 1)
                if self.junc2atse.device != p.device:
                    self.junc2atse = self.junc2atse.to(p.device)
                conc_junc = torch.sparse.mm(self.junc2atse, phi.unsqueeze(1)).squeeze(1)
                #print("Phi is per ATSE")
            elif phi.ndim == 1 and phi.shape[0] == self.junc2atse.size(0):
                # already per-junction
                conc_junc = phi
                #print("Phi is per Junction")
            else:
                raise ValueError(f"Unexpected phi shape {tuple(phi.shape)}")

            # now conc_junc is either a 0-D tensor or a 1-D length-J tensor.
            # broadcast to (N × J) by unsqueezing
            conc = conc_junc.unsqueeze(0) # shape (1, J)
            alpha = p * conc # broadcasting (1, J) → (N, J)

            # 4) feed into our DM‐likelihood helper
            ll = self.dirichlet_multinomial_likelihood(
                counts=junc_counts,
                atse_counts=atse_counts,
                junc2atse=self.junc2atse,
                alpha=alpha,
                mask=mask, #internally sums over all atses and junctions
            )
            # ll is the log‐likelihood per cell, so -ll is your loss
            return -ll
        
        else:
            raise ValueError("`splicing_loss_type` must be one of "
                         "'binomial', 'beta_binomial' or 'dirichlet_multinomial'")




    def _compute_mod_penalty(self, mod_params_expr, mod_params_spl, mask1, mask2):
        """
        Compute the alignment penalty between expression and splicing latent distributions.

        Parameters
        ----------
        mod_params_expr : tuple
            (qzm_expr, qzv_expr) from the expression encoder.
        mod_params_spl : tuple
            (qzm_spl, qzv_spl) from the splicing encoder.
        mask1 : Tensor
            Boolean mask for cells with valid expression data.
        mask2 : Tensor
            Boolean mask for cells with valid splicing data.

        Returns
        -------
        Tensor
            A scalar penalty computed over cells where both modalities are observed.
        """

        mask = torch.logical_and(mask1, mask2)
        if self.modality_weights == "concatenate":
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        if self.modality_penalty == "None":
            return 0
        elif self.modality_penalty == "Jeffreys":
            penalty = sym_kld(mod_params_expr[0], mod_params_expr[1].sqrt(), #why are they doing sqrt twice in multivae (orig had this so i kept it the same??)
                                    mod_params_spl[0], mod_params_spl[1].sqrt())
            return penalty[mask].sum()
        elif self.modality_penalty == "MMD":
            penalty = torch.linalg.norm(mod_params_expr[0] - mod_params_spl[0], dim=1)
            return penalty[mask].sum()
        else:
            raise ValueError("modality penalty not supported")




@auto_move_data
def mix_modalities(Xs, masks, weights, weight_transform: callable = None):
    """Compute the weighted mean of the Xs while masking unmeasured modality values.

    Parameters
    ----------
    Xs
        Sequence of Xs to mix, each should be (N x D)
    masks
        Sequence of masks corresponding to the Xs, indicating whether the values
        should be included in the mix or not (N)
    weights
        Weights for each modality (either K or N x K)
    weight_transform
        Transformation to apply to the weights before using them
    """
    # (batch_size x latent) -> (batch_size x modalities x latent)
    Xs = torch.stack(Xs, dim=1)
    # (batch_size) -> (batch_size x modalities)
    masks = torch.stack(masks, dim=1).float()
    weights = masked_softmax(weights, masks, dim=-1)

    # (batch_size x modalities) -> (batch_size x modalities x latent)
    weights = weights.unsqueeze(-1)
    if weight_transform is not None:
        weights = weight_transform(weights)

    # sum over modalities, so output is (batch_size x latent)
    return (weights * Xs).sum(1)


@auto_move_data
def sym_kld(qzm1, qzv1, qzm2, qzv2):
    """Symmetric KL divergence between two Gaussians."""
    rv1 = Normal(qzm1, qzv1.sqrt())
    rv2 = Normal(qzm2, qzv2.sqrt())

    return kld(rv1, rv2) + kld(rv2, rv1)



