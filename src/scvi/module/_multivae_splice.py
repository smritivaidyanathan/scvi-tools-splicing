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
from scvi.nn import DecoderSCVI, Encoder, FCLayers

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
    """Variational auto-encoder for joint paired and unpaired RNA-seq and alternative splicing data.

    This module is an adaptation of MultiVAE. integrates gene expression and alternative splicing (junction usage ratios)
    by means of two separate encoder-decoder branches that are fused into a joint latent space.
    Reconstruction is performed for gene expression and splicing separately.

    Parameters
    ----------
    n_input_genes : int
        Number of gene expression features.
    n_input_junctions : int
        Number of alternative splicing (junction) features.
    modality_weights : Literal["equal", "cell", "universal"], default "equal"
        Weighting scheme across modalities.
    modality_penalty : Literal["Jeffreys", "MMD", "None"], default "Jeffreys"
        Penalty to align the latent distributions.
    n_batch : int, default 0
        Number of batches (for batch correction).
    n_obs : int, default 0
        Number of observations.
    n_labels : int, default 0
        Number of cell labels.
    gene_likelihood : Literal["zinb", "nb", "poisson"], default "zinb"
        Likelihood model for gene expression.
    gene_dispersion : Literal["gene", "gene-batch", "gene-label", "gene-cell"], default "gene"
        Dispersion configuration for gene expression.
    n_hidden : int or None, default None
        Number of hidden units per layer. If None, a heuristic is used.
    n_latent : int or None, default None
        Dimensionality of the latent space. If None, a heuristic is used.
    n_layers_encoder : int, default 2
        Number of layers in the encoder networks.
    n_layers_decoder : int, default 2
        Number of layers in the decoder networks.
    n_continuous_cov : int, default 0
        Number of continuous covariates.
    n_cats_per_cov : Iterable[int] or None, default None
        List with the number of categories per categorical covariate.
    dropout_rate : float, default 0.1
        Dropout rate.
    region_factors : bool, default True
        Whether to include junction-specific factors.
    use_batch_norm : Literal["encoder", "decoder", "none", "both"], default "none"
        Where to apply batch normalization.
    use_layer_norm : Literal["encoder", "decoder", "none", "both"], default "both"
        Where to apply layer normalization.
    latent_distribution : Literal["normal", "ln"], default "normal"
        Latent distribution type.
    deeply_inject_covariates : bool, default False
        Whether to deeply inject covariate information into decoders.
    encode_covariates : bool, default False
        Whether to provide covariates to the encoders.
    use_size_factor_key : bool, default False
        Whether to use a size-factor field for gene expression.
    splicing_loss_type : Literal["binomial", "beta_binomial"], default "beta_binomial"
        Loss type used for splicing reconstruction.
    splicing_concentration : float or None, default None
        Concentration parameter used for the beta-binomial loss (if applicable).
    **model_kwargs :
        Additional keyword arguments for the encoders/decoders.

    Notes
    -----
    Protein-related functionality has been removed.
    Compatibility with AnnData and MuData is handled via the AnnDataManager in the model’s setup.
    """
    def __init__(
        self,
        n_input_genes: int = 0,
        n_input_junctions: int = 0,
        modality_weights: Literal["equal", "cell", "universal"] = "equal",
        modality_penalty: Literal["Jeffreys", "MMD", "None"] = "Jeffreys",
        n_batch: int = 0,
        n_obs: int = 0,
        n_labels: int = 0,
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        gene_dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        n_hidden: int = None,
        n_latent: int = None,
        n_layers_encoder: int = 2,
        n_layers_decoder: int = 2,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Iterable[int] | None = None,
        dropout_rate: float = 0.1,
        region_factors: bool = True,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        latent_distribution: Literal["normal", "ln"] = "normal",
        deeply_inject_covariates: bool = False,
        encode_covariates: bool = False,
        use_size_factor_key: bool = False,
        splicing_loss_type: Literal["binomial", "beta_binomial"] = "beta_binomial",
        splicing_concentration: float | None = None,
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
        self.n_latent = int(np.sqrt(self.n_hidden)) if n_latent is None else n_latent
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

        # New splicing loss parameters
        self.splicing_loss_type = splicing_loss_type
        self.splicing_concentration = splicing_concentration

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
            n_output=self.n_latent,
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

        # ---------------- Splicing Branch ----------------
        input_spl = n_input_junctions if n_input_junctions > 0 else 1
        n_input_encoder_spl = input_spl + n_continuous_cov * int(encode_covariates)
        self.z_encoder_splicing = Encoder(
            n_input=n_input_encoder_spl,
            n_layers=n_layers_encoder,
            n_output=self.n_latent,
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
        size_factor = tensors.get(REGISTRY_KEYS.SIZE_FACTOR_KEY, None)
        return {
            "x": x,
            "batch_index": batch_index,
            "cont_covs": cont_covs,
            "cat_covs": cat_covs,
            "label": label,
            "cell_idx": cell_idx,
            "size_factor": size_factor,
        }

    @auto_move_data
    def inference(self, x, batch_index, cont_covs, cat_covs, label, cell_idx, size_factor, n_samples=1) -> dict[str, torch.Tensor]:
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
        qzm_spl, qzv_spl, z_spl = self.z_encoder_splicing(encoder_input_spl, batch_index, *categorical_input)

        # L encoder
        libsize_expr = self.l_encoder_expression(encoder_input_expr, batch_index, *categorical_input)

        # mix representations

        if self.modality_weights == "cell":
            weights = self.mod_weights[cell_idx, :]
        else:
            weights = self.mod_weights.unsqueeze(0).expand(len(cell_idx), -1)

        qz_m = mix_modalities((qzm_expr, qzm_spl), (mask_expr, mask_spl), weights)
        qz_v = mix_modalities((qzv_expr, qzv_spl), (mask_expr, mask_spl), weights, torch.sqrt)
        qz_v = torch.clamp(qz_v, min=1e-6)


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
        if cont_covs is None:
            decoder_input = latent
        elif latent.dim() != cont_covs.dim():
            decoder_input = torch.cat(
                [latent, cont_covs.unsqueeze(0).expand(latent.size(0), -1, -1)], dim=-1
            )
        else:
            decoder_input = torch.cat([latent, cont_covs], dim=-1)
        # Splicing Decoder
        p_s = self.z_decoder_splicing(decoder_input, batch_index, *categorical_input)
        # Expression Decoder
        px_scale, _, px_rate, px_dropout = self.z_decoder_expression(
            self.gene_dispersion,
            decoder_input,
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
            "p": p_s,
            "px_scale": px_scale,
            "px_r": torch.exp(self.px_r),
            "px_rate": px_rate,
            "px_dropout": px_dropout,
        }

    def loss(self, tensors, inference_outputs, generative_outputs, kl_weight: float = 1.0):
        """
        Compute the total loss combining gene expression and splicing reconstruction losses,
        latent KL divergence, and the modality alignment penalty.

        For splicing, if count data is provided (via the keys "atse_counts_key" and "junc_counts_key"),
        the loss is computed using the specified binomial or beta-binomial likelihood; otherwise, binary
        cross-entropy is used.

        Returns
        -------
        LossOutput
            A container with total loss, reconstruction losses, and KL divergence details.
        """
        # Get the data
        x = inference_outputs["x"]

        # Split x into gene expression and splicing components
        x_expr = x[:, :self.n_input_genes]
        x_spl = x[:, self.n_input_genes:(self.n_input_genes + self.n_input_junctions)]

        # Retrieve splicing count data if available
        total_counts = tensors.get("atse_counts_key", None)
        junction_counts = tensors.get("junc_counts_key", None)

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
        if total_counts is not None and junction_counts is not None:
            rl_splicing = self.get_reconstruction_loss_splicing(
            x_spl,
            total_counts,
            junction_counts,
            generative_outputs["p"]
        )
        else:
            rl_splicing = torch.nn.BCELoss(reduction="none")(
                generative_outputs["p"], (x_spl > 0).float()
            ).sum(dim=-1)
        
        # Combine both reconstruction losses
        recon_loss_expression = rl_expression * mask_expr
        recon_loss_splicing = rl_splicing
        recon_loss = recon_loss_expression + recon_loss_splicing

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

        # KL WARMUP
        kl_local_for_warmup = kl_div_z
        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_div_paired

        # TOTAL LOSS
        loss = torch.mean(recon_loss + weighted_kl_local)

        recon_losses = {
            "reconstruction_loss_expression": recon_loss_expression,
            "reconstruction_loss_splicing": recon_loss_splicing,
        }

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

    def get_reconstruction_loss_splicing(self, x, atse_counts, junc_counts, p):
        """
        Compute the reconstruction loss for splicing data using a binomial or beta‐binomial log‐likelihood.
        Entries where atse_counts == 0 are ignored.

        Uses the instance attributes `splicing_loss_type` and `splicing_concentration`.

        Parameters
        ----------
        x : Tensor
            Input tensor for splicing (unused, kept for compatibility).
        atse_counts : Tensor
            Total counts per event.
        junc_counts : Tensor
            Observed junction counts.
        p : Tensor
            Decoded splicing probabilities.

        Returns
        -------
        Tensor
            The negative log-likelihood loss for splicing.
        """
        mask = atse_counts != 0
        if self.splicing_loss_type == "binomial":
            log_prob = torch.log(p + 1e-10)
            log_prob_comp = torch.log(1 - p + 1e-10)
            log_likelihood = junc_counts * log_prob + (atse_counts - junc_counts) * log_prob_comp
            log_likelihood_masked = log_likelihood[mask]
            return -log_likelihood_masked.mean()
        elif self.splicing_loss_type == "beta_binomial":
            concentration = self.splicing_concentration
            if concentration is None:
                concentration = torch.tensor(1.0, device=p.device)
            alpha = p * concentration
            beta = (1 - p) * concentration
            log_pm = (torch.lgamma(atse_counts + 1)
                    - torch.lgamma(junc_counts + 1)
                    - torch.lgamma(atse_counts - junc_counts + 1)
                    + torch.lgamma(junc_counts + alpha)
                    + torch.lgamma(atse_counts - junc_counts + beta)
                    - torch.lgamma(atse_counts + alpha + beta)
                    - torch.lgamma(alpha)
                    - torch.lgamma(beta)
                    + torch.lgamma(alpha + beta))
            log_pm_masked = log_pm[mask]
            return -log_pm_masked.mean()
        else:
            raise ValueError("splicing_loss_type must be either 'binomial' or 'beta_binomial'")



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
        if self.modality_penalty == "None":
            return 0
        elif self.modality_penalty == "Jeffreys":
            penalty = sym_kld(mod_params_expr[0], mod_params_expr[1].sqrt(),
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



