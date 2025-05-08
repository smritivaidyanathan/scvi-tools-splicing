from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable as IterableClass
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from mudata import MuData
from scipy.sparse import csr_matrix, vstack
from torch.distributions import Normal

from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager, fields
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
)
from scvi.model._utils import (
    _get_batch_code_from_category,
    scrna_raw_counts_properties,
    use_distributed_sampler,
)
from scvi.model.base import (
    ArchesMixin,
    BaseModelClass,
    UnsupervisedTrainingMixin,
    VAEMixin,
)
from scvi.model.base._de_core import _de_core
from scvi.module import MULTIVAESPLICE
from scvi.train import AdversarialTrainingPlan
from scvi.train._callbacks import SaveBestState
from scvi.utils import track
from scvi.utils._docstrings import de_dsp, devices_dsp, setup_anndata_dsp

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from typing import Literal

    from anndata import AnnData
    from scvi._types import AnnOrMuData, Number

logger = logging.getLogger(__name__)


class MULTIVISPLICE(VAEMixin, UnsupervisedTrainingMixin, BaseModelClass, ArchesMixin):
    """Integration of gene expression and alternative splicing signals.

    MULTIVISPLICE is designed to integrate multiomic data that includes gene
    expression and alternative splicing measurements (e.g. junction usage ratios).
    
    Parameters
    ----------
    adata
        AnnData/MuData object that has been registered via
        :meth:`~scvi.model.MULTIVISPLICE.setup_anndata` or
        :meth:`~scvi.model.MULTIVISPLICE.setup_mudata`.
    n_genes
        The number of gene expression features.
    n_junctions
        The number of alternative splicing features (junctions).
    modality_weights
        Weighting scheme across modalities. Must be one of:
            * ``"equal"``: equal weight per modality,
            * ``"universal"``: learn a universal weight for each modality,
            * ``"cell"``: learn cell-specific weights.
    modality_penalty
        Penalty applied during training. Options are ``"Jeffreys"``, ``"MMD"``, or ``"None"``.
    n_hidden
        Number of nodes per hidden layer. If `None`, defaults to the square root of `n_junctions`.
    n_latent
        Dimensionality of the latent space. If `None`, defaults to the square root of `n_hidden`.
    n_layers_encoder
        Number of hidden layers used in the encoder networks.
    n_layers_decoder
        Number of hidden layers used in the decoder networks.
    dropout_rate
        Dropout rate for the networks.
    region_factors
        Whether to include junction‐specific factors in the model.
    gene_likelihood
        Likelihood for gene expression. One of: ``"zinb"``, ``"nb"``, ``"poisson"``.
    dispersion
        Dispersion configuration. Options are: ``"gene"``, ``"gene-batch"``, ``"gene-label"``, or ``"gene-cell"``.
    use_batch_norm
        Which layers to apply batch normalization to. Options: ``"encoder"``, ``"decoder"``, ``"both"``, or ``"none"``.
    use_layer_norm
        Which layers to apply layer normalization to. Options: ``"encoder"``, ``"decoder"``, ``"both"``, or ``"none"``.
    latent_distribution
        Latent space distribution; either ``"normal"`` or ``"ln"`` (logistic normal).
    deeply_inject_covariates
        Whether to deeply inject covariates into all layers of the decoder.
    encode_covariates
        Whether to encode covariates.
    fully_paired
        Indicates if the data is fully paired across modalities.
    **model_kwargs
        Additional keyword arguments for :class:`~scvi.module.MULTIVISPLICE`.
    
    Notes
    -----
    * This model integrates only gene expression and splicing data.  
    * The splicing modality does not use library size factors.
    """
    _module_cls = MULTIVAESPLICE
    _training_plan_cls = AdversarialTrainingPlan

    def __init__(
        self,
        adata: AnnOrMuData,
        n_genes: int | None = None,
        n_junctions: int | None = None,
        modality_weights: Literal["equal", "cell", "universal"] = "equal",
        modality_penalty: Literal["Jeffreys", "MMD", "None"] = "Jeffreys",
        n_hidden: int | None = None,
        n_latent: int | None = None,
        n_layers_encoder: int = 2,
        n_layers_decoder: int = 2,
        dropout_rate: float = 0.1,
        region_factors: bool = True,
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        latent_distribution: Literal["normal", "ln"] = "normal",
        deeply_inject_covariates: bool = False,
        encode_covariates: bool = False,
        fully_paired: bool = False,
        **model_kwargs,
    ):
        super().__init__(adata)

        if n_genes is None or n_junctions is None:
            assert isinstance(adata, MuData), (
                "n_genes and n_junctions must be provided if using AnnData"
            )
            n_genes = self.summary_stats.get("n_vars", 0)
            n_junctions = self.summary_stats.get("n_junc", 0)

        prior_mean, prior_scale = None, None
        n_cats_per_cov = (
            self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else []
        )

        # Splicing modality does not use library size factors.
        use_size_factor_key = REGISTRY_KEYS.SIZE_FACTOR_KEY in self.adata_manager.data_registry

        self.module = self._module_cls(
            n_input_genes=n_genes,
            n_input_junctions=n_junctions,
            modality_weights=modality_weights,
            modality_penalty=modality_penalty,
            n_batch=self.summary_stats.n_batch,
            n_obs=adata.n_obs,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers_encoder=n_layers_encoder,
            n_layers_decoder=n_layers_decoder,
            n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
            n_cats_per_cov=n_cats_per_cov,
            dropout_rate=dropout_rate,
            region_factors=region_factors,
            gene_likelihood=gene_likelihood,
            gene_dispersion=dispersion,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            use_size_factor_key=use_size_factor_key,
            latent_distribution=latent_distribution,
            deeply_inject_covariates=deeply_inject_covariates,
            encode_covariates=encode_covariates,
            **model_kwargs,
        )

        self._model_summary_string = (
            f"MultiVI Splice Model with the following params: \nn_genes: {n_genes}, "
            f"n_junctions: {n_junctions}, n_hidden: {self.module.n_hidden}, "
            f"n_latent: {self.module.n_latent}, n_layers_encoder: {n_layers_encoder}, "
            f"n_layers_decoder: {n_layers_decoder}, dropout_rate: {dropout_rate}, "
            f"latent_distribution: {latent_distribution}, deep injection: {deeply_inject_covariates}, "
            f"gene_likelihood: {gene_likelihood}, gene_dispersion: {dispersion}, "
            f"Mod.Weights: {modality_weights}, Mod.Penalty: {modality_penalty}"
        )
        self.fully_paired = fully_paired
        self.n_latent = n_latent
        self.init_params_ = self._get_init_params(locals())
        self.n_genes = n_genes
        self.n_junctions = n_junctions
        self.get_normalized_function_name = "get_normalized_splicing"

    @devices_dsp.dedent
    def train(
        self,
        max_epochs: int = 500,
        lr: float = 1e-4,
        accelerator: str = "auto",
        devices: int | list[int] | str = "auto",
        train_size: float | None = None,
        validation_size: float | None = None,
        shuffle_set_split: bool = True,
        batch_size: int = 128,
        weight_decay: float = 1e-3,
        eps: float = 1e-08,
        early_stopping: bool = True,
        save_best: bool = True,
        check_val_every_n_epoch: int | None = None,
        n_steps_kl_warmup: int | None = None,
        n_epochs_kl_warmup: int | None = 50,
        adversarial_mixing: bool = True,
        datasplitter_kwargs: dict | None = None,
        plan_kwargs: dict | None = None,
        **kwargs,
    ):
        r"""Trains the model using amortized variational inference on gene expression and splicing modalities.

        Parameters
        ----------
        max_epochs
            Number of epochs to train over.
        lr
            Learning rate for optimization.
        accelerator, devices
            Hardware acceleration options.
        train_size, validation_size
            Proportions for splitting the data.
        shuffle_set_split
            Whether to shuffle indices before splitting.
        batch_size
            Minibatch size for training.
        weight_decay, eps
            Optimizer hyperparameters.
        early_stopping, save_best
            Early stopping and checkpointing options.
        check_val_every_n_epoch
            Frequency of validation checks.
        n_steps_kl_warmup, n_epochs_kl_warmup
            KL warmup parameters.
        adversarial_mixing
            Whether to use an adversarial classifier.
        datasplitter_kwargs, plan_kwargs, **kwargs
            Additional options for data splitting, training plan, and trainer.
        """
        update_dict = {
            "lr": lr,
            "adversarial_classifier": adversarial_mixing,
            "weight_decay": weight_decay,
            "eps": eps,
            "n_epochs_kl_warmup": n_epochs_kl_warmup,
            "n_steps_kl_warmup": n_steps_kl_warmup,
            "optimizer": "AdamW",
            "scale_adversarial_loss": 1,
        }
        if plan_kwargs is not None:
            plan_kwargs.update(update_dict)
        else:
            plan_kwargs = update_dict

        datasplitter_kwargs = datasplitter_kwargs or {}

        if save_best:
            warnings.warn(
                "`save_best` is deprecated in v1.2 and will be removed in v1.3. "
                "Please use `enable_checkpointing` instead. See "
                "https://github.com/scverse/scvi-tools/issues/2568 for more details.",
                DeprecationWarning,
                stacklevel=settings.warnings_stacklevel,
            )
            if "callbacks" not in kwargs:
                kwargs["callbacks"] = []
            kwargs["callbacks"].append(SaveBestState(monitor="reconstruction_loss_validation"))

        data_splitter = self._data_splitter_cls(
            self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            shuffle_set_split=shuffle_set_split,
            distributed_sampler=use_distributed_sampler(kwargs.get("strategy", None)),
            batch_size=batch_size,
            **datasplitter_kwargs,
        )
        training_plan = self._training_plan_cls(self.module, **plan_kwargs)
        runner = self._train_runner_cls(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            early_stopping=early_stopping,
            check_val_every_n_epoch=check_val_every_n_epoch,
            early_stopping_monitor="reconstruction_loss_validation",
            early_stopping_patience=50,
            **kwargs,
        )
        return runner()

    @torch.inference_mode()
    def get_library_size_factors(
        self,
        adata: AnnOrMuData | None = None,
        indices: Sequence[int] = None,
        batch_size: int = 128,
    ) -> dict[str, np.ndarray]:
        r"""Return library size factors for gene expression.

        Note that the splicing modality does not use library size factors.

        Parameters
        ----------
        adata
            AnnOrMuData object.
        indices
            Cell indices (default: all cells).
        batch_size
            Batch size for processing.

        Returns
        -------
        A dictionary with key "expression" for the gene expression library size factors.
        """
        self._check_adata_modality_weights(adata)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        lib_exp = []
        for tensors in scdl:
            outputs = self.module.inference(**self.module._get_inference_input(tensors))
            lib_exp.append(outputs["libsize_expr"].cpu())
        return {"expression": torch.cat(lib_exp).numpy().squeeze()}

    @torch.inference_mode()
    def get_latent_representation(
        self,
        adata: AnnOrMuData | None = None,
        modality: Literal["joint", "expression", "splicing"] = "joint",
        indices: Sequence[int] | None = None,
        give_mean: bool = True,
        batch_size: int | None = None,
    ) -> np.ndarray:
        r"""Return the latent representation for each cell.

        Parameters
        ----------
        adata
            AnnOrMuData object used in setup.
        modality
            One of:
              - ``"joint"``: joint latent space,
              - ``"expression"``: expression-specific latent space,
              - ``"splicing"``: splicing-specific latent space.
        indices
            Cell indices to use.
        give_mean
            If True, returns the mean of the latent distribution.
        batch_size
            Batch size for processing.

        Returns
        -------
        A NumPy array of the latent representations.
        """
        if not self.is_trained_:
            raise RuntimeError("Please train the model first.")
        self._check_adata_modality_weights(adata)
        keys = {"z": "z", "qz_m": "qz_m", "qz_v": "qz_v"}
        if self.fully_paired and modality != "joint":
            raise RuntimeError("A fully paired model only has a joint latent representation.")
        if not self.fully_paired and modality != "joint":
            if modality == "expression":
                keys = {"z": "z_expr", "qz_m": "qzm_expr", "qz_v": "qzv_expr"}
            elif modality == "splicing":
                keys = {"z": "z_spl", "qz_m": "qzm_spl", "qz_v": "qzv_spl"}
            else:
                raise RuntimeError("Modality must be 'joint', 'expression', or 'splicing'.")

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        latent = []
        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            qz_m = outputs[keys["qz_m"]]
            qz_v = outputs[keys["qz_v"]]
            z = outputs[keys["z"]]
            if give_mean:
                if self.module.latent_distribution == "ln":
                    samples = Normal(qz_m, qz_v.sqrt()).sample([1])
                    z = torch.nn.functional.softmax(samples, dim=-1)
                    z = z.mean(dim=0)
                else:
                    z = qz_m
            latent.append(z.cpu())
        return torch.cat(latent).numpy()

    @torch.inference_mode()
    def get_normalized_expression(
        self,
        adata: AnnOrMuData | None = None,
        indices: Sequence[int] | None = None,
        n_samples_overall: int | None = None,
        transform_batch: Sequence[Number | str] | None = None,
        gene_list: Sequence[str] | None = None,
        use_z_mean: bool = True,
        n_samples: int = 1,
        batch_size: int | None = None,
        return_mean: bool = True,
        return_numpy: bool = False,
        silent: bool = True,
    ) -> np.ndarray | pd.DataFrame:
        r"""Returns the normalized (decoded) gene expression.

        This is denoted as :math:`\rho_n` in the scVI paper.

        Parameters
        ----------
        adata
            AnnOrMuData object with equivalent structure to initial AnnData. If `None`, defaults
            to the AnnOrMuData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        n_samples_overall
            Number of observations to sample from ``indices`` if ``indices`` is provided.
        transform_batch
            Batch to condition on.
            If transform_batch is:

            - None, then real observed batch is used.
            - int, then batch transform_batch is used.
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        use_z_mean
            If True, use the mean of the latent distribution, otherwise sample from it
        n_samples
            Number of posterior samples to use for estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a numpy array instead of a pandas DataFrame.
        %(de_silent)s

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is
        :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        self._check_adata_modality_weights(adata)
        adata = self._validate_anndata(adata)
        adata_manager = self.get_anndata_manager(adata, required=True)
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        transform_batch = _get_batch_code_from_category(adata_manager, transform_batch)

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names[: self.n_genes]
            gene_mask = [gene in gene_list for gene in all_genes]

        exprs = []
        for tensors in scdl:
            per_batch_exprs = []
            for batch in track(transform_batch, disable=silent):
                if batch is not None:
                    batch_indices = tensors[REGISTRY_KEYS.BATCH_KEY]
                    tensors[REGISTRY_KEYS.BATCH_KEY] = torch.ones_like(batch_indices) * batch
                _, generative_outputs = self.module.forward(
                    tensors=tensors,
                    inference_kwargs={"n_samples": n_samples},
                    generative_kwargs={"use_z_mean": use_z_mean},
                    compute_loss=False,
                )
                output = generative_outputs["px_scale"]
                output = output[..., gene_mask]
                output = output.cpu().numpy()
                per_batch_exprs.append(output)
            per_batch_exprs = np.stack(
                per_batch_exprs
            )  # shape is (len(transform_batch) x batch_size x n_var)
            exprs += [per_batch_exprs.mean(0)]

        if n_samples > 1:
            # The -2 axis correspond to cells.
            exprs = np.concatenate(exprs, axis=-2)
        else:
            exprs = np.concatenate(exprs, axis=0)
        if n_samples > 1 and return_mean:
            exprs = exprs.mean(0)

        if return_numpy:
            return exprs
        else:
            return pd.DataFrame(
                exprs,
                columns=adata.var_names[: self.n_genes][gene_mask],
                index=adata.obs_names[indices],
            )

    @torch.inference_mode()
    def get_normalized_splicing(
        self,
        adata: AnnOrMuData | None = None,
        indices: Sequence[int] | None = None,
        n_samples_overall: int | None = None,
        transform_batch: Sequence[Number | str] | None = None,
        junction_list: Sequence[str] | None = None,
        use_z_mean: bool = True,
        n_samples: int = 1,
        batch_size: int | None = None,
        return_mean: bool = True,
        return_numpy: bool = False,
        silent: bool = True,
    ) -> np.ndarray | pd.DataFrame:
        r"""Returns the normalized (decoded) splicing probabilities.

        This is denoted as :math:`p_{nj}` in the MULTIVISPLICE model.

        Parameters
        ----------
        adata
            AnnOrMuData object with the same structure as used in setup. If `None`,
            defaults to the AnnOrMuData object used to initialize the model.
        indices
            Cell indices to use. If `None`, all cells are used.
        n_samples_overall
            Number of observations to sample from `indices` if provided.
        transform_batch
            Batch(s) to condition on:
            - None: use true observed batch
            - int: force all cells to that batch
            - list[int|str]: average over those batches
        junction_list
            Subset of junction names to return. If `None`, returns all junctions.
        use_z_mean
            If True, use the mean of the latent distribution; otherwise sample.
        n_samples
            Number of posterior samples to draw. If >1 and `return_mean` is True,
            the result is averaged over draws.
        batch_size
            Minibatch size for decoding. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to average over posterior samples when `n_samples>1`.
        return_numpy
            If True, returns a NumPy array; otherwise a pandas DataFrame.
        silent
            If True, suppresses the progress bar.

        Returns
        -------
        A NumPy array or pandas DataFrame of shape `(cells, junctions)` containing the
        decoded splicing probabilities.
        """
        # Validate and prepare
        self._check_adata_modality_weights(adata)
        adata = self._validate_anndata(adata)
        adata_manager = self.get_anndata_manager(adata, required=True)

        # Select cells
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, size=n_samples_overall, replace=False)

        # Data loader
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        # Batches to transform over
        transform_batch = _get_batch_code_from_category(adata_manager, transform_batch)

        # Build junction mask
        all_junc = adata.var_names[self.n_genes : self.n_genes + self.n_junctions]
        if junction_list is None:
            junction_mask = slice(None)
        else:
            junction_mask = [j in junction_list for j in all_junc]

        # Decode in mini-batches
        spls = []
        for tensors in scdl:
            per_batch_spls = []
            for batch in track(transform_batch, disable=silent):
                if batch is not None:
                    # override batch
                    bidx = tensors[REGISTRY_KEYS.BATCH_KEY]
                    tensors[REGISTRY_KEYS.BATCH_KEY] = torch.ones_like(bidx) * batch
                _, generative_outputs = self.module.forward(
                    tensors=tensors,
                    inference_kwargs={"n_samples": n_samples},
                    generative_kwargs={"use_z_mean": use_z_mean},
                    compute_loss=False,
                )
                output = generative_outputs["p"]
                output = output[..., junction_mask]
                output = output.cpu().numpy()
                per_batch_spls.append(output)
            per_batch_spls = np.stack(per_batch_spls)  # (len(transform_batch), bs, n_junc)
            spls += [per_batch_spls.mean(0)]

        # Concatenate across minibatches
        spls = np.concatenate(spls, axis=0)  # (n_cells, n_selected_junc)

        # If multiple samples requested and we averaged over them in forward
        if n_samples > 1 and return_mean:
            spls = spls.mean(0)

        if return_numpy:
            return spls

        # Build DataFrame
        cols = (
            all_junc
            if junction_list is None
            else [j for j, keep in zip(all_junc, junction_mask) if keep]
        )
        return pd.DataFrame(spls, index=adata.obs_names[indices], columns=cols)



    @de_dsp.dedent
    def differential_splicing(
        self,
        adata: AnnData | None = None,
        groupby: str | None = None,
        group1: Iterable[str] | None = None,
        group2: str | None = None,
        idx1: Sequence[int] | Sequence[bool] | None = None,
        idx2: Sequence[int] | Sequence[bool] | None = None,
        mode: Literal["vanilla", "change"] = "change",
        delta: float = 0.05,
        batch_size: int | None = None,
        all_stats: bool = True,
        batch_correction: bool = False,
        batchid1: Iterable[str] | None = None,
        batchid2: Iterable[str] | None = None,
        fdr_target: float = 0.05,
        silent: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        r"""Differential splicing analysis.

        Performs a unified differential analysis on junction usage ratios analogous to
        the 'vanilla' and 'change' methods in scVI for gene expression.
        This method compares splicing (junction usage) between groups.

        Parameters
        ----------
        %(de_adata)s
        %(de_groupby)s
        %(de_group1)s
        %(de_group2)s
        %(de_idx1)s
        %(de_idx2)s
        %(de_mode)s
        %(de_delta)s
        %(de_batch_size)s
        %(de_all_stats)s
        %(de_batch_correction)s
        %(de_batchid1)s
        %(de_batchid2)s
        %(de_fdr_target)s
        %(de_silent)s
        two_sided
            Whether to perform a two-sided test.
        **kwargs
            Additional keyword arguments for differential computation.

        Returns
        -------
        A pandas DataFrame with differential splicing results including:
            - prob_da: probability of differential splicing,
            - is_da_fdr: FDR-based significance,
            - bayes_factor, effect_size, empirical effects, etc.
        """
        self._check_adata_modality_weights(adata)
        adata = self._validate_anndata(adata)
        col_names = adata.var_names[self.n_genes : self.n_genes + self.n_junctions]
        model_fn = partial(
            self.get_normalized_splicing, use_z_mean=False, batch_size=batch_size
        )

        def change_fn(a, b):
            return a - b

        two_sided = kwargs.pop("two_sided", True)
        if two_sided:
            def m1_domain_fn(samples):
                return np.abs(samples) >= delta
        else:
            def m1_domain_fn(samples):
                return samples >= delta

        all_stats_fn = partial(
            scrna_raw_counts_properties,
            var_idx=np.arange(adata.shape[1])[self.n_genes : self.n_genes + self.n_junctions],
        )

        result = _de_core(
            adata_manager=self.get_anndata_manager(adata, required=True),
            model_fn=model_fn,
            representation_fn=None,
            groupby=groupby,
            group1=group1,
            group2=group2,
            idx1=idx1,
            idx2=idx2,
            all_stats=all_stats,
            all_stats_fn=all_stats_fn,
            col_names=col_names,
            mode=mode,
            batchid1=batchid1,
            batchid2=batchid2,
            delta=delta,
            batch_correction=batch_correction,
            fdr=fdr_target,
            change_fn=change_fn,
            m1_domain_fn=m1_domain_fn,
            silent=silent,
            **kwargs,
        )
        result = pd.DataFrame(
            {
                "prob_da": result.proba_de,
                "is_da_fdr": result.loc[:, f"is_de_fdr_{fdr_target}"],
                "bayes_factor": result.bayes_factor,
                "effect_size": result.scale2 - result.scale1,
                "emp_effect": result.emp_mean2 - result.emp_mean1,
                "est_prob1": result.scale1,
                "est_prob2": result.scale2,
                "emp_prob1": result.emp_mean1,
                "emp_prob2": result.emp_mean2,
            },
            index=col_names,
        )
        return result

    @torch.inference_mode()
    def differential_expression(
        self,
        adata: AnnData | None = None,
        groupby: str | None = None,
        group1: Iterable[str] | None = None,
        group2: str | None = None,
        idx1: Sequence[int] | Sequence[bool] | None = None,
        idx2: Sequence[int] | Sequence[bool] | None = None,
        mode: Literal["vanilla", "change"] = "change",
        delta: float = 0.25,
        batch_size: int | None = None,
        all_stats: bool = True,
        batch_correction: bool = False,
        batchid1: Iterable[str] | None = None,
        batchid2: Iterable[str] | None = None,
        fdr_target: float = 0.05,
        silent: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        r"""Differential expression analysis.

        Performs differential gene expression analysis using normalized gene expression estimates.

        Parameters
        ----------
        %(de_adata)s
        %(de_groupby)s
        %(de_group1)s
        %(de_group2)s
        %(de_idx1)s
        %(de_idx2)s
        %(de_mode)s
        %(de_delta)s
        %(de_batch_size)s
        %(de_all_stats)s
        %(de_batch_correction)s
        %(de_batchid1)s
        %(de_batchid2)s
        %(de_fdr_target)s
        %(de_silent)s
        **kwargs
            Additional keyword arguments for differential computation.

        Returns
        -------
        A pandas DataFrame with differential expression results.
        """
        self._check_adata_modality_weights(adata)
        adata = self._validate_anndata(adata)

        col_names = adata.var_names[: self.n_genes]
        model_fn = partial(
            self.get_normalized_expression,
            batch_size=batch_size,
        )
        all_stats_fn = partial(
            scrna_raw_counts_properties,
            var_idx=np.arange(adata.shape[1])[: self.n_genes],
        )
        result = _de_core(
            adata_manager=self.get_anndata_manager(adata, required=True),
            model_fn=model_fn,
            representation_fn=None,
            groupby=groupby,
            group1=group1,
            group2=group2,
            idx1=idx1,
            idx2=idx2,
            all_stats=all_stats,
            all_stats_fn=all_stats_fn,
            col_names=col_names,
            mode=mode,
            batchid1=batchid1,
            batchid2=batchid2,
            delta=delta,
            batch_correction=batch_correction,
            fdr=fdr_target,
            silent=silent,
            **kwargs,
        )
        return result

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: str | None = None,
        junc_ratio: str | None = None,
        cell_by_junction_matrix: str | None = None,
        cell_by_cluster_matrix: str | None = None,
        batch_key: str | None = None,
        size_factor_key: str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        **kwargs,
    ):
        r"""Set up an AnnData object for MULTIVISPLICE.

        Parameters
        ----------
        %(param_adata)s
        %(param_layer)s
        junc_ratio
            Key in ``adata.layers`` for junction ratio values.
        cell_by_junction_matrix
            Key in ``adata.layers`` for the cell-by-junction matrix.
        cell_by_cluster_matrix
            Key in ``adata.layers`` for the cell-by-cluster splicing matrix.
        %(param_batch_key)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s

        Notes
        -----
        Use this method if your splicing data is stored in an AnnData object where gene expression and
        splicing features are concatenated.
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        adata.obs["_indices"] = np.arange(adata.n_obs)
        batch_field = CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key)
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            batch_field,
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, None),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            NumericalJointObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
            NumericalObsField(REGISTRY_KEYS.INDICES_KEY, "_indices"),
        ]
        if junc_ratio is not None:
            anndata_fields.append(LayerField("junc_ratio_key", junc_ratio, is_count_data=True))
        if cell_by_junction_matrix is not None:
            anndata_fields.append(LayerField("cell_by_junction_matrix", cell_by_junction_matrix, is_count_data=True))
        if cell_by_cluster_matrix is not None:
            anndata_fields.append(LayerField("cell_by_cluster_matrix", cell_by_cluster_matrix, is_count_data=True))
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_mudata(
        cls,
        mdata: MuData,
        rna_layer: str | None = None,
        junc_ratio_layer: str | None = None,
        atse_counts_layer: str | None = None,
        junc_counts_layer: str | None = None,
        batch_key: str | None = None,
        size_factor_key: str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        idx_layer: str | None = None,
        modalities: dict[str, str] | None = None,
        **kwargs,
    ):
        r"""Set up a MuData object for MULTIVISPLICE.

        Parameters
        ----------
        %(param_mdata)s
        rna_layer
            Key in the RNA AnnData for gene expression counts.
            If `None`, the primary data (`.X`) of that AnnData is used.
        junc_ratio_layer
            Key in the splicing AnnData for junction ratio values.
            If `None`, the primary data (`.X`) of that AnnData is used.
        atse_counts_layer
            Key in the splicing AnnData for total event counts.
            If `None`, defaults to `"cell_by_cluster_matrix"`.
        junc_counts_layer
            Key in the splicing AnnData for observed junction counts.
            If `None`, defaults to `"cell_by_junction_matrix"`.
        %(param_batch_key)s
        size_factor_key
            Key in `mdata.obsm` for size factors.
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        %(idx_layer)s
        %(param_modalities)s

        Examples
        --------
        >>> mdata = mu.MuData({
        ...    "rna": ge_anndata.copy(),
        ...    "splicing": atse_anndata.copy()
        ... })
        >>> scvi.model.MULTIVISPLICE.setup_mudata(
        ...     mdata,
        ...     modalities={"rna_layer": "rna", "junc_ratio_layer": "splicing"},
        ...     rna_layer="raw_counts",            # gene expression data is in the GE AnnData's "raw_counts" layer
        ...     junc_ratio_layer="junc_ratio",     # splicing data is in the ATSE AnnData's "junc_ratio" layer
        ... )
        >>> model = scvi.model.MULTIVISPLICE(mdata)
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        if modalities is None:
            raise ValueError("Modalities cannot be None.")
        modalities = cls._create_modalities_attr_dict(modalities, setup_method_args)
        mdata.obs["_indices"] = np.arange(mdata.n_obs)

        batch_field = fields.MuDataCategoricalObsField(
            REGISTRY_KEYS.BATCH_KEY,
            batch_key,
            mod_key=modalities.batch_key,
        )
        mudata_fields = [
            batch_field,
            fields.MuDataCategoricalObsField(REGISTRY_KEYS.LABELS_KEY, None, mod_key=None),
            fields.MuDataObsmField(REGISTRY_KEYS.SIZE_FACTOR_KEY, attr_key="X_library_size",is_count_data=False),
            fields.MuDataCategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys, mod_key=modalities.categorical_covariate_keys),
            fields.MuDataNumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys, mod_key=modalities.continuous_covariate_keys),
            fields.MuDataNumericalObsField(REGISTRY_KEYS.INDICES_KEY, "_indices", mod_key=modalities.idx_layer, required=False),
        ]

        # RNA modality registration: use rna_layer from the GE AnnData.
        if modalities.rna_layer is not None:
            mudata_fields.append(
                fields.MuDataLayerField(
                    REGISTRY_KEYS.X_KEY,
                    rna_layer,  # e.g. "raw_counts"
                    mod_key=modalities.rna_layer,
                    is_count_data=True,
                    mod_required=True,
                )
            )

        # Splicing modality registration: we expect the ATSE AnnData to hold the relevant splicing layers.
        if modalities.junc_ratio_layer is not None:
            # Register the primary splicing data as X from the specified junc_ratio_layer.
            mudata_fields.append(
                fields.MuDataLayerField(
                    REGISTRY_KEYS.JUNC_RATIO_X_KEY,
                    junc_ratio_layer,  # e.g. "junc_ratio"
                    mod_key=modalities.junc_ratio_layer,
                    is_count_data=True,
                    mod_required=True,
                )
            )
            # Register the additional splicing layers.
            if atse_counts_layer is None:
                atse_counts_layer = "cell_by_cluster_matrix"
            mudata_fields.append(
                fields.MuDataLayerField(
                    "atse_counts_key",  # internal key used by the model
                    atse_counts_layer,
                    mod_key=modalities.junc_ratio_layer,
                    is_count_data=True,
                    mod_required=True,
                )
            )
            if junc_counts_layer is None:
                junc_counts_layer = "cell_by_junction_matrix"
            mudata_fields.append(
                fields.MuDataLayerField(
                    "junc_counts_key",  # internal key used by the model
                    junc_counts_layer,
                    mod_key=modalities.junc_ratio_layer,
                    is_count_data=True,
                    mod_required=True,
                )
            )

        adata_manager = AnnDataManager(fields=mudata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(mdata, **kwargs)
        cls.register_manager(adata_manager)


    def _check_adata_modality_weights(self, adata):
        r"""Checks whether held-out data is provided when using per-cell weights.

        Parameters
        ----------
        adata : AnnData or MuData
            The input data.

        Raises
        ------
        RuntimeError
            If held-out data is provided when per-cell modality weights are used.
        """
        if (adata is not None) and (self.module.modality_weights == "cell"):
            raise RuntimeError("Held out data not permitted when using per cell weights")
