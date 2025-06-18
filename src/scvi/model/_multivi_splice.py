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
from scvi.module.base import (
    BaseModuleClass,
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

import torch
from scvi.train import AdversarialTrainingPlan

from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

class MyAdvTrainingPlan(AdversarialTrainingPlan):

    def __init__(
        self,
        module: BaseModuleClass,
        *,
        # keep all your existing AdversarialTrainingPlan args here…
        lr_scheduler_type: Literal["plateau", "step"] = "plateau",
        step_size: int = 10,
        gradient_clipping: bool = True,
        gradient_clipping_max_norm: float = 5.0,
        **kwargs,
    ):
        super().__init__(module=module, **kwargs)
        # new scheduling params
        self.lr_scheduler_type = lr_scheduler_type
        self.step_size = step_size
        self.gradient_clipping = gradient_clipping
        self.gradient_clipping_max_norm = gradient_clipping_max_norm


    def compute_and_log_metrics(self, loss_output, metrics, mode):
        # 1. original ELBO, total recon, total KL, and extra‐metrics
        super().compute_and_log_metrics(loss_output, metrics, mode)

        # 2. now log each modality’s recon loss as the *batch mean*
        for key, val in loss_output.reconstruction_loss.items():
            if isinstance(val, torch.Tensor):
                # val might be a vector of per-cell losses → take mean
                val = val.mean()
            self.log(
                f"{key}_{mode}",
                val,
                on_step=False,
                on_epoch=True,
                batch_size=loss_output.n_obs_minibatch,
                sync_dist=self.use_sync_dist,
            )

        try:
            # PL’s API: self.lr_schedulers() returns a list of dicts
            schedulers = self.lr_schedulers()
            last_lrs = schedulers.get_last_lr()
            current_lr = last_lrs[0] if isinstance(last_lrs, (list, tuple)) else last_lrs
        except Exception:
            # fallback to optimizer if scheduler isn’t available
            optim = self.optimizers()
            optim = optim[0] if isinstance(optim, list) else optim
            current_lr = optim.param_groups[0]["lr"]

        self.log(
            f"lr_{mode}",
            current_lr,
            on_step=True,    # logs every step
            on_epoch=False,  # not aggregated at epoch end
        )

    def on_validation_epoch_end(self) -> None:
        """Update the learning rate via scheduler steps."""
        if self.lr_scheduler_type == "step":
            sch = self.lr_schedulers()
            sch.step()
            return
        
        if (not self.reduce_lr_on_plateau or "validation" not in self.lr_scheduler_metric):
            return
        else:
            sch = self.lr_schedulers()
            sch.step(self.trainer.callback_metrics[self.lr_scheduler_metric])

    def configure_optimizers(self):
        """Configure optimizers for adversarial training."""
        params1 = filter(lambda p: p.requires_grad, self.module.parameters())
        optimizer1 = self.get_optimizer_creator()(params1)
        config1 = {"optimizer": optimizer1}
        if self.lr_scheduler_type == "step":
            print("Using step LR")
            scheduler = StepLR(
                optimizer1,
                step_size=self.step_size,
                gamma=self.lr_factor,
            )
            config1.update(
                {
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "epoch",
                    },
                },
            )
            
        elif self.reduce_lr_on_plateau and self.lr_scheduler_type == "plateau":
            print("Using plateau LR")
            scheduler1 = ReduceLROnPlateau(
                optimizer1,
                patience=self.lr_patience,
                factor=self.lr_factor,
                threshold=self.lr_threshold,
                min_lr=self.lr_min,
                threshold_mode="abs",
                verbose=True,
            )
            config1.update(
                {
                    "lr_scheduler": {
                        "scheduler": scheduler1,
                        "monitor": self.lr_scheduler_metric,
                    },
                },
            )

        if self.adversarial_classifier is not False:
            params2 = filter(lambda p: p.requires_grad, self.adversarial_classifier.parameters())
            optimizer2 = torch.optim.Adam(
                params2, lr=1e-3, eps=0.01, weight_decay=self.weight_decay
            )
            config2 = {"optimizer": optimizer2}

            # pytorch lightning requires this way to return
            opts = [config1.pop("optimizer"), config2["optimizer"]]
            if "lr_scheduler" in config1:
                scheds = [config1["lr_scheduler"]]
                return opts, scheds
            else:
                return opts

        return config1
    
    def training_step(self, batch, batch_idx):
        """Training step for adversarial training."""
        if "kl_weight" in self.loss_kwargs:
            self.loss_kwargs.update({"kl_weight": self.kl_weight})
            self.log("kl_weight", self.kl_weight, on_step=True, on_epoch=False)
        kappa = (
            1 - self.kl_weight
            if self.scale_adversarial_loss == "auto"
            else self.scale_adversarial_loss
        )
        batch_tensor = batch[REGISTRY_KEYS.BATCH_KEY]

        opts = self.optimizers()
        if not isinstance(opts, list):
            opt1 = opts
            opt2 = None
        else:
            opt1, opt2 = opts

        inference_outputs, _, scvi_loss = self.forward(batch, loss_kwargs=self.loss_kwargs)
        z = inference_outputs["z"]
        loss = scvi_loss.loss
        # fool classifier if doing adversarial training
        if kappa > 0 and self.adversarial_classifier is not False:
            fool_loss = self.loss_adversarial_classifier(z, batch_tensor, False)
            loss += fool_loss * kappa

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.compute_and_log_metrics(scvi_loss, self.train_metrics, "train")
        opt1.zero_grad()
        self.manual_backward(loss)
        if self.gradient_clipping:
            torch.nn.utils.clip_grad_norm_(self.module.parameters(), max_norm=self.gradient_clipping_max_norm)
        opt1.step()

        # train adversarial classifier
        # this condition will not be met unless self.adversarial_classifier is not False
        if opt2 is not None:
            loss = self.loss_adversarial_classifier(z.detach(), batch_tensor, True)
            loss *= kappa
            opt2.zero_grad()
            self.manual_backward(loss)
            if self.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(self.module.parameters(),  max_norm=self.gradient_clipping_max_norm)
            opt2.step()




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
    splicing_architecture
        Which encoder/decoder to use for splicing. One of:
        * ``"vanilla"``: standard SCVI `Encoder` + `DecoderSplice`,
        * ``"partial"``: `PartialEncoder` + `LinearDecoder` with per‐feature embeddings.
    expression_architecture
        Which decoder to use for gene expression. One of:
        * ``"vanilla"``: standard SCVI `Encoder` +  Standard SCVI Decoder
        * ``"linear"``: standard SCVI `Encoder` + Linear SCVI Decoder
    code_dim
        Dimensionality of per‐feature embeddings in the *partial* encoder (only used when `splicing_architecture="partial"`).
    h_hidden_dim
        Hidden size of the shared “h” network in `PartialEncoder` (only for `"partial"`).
    mlp_encoder_hidden_dim
        Hidden size of the final MLP in `PartialEncoder` (only for `"partial"`).
    initialize_embeddings_from_pca
        Whether or not to initalize the embedding layer for the Partial Encoder using PCA of junction ratios (only for `"partial"`).
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
        Additional keyword arguments for :class:`~scvi.module.MULTIVAESPLICE`.
    
    Notes
    -----
    * This model integrates only gene expression and splicing data.  
    * The splicing modality does not use library size factors.
    """
    _module_cls = MULTIVAESPLICE
    _training_plan_cls = MyAdvTrainingPlan

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
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        splicing_loss_type: Literal["binomial", "beta_binomial", "dirichlet_multinomial"] = "beta_binomial",
        splicing_concentration: float | None = None,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        splicing_architecture: Literal["vanilla", "partial"] = "vanilla",
        expression_architecture: Literal["vanilla", "linear"] = "vanilla",
        code_dim: int = 32,
        h_hidden_dim: int = 64,
        mlp_encoder_hidden_dim: int = 128,
        initialize_embeddings_from_pca: bool = True,
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
            splicing_loss_type = splicing_loss_type,
            splicing_concentration = splicing_concentration,
            gene_likelihood=gene_likelihood,
            gene_dispersion=dispersion,
            splicing_architecture = splicing_architecture,
            expression_architecture = expression_architecture, 
            code_dim= code_dim,
            h_hidden_dim= h_hidden_dim,
            mlp_encoder_hidden_dim= mlp_encoder_hidden_dim,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            use_size_factor_key=use_size_factor_key,
            latent_distribution=latent_distribution,
            deeply_inject_covariates=deeply_inject_covariates,
            encode_covariates=encode_covariates,
            **model_kwargs,
        )

        self._model_summary_string = (
            f"MultiVI Splice Model with n_genes={n_genes}, n_junctions={n_junctions}, "
            f"n_hidden={self.module.n_hidden}, n_latent={self.module.n_latent}, "
            f"n_layers_encoder={n_layers_encoder}, n_layers_decoder={n_layers_decoder}, "
            f"dropout_rate={dropout_rate}, latent_distribution={latent_distribution}, "
            f"expression_architecture={expression_architecture}, "
            f"splicing_architecture={splicing_architecture}, code_dim={code_dim}, "
            f"splicing_loss_type={splicing_loss_type}, splicing_concentration={splicing_concentration}, init_from_pca={initialize_embeddings_from_pca}, "
            f"h_hidden_dim={h_hidden_dim}, mlp_encoder_hidden_dim={mlp_encoder_hidden_dim}, "
            f"gene_likelihood={gene_likelihood}, dispersion={dispersion}, "
            f"modality_weights={modality_weights}, modality_penalty={modality_penalty}"
        )
        self.fully_paired = fully_paired
        self.n_latent = n_latent
        self.init_params_ = self._get_init_params(locals())
        self.n_genes = n_genes
        self.n_junctions = n_junctions
        self.get_normalized_function_name = "get_normalized_splicing"

        if self.adata is not None:
                if initialize_embeddings_from_pca and splicing_architecture == "partial":
                    self.init_feature_embedding_from_adata()
                if splicing_loss_type == "dirichlet_multinomial":
                    self.init_junc2atse()

    def make_junc2atse(self, atse_labels):
        print("Making Junc2Atse...")
        num_junctions = len(atse_labels)
        atse_labels = atse_labels.astype('category')
        row_indices = torch.arange(num_junctions, dtype=torch.long)
        col_indices = torch.tensor(atse_labels.cat.codes.values)

        return torch.sparse_coo_tensor(
            indices=torch.stack([row_indices, col_indices]),
            values=torch.ones(len(row_indices), dtype=torch.float32),
            size=(num_junctions, len(atse_labels.cat.categories))
        ).coalesce()

    def init_junc2atse(self) -> None:
        cl_info = self.adata_manager.data_registry["atse_counts_key"]
        cl_key, mod_key = cl_info.attr_key, cl_info.mod_key
        cluster_counts = self.adata[mod_key].layers[cl_key]
        atse_labels = self.adata[mod_key].var["event_id"]
        j2a = self.make_junc2atse(atse_labels)
        self.module.junc2atse = j2a.coalesce().to(self.module.device)

    def init_feature_embedding_from_adata(self) -> None:
        """
        Center the `junc_ratio` layer, run PCA on it, and copy
        the resulting components into the encoder.feature_embedding.
        Entries where ATSE count == 0 are temporarily set to NaN
        so that nanmean skips them.
        """

        from sklearn.decomposition import PCA
        import scipy.sparse as sp
        import numpy as np
        import torch

        print("Initializing Feature Embeddings...")
        jr_info = self.adata_manager.data_registry[REGISTRY_KEYS.JUNC_RATIO_X_KEY]
        jr_key, mod_key = jr_info.attr_key, jr_info.mod_key

        ac_info = self.adata_manager.data_registry["atse_counts_key"]
        ac_key = ac_info.attr_key

        # 2) Grab as CSR matrices
        X = self.adata[mod_key].layers[jr_key]
        C = self.adata[mod_key].layers[ac_key]

        # 2) densify and cast
        if sp.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=float)
    
        if sp.issparse(C):
            C = C.toarray()
        # set all X entries to NaN where C == 0
        X[C == 0] = np.nan
        # ——————————————————————————————————————————————————

        # 3) column‐wise centering (ignores those NaNs)
        col_means = np.nanmean(X, axis=0)
        X_centered = X - col_means[None, :]
        # turn the NaNs (masked spots) into zeros
        X_centered[np.isnan(X_centered)] = 0.0

        # 4) PCA
        pca = PCA(n_components=self.module.z_encoder_splicing.code_dim)
        pca.fit(X_centered)
        comps = pca.components_.T  # (n_vars, code_dim)

        # 5) copy into the encoder
        with torch.no_grad():
            self.module.z_encoder_splicing.feature_embedding.copy_(
                torch.as_tensor(comps, dtype=self.module.z_encoder_splicing.feature_embedding.dtype)
            )
        X[C == 0] = 0
        print("Initialized Feature Embeddings!")


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
        early_stopping_patience: int = 50,
        save_best: bool = True,
        check_val_every_n_epoch: int | None = None,
        n_steps_kl_warmup: int | None = None,
        n_epochs_kl_warmup: int | None = 50,
        adversarial_mixing: bool = True,
        lr_scheduler_type: Literal["plateau", "step"] = "plateau",
        reduce_lr_on_plateau: bool = False,
        lr_factor: float = 0.6,
        lr_patience: int = 30,
        lr_threshold: float = 0.0,
        lr_scheduler_metric: Literal[
            "elbo_validation", "reconstruction_loss_validation", "kl_local_validation"
        ] = "elbo_validation",
        step_size: int = 10,
        gradient_clipping: bool = True,
        gradient_clipping_max_norm: float = 5.0,
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
            Proportions for splitting the data into train and validation sets.
        shuffle_set_split
            Whether to shuffle before splitting.
        batch_size
            Minibatch size for training.
        weight_decay, eps
            Optimizer hyperparameters.
        early_stopping
            Whether to enable early stopping.
        early_stopping_patience
            Number of epochs with no improvement before stopping.
        save_best
            Whether to save the best model checkpoint.
        check_val_every_n_epoch
            How often (in epochs) to run validation.
        n_steps_kl_warmup, n_epochs_kl_warmup
            KL divergence warmup parameters (by steps or epochs).
        adversarial_mixing
            Whether to include adversarial classifier during training.
        lr_scheduler_type
            Scheduler type in TrainingPlan: “plateau” (reduce-on-plateau) or “step” (fixed-step).
        reduce_lr_on_plateau
            If True and using plateau scheduler, enable ReduceLROnPlateau.
        lr_factor
            Multiplicative factor for LR reduction (used for both plateau and step schedulers).
        lr_patience
            Number of epochs with no improvement for plateau scheduler.
        lr_threshold
            Threshold for measuring new optimum (plateau scheduler).
        lr_scheduler_metric
            Metric to monitor for plateau scheduler.
        step_size
            Epoch interval between LR drops (step scheduler).
        gradient_clipping
            Whether or not (true or false) to use gradient norm clipping
        gradient_clipping_max_norm
            Max norm of the gradients to be used in gradient clipping
        datasplitter_kwargs
            Additional kwargs for the data splitter.
        plan_kwargs
            Additional kwargs to pass to the TrainingPlan constructor.
        **kwargs
            Additional Trainer kwargs (callbacks, strategy, etc.).
        """
         
        update_dict = {
            "lr": lr,
            "lr_scheduler_type": lr_scheduler_type,
            "reduce_lr_on_plateau": reduce_lr_on_plateau, 
            "lr_factor": lr_factor,
            "lr_patience": lr_patience,
            "lr_threshold": lr_threshold,
            "lr_scheduler_metric": lr_scheduler_metric,
            "step_size": step_size,
            "adversarial_classifier": adversarial_mixing,
            "weight_decay": weight_decay,
            "eps": eps,
            "n_epochs_kl_warmup": n_epochs_kl_warmup,
            "n_steps_kl_warmup": n_steps_kl_warmup,
            "optimizer": "AdamW",
            "scale_adversarial_loss": 1,
            "gradient_clipping": gradient_clipping,
            "gradient_clipping_max_norm": gradient_clipping_max_norm,
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
            early_stopping_patience=early_stopping_patience,
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
        psi_mask_layer: str | None = None,
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
        psi_mask_layer
            Layer with binary mask (1=observed, 0=missing) per junction.
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
        if psi_mask_layer is not None:
            anndata_fields.append(LayerField(REGISTRY_KEYS.PSI_MASK_KEY, psi_mask_layer, is_count_data=False))
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
        psi_mask_layer: str | None = None,
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
        psi_mask_layer
            Layer with binary mask (1=observed, 0=missing) per junction.
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

        if size_factor_key is None:
            size_factor_key = "X_library_size"

        mudata_fields = [
            batch_field,
            fields.MuDataCategoricalObsField(REGISTRY_KEYS.LABELS_KEY, None, mod_key=None),
            fields.MuDataObsmField(REGISTRY_KEYS.SIZE_FACTOR_KEY, attr_key=size_factor_key,is_count_data=False),
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

            if psi_mask_layer is None:
                psi_mask_layer = "mask"
            mudata_fields.append(
                fields.MuDataLayerField(
                    REGISTRY_KEYS.PSI_MASK_KEY,  # internal key used by the model
                    psi_mask_layer,
                    mod_key=modalities.junc_ratio_layer,
                    is_count_data=False,
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
