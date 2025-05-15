from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Literal, Dict, Any, Sequence

import numpy as np
import torch
from torch.distributions import Normal
from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager
from scvi.data._constants import ADATA_MINIFY_TYPE
from scvi.data.fields import (
    LayerField,
    CategoricalObsField,
    NumericalObsField,
    NumericalJointObsField,
)
from scvi.model.base import (
    BaseModelClass,
    UnsupervisedTrainingMixin,
    VAEMixin,
)
from scvi.module import PARTIALVAE
from scvi.utils import setup_anndata_dsp

from scvi.train import TrainRunner, TrainingPlan
from scvi.utils._docstrings import de_dsp, devices_dsp, setup_anndata_dsp
from scvi.train._callbacks import SaveBestState

from scvi.model._utils import (
    use_distributed_sampler,
)

if TYPE_CHECKING:
    from anndata import AnnData

logger = logging.getLogger(__name__)


class SPLICEVI(VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):
    """
    Variational autoencoder for splicing data (SCpliceVAE).

    Parameters
    ----------
    adata
        AnnData object registered via :meth:`~SpliceVI.setup_anndata`.
    code_dim
        Dimensionality of feature embeddings.
    h_hidden_dim
        Hidden size for shared PointNet-like layer.
    encoder_hidden_dim
        Hidden size for encoder MLP.
    latent_dim
        Dimensionality of latent space.
    decoder_hidden_dim
        Hidden size for decoder layers.
    dropout_rate
        Dropout probability.
    learn_concentration
        Whether to learn Beta-Binomial concentration.
    """
    _module_cls = PARTIALVAE

    def __init__(
        self,
        adata: AnnData | None = None,
        code_dim: int = 32,
        h_hidden_dim: int = 64,
        encoder_hidden_dim: int = 128,
        latent_dim: int = 10,
        dropout_rate: float = 0.1, 
        learn_concentration: bool = True,
        splice_likelihood: Literal["binomial", "beta_binomial"] = "beta_binomial",
        **kwargs,
    ):
        super().__init__(adata)
        # Store hyperparameters for module init
        self._module_kwargs = dict(
            code_dim=code_dim,
            h_hidden_dim=h_hidden_dim,
            encoder_hidden_dim=encoder_hidden_dim,
            n_latent=latent_dim,
            dropout_rate=dropout_rate,
            learn_concentration=learn_concentration,
            splice_likelihood = splice_likelihood,
            **kwargs,
        )
        self._model_summary_string = (
            f"SpliceVI PartialVAE with "
            f"h_hidden_dim={h_hidden_dim}, "
            f"encoder_hidden_dim={encoder_hidden_dim}, "
            f"latent_dim={latent_dim}, "
            f"learn_concentration={learn_concentration}, "
            f"splice_likelihood={splice_likelihood}."
        )
        
        if self._module_init_on_train:
            self.module = None
            warnings.warn(
                "Model initialized without adata; will init on train.",
                UserWarning,
                stacklevel=settings.warnings_stacklevel,
            )
        else:
            n_batch = self.summary_stats.n_batch
            # instantiate module
            self.module = self._module_cls(
                n_input=self.summary_stats.n_vars,
                n_batch=n_batch,
                **self._module_kwargs,
            )

        self.init_params_ = self._get_init_params(locals())

    
    @devices_dsp.dedent
    def train(
        self,
        max_epochs: int = 200,
        lr: float = 1e-4, # 0.001
        accelerator: str = "auto",
        devices: int | list[int] | str = "auto",
        train_size: float | None = None,
        validation_size: float | None = None,
        shuffle_set_split: bool = True,
        batch_size: int = 512, #bigger batch size
        weight_decay: float = 1e-3, #remove weight decay
        eps: float = 1e-8,
        early_stopping: bool = True,
        save_best: bool = True,
        check_val_every_n_epoch: int | None = None,
        n_steps_kl_warmup: int | None = None,
        n_epochs_kl_warmup: int | None = 10,
        reduce_lr_on_plateau: bool = False,
        lr_factor: float = 0.6,
        lr_patience: int = 30,
        lr_threshold: float = 0.0,
        lr_min: float = 0.0,
        datasplitter_kwargs: dict | None = None,
        plan_kwargs: dict | None = None,
        **kwargs,
    ):
        """
        Trains the model using amortized variational inference on splicing data.

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
        datasplitter_kwargs, plan_kwargs, **kwargs
            Additional options for data splitting, training plan, and trainer.
        """
        update_dict = {
            "lr": lr,
            "weight_decay": weight_decay,
            "eps": eps,
            "n_epochs_kl_warmup": n_epochs_kl_warmup,
            "n_steps_kl_warmup": n_steps_kl_warmup,
            "optimizer": "AdamW",
            "reduce_lr_on_plateau": reduce_lr_on_plateau,
            "lr_factor": lr_factor,
            "lr_patience": lr_patience,
            "lr_threshold": lr_threshold,
            "lr_min": lr_min,
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
            early_stopping_patience=10,
            **kwargs,
        )

        return runner()


    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        junc_ratio_layer: str,
        junc_counts_layer: str,
        cluster_counts_layer: str,
        psi_mask_layer: str,
        batch_key: str | None = None,
        size_factor_key: str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        **kwargs,
    ):
        """
        Set up AnnData for SpliceVI.

        Parameters
        ----------
        adata
            AnnData to register.
        junc_ratio_layer
            Layer with junction usage ratios (X input).
        junc_counts_layer
            Layer with junction counts (successes).
        cluster_counts_layer
            Layer with total cluster counts (trials).
        psi_mask_layer
            Layer with binary mask (1=observed, 0=missing) per junction.
        batch_key
            Column in obs for batch.
        size_factor_key
            If provided, registers size factor but unused for splicing.
        categorical_covariate_keys
        continuous_covariate_keys
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            # Ratios as input
            LayerField(REGISTRY_KEYS.X_KEY, junc_ratio_layer, is_count_data=False),
            # Counts for likelihood
            LayerField("junction_counts", junc_counts_layer, is_count_data=True),
            LayerField("cluster_counts", cluster_counts_layer, is_count_data=True),
            # Mask layer for partial VAE
            LayerField(REGISTRY_KEYS.PSI_MASK_KEY, psi_mask_layer, is_count_data=False),
            # batch covariate
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
        ]
        # optional size factor (unused)
        if size_factor_key is not None:
            anndata_fields.append(
                NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False)
            )
        # additional covariates if desired
        if categorical_covariate_keys:
            anndata_fields.append(
                CategoricalObsField(REGISTRY_KEYS.CAT_COVS_KEY, None)
            )
        if continuous_covariate_keys:
            anndata_fields.append(
                NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, None)
            )
        # register
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @torch.inference_mode()
    def get_latent_representation(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        give_mean: bool = True,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """
        Return latent embeddings of splicing VAE.

        Parameters
        ----------
        adata
            AnnData for inference (defaults to init adata).
        indices
            Cell indices to use.
        give_mean
            If True, use posterior mean; else sample.
        batch_size
            Batch size.

        Returns
        -------
        Array of shape (cells, latent_dim).
        """
        if not self.is_trained_:
            raise RuntimeError("Train the model before extracting latent.")
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        latents = []
        for tensors in scdl:
            inf_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inf_inputs)
            qz_m = outputs["qz_m"]  # Underlying Normal distribution
            z = outputs["z"]
            if give_mean:
                z = qz_m
            latents.append(z.cpu())
        return torch.cat(latents).numpy()
