from typing import NamedTuple


class _REGISTRY_KEYS_NT(NamedTuple):
    X_KEY: str = "X"
    ATAC_X_KEY: str = "atac"
    JUNC_RATIO_X_KEY: str = "junc_ratio"
    BATCH_KEY: str = "batch"
    SAMPLE_KEY: str = "sample"
    LABELS_KEY: str = "labels"
    PROTEIN_EXP_KEY: str = "proteins"
    CAT_COVS_KEY: str = "extra_categorical_covs"
    CONT_COVS_KEY: str = "extra_continuous_covs"
    INDICES_KEY: str = "ind_x"
    SIZE_FACTOR_KEY: str = "size_factor"
    MINIFY_TYPE_KEY: str = "minify_type"
    LATENT_QZM_KEY: str = "latent_qzm"
    LATENT_QZV_KEY: str = "latent_qzv"
    OBSERVED_LIB_SIZE: str = "observed_lib_size"
    PSI_MASK_KEY: str = "psi_observed_mask"


REGISTRY_KEYS = _REGISTRY_KEYS_NT()
