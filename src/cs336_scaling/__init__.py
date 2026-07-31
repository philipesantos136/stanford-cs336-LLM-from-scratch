"""Stanford CS336 Scaling Laws module."""

from __future__ import annotations

from cs336_scaling.api import ScalingApiClient, ScalingApiError, load_runs, merge_runs, save_runs
from cs336_scaling.api_fit import ApiIsoFlopsOptimum, LossFit, fit_loss_scaling, fit_model_size_scaling, select_best_by_compute
from cs336_scaling.isoflops import IsoFlopsOptimum, PowerLawFit, fit_isoflops_scaling_laws, fit_power_law, load_isoflops_runs, select_isoflops_optima
from cs336_scaling.local_api import LocalTrainingApi
from cs336_scaling.scaling_config import (
    ALLOWED_TRAIN_FLOPS,
    ModelShape,
    TrainingConfig,
    all_valid_model_shapes,
    choose_num_heads,
    default_model_shapes,
    estimate_non_embedding_params,
    make_config,
    nearest_shape,
)
from cs336_scaling.scaling_plan import QueryPlan, build_initial_query_plan, build_local_self_study_query_plan, summarize_plan

__version__ = "0.1.0"

__all__ = [
    "ALLOWED_TRAIN_FLOPS",
    "ApiIsoFlopsOptimum",
    "IsoFlopsOptimum",
    "LocalTrainingApi",
    "LossFit",
    "ModelShape",
    "PowerLawFit",
    "QueryPlan",
    "ScalingApiClient",
    "ScalingApiError",
    "TrainingConfig",
    "all_valid_model_shapes",
    "build_initial_query_plan",
    "build_local_self_study_query_plan",
    "choose_num_heads",
    "default_model_shapes",
    "estimate_non_embedding_params",
    "fit_isoflops_scaling_laws",
    "fit_loss_scaling",
    "fit_model_size_scaling",
    "fit_power_law",
    "load_isoflops_runs",
    "load_runs",
    "make_config",
    "merge_runs",
    "nearest_shape",
    "save_runs",
    "select_best_by_compute",
    "select_isoflops_optima",
    "summarize_plan",
]
