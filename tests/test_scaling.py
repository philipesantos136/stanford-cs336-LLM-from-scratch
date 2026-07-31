"""Unit tests for CS336 Assignment 3: Scaling Laws, IsoFLOPs Analysis, and Training API."""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest

from cs336_scaling import (
    ALLOWED_TRAIN_FLOPS,
    LocalTrainingApi,
    ModelShape,
    PowerLawFit,
    TrainingConfig,
    build_initial_query_plan,
    build_local_self_study_query_plan,
    choose_num_heads,
    default_model_shapes,
    estimate_non_embedding_params,
    fit_isoflops_scaling_laws,
    fit_loss_scaling,
    fit_model_size_scaling,
    fit_power_law,
    load_isoflops_runs,
    make_config,
    nearest_shape,
    select_best_by_compute,
    select_isoflops_optima,
)


def test_estimate_non_embedding_params():
    # Formula: 12 * num_layers * d_model^2
    params = estimate_non_embedding_params(num_layers=12, d_model=512)
    expected = 12 * 12 * (512**2)
    assert params == expected
    assert params == 37748736


def test_model_shape_and_head_selection():
    shape = ModelShape(num_layers=8, d_model=256, num_heads=choose_num_heads(256))
    assert shape.num_heads == 4
    assert shape.non_embedding_params == 12 * 8 * 256 * 256

    # Test nearest_shape matching in log space
    target_params = 1e7
    closest = nearest_shape(target_params)
    assert closest.non_embedding_params > 0
    assert abs(math.log(closest.non_embedding_params) - math.log(target_params)) < 1.0


def test_power_law_fit_synthetic():
    # Synthetic relationship y = 2.5 * x^0.5
    xs = np.array([1e10, 1e11, 1e12, 1e13], dtype=np.float64)
    ys = 2.5 * (xs ** 0.5)

    fit = fit_power_law(xs, ys)
    assert pytest.approx(fit.coefficient, rel=1e-3) == 2.5
    assert pytest.approx(fit.exponent, rel=1e-3) == 0.5

    pred = fit.predict(1e14)
    assert pytest.approx(pred, rel=1e-3) == 2.5 * (1e14 ** 0.5)


def test_isoflops_curves_data_loading_and_optima():
    data_path = Path("data/isoflops_curves.json")
    assert data_path.exists(), "data/isoflops_curves.json must exist"

    runs = load_isoflops_runs(data_path)
    assert len(runs) == 72

    optima = select_isoflops_optima(runs)
    # 9 distinct compute budgets in benchmark dataset
    assert len(optima) == 9

    param_fit, data_fit = fit_isoflops_scaling_laws(optima)

    # In Chinchilla scaling laws, N and D scale roughly equally with compute (exponents near 0.5)
    assert 0.40 <= param_fit.exponent <= 0.60
    assert 0.40 <= data_fit.exponent <= 0.60

    # Test extrapolation to 1e22 FLOPs
    c_target = 1e22
    n_pred = param_fit.predict(c_target)
    d_pred = data_fit.predict(c_target)

    # Total compute check: 6 * N * D should closely match C
    approx_flops = 6.0 * n_pred * d_pred
    assert pytest.approx(approx_flops, rel=0.15) == c_target


def test_local_training_api_surrogate():
    api = LocalTrainingApi()
    shape = ModelShape(num_layers=12, d_model=512, num_heads=8)
    config = make_config(
        shape=shape,
        train_flops=int(1e15),
        batch_size=128,
        learning_rate=6e-4,
    )

    result = api.loss(config)
    assert "loss" in result
    assert result["loss"] > api.irreducible_loss
    assert result["non_embedding_params"] == shape.non_embedding_params
    assert pytest.approx(result["dataset_tokens"]) == 1e15 / (6.0 * shape.non_embedding_params)


def test_query_plan_flops_budget():
    plan = build_initial_query_plan()
    assert plan.total_flops > 0
    # Must be bounded by FLOPs cap
    assert plan.total_flops <= 2e18

    stages = plan.by_stage()
    assert "pilot_hparams" in stages
    assert "stage1_isoflops" in stages

    local_plan = build_local_self_study_query_plan()
    assert local_plan.total_flops > plan.total_flops
    assert local_plan.total_flops <= 2e18


def test_api_loss_fit():
    # Evaluate surrogate on a few configs to simulate API runs
    api = LocalTrainingApi()
    shapes = default_model_shapes()[:4]
    runs = []
    for flops in [int(1e15), int(3e15), int(1e16)]:
        for shape in shapes:
            config = make_config(
                shape=shape,
                train_flops=flops,
                batch_size=128,
                learning_rate=6e-4,
            )
            runs.append(api.loss(config))

    optima = select_best_by_compute(runs)
    assert len(optima) == 3

    loss_fit = fit_loss_scaling(optima)
    assert loss_fit.irreducible_loss >= 0.0
    assert loss_fit.exponent < 0.0  # Loss decreases as compute increases
    assert loss_fit.sse >= 0.0

    model_fit = fit_model_size_scaling(optima)
    assert model_fit.exponent > 0.0  # Optimal model size grows with compute
