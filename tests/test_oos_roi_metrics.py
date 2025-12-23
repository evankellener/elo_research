"""Additional ROI metric regression tests using the GA example helper."""

import importlib.util
import pathlib

import numpy as np
import pandas as pd


def _load_example_module():
    root = pathlib.Path(__file__).resolve().parents[1]
    module_path = root / "examples" / "example_ga_optimization.py"
    spec = importlib.util.spec_from_file_location("example_ga_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


example_module = _load_example_module()


def _make_training_df(fav_rating: float, dog_rating: float):
    return pd.DataFrame(
        {
            "DATE": pd.to_datetime(["2024-01-01"]),
            "FIGHTER": ["Favorite"],
            "opp_FIGHTER": ["Underdog"],
            "postcomp_elo": [fav_rating],
            "opp_postcomp_elo": [dog_rating],
            "result": [1],
        }
    )


def _make_test_df(results):
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-02-01"] * len(results)),
            "fighter": ["Favorite"] * len(results),
            "opp_fighter": ["Underdog"] * len(results),
            "result": results,
        }
    )


def _expected_roi(rating_fav: float, rating_dog: float, results, denominator: float):
    # Mirror test_out_of_sample_metrics ROI logic for deterministic checks
    probs = []
    for _ in results:
        elo_diff = rating_fav - rating_dog
        probs.append(1.0 / (1.0 + 10.0 ** (-elo_diff / denominator)))

    total_profit = 0.0
    total_bets = 0
    for pred, actual in zip(probs, results):
        pred_winner = 1 if pred > 0.5 else 0
        pred_prob = pred if pred_winner == 1 else (1 - pred)
        pred_prob_clamped = max(pred_prob, example_module.MIN_BET_PROBABILITY)
        payout_multiplier = (1.0 / pred_prob_clamped) - 1.0
        if pred_winner == actual:
            total_profit += payout_multiplier
        else:
            total_profit -= 1.0
        total_bets += 1

    return total_profit / total_bets if total_bets else -1.0


def test_roi_positive_run_matches_expected_value():
    denominator = 400
    training_df = _make_training_df(1700, 1500)
    results = [1, 1]
    test_df = _make_test_df(results)

    expected_roi = _expected_roi(1700, 1500, results, denominator)
    metrics = example_module.test_out_of_sample_metrics(
        training_df, test_df, denominator=denominator, base_elo=1500, verbose=False
    )

    assert np.isclose(metrics["roi"], expected_roi, atol=1e-6)
    assert metrics["n_bets"] == len(results)


def test_roi_negative_run_matches_expected_value():
    denominator = 400
    training_df = _make_training_df(1700, 1500)
    results = [0, 0]
    test_df = _make_test_df(results)

    expected_roi = _expected_roi(1700, 1500, results, denominator)
    metrics = example_module.test_out_of_sample_metrics(
        training_df, test_df, denominator=denominator, base_elo=1500, verbose=False
    )

    assert np.isclose(metrics["roi"], expected_roi, atol=1e-6)
    assert metrics["n_bets"] == len(results)


def test_roi_returns_worst_case_when_no_predictions():
    training_df = pd.DataFrame(
        {
            "DATE": pd.to_datetime([]),
            "FIGHTER": [],
            "opp_FIGHTER": [],
            "postcomp_elo": [],
            "opp_postcomp_elo": [],
            "result": [],
        }
    )
    test_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-02-01"]),
            "fighter": ["Unknown"],
            "opp_fighter": ["Unknown"],
            "result": [1],
        }
    )

    metrics = example_module.test_out_of_sample_metrics(
        training_df, test_df, denominator=400, base_elo=1500, verbose=False
    )

    assert metrics["roi"] == -1.0
    assert metrics["n_predictions"] == 0
