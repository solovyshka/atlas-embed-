from pathlib import Path

import benchmark_okved
import pytest
from synthetic_data.generate_okved import generate_okved_data


def test_benchmark_runs_all_neural_arms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, _ = generate_okved_data(
        rows=3_000,
        clients=1_500,
        start_date="2024-01-01",
        end_date="2026-08-31",
        bad_rate=0.12,
        regime="hierarchical",
        oov_rate=0.1,
        oov_cutoff="2026-01-01",
        seed=9,
    )
    data_path = tmp_path / "okved.csv"
    output_dir = tmp_path / "benchmark"
    data.to_csv(data_path, index=False)

    monkeypatch.setattr(benchmark_okved, "DATA_PATH", data_path)
    monkeypatch.setattr(benchmark_okved, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(benchmark_okved, "RARE_THRESHOLD", 1)
    monkeypatch.setattr(benchmark_okved, "EPOCHS", 2)
    monkeypatch.setattr(benchmark_okved, "BATCH_SIZE", 512)
    monkeypatch.setattr(benchmark_okved, "EARLY_STOPPING_PATIENCE", 2)
    monkeypatch.setattr(benchmark_okved, "BOOTSTRAP_SAMPLES", 10)
    monkeypatch.setattr(benchmark_okved, "MATCH_PARAMETER_BUDGET", True)
    monkeypatch.setattr(
        benchmark_okved, "resolve_device", lambda: benchmark_okved.torch.device("cpu")
    )
    result = benchmark_okved.run_benchmark()

    assert [arm["arm"] for arm in result["arms"]] == [
        "boost_score",
        "flat_residual_nn",
        "hierarchical_residual_nn",
    ]
    assert sum(result["coverage"].values()) == result["rows"]["test"]
    assert result["config"]["match_parameter_budget"] is True
    assert set(result["trainable_parameters"]) == {"flat", "hierarchical"}
    assert (output_dir / "results.json").is_file()
    assert (output_dir / "results.csv").is_file()
    assert (output_dir / "flat_model.pt").is_file()
    assert (output_dir / "hierarchical_model.pt").is_file()
