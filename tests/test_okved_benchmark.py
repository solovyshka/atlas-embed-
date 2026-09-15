from pathlib import Path

from benchmark_okved import parse_args, run_benchmark
from synthetic_data.generate_okved import generate_okved_data


def test_benchmark_runs_all_neural_arms(tmp_path: Path) -> None:
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

    args = parse_args(
        [
            "--data",
            str(data_path),
            "--output-dir",
            str(output_dir),
            "--rare-threshold",
            "1",
            "--epochs",
            "2",
            "--batch-size",
            "512",
            "--patience",
            "2",
            "--bootstrap-samples",
            "10",
            "--match-parameter-budget",
            "--device",
            "cpu",
        ]
    )
    result = run_benchmark(args)

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
