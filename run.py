"""Run the six-robot localization and fencing simulation."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import statistics
import sys
from tempfile import mkdtemp

from fencing.study import StudyConfig, run_study_case


ROOT = Path(__file__).resolve().parent
CASES = ("proposed", "current_only", "published_global")


def prepare_output(directory):
    if directory is None:
        results = ROOT / "results"
        results.mkdir(exist_ok=True)
        prefix = datetime.now().strftime("%Y%m%d-%H%M%S-")
        directory = Path(mkdtemp(prefix=prefix, dir=results))
    else:
        directory = directory.expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        if any(directory.iterdir()):
            raise FileExistsError(f"Choose an empty or new output directory: {directory}")

    # Reserve the run even if another invocation selected the same empty directory.
    with (directory / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump({"status": "incomplete"}, stream)
    return directory


def trial_metrics(case, summary):
    performance = summary["performance"]
    means = performance["full_mean"]
    return {
        "E_q_mean_m": means["localization_max"] if case != "published_global" else None,
        "E_s_mean_m": means["observer_max"] if case != "published_global" else None,
        "E_v_mean_mps": means["speed_max"],
        "E_c_mean_m": means["centroid_error"],
        "E_c_tail_mean_m": performance["tail_mean"]["centroid_error"],
        "inside_hull_percent": 100 * performance["containment_fraction"],
        "minimum_pair_distance_m": summary["safety"]["minimum_pair_distance"],
    }


def paired_statistics(trials, cases):
    paired = {}
    for case in cases:
        if case == "current_only":
            continue
        scheduled = [r for r in trials if r["case"] == case]
        completed = [r for r in scheduled if r["status"] == "completed"]
        result = {
            "completed_seeds": [r["seed"] for r in completed],
            "completed_count": len(completed),
            "scheduled_count": len(scheduled),
        }
        for name in ("E_c_mean_m", "E_c_tail_mean_m"):
            values = [r["metrics"][name] for r in completed]
            result[name] = {
                "mean": statistics.mean(values) if values else None,
                "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            }
        paired[case] = result
    return paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("all", *CASES), default="all")
    parser.add_argument("--repeats", action="store_true",
                        help="use five paired seeds; current_only uses the primary seed only")
    parser.add_argument("--duration", type=float, help="override the simulation time in seconds")
    parser.add_argument("--jobs", type=int, default=3, help="maximum parallel simulations")
    parser.add_argument("--output", type=Path, help="empty or new destination (default: a new run in results/)")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be a positive integer")
    settings = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    if args.duration is not None:
        settings["duration"] = args.duration
    try:
        config = StudyConfig(**settings)
        output = prepare_output(args.output)
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))

    cases = CASES if args.case == "all" else (args.case,)
    tasks = []
    for case in cases:
        repeats = 5 if args.repeats and case != "current_only" else 1
        for seed in range(config.seed, config.seed + repeats):
            folder = Path(case) if seed == config.seed else Path("paired", str(seed), case)
            tasks.append((case, seed, folder))

    print(f"Output: {output}", flush=True)
    trials = []
    with ProcessPoolExecutor(max_workers=min(args.jobs, len(tasks))) as pool:
        futures = {
            pool.submit(run_study_case, case, replace(config, seed=seed), output / folder):
            (case, seed, folder) for case, seed, folder in tasks
        }
        for future in as_completed(futures):
            case, seed, folder = futures[future]
            row = {"case": case, "seed": seed, "directory": folder.as_posix()}
            try:
                row["metrics"] = trial_metrics(case, future.result())
                row["status"] = "completed"
                print(f"{case} [{seed}] mean E_c = {row['metrics']['E_c_mean_m']:.6g} m", flush=True)
            except Exception as error:
                row.update(status="failed", error=str(error))
                print(f"{case} [{seed}] failed: {error}", file=sys.stderr, flush=True)
            trials.append(row)

    trials.sort(key=lambda r: (r["seed"], cases.index(r["case"])))
    complete = all(r["status"] == "completed" for r in trials)
    report = {
        "status": "completed" if complete else "failed",
        "primary_seed": config.seed,
        "duration_seconds": config.duration,
        "tail_seconds": min(40.0, config.duration),
        "trials": trials,
    }
    if args.repeats:
        report["paired_statistics"] = paired_statistics(trials, cases)
    temporary = output / "summary.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output / "summary.json")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
