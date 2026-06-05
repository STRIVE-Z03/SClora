"""
SCALE-FG result table builder.

The public table contains two rows:
  - LASRC: the primary-view dense LASRC candidate.
  - SCALE_FG: the support-loss-selected candidate over the finite grid.
"""

import argparse
import csv
import json
import math
import random
from pathlib import Path

import utils


TABLE_ROWS = [
    ("LASRC", "Primary-view dense LASRC candidate"),
    ("SCALE_FG", "Support-loss-selected candidate over the SCALE-FG grid"),
]


def load_task_payloads(input_dir):
    paths = sorted(Path(input_dir).glob("task_*.json"))
    if not paths:
        raise FileNotFoundError("No task_*.json files under %s" % input_dir)
    payloads = []
    for path in paths:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            raise ValueError("%s: expected a JSON object" % path)
        payloads.append(payload)
    return payloads


def load_run_meta(input_dir):
    meta_path = Path(input_dir) / "_run_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError("Missing _run_meta.json under %s" % input_dir)
    with meta_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def find_point(points, view_name, sparse_rate, op, tol=1e-9):
    for point in points:
        if (
            point.get("view") == view_name
            and abs(float(point.get("sparse_rate", float("nan"))) - float(sparse_rate)) <= tol
            and point.get("op") == op
        ):
            return point
    return None


def point_result(point):
    if point is None:
        return {
            "accuracy": float("nan"),
            "selection": "missing",
            "support_loss": float("nan"),
            "correct_per_query": [],
        }
    return {
        "accuracy": float(point["query_accuracy"]),
        "selection": point["id"],
        "support_loss": float(point.get("support_loss", float("nan"))),
        "correct_per_query": grid_correctness_vector(point),
    }


def grid_correctness_vector(point):
    n_queries = int(point.get("query_total", len(point.get("query_predictions", []))))
    correct = [0] * n_queries
    for record in point.get("query_predictions", []):
        query_id = int(record["query_id"])
        if 0 <= query_id < n_queries:
            correct[query_id] = 1 if bool(record["correct"]) else 0
    return correct


def apply_rules(task_payload, primary_view):
    points = task_payload.get("primitive_grid")
    if not isinstance(points, list) or not points:
        raise ValueError("%s: missing primitive_grid" % task_payload.get("task_name", "<unknown>"))

    lasrc_point = find_point(points, primary_view, 0.0, "lasrc")
    if lasrc_point is None:
        raise ValueError(
            "%s: missing LASRC primitive for primary_view=%s"
            % (task_payload.get("task_name", "<unknown>"), primary_view)
        )

    method_points = [point for point in points if point.get("op") in {"linear", "lasrc"}]
    if not method_points:
        raise ValueError("%s: no SCALE-FG candidate points" % task_payload.get("task_name", "<unknown>"))
    best_point = min(method_points, key=lambda point: float(point["support_loss"]))

    return {
        "LASRC": point_result(lasrc_point),
        "SCALE_FG": point_result(best_point),
    }


def finite_mean(values):
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else float("nan")


def finite_pair_vectors(rule_accs, ref_accs):
    pairs = [
        (float(a), float(b))
        for a, b in zip(rule_accs, ref_accs)
        if math.isfinite(float(a)) and math.isfinite(float(b))
    ]
    if not pairs:
        return [], []
    return [a for a, _ in pairs], [b for _, b in pairs]


def paired_bootstrap(rule_accs, ref_accs, n_boot=10000, ci=0.95, seed=20240514):
    rng = random.Random(int(seed))
    n = len(rule_accs)
    if n == 0 or n != len(ref_accs):
        return float("nan"), float("nan"), float("nan"), float("nan")
    deltas = []
    for _ in range(int(n_boot)):
        indices = [rng.randrange(n) for _ in range(n)]
        deltas.append(sum(rule_accs[index] - ref_accs[index] for index in indices) / n)
    deltas.sort()
    alpha = (1.0 - float(ci)) / 2.0
    lower = deltas[int(math.floor(alpha * len(deltas)))]
    upper = deltas[min(len(deltas) - 1, int(math.ceil((1.0 - alpha) * len(deltas))) - 1)]
    observed = sum(rule_accs[index] - ref_accs[index] for index in range(n)) / n
    above = sum(1 for delta in deltas if delta >= 0.0)
    below = sum(1 for delta in deltas if delta <= 0.0)
    p_value = max(0.0, min(1.0, 2.0 * min(above, below) / len(deltas)))
    return float(observed), float(lower), float(upper), float(p_value)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate LASRC and SCALE-FG task results.",
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--bootstrap_iters", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=20240514)
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = load_run_meta(input_dir)
    primary_view = meta["primary_view"]
    payloads = load_task_payloads(input_dir)

    per_task = {}
    for payload in payloads:
        task_name = payload.get("task_name")
        if not task_name:
            raise ValueError("A task payload is missing task_name")
        per_task[task_name] = apply_rules(payload, primary_view)

    task_names = sorted(per_task.keys())
    rule_names = [name for name, _ in TABLE_ROWS]
    rule_to_accs = {
        name: [float(per_task[task_name][name]["accuracy"]) for task_name in task_names]
        for name in rule_names
    }
    ref_accs = rule_to_accs["LASRC"]

    rows = []
    descriptions = dict(TABLE_ROWS)
    for name in rule_names:
        accs = rule_to_accs[name]
        paired_rule, paired_ref = finite_pair_vectors(accs, ref_accs)
        delta, lower, upper, p_value = paired_bootstrap(
            paired_rule,
            paired_ref,
            n_boot=int(args.bootstrap_iters),
            seed=int(args.bootstrap_seed),
        )
        rows.append(
            {
                "rule": name,
                "description": descriptions[name],
                "num_tasks": len([value for value in accs if math.isfinite(float(value))]),
                "average": finite_mean(accs),
                "delta_vs_LASRC": delta,
                "ci95_lo_vs_LASRC": lower,
                "ci95_hi_vs_LASRC": upper,
                "p_vs_LASRC": p_value,
            }
        )

    csv_path = output_dir / "main_table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md_lines = [
        "# SCALE-FG Main Table",
        "",
        "Reference row: **LASRC**.",
        "Tasks: %d. Grid size: %d." % (len(task_names), int(meta.get("primitive_grid_size", -1))),
        "Primary view: `%s`." % primary_view,
        "",
        "| Rule | Avg | Delta vs LASRC | 95% CI | p |",
        "|---|---:|---:|---|---:|",
    ]
    for row in rows:
        md_lines.append(
            "| `%s` | %.4f | %+.4f | [%+.4f, %+.4f] | %.4f |"
            % (
                row["rule"],
                row["average"],
                row["delta_vs_LASRC"],
                row["ci95_lo_vs_LASRC"],
                row["ci95_hi_vs_LASRC"],
                row["p_vs_LASRC"],
            )
        )
    md_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `LASRC` is the primary-view dense LASRC candidate.",
            "- `SCALE_FG` selects one candidate by support loss before query evaluation.",
        ]
    )
    md_path = output_dir / "main_table.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    json_path = output_dir / "main_table.json"
    utils.save_json(
        json_path,
        {
            "input_dir": "xx/xx",
            "primary_view": primary_view,
            "bootstrap_iters": int(args.bootstrap_iters),
            "bootstrap_seed": int(args.bootstrap_seed),
            "tasks": task_names,
            "per_task_rule_accuracy": {
                task_name: {
                    rule_name: float(per_task[task_name][rule_name]["accuracy"])
                    for rule_name in rule_names
                }
                for task_name in task_names
            },
            "per_task_rule_selection": {
                task_name: {
                    rule_name: per_task[task_name][rule_name]["selection"]
                    for rule_name in rule_names
                }
                for task_name in task_names
            },
            "rows": rows,
        },
    )

    print("Tasks evaluated : %d" % len(task_names))
    print("CSV  : %s" % csv_path)
    print("MD   : %s" % md_path)
    print("JSON : %s" % json_path)


if __name__ == "__main__":
    main()
