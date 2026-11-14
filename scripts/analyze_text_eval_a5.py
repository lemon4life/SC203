#!/usr/bin/env python3
"""Analyze Assignment 5 baseline text-evaluation CSVs across HR-solvability cohorts.

This stdlib-only script audits the 193-box annotated dataset by partitioning boxes
into OCR-solvable (HR exact match == 1, 136 boxes) and HR-unsolvable (HR exact
match == 0, 57 boxes) cohorts. It reports box-weighted and image-weighted means,
standard errors, capped CER (min(CER, 1.0)), medians/IQRs, 10,000-resample
image-level bootstrap 95% confidence intervals, paired prompt deltas, and the
Canon_001 ("eduroam") 10-seed audit.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
from pathlib import Path
from typing import Any


COHORTS = ("ALL_193_BOXES", "OCR_SOLVABLE_136_BOXES", "OCR_UNSOLVABLE_57_BOXES")


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load all rows from a UTF-8 CSV file as dictionaries."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def mean(values: list[float]) -> float:
    """Compute arithmetic mean; raises ValueError on empty input."""
    if not values:
        raise ValueError("Cannot compute mean of empty list")
    return sum(values) / len(values)


def mean_and_se(values: list[float]) -> tuple[float, float]:
    """Compute sample mean and standard error (ddof=1)."""
    n = len(values)
    m = mean(values)
    if n <= 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in values) / (n - 1)
    return m, math.sqrt(var) / math.sqrt(n)


def quantile_sorted(sorted_vals: list[float], q: float) -> float:
    """Compute linear-interpolated quantile q in [0, 1] on a pre-sorted list."""
    if not sorted_vals:
        raise ValueError("Cannot compute quantile of empty list")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def bootstrap_ci(
    values: list[float],
    *,
    resamples: int,
    rng: random.Random,
) -> tuple[float, float]:
    """Compute percentile 95% CI by resampling values with replacement."""
    n = len(values)
    if n == 0:
        raise ValueError("Cannot bootstrap empty list")
    if n == 1:
        return values[0], values[0]
    means: list[float] = []
    for _ in range(resamples):
        sample_sum = sum(values[rng.randrange(n)] for _ in range(n))
        means.append(sample_sum / n)
    means.sort()
    return quantile_sorted(means, 0.025), quantile_sorted(means, 0.975)


def compute_box_disagreement(rows: list[dict[str, str]]) -> dict[tuple[str, int], dict[str, Any]]:
    """Compute pairwise cross-seed OCR disagreement and distinct string counts per box."""
    by_box: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["stem"], int(row["box_id"]))
        by_box.setdefault(key, []).append(row)

    stats: dict[tuple[str, int], dict[str, Any]] = {}
    for key, box_rows in by_box.items():
        sorted_rows = sorted(box_rows, key=lambda r: int(r["seed"]))
        transcripts = [r["ocr_norm"] for r in sorted_rows]
        pairs = list(itertools.combinations(transcripts, 2))
        disagree = (
            sum(1 for a, b in pairs if a != b) / len(pairs) if pairs else 0.0
        )
        stats[key] = {
            "stem": key[0],
            "box_id": key[1],
            "num_seeds": len(sorted_rows),
            "num_pairs": len(pairs),
            "disagreement_rate": disagree,
            "distinct_ocr_norm": len(set(transcripts)),
        }
    return stats


def filter_keys_for_cohort(
    solvable_map: dict[tuple[str, int], int],
    cohort: str,
) -> set[tuple[str, int]]:
    """Return the set of (stem, box_id) keys belonging to the named cohort."""
    if cohort == "ALL_193_BOXES":
        return set(solvable_map.keys())
    if cohort == "OCR_SOLVABLE_136_BOXES":
        return {k for k, v in solvable_map.items() if v == 1}
    if cohort == "OCR_UNSOLVABLE_57_BOXES":
        return {k for k, v in solvable_map.items() if v == 0}
    raise ValueError(f"Unknown cohort: {cohort}")


def analyze_single_condition(
    *,
    condition: str,
    cohort: str,
    cohort_keys: set[tuple[str, int]],
    rows: list[dict[str, str]],
    box_disagreement_map: dict[tuple[str, int], dict[str, Any]] | None,
    resamples: int,
    rng: random.Random,
) -> dict[str, Any]:
    """Compute exact cohort statistics and bootstrap 95% CIs for one condition."""
    filtered = [r for r in rows if (r["stem"], int(r["box_id"])) in cohort_keys]
    if not filtered:
        raise ValueError(f"No rows matched cohort={cohort} for condition={condition}")

    wrong_vals = [int(r["wrong_char"]) for r in filtered]
    cer_vals = [float(r["cer"]) for r in filtered]
    cer_capped_vals = [min(c, 1.0) for c in cer_vals]
    cer_sorted = sorted(cer_vals)
    unique_boxes = {(r["stem"], int(r["box_id"])) for r in filtered}

    by_img_rows: dict[str, list[dict[str, str]]] = {}
    for row in filtered:
        by_img_rows.setdefault(row["stem"], []).append(row)

    img_wrong_rates = [
        mean([float(int(r["wrong_char"])) for r in img_rows])
        for _, img_rows in sorted(by_img_rows.items())
    ]
    img_cer_means = [
        mean([float(r["cer"]) for r in img_rows])
        for _, img_rows in sorted(by_img_rows.items())
    ]
    img_cer_capped_means = [
        mean([min(float(r["cer"]), 1.0) for r in img_rows])
        for _, img_rows in sorted(by_img_rows.items())
    ]

    img_wcr_m, img_wcr_se = mean_and_se(img_wrong_rates)
    img_cer_m, img_cer_se = mean_and_se(img_cer_means)
    img_cer_cap_m, img_cer_cap_se = mean_and_se(img_cer_capped_means)

    img_wcr_ci = bootstrap_ci(img_wrong_rates, resamples=resamples, rng=rng)
    img_cer_ci = bootstrap_ci(img_cer_means, resamples=resamples, rng=rng)
    img_cer_cap_ci = bootstrap_ci(img_cer_capped_means, resamples=resamples, rng=rng)

    record: dict[str, Any] = {
        "cohort": cohort,
        "condition": condition,
        "num_boxes": len(unique_boxes),
        "num_images": len(by_img_rows),
        "num_records": len(filtered),
        "exact_matches": sum(1 - w for w in wrong_vals),
        "box_wrong_char_rate": mean([float(w) for w in wrong_vals]),
        "box_cer_mean": mean(cer_vals),
        "box_cer_capped_mean": mean(cer_capped_vals),
        "box_cer_median": quantile_sorted(cer_sorted, 0.5),
        "box_cer_q25": quantile_sorted(cer_sorted, 0.25),
        "box_cer_q75": quantile_sorted(cer_sorted, 0.75),
        "image_wrong_char_rate": img_wcr_m,
        "image_wrong_char_se": img_wcr_se,
        "image_wrong_char_ci95_low": img_wcr_ci[0],
        "image_wrong_char_ci95_high": img_wcr_ci[1],
        "image_cer_mean": img_cer_m,
        "image_cer_se": img_cer_se,
        "image_cer_ci95_low": img_cer_ci[0],
        "image_cer_ci95_high": img_cer_ci[1],
        "image_cer_capped_mean": img_cer_cap_m,
        "image_cer_capped_se": img_cer_cap_se,
        "image_cer_capped_ci95_low": img_cer_cap_ci[0],
        "image_cer_capped_ci95_high": img_cer_cap_ci[1],
        "box_disagreement_rate": "",
        "box_distinct_ocr_mean": "",
        "image_disagreement_rate": "",
        "image_disagreement_se": "",
        "image_disagreement_ci95_low": "",
        "image_disagreement_ci95_high": "",
        "image_distinct_ocr_mean": "",
    }

    if box_disagreement_map is not None:
        cohort_box_stats = [
            box_disagreement_map[k] for k in sorted(cohort_keys) if k in box_disagreement_map
        ]
        by_img_dis: dict[str, list[dict[str, Any]]] = {}
        for item in cohort_box_stats:
            by_img_dis.setdefault(item["stem"], []).append(item)

        img_dis_rates = [
            mean([x["disagreement_rate"] for x in items])
            for _, items in sorted(by_img_dis.items())
        ]
        img_distinct_means = [
            mean([float(x["distinct_ocr_norm"]) for x in items])
            for _, items in sorted(by_img_dis.items())
        ]
        img_dis_m, img_dis_se = mean_and_se(img_dis_rates)
        img_dis_ci = bootstrap_ci(img_dis_rates, resamples=resamples, rng=rng)

        record.update(
            {
                "box_disagreement_rate": mean(
                    [x["disagreement_rate"] for x in cohort_box_stats]
                ),
                "box_distinct_ocr_mean": mean(
                    [float(x["distinct_ocr_norm"]) for x in cohort_box_stats]
                ),
                "image_disagreement_rate": img_dis_m,
                "image_disagreement_se": img_dis_se,
                "image_disagreement_ci95_low": img_dis_ci[0],
                "image_disagreement_ci95_high": img_dis_ci[1],
                "image_distinct_ocr_mean": mean(img_distinct_means),
            }
        )

    return record


def build_eduroam_audit(
    per_box_seed_rows: list[dict[str, str]],
    summary_by_image_rows: list[dict[str, str]],
) -> dict[str, Any]:
    """Extract Canon_001 ("eduroam") 10-seed OCR transcripts and spread metrics."""
    canon_rows = sorted(
        [r for r in per_box_seed_rows if r["stem"] == "Canon_001_LR4"],
        key=lambda r: int(r["seed"]),
    )
    img_row = next(
        (r for r in summary_by_image_rows if r["stem"] == "Canon_001_LR4"), None
    )
    seeds_data = []
    for r in canon_rows:
        tokens = r["ocr_norm"].split()
        seeds_data.append(
            {
                "seed": int(r["seed"]),
                "ocr_raw": r["ocr_raw"],
                "ocr_norm": r["ocr_norm"],
                "last_token": tokens[-1] if tokens else "",
                "wrong_char": int(r["wrong_char"]),
                "cer": float(r["cer"]),
            }
        )
    full_transcripts = [x["ocr_norm"] for x in seeds_data]
    last_tokens = [x["last_token"] for x in seeds_data]
    return {
        "stem": "Canon_001_LR4",
        "gt_transcript": canon_rows[0]["gt_transcript"] if canon_rows else "",
        "box_coords": {
            "x": int(canon_rows[0]["x"]),
            "y": int(canon_rows[0]["y"]),
            "w": int(canon_rows[0]["w"]),
            "h": int(canon_rows[0]["h"]),
        }
        if canon_rows
        else {},
        "num_seeds": len(seeds_data),
        "num_distinct_full_ocr_norm": len(set(full_transcripts)),
        "num_distinct_last_tokens": len(set(last_tokens)),
        "crop_pixel_seed_spread_l1": float(img_row["crop_pixel_seed_spread_l1"])
        if img_row is not None
        else None,
        "seeds": seeds_data,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of flat dictionaries to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Audit Assignment 5 text_eval CSVs by HR-solvability cohorts."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing text_eval_*.csv files (default: results).",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=10000,
        help="Number of image-level bootstrap resamples for 95%% CIs (default: 10000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap resampling (default: 42).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/text_eval_a5_cohort_breakdown.csv"),
        help="Output CSV path for cohort breakdown.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("results/text_eval_a5_summary.json"),
        help="Output JSON path for full structured summary.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run cohort analysis and save CSV and JSON summaries."""
    args = parse_args(argv)
    results_dir = args.results_dir

    hr_rows = load_csv_rows(results_dir / "text_eval_hr_ceiling_per_box.csv")
    lr_rows = load_csv_rows(results_dir / "text_eval_bicubic_floor_per_box.csv")
    dossr_rows = load_csv_rows(results_dir / "text_eval_per_box_seed.csv")
    paired_rows = load_csv_rows(results_dir / "text_eval_paired_comparison_by_image.csv")
    summary_img_rows = load_csv_rows(results_dir / "text_eval_summary_by_image.csv")

    solvable_map: dict[tuple[str, int], int] = {
        (r["stem"], int(r["box_id"])): int(r["exact_match"]) for r in hr_rows
    }
    if len(solvable_map) != len(hr_rows):
        raise ValueError("Duplicate (stem, box_id) entries in HR ceiling CSV")

    box_disagreement_map = compute_box_disagreement(dossr_rows)
    rng = random.Random(args.seed)

    table_rows: list[dict[str, Any]] = []
    for cohort in COHORTS:
        cohort_keys = filter_keys_for_cohort(solvable_map, cohort)
        for cond_name, cond_rows, dis_map in (
            ("HR_CEILING", hr_rows, None),
            ("BICUBIC_X4_FLOOR", lr_rows, None),
            ("DOSSR_DEFAULT_10SEED", dossr_rows, box_disagreement_map),
        ):
            table_rows.append(
                analyze_single_condition(
                    condition=cond_name,
                    cohort=cohort,
                    cohort_keys=cohort_keys,
                    rows=cond_rows,
                    box_disagreement_map=dis_map,
                    resamples=args.bootstrap_resamples,
                    rng=rng,
                )
            )

    delta_wcr = [
        float(r["delta_wrong_char_rate_compare_minus_base"]) for r in paired_rows
    ]
    delta_cer = [float(r["delta_cer_compare_minus_base"]) for r in paired_rows]
    pair_counts = [int(r["num_paired_box_seeds"]) for r in paired_rows]
    total_pairs = sum(pair_counts)
    box_dw_m = (
        sum(n * dw for n, dw in zip(pair_counts, delta_wcr)) / total_pairs
        if total_pairs
        else 0.0
    )
    box_dc_m = (
        sum(n * dc for n, dc in zip(pair_counts, delta_cer)) / total_pairs
        if total_pairs
        else 0.0
    )
    dw_m, dw_se = mean_and_se(delta_wcr)
    dc_m, dc_se = mean_and_se(delta_cer)
    dw_ci = bootstrap_ci(delta_wcr, resamples=args.bootstrap_resamples, rng=rng)
    dc_ci = bootstrap_ci(delta_cer, resamples=args.bootstrap_resamples, rng=rng)

    paired_summary = {
        "comparison": "empty_minus_default",
        "num_images": len(paired_rows),
        "num_paired_box_seeds": total_pairs,
        "box_delta_wrong_char_rate_mean": box_dw_m,
        "box_delta_cer_mean": box_dc_m,
        "image_delta_wrong_char_rate_mean": dw_m,
        "image_delta_wrong_char_rate_se": dw_se,
        "image_delta_wrong_char_rate_ci95": [dw_ci[0], dw_ci[1]],
        "image_delta_cer_mean": dc_m,
        "image_delta_cer_se": dc_se,
        "image_delta_cer_ci95": [dc_ci[0], dc_ci[1]],
    }

    eduroam_audit = build_eduroam_audit(dossr_rows, summary_img_rows)

    write_csv(args.out_csv, table_rows)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.seed,
        "cohorts": table_rows,
        "paired_prompt_comparison": paired_summary,
        "eduroam_audit": eduroam_audit,
    }
    args.out_json.write_text(
        json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8"
    )

    for row in table_rows:
        dis_str = (
            f" dis={row['box_disagreement_rate']:.4f}(img {row['image_disagreement_rate']:.4f})"
            if row["box_disagreement_rate"] != ""
            else ""
        )
        print(
            f"[{row['cohort']}] {row['condition']}: "
            f"exact={row['exact_matches']}/{row['num_records']} "
            f"WCR={row['box_wrong_char_rate']*100:.2f}%(img {row['image_wrong_char_rate']*100:.2f}% "
            f"[{row['image_wrong_char_ci95_low']*100:.2f}%, {row['image_wrong_char_ci95_high']*100:.2f}%]) "
            f"CER={row['box_cer_mean']:.4f}(cap {row['box_cer_capped_mean']:.4f}, "
            f"med {row['box_cer_median']:.4f}, img {row['image_cer_mean']:.4f})"
            f"{dis_str}"
        )
    print(
        f"[PAIRED empty-default] imgs={paired_summary['num_images']} "
        f"box_seeds={paired_summary['num_paired_box_seeds']} "
        f"box_dWCR={box_dw_m*100:+.2f}% box_dCER={box_dc_m:+.4f} "
        f"(img_dWCR={dw_m*100:+.2f}% img_dCER={dc_m:+.4f})"
    )
    print(f"Wrote: {args.out_csv}")
    print(f"Wrote: {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
