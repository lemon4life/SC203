#!/usr/bin/env python3
"""Unified evaluation & category audit script for Phase-2 Real-CE LoRA sweeps and 2,000-step curves.

Supports:
1. Auditing the 30 validation patch IDs, their 30 scene keys, and the 945 = 645 train + 300 val split source.
2. Classifying all 154 Real-CE validation boxes (and the 63 HR-solvable subset) into diagnostic slices:
   - numeric_or_date : transcript contains ASCII digits (e.g. 2024/06/17, 24h)
   - latin_only      : contains ASCII letters [A-Za-z] and zero CJK characters (e.g. paas)
   - border_touching : box within `border_margin` px (default 16) of the 512x512 patch border
   - cjk_signage     : contains CJK Unified Ideographs
3. Computing Box Error Rate (1 - WCR), Mean CER, Cross-Seed OCR Disagreement, Win/Tie/Loss vs. LoRA 2,
   and optional whole-patch LPIPS / PSNR_Y / SSIM_Y / VRAM / s_per_step across arbitrary methods/checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.split_realce_by_scene import extract_scene_key  # noqa: E402


CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
DIGIT_RE = re.compile(r"[0-9]")
LATIN_RE = re.compile(r"[A-Za-z]")


def classify_box_categories(
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    transcript: str,
    patch_size: int = 512,
    border_margin: int = 16,
) -> list[str]:
    """Return non-exclusive diagnostic categories matching a text bounding box."""
    cats: list[str] = []
    if DIGIT_RE.search(transcript):
        cats.append("numeric_or_date")
    has_cjk = bool(CJK_RE.search(transcript))
    has_latin = bool(LATIN_RE.search(transcript))
    if has_latin and not has_cjk:
        cats.append("latin_only")
    if has_cjk:
        cats.append("cjk_signage")
    if (
        x <= border_margin
        or y <= border_margin
        or (x + w) >= (patch_size - border_margin)
        or (y + h) >= (patch_size - border_margin)
    ):
        cats.append("border_touching")
    return cats


def _pairwise_disagreement(preds: list[str]) -> float:
    n = len(preds)
    if n < 2:
        return 0.0
    total_pairs = n * (n - 1) // 2
    dis = 0
    for i in range(n):
        for j in range(i + 1, n):
            if preds[i] != preds[j]:
                dis += 1
    return dis / float(total_pairs)


def _mean(vals: list[float]) -> float:
    return sum(vals) / float(len(vals)) if vals else 0.0


def load_reference_single_eval(csv_path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Load per-box single-evaluation reference CSV (HR ceiling or Bicubic floor)."""
    out: dict[tuple[str, int], dict[str, Any]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["image"].strip(), int(row["box_id"]))
            out[key] = {
                "wrong_char": float(row["wrong_char"]),
                "cer": float(row["cer"]),
                "ocr_norm": row["ocr_norm"],
                "ocr_raw": row.get("ocr_raw", ""),
                "exact_match": int(row.get("exact_match", 1 if float(row["wrong_char"]) == 0.0 else 0)),
            }
    return out


def load_multi_seed_eval(per_box_seed_csv: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Aggregate 10-seed text_eval_per_box_seed.csv into per-box metrics."""
    grouped: dict[tuple[str, int], list[dict[str, str]]] = {}
    with per_box_seed_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["image"].strip(), int(row["box_id"]))
            grouped.setdefault(key, []).append(row)

    out: dict[tuple[str, int], dict[str, Any]] = {}
    for key, rows in grouped.items():
        rows_sorted = sorted(rows, key=lambda r: int(r["seed"]))
        preds = [r["ocr_norm"] for r in rows_sorted]
        wrs = [float(r["wrong_char"]) for r in rows_sorted]
        cers = [float(r["cer"]) for r in rows_sorted]
        out[key] = {
            "num_seeds": len(rows_sorted),
            "wrong_char_rate": _mean(wrs),
            "cer_mean": _mean(cers),
            "seed_disagreement": _pairwise_disagreement(preds),
            "num_distinct_ocr": len(set(preds)),
            "preds_by_seed": {str(r["seed"]): r["ocr_norm"] for r in rows_sorted},
        }
    return out


def load_summary_by_image_spread(summary_by_image_csv: Path) -> float | None:
    """Read image-mean crop_pixel_seed_spread_l1 from text_eval_summary_by_image.csv if available."""
    if not summary_by_image_csv.is_file():
        return None
    with summary_by_image_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    vals = [float(r["crop_pixel_seed_spread_l1"]) for r in rows if r.get("crop_pixel_seed_spread_l1")]
    return _mean(vals) if vals else None


def load_whole_patch_metrics(metrics_csv: Path | None) -> dict[str, float] | None:
    """Read ALL_IMAGES_MEAN whole-patch metrics from compute_metrics.py summary CSV."""
    if metrics_csv is None or not metrics_csv.is_file():
        return None
    with metrics_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    target = next((r for r in rows if r.get("image_stem") == "ALL_IMAGES_MEAN"), None)
    if target is None:
        return None
    return {
        "psnr_y_mean": float(target["psnr_y_mean"]),
        "ssim_y_mean": float(target["ssim_y_mean"]),
        "lpips_mean": float(target["lpips_mean"]),
        "pairwise_seed_px_mean_abs_diff": float(target.get("pairwise_seed_px_mean_abs_diff", 0.0)),
        "pairwise_seed_lpips_mean": float(target.get("pairwise_seed_lpips_mean", 0.0)),
    }


def load_manifest_cost(manifest_json: Path | None) -> dict[str, Any] | None:
    """Read VRAM, step time, lambda_cons, cons_t_max, and cons_active_ratio from lora_run_manifest.json."""
    if manifest_json is None or not manifest_json.is_file():
        return None
    m = json.loads(manifest_json.read_text(encoding="utf-8"))
    t_stats = m.get("timestep_and_consistency_stats") or {}
    return {
        "max_steps": m.get("max_steps"),
        "grad_accum": m.get("grad_accum"),
        "total_minibatches": m.get("total_minibatches"),
        "seed": m.get("seed"),
        "lambda_cons": m.get("lambda_cons"),
        "cons_t_min": m.get("cons_t_min"),
        "cons_t_max": m.get("cons_t_max"),
        "cons_active_ratio": t_stats.get("cons_active_ratio"),
        "peak_vram_allocated_gib": m.get("peak_vram_allocated_gib"),
        "peak_vram_reserved_gib": m.get("peak_vram_reserved_gib"),
        "mean_step_time_sec": m.get("mean_step_time_sec"),
        "p95_step_time_sec": m.get("p95_step_time_sec"),
    }


def summarize_subset(
    keys: list[tuple[str, int]],
    method_boxes: dict[tuple[str, int], dict[str, Any]],
    ref_boxes: dict[tuple[str, int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute box-weighted error, CER, disagreement, and Win/Tie/Loss vs ref on `keys`."""
    present = [k for k in keys if k in method_boxes]
    if not present:
        return {"num_boxes": 0}
    wrs = [method_boxes[k].get("wrong_char_rate", method_boxes[k].get("wrong_char", 0.0)) for k in present]
    cers = [method_boxes[k].get("cer_mean", method_boxes[k].get("cer", 0.0)) for k in present]
    dis_vals = [method_boxes[k]["seed_disagreement"] for k in present if "seed_disagreement" in method_boxes[k]]

    res: dict[str, Any] = {
        "num_boxes": len(present),
        "box_error_rate": round(_mean(wrs), 4),
        "mean_cer": round(_mean(cers), 4),
        "seed_disagreement": round(_mean(dis_vals), 4) if dis_vals else None,
    }
    if ref_boxes is not None:
        cer_w = cer_t = cer_l = 0
        wr_w = wr_t = wr_l = 0
        for k in present:
            if k not in ref_boxes:
                continue
            dc = method_boxes[k].get("cer_mean", method_boxes[k].get("cer", 0.0)) - ref_boxes[k].get(
                "cer_mean", ref_boxes[k].get("cer", 0.0)
            )
            dw = method_boxes[k].get("wrong_char_rate", method_boxes[k].get("wrong_char", 0.0)) - ref_boxes[k].get(
                "wrong_char_rate", ref_boxes[k].get("wrong_char", 0.0)
            )
            if dc < -1e-6:
                cer_w += 1
            elif dc > 1e-6:
                cer_l += 1
            else:
                cer_t += 1
            if dw < -1e-6:
                wr_w += 1
            elif dw > 1e-6:
                wr_l += 1
            else:
                wr_t += 1
        res["cer_win_tie_loss_vs_ref"] = f"{cer_w}/{cer_t}/{cer_l}"
        res["box_err_win_tie_loss_vs_ref"] = f"{wr_w}/{wr_t}/{wr_l}"
    return res


def parse_method_spec(spec: str) -> tuple[str, Path, Path | None, Path | None]:
    """Parse `--method label=ocr_dir[:metrics_summary_csv[:lora_manifest_json]]`."""
    if "=" not in spec:
        raise ValueError(f"Expected label=ocr_dir[:metrics_csv[:manifest_json]], got {spec!r}")
    label, rest = spec.rsplit("=", 1)
    parts = rest.split(":")
    ocr_dir = Path(parts[0]).expanduser()
    metrics_csv = Path(parts[1]).expanduser() if len(parts) > 1 and parts[1] else None
    manifest_json = Path(parts[2]).expanduser() if len(parts) > 2 and parts[2] else None
    return label.strip(), ocr_dir, metrics_csv, manifest_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze Real-CE LoRA sweep, categories, and split audit.")
    parser.add_argument("--val-boxes-csv", type=Path, default=REPO_ROOT / "data" / "realce_val_boxes.csv")
    parser.add_argument(
        "--hr-ceiling-csv",
        type=Path,
        default=REPO_ROOT / "results" / "pilot_lora2_vs_lora3_reconly" / "text_eval_hr_ceiling_per_box.csv",
    )
    parser.add_argument(
        "--bicubic-floor-csv",
        type=Path,
        default=REPO_ROOT / "results" / "pilot_lora2_vs_lora3_reconly" / "text_eval_bicubic_floor_per_box.csv",
    )
    parser.add_argument(
        "--lora2-ref-dir",
        type=Path,
        default=REPO_ROOT / "results" / "pilot_lora2_vs_lora3_reconly",
    )
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        help="Method spec: label=ocr_dir[:metrics_summary_csv[:lora_manifest_json]]",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "results" / "realce_val_split_and_category_audit.json",
    )
    args = parser.parse_args(argv)

    with args.val_boxes_csv.open("r", encoding="utf-8", newline="") as handle:
        val_rows = list(csv.DictReader(handle))

    hr_boxes = load_reference_single_eval(args.hr_ceiling_csv)
    bic_boxes = load_reference_single_eval(args.bicubic_floor_csv)
    lora2_ref_boxes = load_multi_seed_eval(args.lora2_ref_dir / "text_eval_per_box_seed.csv")

    all_keys: list[tuple[str, int]] = []
    solvable_keys: list[tuple[str, int]] = []
    cat_keys_all: dict[str, list[tuple[str, int]]] = {
        "numeric_or_date": [],
        "latin_only": [],
        "border_touching": [],
        "cjk_signage": [],
    }
    cat_keys_solvable: dict[str, list[tuple[str, int]]] = {k: [] for k in cat_keys_all}
    val_patches: dict[str, dict[str, Any]] = {}
    box_metadata: dict[tuple[str, int], dict[str, Any]] = {}

    for row in val_rows:
        img = row["image"].strip()
        bid = int(row["box_id"])
        key = (img, bid)
        x = float(row["x"])
        y = float(row["y"])
        w = float(row["w"])
        h = float(row["h"])
        transcript = row.get("transcript", "")
        cats = classify_box_categories(x=x, y=y, w=w, h=h, transcript=transcript)
        is_solvable = key in hr_boxes and hr_boxes[key]["exact_match"] == 1

        all_keys.append(key)
        if is_solvable:
            solvable_keys.append(key)
        for c in cats:
            cat_keys_all[c].append(key)
            if is_solvable:
                cat_keys_solvable[c].append(key)

        box_metadata[key] = {
            "image": img,
            "box_id": bid,
            "xywh": [x, y, w, h],
            "transcript": transcript,
            "categories": cats,
            "hr_solvable": is_solvable,
        }
        scene_id = extract_scene_key(img)
        entry = val_patches.setdefault(
            img, {"val_patch_image": img, "parent_scene_id": scene_id, "num_boxes": 0, "num_hr_solvable": 0}
        )
        entry["num_boxes"] += 1
        if is_solvable:
            entry["num_hr_solvable"] += 1

    # Default built-in methods when user didn't pass custom --method args (plus include custom methods)
    method_records: list[tuple[str, dict[tuple[str, int], dict[str, Any]], float | None, dict[str, float] | None, dict[str, Any] | None]] = [
        ("HR Ceiling", hr_boxes, None, None, None),
        ("Bicubic x4", bic_boxes, None, None, None),
    ]

    if not args.method:
        pilot_json_path = REPO_ROOT / "results" / "lora_pilot_summary.json"
        pilot_json = json.loads(pilot_json_path.read_text(encoding="utf-8")) if pilot_json_path.is_file() else {}
        l2_patch = pilot_json.get("whole_patch_10seed_val", {}).get("lora2")
        l3_patch = pilot_json.get("whole_patch_10seed_val", {}).get("lora3")
        l2_cost = pilot_json.get("training_cost_and_vram", {}).get("lora2")
        l3_cost = pilot_json.get("training_cost_and_vram", {}).get("lora3")

        method_records.append(
            (
                "LoRA 2 (lambda=0.0, 400st)",
                lora2_ref_boxes,
                load_summary_by_image_spread(args.lora2_ref_dir / "text_eval_summary_by_image.csv"),
                l2_patch,
                l2_cost,
            )
        )
        l3_dir = REPO_ROOT / "results" / "pilot_lora3_reconly"
        if (l3_dir / "text_eval_per_box_seed.csv").is_file():
            method_records.append(
                (
                    "LoRA 3 (lambda=0.5, 400st)",
                    load_multi_seed_eval(l3_dir / "text_eval_per_box_seed.csv"),
                    load_summary_by_image_spread(l3_dir / "text_eval_summary_by_image.csv"),
                    l3_patch,
                    l3_cost,
                )
            )
    else:
        pilot_json_path = REPO_ROOT / "results" / "lora_pilot_summary.json"
        pilot_json = json.loads(pilot_json_path.read_text(encoding="utf-8")) if pilot_json_path.is_file() else {}
        for spec in args.method:
            label, ocr_dir, metrics_csv, manifest_json = parse_method_spec(spec)
            boxes = load_multi_seed_eval(ocr_dir / "text_eval_per_box_seed.csv")
            spread = load_summary_by_image_spread(ocr_dir / "text_eval_summary_by_image.csv")
            whole_m = load_whole_patch_metrics(metrics_csv)
            if whole_m is None:
                if "lam=0.0" in label:
                    whole_m = pilot_json.get("whole_patch_10seed_val", {}).get("lora2")
                elif "lam=0.5" in label:
                    whole_m = pilot_json.get("whole_patch_10seed_val", {}).get("lora3")
            cost_m = load_manifest_cost(manifest_json)
            if cost_m is None:
                if "lam=0.0" in label:
                    cost_m = pilot_json.get("training_cost_and_vram", {}).get("lora2")
                elif "lam=0.5" in label:
                    cost_m = pilot_json.get("training_cost_and_vram", {}).get("lora3")
            method_records.append((label, boxes, spread, whole_m, cost_m))

    methods_summary: list[dict[str, Any]] = []
    for label, m_boxes, crop_spread, whole_m, cost_m in method_records:
        use_ref = lora2_ref_boxes if label not in {"HR Ceiling", "Bicubic x4"} else None
        all_stats = summarize_subset(all_keys, m_boxes, ref_boxes=use_ref)
        sol_stats = summarize_subset(solvable_keys, m_boxes, ref_boxes=use_ref)
        cat_stats = {
            "all_154": {c: summarize_subset(c_keys, m_boxes, ref_boxes=use_ref) for c, c_keys in cat_keys_all.items()},
            "solvable_63": {
                c: summarize_subset(c_keys, m_boxes, ref_boxes=use_ref) for c, c_keys in cat_keys_solvable.items()
            },
        }
        methods_summary.append(
            {
                "label": label,
                "all_154_boxes": all_stats,
                "hr_solvable_63_boxes": sol_stats,
                "crop_pixel_seed_spread_l1": round(crop_spread, 4) if crop_spread is not None else None,
                "whole_patch_metrics": whole_m,
                "training_cost": cost_m,
                "category_breakdown": cat_stats,
            }
        )

    # Highlight inspection cases (date, latin, border-touching)
    highlight_targets = [
        ("ip12pro_output_100APPLE_renew_29_patch000.png", 1),  # 2024/06/17 numeric/date
        ("ip11pro_output_TextSR_renew_114_patch000.png", 1),  # paas latin
    ]
    border_solvable = [k for k in cat_keys_solvable["border_touching"] if k not in highlight_targets][:3]
    highlight_keys = highlight_targets + border_solvable
    highlights: list[dict[str, Any]] = []
    for key in highlight_keys:
        if key not in box_metadata:
            continue
        item = dict(box_metadata[key])
        item["per_method"] = {}
        for label, m_boxes, _, _, _ in method_records:
            if key in m_boxes:
                b = m_boxes[key]
                item["per_method"][label] = {
                    "box_error_rate": round(b.get("wrong_char_rate", b.get("wrong_char", 0.0)), 4),
                    "mean_cer": round(b.get("cer_mean", b.get("cer", 0.0)), 4),
                    "seed_disagreement": round(b["seed_disagreement"], 4) if "seed_disagreement" in b else None,
                    "sample_ocr": b.get("ocr_norm") or b.get("preds_by_seed", {}).get("42"),
                }
        highlights.append(item)

    feasibility_path = REPO_ROOT / "results" / "realce_feasibility_summary.json"
    feasibility = json.loads(feasibility_path.read_text(encoding="utf-8")) if feasibility_path.is_file() else {}
    splits_info = feasibility.get("splits", {})

    audit_payload = {
        "realce_dataset_provenance": {
            "total_4x_pairs": feasibility.get("total_4x_pairs", 945),
            "official_train_pairs_13mm_52mm": splits_info.get("train", {}).get("matched_4x", 645),
            "official_val_pairs_13mm_52mm_held_out": splits_info.get("val", {}).get("matched_4x", 300),
            "explanation": (
                "Real-CE contains 945 paired (13mm, 52mm) full images in total: 645 pairs in official "
                "RealCE/train/ and 300 pairs in official RealCE/val/. scripts/split_realce_by_scene.py "
                "--split train ONLY splits the 645 official RealCE/train/ pairs by root scene ID (80% train / "
                "20% internal val, seed=42), leaving all 300 official RealCE/val/ pairs 100% untouched for final test evaluation."
            ),
        },
        "validation_split_summary": {
            "num_val_patches": len(val_patches),
            "num_distinct_parent_scenes": len({v["parent_scene_id"] for v in val_patches.values()}),
            "total_boxes": len(all_keys),
            "hr_solvable_boxes": len(solvable_keys),
            "val_patches": sorted(val_patches.values(), key=lambda x: x["val_patch_image"]),
        },
        "category_counts": {
            "all_154": {c: len(ks) for c, ks in cat_keys_all.items()},
            "solvable_63": {c: len(ks) for c, ks in cat_keys_solvable.items()},
        },
        "methods": methods_summary,
        "highlight_boxes": highlights,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Print readable Markdown tables
    print("\n=== Validation Cohort Overview (154 All / 63 HR-Solvable) ===")
    print("| Method | All-154 BoxErr | All-154 CER | All-154 Disag | Solv-63 BoxErr | Solv-63 CER | Solv-63 Disag | Solv CER W/T/L | Solv BoxErr W/T/L | LPIPS |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for m in methods_summary:
        a = m["all_154_boxes"]
        s = m["hr_solvable_63_boxes"]
        wp = m["whole_patch_metrics"] or {}
        lpips_val = wp.get("lpips_mean", wp.get("lpips"))
        lpips_s = f"{lpips_val:.4f}" if lpips_val is not None else "---"
        a_dis = f"{100*a['seed_disagreement']:.2f}%" if a.get("seed_disagreement") is not None else "---"
        s_dis = f"{100*s['seed_disagreement']:.2f}%" if s.get("seed_disagreement") is not None else "---"
        print(
            f"| {m['label']} | {100*a['box_error_rate']:.2f}% | {a['mean_cer']:.4f} | {a_dis} "
            f"| {100*s['box_error_rate']:.2f}% | {s['mean_cer']:.4f} | {s_dis} "
            f"| {s.get('cer_win_tie_loss_vs_ref', '---')} | {s.get('box_err_win_tie_loss_vs_ref', '---')} | {lpips_s} |"
        )

    print("\n=== Diagnostic Category Breakdown (63 HR-Solvable Boxes: Numeric/Date, Latin-Only, Border<=16px, CJK) ===")
    print(f"Category counts (All 154): {audit_payload['category_counts']['all_154']}")
    print(f"Category counts (Solvable 63): {audit_payload['category_counts']['solvable_63']}")
    for cat in ("numeric_or_date", "latin_only", "border_touching", "cjk_signage"):
        print(f"\nCategory: {cat} (solvable={len(cat_keys_solvable[cat])}, all={len(cat_keys_all[cat])})")
        for m in methods_summary:
            cs = m["category_breakdown"]["solvable_63"][cat]
            ca = m["category_breakdown"]["all_154"][cat]
            s_dis = f"{100*cs['seed_disagreement']:.2f}%" if cs.get("seed_disagreement") is not None else "---"
            print(
                f"  {m['label']:28s} | Solv CER={cs['mean_cer']:.4f} BoxErr={100*cs['box_error_rate']:5.1f}% Dis={s_dis:>6s} "
                f"| All CER={ca['mean_cer']:.4f} BoxErr={100*ca['box_error_rate']:5.1f}%"
            )

    print(f"\nSaved audit JSON: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
