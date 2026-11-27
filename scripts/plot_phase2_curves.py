import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEPS = [0, 400, 800, 1200, 1600, 2000]
RUNS = {
    "lora2": {"label": "LoRA 2 (λ=0, L_rec only)", "color": "#257de0", "marker": "o"},
    "lora3": {"label": "LoRA 3 (λ=0.2, t≤499)", "color": "#f47230", "marker": "s"},
}
TEXT = "#191919"
MUTED = "#666666"
GRID = "#e6e6e6"
REF = "#8a8a8a"


def load(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {m["label"]: m for m in data["methods"]}


def metric(m, key):
    s = m["hr_solvable_63_boxes"]
    wp = m["whole_patch_metrics"] or {}
    return {
        "solv_boxerr": s["box_error_rate"] * 100.0,
        "solv_cer": s["mean_cer"],
        "solv_disag": (s["seed_disagreement"] or 0.0) * 100.0,
        "lpips": wp.get("lpips_mean"),
        "pair_px": wp.get("pairwise_seed_px_mean_abs_diff"),
    }[key]


def series(methods, run, key):
    base = methods["Base DoSSR"]
    ys = [metric(base, key)]
    for st in STEPS[1:]:
        ys.append(metric(methods[f"{run}_s231_st{st:04d}"], key))
    return ys


def style_axes(ax, ylabel):
    ax.set_xticks(STEPS)
    ax.set_xlabel("Optimizer step (×4 minibatches)", color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT, fontsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#bdbdbd")
    ax.tick_params(colors=MUTED, labelsize=8.5)


def draw_runs(ax, methods, key, fmt):
    ends = {}
    for run, spec in RUNS.items():
        ys = series(methods, run, key)
        ax.plot(
            STEPS, ys, color=spec["color"], linewidth=2, marker=spec["marker"],
            markersize=7, markeredgecolor="white", markeredgewidth=1.5,
            label=spec["label"], zorder=3,
        )
        ends[run] = ys
    return ends


def ref_line(ax, y, text, style="--"):
    ax.axhline(y, color=REF, linewidth=1.2, linestyle=style, zorder=1)
    ax.annotate(
        text, xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(4, 0),
        textcoords="offset points", va="center", ha="left", fontsize=8, color=MUTED,
    )


def callout(ax, x, y, text, dy, color):
    ax.annotate(
        text, xy=(x, y), xytext=(0, dy), textcoords="offset points", ha="center",
        fontsize=8.5, color=TEXT,
        arrowprops=dict(arrowstyle="-", color=color, linewidth=1),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, lw=1),
        zorder=5,
    )


def plot_ocr(methods, out):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    base = methods["Base DoSSR"]
    bic = methods["Bicubic x4"]

    ax = axes[0]
    ends = draw_runs(ax, methods, "solv_boxerr", "{:.2f}%")
    ref_line(ax, metric(base, "solv_boxerr"), f"Base DoSSR\n{metric(base, 'solv_boxerr'):.2f}%")
    ref_line(ax, metric(bic, "solv_boxerr"), f"Bicubic ×4\n{metric(bic, 'solv_boxerr'):.2f}%", ":")
    l3 = ends["lora3"]
    callout(ax, 1600, l3[4], f"peak st1600\n{l3[4]:.2f}%", -38, RUNS["lora3"]["color"])
    callout(ax, 2000, l3[5], f"st2000\n{l3[5]:.2f}%", -34, RUNS["lora3"]["color"])
    style_axes(ax, "Solvable-63 box error rate (%) ↓")
    ax.set_ylim(50, 88)
    ax.set_title("Box error rate (any char wrong)", loc="left", fontsize=11, color=TEXT)

    ax = axes[1]
    ends = draw_runs(ax, methods, "solv_cer", "{:.4f}")
    ref_line(ax, metric(base, "solv_cer"), f"Base DoSSR\n{metric(base, 'solv_cer'):.4f}")
    ref_line(ax, metric(bic, "solv_cer"), f"Bicubic ×4\n{metric(bic, 'solv_cer'):.4f}", ":")
    l3 = ends["lora3"]
    callout(ax, 1600, l3[4], f"peak st1600\n{l3[4]:.4f}", -38, RUNS["lora3"]["color"])
    callout(ax, 2000, l3[5], f"st2000\n{l3[5]:.4f}", -34, RUNS["lora3"]["color"])
    style_axes(ax, "Solvable-63 mean CER ↓")
    ax.set_ylim(0.38, 0.66)
    ax.set_title("Mean character error rate", loc="left", fontsize=11, color=TEXT)

    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside upper center", ncol=2, frameon=False, fontsize=9.5)
    fig.suptitle(
        "Real-CE internal val (20% scene split of train), 63 HR-solvable boxes × 10 seeds — step 0 = Base DoSSR (LoRA B=0 at init)",
        fontsize=10, color=MUTED, y=-0.02, va="top",
    )
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_stability(methods, out):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), constrained_layout=True)
    base = methods["Base DoSSR"]
    specs = [
        ("solv_disag", "Solvable-63 seed OCR disagreement (%) ↓", "{:.2f}%", "OCR disagreement across 10 seeds"),
        ("lpips", "Whole-patch LPIPS ↓", "{:.4f}", "Perceptual distance to HR"),
        ("pair_px", "Pairwise seed |Δpx| (0–255) ↓", "{:.3f}", "Pixel spread across 10 seeds"),
    ]
    for ax, (key, ylabel, fmt, title) in zip(axes, specs):
        ends = draw_runs(ax, methods, key, fmt)
        ref_line(ax, metric(base, key), "Base DoSSR\n" + fmt.format(metric(base, key)))
        style_axes(ax, ylabel)
        ax.set_title(title, loc="left", fontsize=11, color=TEXT)
        if key == "pair_px":
            l3 = ends["lora3"]
            callout(ax, 1600, l3[4], f"st1600\n{l3[4]:.3f}", 30, RUNS["lora3"]["color"])
            ax.set_ylim(4.5, 11)
        if key == "solv_disag":
            ax.set_ylim(30, 56)
        if key == "lpips":
            ax.set_ylim(0.37, 0.47)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside upper center", ncol=2, frameon=False, fontsize=9.5)
    fig.suptitle(
        "Real-CE internal val (20% scene split of train), 30 patches × 10 seeds (42..51) — step 0 = Base DoSSR (LoRA B=0 at init)",
        fontsize=10, color=MUTED, y=-0.02, va="top",
    )
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--curves-json", default="results/phase2/phase2_2000step_curves_s231.json")
    p.add_argument("--out-dir", default="results/phase2/figures")
    a = p.parse_args()
    methods = load(a.curves_json)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_ocr(methods, out / "phase2_curve1_ocr_boxerr_cer.png")
    plot_stability(methods, out / "phase2_curve2_disag_lpips_pairpx.png")
    print(f"wrote {out / 'phase2_curve1_ocr_boxerr_cer.png'}")
    print(f"wrote {out / 'phase2_curve2_disag_lpips_pairpx.png'}")


if __name__ == "__main__":
    main()
