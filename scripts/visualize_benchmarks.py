"""
Visualize benchmark metric progression across benchmark versions.

Usage (run from project root):
    python scripts/visualize_benchmarks.py

Reads all data/08_reporting/benchmark_report_v*.json files, sorts them by
version, and saves one PNG per metric to data/08_reporting/plots/.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


REPORTS_DIR = Path("data/08_reporting")
PLOTS_DIR = REPORTS_DIR / "plots"


def version_key(version_str: str) -> tuple:
    """Sort '1.10' > '1.9' > '1.0' correctly."""
    try:
        return tuple(int(x) for x in version_str.split("."))
    except ValueError:
        return (0,)


def load_reports() -> list[dict]:
    files = sorted(REPORTS_DIR.glob("benchmark_report_v*.json"))
    if not files:
        print("No benchmark report files found in", REPORTS_DIR)
        sys.exit(0)

    reports = []
    for f in files:
        data = json.loads(f.read_text())
        reports.append(data)

    reports.sort(key=lambda r: version_key(r.get("benchmark_version", "0")))
    return reports


def plot_metric(
    versions: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    output_name: str,
    target: float | None = None,
    target_label: str = "Target",
    higher_is_better: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))

    # Marker colors based on pass/fail against target
    if target is not None:
        colors = []
        for v in values:
            if higher_is_better:
                colors.append("green" if v >= target else "red")
            else:
                colors.append("green" if v <= target else "red")
    else:
        colors = ["steelblue"] * len(values)

    ax.plot(versions, values, color="steelblue", linewidth=1.5, zorder=1)
    ax.scatter(versions, values, color=colors, s=60, zorder=2)

    if target is not None:
        ax.axhline(
            target,
            color="crimson",
            linestyle="--",
            linewidth=1.2,
            label=f"{target_label} ({target})",
        )
        ax.legend(fontsize=9)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Benchmark Version")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(versions)))
    ax.set_xticklabels(versions)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / f"{output_name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    reports = load_reports()
    versions = [r.get("benchmark_version", "?") for r in reports]

    def extract(reports, *keys):
        result = []
        for r in reports:
            node = r
            for k in keys:
                node = node[k]
            result.append(float(node))
        return result

    def extract_optional(reports, *keys) -> tuple[list[str], list[float]]:
        """Like extract but skips reports where the key path is missing."""
        vers, vals = [], []
        for r in reports:
            node = r
            try:
                for k in keys:
                    node = node[k]
                vers.append(r.get("benchmark_version", "?"))
                vals.append(float(node))
            except (KeyError, TypeError):
                pass
        return vers, vals

    plot_metric(
        versions,
        extract(reports, "results", "retrieval", "hit_rate_at_5"),
        title="Hit Rate @ 5",
        ylabel="Hit Rate",
        output_name="hit_rate",
        target=0.70,
        target_label="Target ≥",
        higher_is_better=True,
    )

    plot_metric(
        versions,
        extract(reports, "results", "retrieval", "mrr"),
        title="Mean Reciprocal Rank (MRR)",
        ylabel="MRR",
        output_name="mrr",
        target=0.55,
        target_label="Target ≥",
        higher_is_better=True,
    )

    plot_metric(
        versions,
        extract(reports, "results", "quality", "avg_score"),
        title="Avg Quality Score (1–5)",
        ylabel="Score",
        output_name="quality_score",
        target=3.5,
        target_label="Target ≥",
        higher_is_better=True,
    )

    plot_metric(
        versions,
        extract(reports, "results", "latency", "p50_seconds"),
        title="P50 Latency",
        ylabel="Seconds",
        output_name="latency_p50",
        target=None,
        higher_is_better=False,
    )

    plot_metric(
        versions,
        extract(reports, "results", "latency", "p95_seconds"),
        title="P95 Latency",
        ylabel="Seconds",
        output_name="latency_p95",
        target=5.0,
        target_label="Target ≤",
        higher_is_better=False,
    )

    plot_metric(
        versions,
        extract(reports, "results", "latency", "p99_seconds"),
        title="P99 Latency",
        ylabel="Seconds",
        output_name="latency_p99",
        target=None,
        higher_is_better=False,
    )

    plot_metric(
        versions,
        extract(reports, "results", "latency", "avg_seconds"),
        title="Avg Latency",
        ylabel="Seconds",
        output_name="latency_avg",
        target=None,
        higher_is_better=False,
    )

    prec_vers, prec_vals = extract_optional(reports, "results", "retrieval", "avg_precision")
    plot_metric(
        prec_vers,
        prec_vals,
        title="Precision @ k",
        ylabel="Precision",
        output_name="precision_at_k",
        target=0.15,
        target_label="Target ≥",
        higher_is_better=True,
    )

    rec_vers, rec_vals = extract_optional(reports, "results", "retrieval", "avg_recall")
    plot_metric(
        rec_vers,
        rec_vals,
        title="Recall @ k",
        ylabel="Recall",
        output_name="recall_at_k",
        target=0.35,
        target_label="Target ≥",
        higher_is_better=True,
    )


if __name__ == "__main__":
    main()
