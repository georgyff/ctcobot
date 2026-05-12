"""Corpus EDA for ctcobot: ctco_policies + gitlab_handbook."""

import os
import re
import json
import random
import hashlib
import warnings
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import tiktoken
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path("/Users/georgyfurs/Documents/DevProjects/ctcobot")
RAW = ROOT / "data/01_raw"
HANDBOOK = RAW / "gitlab_handbook"
POLICIES = RAW / "ctco_policies"
OUT = ROOT / "data/08_reporting/eda"
OUT.mkdir(parents=True, exist_ok=True)

TOKENIZER = tiktoken.get_encoding("cl100k_base")

# ── Helpers ────────────────────────────────────────────────────────────────
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
HUGO_SHORTCODE_RE = re.compile(r"\{\{[<%%].*?[>%%]\}\}", re.DOTALL)
HUGO_INLINE_RE = re.compile(r"\{\{[<%%].*?[>%%]\}\}")


def strip_frontmatter(text: str) -> tuple[str, str]:
    """Return (frontmatter_block, body)."""
    m = FRONTMATTER_RE.match(text)
    if m:
        return m.group(1), text[m.end():]
    return "", text


def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text, disallowed_special=()))


def file_stats(path: Path) -> dict:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    fm, body = strip_frontmatter(raw)
    shortcodes = HUGO_SHORTCODE_RE.findall(raw)
    body_stripped = HUGO_SHORTCODE_RE.sub("", body).strip()
    tokens_full = count_tokens(raw)
    tokens_body = count_tokens(body_stripped)
    size_bytes = path.stat().st_size
    word_count = len(body_stripped.split())
    # simple near-dup fingerprint: first 500 chars of body
    fingerprint = hashlib.md5(body_stripped[:500].encode()).hexdigest()
    return {
        "path": str(path.relative_to(RAW)),
        "folder": path.relative_to(HANDBOOK).parts[0] if HANDBOOK in path.parents else "ctco_policies",
        "size_bytes": size_bytes,
        "tokens_full": tokens_full,
        "tokens_body": tokens_body,
        "word_count": word_count,
        "has_frontmatter": bool(fm),
        "frontmatter_only": (tokens_full > 0 and tokens_body == 0),
        "empty_file": (size_bytes == 0),
        "shortcode_count": len(shortcodes),
        "has_shortcodes": len(shortcodes) > 0,
        "fingerprint": fingerprint,
    }


# ── 1. Collect all files ───────────────────────────────────────────────────
print("Scanning files...")
records = []

for md_file in sorted(HANDBOOK.rglob("*.md")):
    stats = file_stats(md_file)
    if stats:
        records.append(stats)

# CTCO policies – markdown not present; note as special entries
policy_files = list(POLICIES.iterdir())
policy_summary = [{"name": f.name, "size_bytes": f.stat().st_size,
                   "suffix": f.suffix} for f in policy_files]

df = pd.DataFrame(records)
print(f"Total markdown files indexed: {len(df)}")

# ── 2. Per-folder counts & sizes ──────────────────────────────────────────
folder_stats = (
    df.groupby("folder")
    .agg(
        file_count=("path", "count"),
        total_bytes=("size_bytes", "sum"),
        median_tokens=("tokens_body", "median"),
        mean_tokens=("tokens_body", "mean"),
    )
    .sort_values("file_count", ascending=False)
    .reset_index()
)
folder_stats["total_kb"] = folder_stats["total_bytes"] / 1024

# ── 3. Token distribution stats ───────────────────────────────────────────
# Sample up to 2000 files for speed (stratified by folder)
sample_size = min(2000, len(df))
sample_df = df.sample(n=sample_size, random_state=42)

token_stats = {
    "n_files": len(df),
    "min": int(df["tokens_body"].min()),
    "p5": float(df["tokens_body"].quantile(0.05)),
    "p25": float(df["tokens_body"].quantile(0.25)),
    "median": float(df["tokens_body"].median()),
    "mean": float(df["tokens_body"].mean()),
    "p75": float(df["tokens_body"].quantile(0.75)),
    "p95": float(df["tokens_body"].quantile(0.95)),
    "p99": float(df["tokens_body"].quantile(0.99)),
    "max": int(df["tokens_body"].max()),
}
print("Token stats:", token_stats)

# ── 4. Data quality issues ─────────────────────────────────────────────────
empty_files = df[df["empty_file"]]
frontmatter_only = df[df["frontmatter_only"] & ~df["empty_file"]]
very_short = df[(df["tokens_body"] < 30) & ~df["empty_file"] & ~df["frontmatter_only"]]
shortcode_heavy = df[df["shortcode_count"] > 5]
near_dups = df[df.duplicated(subset="fingerprint", keep=False)]

# Files with no frontmatter
no_frontmatter = df[~df["has_frontmatter"]]

print(f"Empty files: {len(empty_files)}")
print(f"Frontmatter-only: {len(frontmatter_only)}")
print(f"Very short (<30 tokens body): {len(very_short)}")
print(f"Shortcode-heavy (>5 shortcodes): {len(shortcode_heavy)}")
print(f"Near-duplicate fingerprint pairs: {len(near_dups)}")
print(f"No frontmatter: {len(no_frontmatter)}")

# ── 5. Style setup ────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE = sns.color_palette("muted")
ACCENT = "#4C72B0"
WARN = "#DD8452"
BAD = "#C44E52"
GOOD = "#55A868"

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 1 — File count per top-level folder                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
fig1, ax1 = plt.subplots(figsize=(14, 7))
bars = ax1.barh(
    folder_stats["folder"],
    folder_stats["file_count"],
    color=[ACCENT if v > 100 else PALETTE[1] for v in folder_stats["file_count"]],
    edgecolor="white", linewidth=0.5,
)
ax1.bar_label(bars, fmt="%d", padding=3, fontsize=9)
ax1.set_xlabel("Markdown file count", fontsize=11)
ax1.set_title("Fig 1 — Markdown file count per top-level folder\n(gitlab_handbook, 39 folders, 4,019 files total)", fontsize=13)
ax1.invert_yaxis()
ax1.axvline(folder_stats["file_count"].median(), color=WARN, linestyle="--", lw=1.5, label=f"Median={folder_stats['file_count'].median():.0f}")
ax1.legend(fontsize=10)
plt.tight_layout()
fig1.savefig(OUT / "fig1_file_count_per_folder.png", dpi=150)
plt.close(fig1)
print("Saved fig1")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 2 — Total KB per folder (log scale)                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
fs_sorted = folder_stats.sort_values("total_kb", ascending=False)
fig2, ax2 = plt.subplots(figsize=(14, 7))
ax2.barh(fs_sorted["folder"], fs_sorted["total_kb"], color=ACCENT, edgecolor="white")
ax2.set_xscale("log")
ax2.set_xlabel("Total size (KB, log scale)", fontsize=11)
ax2.set_title("Fig 2 — Total corpus size per top-level folder (KB, log scale)", fontsize=13)
ax2.invert_yaxis()
plt.tight_layout()
fig2.savefig(OUT / "fig2_total_kb_per_folder.png", dpi=150)
plt.close(fig2)
print("Saved fig2")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 3 — Token distribution histogram (log-capped at 4000)          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
tokens_capped = df["tokens_body"].clip(upper=4000)
fig3, ax3 = plt.subplots(figsize=(12, 5))
ax3.hist(tokens_capped, bins=80, color=ACCENT, edgecolor="white", alpha=0.85)
pcts = [("P25", token_stats["p25"], "green"),
        ("Median", token_stats["median"], WARN),
        ("P75", token_stats["p75"], "darkorange"),
        ("P95", token_stats["p95"], BAD)]
for label, val, col in pcts:
    xval = min(val, 4000)
    ax3.axvline(xval, color=col, linestyle="--", lw=1.5, label=f"{label}={val:.0f}")
ax3.set_xlabel("Body tokens (capped at 4,000 for readability)", fontsize=11)
ax3.set_ylabel("File count", fontsize=11)
ax3.set_title("Fig 3 — Token count distribution across 4,019 markdown files", fontsize=13)
ax3.legend(fontsize=10)
plt.tight_layout()
fig3.savefig(OUT / "fig3_token_distribution.png", dpi=150)
plt.close(fig3)
print("Saved fig3")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 4 — Log-scale token histogram (full range)                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
fig4, ax4 = plt.subplots(figsize=(12, 5))
bins_log = np.logspace(0, np.log10(df["tokens_body"].clip(lower=1).max() + 1), 70)
ax4.hist(df["tokens_body"].clip(lower=1), bins=bins_log, color=ACCENT, edgecolor="white", alpha=0.85)
ax4.set_xscale("log")
for label, val, col in pcts:
    ax4.axvline(val if val > 0 else 1, color=col, linestyle="--", lw=1.5, label=f"{label}={val:.0f}")
ax4.set_xlabel("Body tokens (log scale)", fontsize=11)
ax4.set_ylabel("File count", fontsize=11)
ax4.set_title("Fig 4 — Token distribution (log scale, full range)", fontsize=13)
ax4.legend(fontsize=10)
plt.tight_layout()
fig4.savefig(OUT / "fig4_token_distribution_log.png", dpi=150)
plt.close(fig4)
print("Saved fig4")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 5 — Box-plot: token body per folder (top-15)                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
top15 = folder_stats.nlargest(15, "file_count")["folder"].tolist()
df_top15 = df[df["folder"].isin(top15)]
order15 = df_top15.groupby("folder")["tokens_body"].median().sort_values(ascending=False).index

fig5, ax5 = plt.subplots(figsize=(14, 6))
sns.boxplot(
    data=df_top15, y="folder", x="tokens_body",
    order=order15, palette="muted", fliersize=2,
    ax=ax5, orient="h",
)
ax5.set_xlim(0, 3000)
ax5.set_xlabel("Body tokens (capped display at 3,000)", fontsize=11)
ax5.set_title("Fig 5 — Token body distribution by folder (top-15 by file count)", fontsize=13)
plt.tight_layout()
fig5.savefig(OUT / "fig5_token_boxplot_per_folder.png", dpi=150)
plt.close(fig5)
print("Saved fig5")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 6 — CDF of body tokens                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
sorted_tokens = np.sort(df["tokens_body"].values)
cdf = np.arange(1, len(sorted_tokens) + 1) / len(sorted_tokens)
fig6, ax6 = plt.subplots(figsize=(10, 5))
ax6.plot(sorted_tokens, cdf, color=ACCENT, lw=2)
for label, val, col in pcts:
    ax6.axvline(val, color=col, linestyle="--", lw=1.5, label=f"{label}={val:.0f}")
ax6.axvline(token_stats["mean"], color="purple", linestyle=":", lw=1.5, label=f"Mean={token_stats['mean']:.0f}")
ax6.set_xscale("log")
ax6.set_xlim(1, sorted_tokens.max())
ax6.set_ylim(0, 1.02)
ax6.set_xlabel("Body tokens (log scale)", fontsize=11)
ax6.set_ylabel("Cumulative fraction of files", fontsize=11)
ax6.set_title("Fig 6 — CDF of body token counts", fontsize=13)
ax6.legend(fontsize=10)
plt.tight_layout()
fig6.savefig(OUT / "fig6_cdf_tokens.png", dpi=150)
plt.close(fig6)
print("Saved fig6")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 7 — Data quality issues stacked bar                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
dq_categories = {
    "Empty files (0 bytes)": len(empty_files),
    "Frontmatter-only\n(body=0 tokens)": len(frontmatter_only),
    "Very short\n(<30 body tokens)": len(very_short),
    "Shortcode-heavy\n(>5 shortcodes)": len(shortcode_heavy),
    "Near-duplicate\nfingerprint": len(near_dups),
    "No YAML frontmatter": len(no_frontmatter),
}
fig7, ax7 = plt.subplots(figsize=(10, 5))
colors7 = [BAD, BAD, WARN, WARN, "#9467BD", PALETTE[3]]
bars7 = ax7.bar(dq_categories.keys(), dq_categories.values(), color=colors7, edgecolor="white")
ax7.bar_label(bars7, fmt="%d", padding=3, fontsize=11, fontweight="bold")
ax7.set_ylabel("File count", fontsize=11)
ax7.set_title("Fig 7 — Data quality issues across corpus", fontsize=13)
ax7.set_ylim(0, max(dq_categories.values()) * 1.2)
plt.xticks(fontsize=9)
plt.tight_layout()
fig7.savefig(OUT / "fig7_data_quality_issues.png", dpi=150)
plt.close(fig7)
print("Saved fig7")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 8 — Shortcode count per folder (heatmap-style)                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝
sc_stats = (
    df.groupby("folder")
    .agg(
        pct_has_shortcode=("has_shortcodes", "mean"),
        total_shortcodes=("shortcode_count", "sum"),
        n_files=("path", "count"),
    )
    .sort_values("pct_has_shortcode", ascending=False)
    .reset_index()
)
sc_stats["pct_has_shortcode_pct"] = sc_stats["pct_has_shortcode"] * 100

fig8, ax8 = plt.subplots(figsize=(14, 7))
colors8 = [BAD if v > 40 else (WARN if v > 20 else GOOD) for v in sc_stats["pct_has_shortcode_pct"]]
bars8 = ax8.barh(sc_stats["folder"], sc_stats["pct_has_shortcode_pct"], color=colors8, edgecolor="white")
ax8.set_xlabel("% of files containing Hugo shortcodes", fontsize=11)
ax8.set_title("Fig 8 — Hugo shortcode prevalence per folder\n(red >40%, amber >20%, green ≤20%)", fontsize=13)
ax8.axvline(20, color=WARN, linestyle="--", lw=1, label="20% threshold")
ax8.invert_yaxis()
ax8.legend(fontsize=10)
plt.tight_layout()
fig8.savefig(OUT / "fig8_shortcode_prevalence.png", dpi=150)
plt.close(fig8)
print("Saved fig8")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 9 — Chunk size decision: token histogram with candidate sizes  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
fig9, ax9 = plt.subplots(figsize=(12, 5))
ax9.hist(tokens_capped, bins=80, color=ACCENT, edgecolor="white", alpha=0.75, label="Files")
chunk_candidates = [(128, "128", "#2ca02c"), (256, "256 (proposed)", "#ff7f0e"), (512, "512", "#d62728")]
for val, lbl, col in chunk_candidates:
    ax9.axvline(val, color=col, linestyle="-", lw=2, label=f"chunk={lbl}")
ax9.axvline(token_stats["median"], color="grey", linestyle=":", lw=1.5, label=f"Median={token_stats['median']:.0f}")
ax9.set_xlabel("Body tokens (capped at 4,000)", fontsize=11)
ax9.set_ylabel("File count", fontsize=11)
ax9.set_title("Fig 9 — Chunk size candidates vs. token distribution\n(chunk=256 keeps ~P50 files as 1-2 chunks)", fontsize=13)
ax9.legend(fontsize=10)
plt.tight_layout()
fig9.savefig(OUT / "fig9_chunk_size_decision.png", dpi=150)
plt.close(fig9)
print("Saved fig9")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 10 — Min-token threshold decision                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝
thresholds = range(0, 201, 5)
counts_above = [len(df[df["tokens_body"] >= t]) for t in thresholds]
pct_above = [c / len(df) * 100 for c in counts_above]
pct_removed = [100 - p for p in pct_above]

fig10, ax10 = plt.subplots(figsize=(10, 5))
ax10.plot(thresholds, pct_above, color=GOOD, lw=2, label="% files kept")
ax10.plot(thresholds, pct_removed, color=BAD, lw=2, linestyle="--", label="% files removed")
threshold_candidates = [30, 50, 100]
for t in threshold_candidates:
    kept = len(df[df["tokens_body"] >= t]) / len(df) * 100
    ax10.axvline(t, color="grey", linestyle=":", lw=1.2)
    ax10.annotate(f"t={t}\n{kept:.1f}% kept", xy=(t, kept), xytext=(t + 5, kept - 8),
                  fontsize=9, color="grey")
ax10.set_xlabel("Minimum body token threshold", fontsize=11)
ax10.set_ylabel("% of corpus files", fontsize=11)
ax10.set_title("Fig 10 — Effect of minimum token threshold on corpus coverage", fontsize=13)
ax10.legend(fontsize=10)
ax10.set_ylim(0, 105)
plt.tight_layout()
fig10.savefig(OUT / "fig10_min_token_threshold.png", dpi=150)
plt.close(fig10)
print("Saved fig10")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 11 — Priority folder selection for eval set                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# Score folders: file_count weight + mean_tokens weight + < 30% shortcode penalty
sc_stats_idx = sc_stats.set_index("folder")
folder_stats_eval = folder_stats.copy()
folder_stats_eval["pct_shortcodes"] = folder_stats_eval["folder"].map(
    sc_stats_idx["pct_has_shortcode"]
).fillna(0)
folder_stats_eval["mean_tokens_norm"] = (
    folder_stats_eval["mean_tokens"] / folder_stats_eval["mean_tokens"].max()
)
folder_stats_eval["file_count_norm"] = (
    folder_stats_eval["file_count"] / folder_stats_eval["file_count"].max()
)
folder_stats_eval["sc_penalty"] = folder_stats_eval["pct_shortcodes"].clip(upper=0.5)
folder_stats_eval["eval_score"] = (
    0.4 * folder_stats_eval["file_count_norm"]
    + 0.4 * folder_stats_eval["mean_tokens_norm"]
    - 0.2 * folder_stats_eval["sc_penalty"]
)
folder_stats_eval_sorted = folder_stats_eval.sort_values("eval_score", ascending=False).head(20)

fig11, axes11 = plt.subplots(1, 3, figsize=(18, 7))
# (a) file count
ax_a = axes11[0]
bars_a = ax_a.barh(folder_stats_eval_sorted["folder"], folder_stats_eval_sorted["file_count"],
                   color=[GOOD if v > 100 else PALETTE[1] for v in folder_stats_eval_sorted["file_count"]])
ax_a.invert_yaxis()
ax_a.set_xlabel("File count")
ax_a.set_title("(a) File count")
# (b) mean tokens
ax_b = axes11[1]
ax_b.barh(folder_stats_eval_sorted["folder"], folder_stats_eval_sorted["mean_tokens"], color=ACCENT)
ax_b.invert_yaxis()
ax_b.set_xlabel("Mean body tokens")
ax_b.set_title("(b) Mean body tokens")
# (c) eval score
ax_c = axes11[2]
colors_c = [GOOD if v >= folder_stats_eval_sorted["eval_score"].quantile(0.6) else WARN
            for v in folder_stats_eval_sorted["eval_score"]]
bars_c = ax_c.barh(folder_stats_eval_sorted["folder"], folder_stats_eval_sorted["eval_score"],
                   color=colors_c)
ax_c.bar_label(bars_c, fmt="%.2f", padding=2, fontsize=8)
ax_c.invert_yaxis()
ax_c.set_xlabel("Composite eval score")
ax_c.set_title("(c) Composite eval score")
fig11.suptitle("Fig 11 — Priority folder selection for evaluation Q&A dataset (top-20)", fontsize=13)
plt.tight_layout()
fig11.savefig(OUT / "fig11_priority_folders.png", dpi=150)
plt.close(fig11)
print("Saved fig11")

# ── Save summary tables ────────────────────────────────────────────────────
folder_stats.to_csv(OUT / "table_folder_stats.csv", index=False)
pd.DataFrame([token_stats]).to_csv(OUT / "table_token_stats.csv", index=False)

dq_df = pd.DataFrame([
    {"category": k, "file_count": v, "pct_of_corpus": round(v / len(df) * 100, 2)}
    for k, v in dq_categories.items()
])
dq_df.to_csv(OUT / "table_data_quality.csv", index=False)

folder_stats_eval_sorted.to_csv(OUT / "table_eval_priority_folders.csv", index=False)

# ── Print summary ──────────────────────────────────────────────────────────
print("\n=== CORPUS SUMMARY ===")
print(f"Total MD files: {len(df)}")
print(f"Total folders: {df['folder'].nunique()}")
total_mb = df['size_bytes'].sum() / (1024**2)
print(f"Total size: {total_mb:.1f} MB")
print(f"\nToken stats (body, post-shortcode strip):")
for k, v in token_stats.items():
    print(f"  {k}: {v:.0f}")
print(f"\nData quality:")
for k, v in dq_categories.items():
    print(f"  {k.replace(chr(10),' ')}: {v} ({v/len(df)*100:.1f}%)")
print(f"\nTop-5 priority folders for eval:")
print(folder_stats_eval_sorted[["folder","file_count","mean_tokens","eval_score"]].head(5).to_string())
print("\nAll figures saved to:", OUT)
