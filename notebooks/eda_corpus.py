"""
T-05: Corpus EDA — GitLab Handbook
Profiles the raw corpus before ingestion.
"""
import os
from collections import defaultdict
from pathlib import Path

import tiktoken

RAW_PATH = Path("data/01_raw/gitlab_handbook")
ENCODING = tiktoken.get_encoding("cl100k_base")
MIN_TOKENS = 50


def token_count(text: str) -> int:
    return len(ENCODING.encode(text))


def main():
    files = list(RAW_PATH.rglob("*.md"))
    print(f"\n{'='*60}")
    print(f"CORPUS OVERVIEW")
    print(f"{'='*60}")
    print(f"Total .md files found:     {len(files)}")

    # Per top-level folder counts
    folder_counts = defaultdict(int)
    folder_sizes = defaultdict(int)
    for f in files:
        # top-level folder relative to gitlab_handbook/
        parts = f.relative_to(RAW_PATH).parts
        top = parts[0] if len(parts) > 1 else "__root__"
        folder_counts[top] += 1
        folder_sizes[top] += f.stat().st_size

    print(f"\nTop-level folders ({len(folder_counts)} total):")
    print(f"  {'Folder':<40} {'Files':>6}  {'Size (KB)':>10}")
    print(f"  {'-'*40} {'-'*6}  {'-'*10}")
    for folder, count in sorted(folder_counts.items(), key=lambda x: -x[1]):
        kb = folder_sizes[folder] / 1024
        print(f"  {folder:<40} {count:>6}  {kb:>10.1f}")

    # Token distribution (sample up to 500 files for speed)
    print(f"\nToken distribution (sampling up to 500 files)...")
    sample = files[:500]
    token_counts = []
    empty = 0
    tiny = 0

    for f in sample:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                empty += 1
                continue
            tc = token_count(text)
            token_counts.append(tc)
            if tc < MIN_TOKENS:
                tiny += 1
        except Exception as e:
            print(f"  [WARN] Could not read {f}: {e}")

    if token_counts:
        token_counts.sort()
        n = len(token_counts)
        print(f"\n  Files sampled:             {len(sample)}")
        print(f"  Empty files:               {empty}")
        print(f"  Files < {MIN_TOKENS} tokens:          {tiny}")
        print(f"  Min tokens:                {token_counts[0]}")
        print(f"  Max tokens:                {token_counts[-1]}")
        print(f"  Median tokens:             {token_counts[n//2]}")
        print(f"  P25 tokens:                {token_counts[n//4]}")
        print(f"  P75 tokens:                {token_counts[3*n//4]}")
        print(f"  P95 tokens:                {token_counts[int(n*0.95)]}")
        avg = sum(token_counts) / n
        print(f"  Mean tokens:               {avg:.0f}")

    # Priority folders deep-dive
    print(f"\n{'='*60}")
    print(f"PRIORITY FOLDERS: it/ and legal/")
    print(f"{'='*60}")
    for priority in ["it", "legal"]:
        folder_path = RAW_PATH / priority
        if not folder_path.exists():
            print(f"\n  [{priority}/] NOT FOUND at {folder_path}")
            continue
        pfiles = list(folder_path.rglob("*.md"))
        print(f"\n  {priority}/ — {len(pfiles)} files")
        sub_counts = defaultdict(int)
        for pf in pfiles:
            parts = pf.relative_to(folder_path).parts
            sub = parts[0] if len(parts) > 1 else "__root__"
            sub_counts[sub] += 1
        for sub, cnt in sorted(sub_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"    {sub:<35} {cnt:>4} files")

    print(f"\n{'='*60}")
    print("EDA complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()