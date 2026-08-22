#!/usr/bin/env python3
"""Analyze training data distribution."""

import sys
from collections import Counter


def analyze(filepath):
    answers = Counter()
    total = 0

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '|' not in line:
                continue

            _query, a = line.rsplit('|', 1)
            answers[a.upper()] += 1
            total += 1

    print(f"Total pairs: {total:,}")
    print("\nAnswer distribution:")
    print("-" * 40)

    for answer, count in answers.most_common():
        pct = 100 * count / total
        bar = '█' * int(pct / 2)
        print(f"  {answer:6} {count:6,} ({pct:5.1f}%) {bar}")

    print("-" * 40)

    # Imbalance warnings
    if answers:
        max_pct = 100 * max(answers.values()) / total
        min_pct = 100 * min(answers.values()) / total

        if max_pct > 60:
            top = answers.most_common(1)[0][0]
            print(f"\n⚠️  Heavy imbalance: {top} is {max_pct:.1f}% of data")
            print(f"   Model may default to {top} when uncertain")

        if min_pct < 5:
            rare = answers.most_common()[-1][0]
            print(f"\n⚠️  Rare class: {rare} is only {min_pct:.1f}% of data")
            print(f"   Model may struggle to learn {rare}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <training.txt>")
        sys.exit(1)

    analyze(sys.argv[1])
