"""
Training-data handling: parsing, splitting, and the checks worth running first.

Deliberately free of torch, so the test suite and `data/lint.py` can use it
without pulling in a training dependency - the same reason `libinfer.py` is
NumPy-only.  `feedme.py` imports from here rather than keeping its own copies.
"""

from __future__ import annotations

import gzip
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

Pair = tuple[str, str]

#: Longest query and response feedme will keep. Both are truncated, not
#: dropped, so data past these limits silently changes meaning.
MAX_QUERY_LEN = 60
MAX_RESPONSE_LEN = 50


def parse_pair(line: str) -> Pair | None:
    """Parse ``query|response`` into an uppercased pair, or None if unusable.

    Blank lines and ``#`` comments are skipped.  Comments only survived before
    because none of them happened to contain a pipe; a comment that did would
    have become a training pair.
    """
    line = line.strip()
    if not line or line.startswith('#') or '|' not in line:
        return None

    query, response = line.split('|', 1)
    query = query.strip().upper()
    response = response.strip().upper()

    if len(query) < 2 or len(response) < 1:
        return None

    if len(query) > MAX_QUERY_LEN:
        head = query[:MAX_QUERY_LEN]
        query = head.rsplit(' ', 1)[0] if ' ' in query[40:MAX_QUERY_LEN] else head
    if len(response) > MAX_RESPONSE_LEN:
        head = response[:MAX_RESPONSE_LEN]
        response = head.rsplit(' ', 1)[0] if ' ' in response[30:MAX_RESPONSE_LEN] else head
    return query, response


def load_pairs(lines: Iterable[str], limit: int = 0) -> list[Pair]:
    """Parse every line, stopping after ``limit`` pairs (0 = no limit)."""
    pairs: list[Pair] = []
    for line in lines:
        pair = parse_pair(line)
        if pair:
            pairs.append(pair)
            if limit and len(pairs) >= limit:
                break
    return pairs


def read_files(paths: Sequence[str]) -> list[Pair]:
    """Read pairs from files (``.gz`` transparently) or stdin when empty."""
    if not paths:
        return load_pairs(sys.stdin)
    pairs: list[Pair] = []
    for path in paths:
        opener = gzip.open if path.endswith('.gz') else open
        with opener(path, 'rt') as fh:
            pairs.extend(load_pairs(fh))
    return pairs


def build_charset(pairs: Sequence[Pair]) -> str:
    """Every character appearing in a response, which is what feedme encodes.

    Queries never contribute: they are hashed into buckets, not spelled out.
    """
    return ''.join(sorted({c for _, r in pairs for c in r}))


def split_pairs(pairs: Sequence[Pair], val_frac: float = 0.1,
                seed: int = 0) -> tuple[list[Pair], list[Pair]]:
    """Hold out a fraction of the *unique queries* for validation.

    Splitting by line instead would put the same query on both sides - 35% of
    the shipped guess data is exact duplicate pairs - and the validation score
    would be the training score under another name.

    Templated paraphrases of a held-out query ("can it cut you" / "can this cut
    you") do still land in the training half, so this is an optimistic estimate
    of generalization.  It is a much better one than measuring on the training
    set, which is what happened before.
    """
    if val_frac <= 0:
        return list(pairs), []
    if not 0 < val_frac < 1:
        raise ValueError(f"val_frac must be in [0, 1), got {val_frac}")

    queries = sorted({q for q, _ in pairs})  # sorted, then seeded: reproducible
    random.Random(seed).shuffle(queries)
    held_out = set(queries[:int(len(queries) * val_frac)])

    train = [p for p in pairs if p[0] not in held_out]
    val = [p for p in pairs if p[0] in held_out]
    return train, val


def accuracy_ceiling(pairs: Sequence[Pair]) -> float:
    """Best pair-level accuracy any model could reach, given contradictions.

    A query that appears with two different responses cannot be got right both
    times.  Worth printing beside the accuracy: otherwise a model that has
    learned everything learnable looks like it is underperforming, and the fix
    looks like "train longer" when it is "fix the data".
    """
    if not pairs:
        return 1.0
    by_query: dict[str, Counter] = defaultdict(Counter)
    for query, response in pairs:
        by_query[query][response] += 1
    return sum(c.most_common(1)[0][1] for c in by_query.values()) / len(pairs)


def contradictions(pairs: Sequence[Pair]) -> dict[str, set[str]]:
    """Queries that appear with more than one response, and what they map to."""
    by_query: dict[str, set[str]] = defaultdict(set)
    for query, response in pairs:
        by_query[query].add(response)
    return {q: rs for q, rs in by_query.items() if len(rs) > 1}
