"""Training-data handling: parsing, the validation split, and the data checks.

These are deliberately torch-free so they run in CI, where training is not
installed.  The split in particular is worth pinning: measuring accuracy on the
training set is what hid an 18-point generalization gap in the tinychat data,
and a split that leaks queries would hide it just as well.
"""

from __future__ import annotations

import pytest

import libdata

# --- parsing -----------------------------------------------------------------


@pytest.mark.parametrize(
    "line,expected",
    [
        ("hello|HI", ("HELLO", "HI")),
        ("  spaced  |  out  ", ("SPACED", "OUT")),
        ("lower|case", ("LOWER", "CASE")),
        ("a|b", None),                      # query shorter than 2 characters
        ("query|", None),                   # empty response
        ("no pipe here", None),
        ("", None),
        ("   ", None),
        ("# a comment", None),
        ("take the lamp|TAKE|EXTRA", ("TAKE THE LAMP", "TAKE|EXTRA")),
    ],
)
def test_parse_pair(line, expected):
    assert libdata.parse_pair(line) == expected


def test_comments_are_skipped_even_when_they_contain_a_pipe():
    """The old parser only ignored comments that happened to have no pipe."""
    assert libdata.parse_pair("# vocabulary: YES|NO|MAYBE") is None


def test_long_queries_are_truncated_not_dropped():
    query = "WORD " * 20
    parsed = libdata.parse_pair(f"{query}|OK")
    assert parsed is not None
    assert len(parsed[0]) <= libdata.MAX_QUERY_LEN


def test_load_pairs_honours_a_limit():
    lines = [f"a{i}|X" for i in range(10)]
    assert len(libdata.load_pairs(lines)) == 10
    assert len(libdata.load_pairs(lines, 3)) == 3


# --- charset -----------------------------------------------------------------


def test_charset_comes_from_responses_only():
    """Queries are hashed into buckets, never spelled out, so they add nothing."""
    pairs = [("ZZZ QUERY", "OK"), ("ANOTHER", "NO")]
    assert libdata.build_charset(pairs) == "KNO"


def test_a_single_rare_character_still_enters_the_charset():
    """One line with a slash costs an output neuron for the whole model."""
    pairs = [("QUERY ONE", "OK")] * 100 + [("QUERY TWO", "K W/E")]
    assert "/" in libdata.build_charset(pairs)


# --- the split ---------------------------------------------------------------


def _pairs(n: int) -> list[tuple[str, str]]:
    return [(f"QUERY {i}", "YES" if i % 2 else "NO") for i in range(n)]


def test_split_holds_out_roughly_the_requested_fraction():
    train, val = libdata.split_pairs(_pairs(100), val_frac=0.2)
    assert len(val) == 20
    assert len(train) == 80


def test_split_never_leaks_a_query_across_the_boundary():
    """The whole point: a query in both halves makes validation meaningless."""
    pairs = _pairs(50) * 3  # every query appears three times
    train, val = libdata.split_pairs(pairs, val_frac=0.2)
    assert {q for q, _ in train}.isdisjoint({q for q, _ in val})
    assert len(train) + len(val) == len(pairs)


def test_split_is_deterministic_for_a_given_seed():
    pairs = _pairs(200)
    assert libdata.split_pairs(pairs, 0.1, 7) == libdata.split_pairs(pairs, 0.1, 7)


def test_split_seed_actually_changes_the_split():
    pairs = _pairs(200)
    assert libdata.split_pairs(pairs, 0.1, 0) != libdata.split_pairs(pairs, 0.1, 1)


def test_split_does_not_depend_on_input_order():
    """Otherwise a shuffled data file would silently change the held-out set."""
    pairs = _pairs(200)
    _, val_a = libdata.split_pairs(pairs, 0.1, 3)
    _, val_b = libdata.split_pairs(list(reversed(pairs)), 0.1, 3)
    assert sorted(val_a) == sorted(val_b)


@pytest.mark.parametrize("off", [0, -0.5])
def test_a_non_positive_fraction_disables_the_split(off):
    pairs = _pairs(10)
    train, val = libdata.split_pairs(pairs, val_frac=off)
    assert val == []
    assert train == pairs


@pytest.mark.parametrize("bad", [1.0, 1.5])
def test_fractions_of_one_or_more_are_rejected(bad):
    """Holding out everything would leave nothing to train on."""
    with pytest.raises(ValueError, match="val_frac"):
        libdata.split_pairs(_pairs(10), val_frac=bad)


# --- contradictions ----------------------------------------------------------


def test_accuracy_ceiling_is_one_when_labels_are_consistent():
    assert libdata.accuracy_ceiling([("A B", "YES"), ("C D", "NO")]) == 1.0


def test_accuracy_ceiling_reflects_a_contradiction():
    """Two of three pairs agree, so the best any model can do is 2/3."""
    pairs = [("A B", "YES"), ("A B", "YES"), ("A B", "NO")]
    assert libdata.accuracy_ceiling(pairs) == pytest.approx(2 / 3)


def test_accuracy_ceiling_of_an_empty_set_is_one():
    assert libdata.accuracy_ceiling([]) == 1.0


def test_contradictions_reports_the_conflicting_labels():
    pairs = [("A B", "YES"), ("A B", "NO"), ("C D", "OK")]
    assert libdata.contradictions(pairs) == {"A B": {"YES", "NO"}}


