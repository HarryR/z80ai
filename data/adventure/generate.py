#!/usr/bin/env python3
"""
Generate the adventure command dataset.

Deterministic and dependency-free on purpose.  The other generators in this
repo need an Anthropic key (`examples/guess/gendata.py`) or three local Ollama
models (`examples/tinychat/genpairs.py`), neither is seeded, and neither one's
output is checked in - so nobody can reproduce the data the shipped models were
trained on.  This one is a pure function of its source: same seed, same bytes.

    python data/adventure/generate.py > data/adventure/commands.txt
    python data/lint.py data/adventure/commands.txt --strict

The task is intent classification for a text adventure parser: map whatever the
player typed onto one of a small closed set of commands.  That shape suits this
architecture better than open-ended chat does.  Trigram hashing over a short
query rewards distinctive keywords, and the labels share most of their letters,
so the charset - and with it the output layer - stays small.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict

# Politeness and filler the player might type around the real command. The
# empty string matters most: bare "north" has to work.
PREFIXES = [
    "", "", "", "", "please ", "i want to ", "id like to ", "lets ", "can i ",
    "could you ", "now ", "just ", "ok ", "try to ", "i will ", "im going to ",
]
SUFFIXES = ["", "", "", "", "", " please", " now", " ok", " thanks", " then"]

# Objects the manipulation verbs act on. Kept concrete and short: long noun
# phrases dilute the keyword signal across the 128 trigram buckets.
OBJECTS = [
    "lamp", "sword", "key", "coin", "rope", "book", "torch", "gem", "bottle",
    "map", "knife", "shield", "bag", "ring", "candle", "box", "chest", "door",
    "letter", "stone", "axe", "cloak", "flask", "scroll", "helmet",
]
ARTICLES = ["", "the ", "a ", "my ", "that ", "this "]

# label -> list of (verbs, arguments, articles-apply). Articles only make sense
# in front of a noun: "take the lamp" is something a player types, "go a
# northward" is not, and generating it would spend capacity on patterns that
# never appear at inference.
COMMANDS: dict[str, list[tuple[list[str], list[str], bool]]] = {
    "GO N": [(["go", "walk", "head", "move", "run", "travel", "", "go to",
               "head to", "walk to", "move to"],
              ["north", "n", "northward", "the north", "up north"], False)],
    "GO S": [(["go", "walk", "head", "move", "run", "travel", "", "go to",
               "head to", "walk to", "move to"],
              ["south", "s", "southward", "the south", "down south"], False)],
    "GO E": [(["go", "walk", "head", "move", "run", "travel", "", "go to",
               "head to", "walk to", "move to"],
              ["east", "e", "eastward", "the east"], False)],
    "GO W": [(["go", "walk", "head", "move", "run", "travel", "", "go to",
               "head to", "walk to", "move to"],
              ["west", "w", "westward", "the west"], False)],
    "UP": [(["go", "climb", "walk", "head", "move", "", "climb the",
             "go to the", "head"],
            ["up", "upward", "upstairs", "u", "ladder", "stairs", "above"],
            False)],
    "DOWN": [(["go", "climb", "walk", "head", "move", "", "climb the",
               "go to the", "head"],
              ["down", "downward", "downstairs", "d", "below"], False)],
    "LOOK": [(["look", "look at", "examine", "inspect", "study", "check",
               "look around", "survey", "observe", "search", "peer at",
               "stare at", "glance at"],
              ["", "room", "surroundings", "area", "place", "here", "around"],
              False)],
    "READ": [(["read", "study", "peruse", "scan", "look over", "decipher"],
              ["book", "letter", "scroll", "sign", "note", "map", "inscription",
               "writing", "label", "page"], True)],
    "TAKE": [(["take", "get", "grab", "pick up", "acquire", "collect", "seize",
               "snatch", "lift", "carry", "pocket", "steal"], OBJECTS, True)],
    "DROP": [(["drop", "put down", "discard", "release", "leave", "throw away",
               "abandon", "let go of", "set down", "ditch"], OBJECTS, True)],
    "OPEN": [(["open", "unlock", "unseal", "pry open", "unlatch", "force open",
               "unfasten"],
              ["door", "box", "chest", "bag", "gate", "window", "drawer",
               "cabinet", "lid", "hatch"], True)],
    "CLOSE": [(["close", "shut", "seal", "lock", "latch", "slam", "fasten"],
               ["door", "box", "chest", "bag", "gate", "window", "drawer",
                "cabinet", "lid", "hatch"], True)],
    "INV": [(["inventory", "i", "check inventory", "what am i carrying",
              "what do i have", "show inventory", "list items", "my items",
              "whats in my bag", "what am i holding", "show my stuff",
              "check my pack", "inv"], [""], False)],
    "EAT": [(["eat", "consume", "devour", "taste", "bite", "swallow",
              "munch on", "chew"],
             ["bread", "apple", "food", "meal", "ration", "cheese", "berry",
              "mushroom", "fish", "stew"], True)],
    "WAIT": [(["wait", "do nothing", "rest", "pause", "hold on", "stay put",
               "sit tight", "linger", "stand still", "z", "bide time",
               "hang on", "take a moment"], [""], False)],
    "HELP": [(["help", "what can i do", "commands", "instructions", "how do i play",
               "list commands", "what are my options", "h", "hint", "show help",
               "im stuck", "what now", "guide me"], [""], False)],
}

# Deliberate out-of-distribution catch-all. Without this the model fires
# whatever neurons a strange input happens to excite; TRAINING.md has said to
# train one of these for a while, but no shipped dataset actually does - the
# guess charset cannot even spell IDK.
NONSENSE = [
    "asdf", "qwerty", "blorp", "xyzzy plugh", "flibbertigibbet", "zzzz",
    "hello world", "what is the meaning of life", "sing me a song",
    "tell me a joke", "banana phone", "the quick brown fox", "42",
    "wibble wobble", "colorless green ideas", "purple monkey dishwasher",
    "who are you", "what year is it", "compute pi", "sudo make me a sandwich",
    "knock knock", "beep boop", "lorem ipsum", "how much wood",
    "spam spam spam", "mary had a little lamb", "supercalifragilistic",
    "glorp znak", "fizzbuzz", "to be or not to be", "e equals mc squared",
    "roses are red", "one two three four", "abcdefg", "mumble mumble",
    "the rain in spain", "hocus pocus", "abracadabra", "shazam", "kerfuffle",
]
NONSENSE_LABEL = "IDK"


def phrasings(verbs: list[str], args: list[str], use_articles: bool,
              decorate: bool = True) -> set[str]:
    """Every query this verb/argument family can produce.

    With ``decorate=False`` only the bare forms - no politeness prefix, no
    trailing filler - which are the ones a player is most likely to actually
    type.  They are kept unconditionally; see :func:`build`.
    """
    prefixes = PREFIXES if decorate else [""]
    suffixes = SUFFIXES if decorate else [""]
    out: set[str] = set()
    for verb in verbs:
        for arg in args:
            articles = ARTICLES if (use_articles and arg) else [""]
            for article in articles:
                stem = f"{verb} {article}{arg}".strip()
                if not stem:
                    continue
                for prefix in prefixes:
                    for suffix in suffixes:
                        query = f"{prefix}{stem}{suffix}".strip()
                        # feedme drops anything shorter than 2 characters and
                        # truncates past 60, so do not emit either.
                        if 2 <= len(query) <= 60:
                            out.add(query.upper())
    return out


def build(per_label: int, seed: int) -> list[tuple[str, str]]:
    """Generate a balanced, contradiction-free set of (query, label) pairs."""
    candidates: dict[str, set[str]] = {}
    core: dict[str, set[str]] = {}
    for label, families in COMMANDS.items():
        queries: set[str] = set()
        bare: set[str] = set()
        for verbs, args, use_articles in families:
            queries |= phrasings(verbs, args, use_articles)
            bare |= phrasings(verbs, args, use_articles, decorate=False)
        candidates[label] = queries
        core[label] = bare

    candidates[NONSENSE_LABEL] = phrasings(NONSENSE, [""], False)
    core[NONSENSE_LABEL] = phrasings(NONSENSE, [""], False, decorate=False)

    # A query that two labels both claim is a contradiction the model cannot
    # resolve. Drop it from both rather than let it cap accuracy.
    owners: dict[str, set[str]] = defaultdict(set)
    for label, queries in candidates.items():
        for query in queries:
            owners[query].add(label)
    ambiguous = {q for q, labels in owners.items() if len(labels) > 1}
    if ambiguous:
        print(f"# dropped {len(ambiguous)} ambiguous queries", file=sys.stderr)

    rng = random.Random(seed)
    pairs: list[tuple[str, str]] = []
    for label in sorted(candidates):
        # Keep every bare phrasing, then fill the quota with decorated ones.
        # Sampling the whole pool uniformly instead drops canonical commands:
        # at per_label=700 out of ~50,000 candidates, "take the lamp" itself
        # did not survive, and the trained model got it wrong.
        bare = sorted(core[label] - ambiguous)          # sorted: reproducible
        extra = sorted((candidates[label] - ambiguous).difference(bare))

        if len(bare) > per_label:
            queries = rng.sample(bare, per_label)
        else:
            queries = bare + rng.sample(extra, min(per_label - len(bare), len(extra)))
        pairs.extend((q, label) for q in queries)

    rng.shuffle(pairs)
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--per-label', type=int, default=700,
                        help='Cap on pairs per label (balance is by construction)')
    parser.add_argument('--seed', type=int, default=0, help='Sampling seed')
    args = parser.parse_args()

    pairs = build(args.per_label, args.seed)
    counts = Counter(label for _, label in pairs)

    print("# Text adventure command parser.")
    print("# Generated by data/adventure/generate.py - do not edit by hand.")
    print(f"# {len(pairs)} pairs, {len(counts)} labels, "
          f"per_label={args.per_label}, seed={args.seed}")
    for label, count in sorted(counts.items()):
        print(f"#   {label:<6} {count}")
    for query, label in pairs:
        print(f"{query}|{label}")


if __name__ == '__main__':
    main()
