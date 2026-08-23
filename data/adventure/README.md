# Adventure command parser

Map whatever the player typed onto one of 17 commands.

```
LETS SEIZE MY TORCH THANKS|TAKE
NOW GO TO THE LADDER PLEASE|UP
JUST HEAD ABOVE PLEASE|UP
COULD YOU MOVE WEST OK|GO W
LETS KERFUFFLE NOW|IDK
```

```bash
python data/lint.py data/adventure/commands.txt --strict
cat data/adventure/commands.txt | ./feedme.py --epochs 150 --chat
```

Reaches **99.8% on held-out queries** against 100.0% on its training data — so
it is generalizing, not memorizing. (Training is not seeded, so runs land around
98.5–99.8%.) 11,900 pairs, 17 labels balanced at 700 each, no duplicates, no
contradictions, 20-character charset.

## Why this task

It fits what the architecture is good at, which open-ended chat does not:

- **17 labels.** The character decoder is a label decoder; distinct responses
  are the real complexity parameter, and 17 is comfortably inside what 2-bit
  weights handle.
- **Keyword-driven.** Trigram hashing over a short query is a good match for
  "which verb is this", and a poor one for anything needing careful word-order
  reasoning.
- **Labels share letters.** `GO N`/`GO S`/`GO E`/`GO W`/`TAKE`/`DROP` come to 20
  characters — 2,560 output weights, against `tinychat`'s 5,120 for 40.
- **Phrasings expand cheaply.** Volume goes into ways of saying the same
  command, not into more commands.
- **`IDK` is a real class.** Out-of-distribution input produces *something*
  regardless; training a catch-all decides what. `guess` cannot do this — its
  11-character charset cannot even spell `IDK`.

## Generating

```bash
python data/adventure/generate.py > data/adventure/commands.txt
```

Deterministic: no network, no LLM, seeded sampling, sorted iteration
everywhere. `test_adventure_dataset_matches_its_generator` asserts the checked-in
file is exactly what the generator produces, so the two cannot drift.

Queries come from `prefix + verb + article + argument + suffix` over per-label
verb and object lists. Articles apply only to nouns — `take the lamp` is
something a player types, `go a northward` is not, and generating it would spend
capacity on patterns that never occur at inference.

A query that two labels both claim is dropped from both, rather than left in to
cap accuracy. The generator reports how many it dropped on stderr.

`--per-label` sets the cap (default 700), so balance holds by construction
rather than by a rebalancing pass afterwards. **Bare phrasings are kept
unconditionally** and the quota is filled with decorated ones: sampling the
whole pool uniformly drops canonical commands, and at 700 out of ~50,000
candidates `take the lamp` itself did not survive — the trained model answered
`WAIT`.

One thing the format cannot express: `feedme` drops queries shorter than two
characters, so single-letter commands (`n`, `i`, `z`) can never be trained.
`go n` and `head n` work; bare `n` does not.

## Labels

`GO N` `GO S` `GO E` `GO W` `UP` `DOWN` `LOOK` `READ` `TAKE` `DROP` `OPEN`
`CLOSE` `INV` `EAT` `WAIT` `HELP` `IDK`
