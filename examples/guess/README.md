# Guess What - 20 Questions for Z80

A 20 Questions game where a tiny neural network learns to answer YES/NO/MAYBE about a secret topic.

## Quick Start

```bash
# Generate training data (requires ANTHROPIC_API_KEY for --claude)
./gendata.py -t 'elephant' -n 500 --claude > elephant.txt

# Add distractors and nonsense responses
./gendata.py -t 'elephant' -d 30 --nonsense --claude >> elephant.txt

# Generate YES-only questions (helps balance)
./gendata.py -t 'elephant' --yes-only --claude >> elephant.txt

# Generate WIN guesses (ways to guess the answer)
./gendata.py -t 'elephant' --win-only --claude >> elephant.txt

# Balance, train, and build
cat *.txt | ./balance.py -t 5000 -o --stats | ../../feedme.py
../../buildz80com.py -o GUESS.COM

# Play!
../../cpm GUESS.COM
```

## Training Data Format

```
is it big|YES
can it fly|NO
does it have a trunk|YES
is it grey|MAYBE
elephant|WIN
hello|IDK
```

## Tools

### gendata.py

Generates Q&A pairs using an LLM.

```bash
# Local Ollama
./gendata.py --topic elephant -n 100

# Claude API (faster, better quality)
./gendata.py --topic elephant --claude -n 500

# MiniMax API (set MINIMAX_API_KEY env var)
./gendata.py --topic elephant --minimax -n 500

# With paraphrases (5 variations per question)
./gendata.py --topic elephant --claude -p 5 -n 500
```

Options:
- `-t, --topic` - The secret topic
- `-n, --num` - Number of questions to generate
- `-d, --distractors` - Add wrong-topic questions (answered NO)
- `--nonsense` - Add off-topic phrases (answered IDK)
- `--claude` - Use Claude API instead of Ollama
- `--minimax` - Use MiniMax API instead of Ollama
- `--yes-only` - Only generate YES-answer questions
- `--win-only` - Generate winning guesses
- `-p, --paraphrase` - Generate N paraphrases per question

### balance.py

Balances training data by over/undersampling classes.

```bash
# Balance to 5000 per class, oversample minorities
cat *.txt | ./balance.py -t 5000 -o --stats > balanced.txt

# Check distribution without balancing
cat *.txt | ./balance.py --stats > /dev/null
```

### analyze.py

Analyzes answer distribution in training data.

```bash
./analyze.py training.txt
```

## Limitations

The Z80 model uses trigram hashing with 128 buckets:
- Short distinctive phrases work best
- Long similar phrases may collide ("bigger than X" vs "smaller than X")
- Out-of-distribution questions get unpredictable answers
