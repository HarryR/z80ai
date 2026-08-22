#!/usr/bin/env python3
"""
Synthetic training data generator using Ollama.

Pipeline:
1. Generate seed conversation intents
2. Generate variations of each intent (txt-speak, casual, typos, etc.)
3. Generate constrained responses using GBNF grammar

Usage:
    ./genpairs.py --seeds 20 --variations 10 > training_data.txt
    ./genpairs.py --vocab-file vocab.txt  # custom vocab
"""

import argparse
import json
import random
import sys
import urllib.request

# Base vocabulary - model can combine 1-2 of these
_BASE_VOCAB = [
    # Questions/prompts
    "WHY", "HOW", "WHO", "WHAT", "WHERE", "WHEN", "WHICH",
    # Affirmative
    "YES", "YEA", "YEP", "YEAH", "OK", "OKAY", "SURE", "TRUE", "COOL", "NICE",
    # Negative
    "NO", "NAH", "NOPE", "NAW",
    # Uncertain
    "IDK", "HMM", "MEH", "MAYBE", "DUNNO",
    # Reactions
    "LOL", "HAHA", "WOW", "OMG", "AWW", "OOF", "YAY", "HUH", "EH",
    # Social
    "HI", "HEY", "BYE", "THX", "SRY", "PLZ", "SUP",
    # Connectors (for 2-word combos)
    "AND", "BUT", "SO", "THO", "RLY",
]

def expand_vocab(base: list[str]) -> list[str]:
    """Expand vocab with ? and ! suffixes."""
    expanded = []
    for word in base:
        expanded.append(word)
        expanded.append(word + "?")
        expanded.append(word + "!")
    return expanded

DEFAULT_VOCAB = expand_vocab(_BASE_VOCAB)

# Seed intents - what kinds of messages to generate variations for
SEED_INTENTS = [
    "sex chat",
    "text sex",
    "sexting",
    "inappropriately flirty",
    "dirty jokes",
    "flirting",
    "being sexy",
    "greeting someone casually",
    "saying goodbye",
    "agreeing with something",
    "disagreeing politely",
    "expressing confusion",
    "asking how someone is",
    "saying thank you",
    "apologizing",
    "asking for help",
    "about depression",
    "expressing happiness",
    "expressing sadness",
    "expressing frustration",
    "inappropriate subjects",
    "telling a secret",
    "sharing good news",
    "sharing bad news",
    "giving a compliment",
    "giving criticism",
    "asking a question",
    "making small talk",
    "trying to be funny",
    "expressing uncertainty",
    "confirming something",
    "denying something",
    "trying to be annoying",
    "expressing surprise",
    "confessing a crush",
    "asking for a favor",
    "making plans",
    "canceling plans",
    "asking for forgiveness",
    "expressing excitement",
    "expressing disappointment",
    "telling a joke",
    "making a pun",
    "being sarcastic",
    "expressing boredom",
    "checking if someone is there",
    "asking someone to repeat",
    "being creepy",
    "teasing someone playfully",
    "giving a warning",
    "expressing excitement",
    "changing the subject",
    "expressing agreement strongly",
    "being skeptical",
    "being angry",
    "asking for an opinion",
    "giving encouragement",
    "expressing tiredness",
    "reacting to good news",
    "reacting to bad news",
    "flirting playfully",
    "trying to be sexy",
    "being silly or random",
    "expressing impatience",
    "being supportive",
    "expressing love or affection",
    "making a joke",
    "telling someone to relax",
    "expressing excitement",
    "expressing disappointment",
    "asking for clarification",
    "giving a compliment",
    "making a suggestion",
    "expressing gratitude",
    "expressing sympathy",
    "asking for advice",
    "giving advice",
    "expressing curiosity",
    "telling someone to wait",
    "expressing relief",
    "expressing pride",
    "expressing jealousy",
    "asking why",
    "asking what something means",
    "saying you dont know",
    "saying you understand",
    # More variety
    "typing gibberish",
    "keyboard smashing",
    "one word reply",
    "asking about weather",
    "complaining about something",
    "bragging",
    "being passive aggressive",
    "making excuses",
    "being dramatic",
    "gossiping",
    "being nosy",
    "changing their mind",
    "being indecisive",
    "rambling incoherently",
    # Edgier stuff
    "trolling",
    "being rude",
    "insulting someone",
    "crude humor",
    "oversharing personal info",
    "being weird on purpose",
    "random nonsense",
    "testing if bot is real",
    "trying to confuse the bot",
    "asking inappropriate questions",
    "being demanding",
    "threatening to leave",
    "acting drunk",
    "pretending to be someone else",
    "roleplaying",
    "speaking in third person",
    "using too many emojis described in text",
    "being clingy",
    "playing hard to get",
    "negging",
    "love bombing",
    "guilt tripping",
    "being manipulative",
    "asking for nudes",
    "sending unsolicited compliments",
    "being thirsty",
    "sliding into dms",
    "asking asl",
    "old school chat room vibes",
    "90s internet slang",
    "zoomer slang",
    "boomer texting style",
    "asking to continue",
    "wrapping up conversation",
    # Internet/tech culture
    "copypasta style message",
    "spam message",
    "asking for tech support",
    "complaining about lag",
    "referencing memes",
    "l33t speak",
    "uwu speak",
    "ALL CAPS YELLING",
    # Emotional extremes
    "having existential crisis",
    "being nihilistic",
    "toxic positivity",
    "having a breakdown",
    "venting frustration",
    # Social dynamics
    "humble bragging",
    "one-upping someone",
    "simping",
    "gatekeeping",
    "fomo",
    "fishing for compliments",
    "attention seeking",
    "playing victim",
    # Communication quirks
    "being cryptic",
    "speaking in riddles",
    "overexplaining everything",
    "no punctuation stream of consciousness",
    "baby talk",
    "voice to text errors",
    "autocorrect fails",
    # Wrong context
    "texting wrong person",
    "continuing previous conversation",
    "butt dial text",
    "quoting song lyrics randomly",
    "inside joke that makes no sense",
    # Scammy/sales
    "asking for money",
    "promoting something",
    "chain message style",
    "mlm pitch",
]

random.shuffle(SEED_INTENTS)


def ollama_generate(model: str, prompt: str, max_tokens: int = 50, retries: int = 3) -> dict:
    """Call Ollama API with JSON format output."""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.8,
        }
    }

    last_error = None
    for attempt in range(retries):
        try:
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                response_text = result.get("response", "").strip()
                if not response_text:
                    raise ValueError(f"Empty response from {model}")
                return json.loads(response_text)
        except json.JSONDecodeError as e:
            last_error = e
            print(f"# Retry {attempt+1}/{retries}: JSON parse error", file=sys.stderr)
            continue

    raise last_error or ValueError("Failed to get valid JSON")


def generate_seeds(count: int) -> list[str]:
    """Generate seed conversation intents. count=0 means all."""
    if count <= 0:
        return SEED_INTENTS.copy()
    return random.sample(SEED_INTENTS, min(count, len(SEED_INTENTS)))


def generate_variations(model: str, intent: str, count: int) -> list[str]:
    """Generate variations of a conversational intent."""

    prompt = f"""Write {count} example text messages that demonstrate: "{intent}"

NOT messages about the topic - messages that ARE examples of it.
Example: "being demanding" -> "do it NOW", "i need this asap", "hurry up!!"
Example: "greeting" -> "hey", "yo wats up", "hiii"
Example: "flirting" -> "ur cute", "hey cutie ;)", "u single?"

Use casual texting style. NO EMOJIS. ASCII only. Short messages.

Return JSON: {{"messages": ["msg1", "msg2", ...]}}"""

    data = ollama_generate(model, prompt, max_tokens=400)

    variations = []
    items = data.get("messages", data.get("variations", []))
    if isinstance(items, list):
        for item in items:
            if isinstance(item, str):
                # Strip non-ASCII characters
                msg = ''.join(c for c in item if ord(c) < 128).strip()
                if msg and 1 <= len(msg) <= 80:
                    variations.append(msg)

    return variations[:count]


def validate_pair(model: str, message: str, response: str) -> bool:
    """Check if a response is not completely wrong."""

    prompt = f"""Is this chatbot reply remotely plausible? Be lenient.

Only reject if COMPLETELY wrong (e.g., "BYE" as greeting, "YAY" to tragedy).
Accept quirky, unexpected, or unusual responses - they add personality.

Message: "{message}"
Reply: "{response}"

Return JSON: {{"ok": true}} or {{"ok": false}}"""

    data = ollama_generate(model, prompt, max_tokens=20)
    return data.get("ok", True) is True  # Default to accepting


def generate_natural_response(model: str, message: str) -> str:
    """Stage 1: Generate a natural, unconstrained response."""

    prompt = f"""You are a cryptic, wise chatbot - part zen master, part mischievous friend.
You speak in short, ambiguous phrases. Sometimes profound, sometimes playful, always brief.

Someone says: "{message}"

Reply in 1-5 words. Be enigmatic."""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "num_predict": 60,
            "temperature": 0.8,
        },
    }

    try:
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            response_text = result.get("response", "").strip()
            parsed = json.loads(response_text)
            # Try common keys
            for key in ["response", "reply", "r", "message", "text"]:
                if parsed.get(key):
                    return str(parsed[key]).strip()
            # Return first string value found
            for v in parsed.values():
                if isinstance(v, str) and v.strip():
                    return v.strip()
    except Exception as e:
        raw = response_text[:200] if 'response_text' in locals() else 'N/A'
        print(f"# Natural response error: {e} | RAW: {raw}", file=sys.stderr)
    return ""


def generate_candidates(model: str, natural: str, vocab: list[str], n: int = 5) -> list[str]:
    """Generate N candidate compressions."""
    candidates = []

    # Filter out short words - model fixates on them
    # Strip punctuation to get base length
    filtered_vocab = [w for w in vocab if len(w.rstrip('?!')) > 1]

    for _ in range(n):
        shuffled = filtered_vocab.copy()
        random.shuffle(shuffled)

        schema = {
            "type": "object",
            "properties": {
                "r": {
                    "type": "array",
                    "items": {"enum": shuffled},
                    "minItems": 1,
                    "maxItems": 2
                }
            },
            "required": ["r"]
        }

        prompt = f"""Summarize this reply in 1-2 short reaction words: "{natural}" """

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": schema,
            "options": {
                "num_predict": 20,
                "temperature": 1.0,  # High temp for variety
            },
        }

        try:
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                response_text = result.get("response", "").strip()
                parsed = json.loads(response_text)
                r = parsed.get("r", [])
                if isinstance(r, list):
                    candidate = " ".join(str(w).strip().upper() for w in r)
                    if candidate and candidate not in candidates:
                        candidates.append(candidate)
        except:
            pass

    return candidates


def pick_best_candidate(model: str, natural: str, candidates: list[str]) -> str:
    """Ask model to pick the best matching candidate."""
    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0]

    options = "\n".join(f"{i+1}. {c}" for i, c in enumerate(candidates))

    prompt = f"""Which reply best matches the meaning of: "{natural}"

{options}

Return the number of the best match."""

    schema = {
        "type": "object",
        "properties": {
            "choice": {"type": "integer", "minimum": 1, "maximum": len(candidates)}
        },
        "required": ["choice"]
    }

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": schema,
        "options": {
            "num_predict": 10,
            "temperature": 0.3,
        },
    }

    try:
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            response_text = result.get("response", "").strip()
            parsed = json.loads(response_text)
            choice = int(parsed.get("choice", 1))
            if 1 <= choice <= len(candidates):
                return candidates[choice - 1]
    except:
        pass

    return candidates[0]  # Fallback to first


def compress_response(model: str, natural: str, vocab: list[str]) -> str:
    """Stage 2: Generate candidates and pick best."""
    candidates = generate_candidates(model, natural, vocab, n=5)
    if not candidates:
        return ""
    return pick_best_candidate(model, natural, candidates)


def generate_response(model: str, message: str, vocab: list[str], debug: bool = False) -> str:
    """Direct response: pick 1-2 vocab words based on input."""

    # Filter out single-char words
    filtered_vocab = [w for w in vocab if len(w.rstrip('?!')) > 1]

    shuffled = filtered_vocab.copy()
    random.shuffle(shuffled)

    schema = {
        "type": "object",
        "properties": {
            "r": {
                "type": "array",
                "items": {"enum": shuffled},
                "minItems": 1,
                "maxItems": 1
            }
        },
        "required": ["r"]
    }

    prompt = f"""You're a chatbot. Someone says: "{message}"

Pick 1-2 short words as your reply."""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": schema,
        "options": {
            "num_predict": 20,
            "temperature": 0.8,
        },
    }

    try:
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            response_text = result.get("response", "").strip()
            parsed = json.loads(response_text)
            r = parsed.get("r", [])
            if isinstance(r, list):
                response = " ".join(str(w).strip().upper() for w in r)
                if debug:
                    print(f"#   {message} → {response}", file=sys.stderr)
                return response
    except Exception as e:
        if debug:
            print(f"# Response error: {e}", file=sys.stderr)
    return ""


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic training pairs')
    parser.add_argument('--seeds', '-s', type=int, default=0,
                        help='Number of seed intents (0 = all)')
    parser.add_argument('--variations', '-v', type=int, default=5,
                        help='Variations per seed')
    parser.add_argument('--vocab-file', '-V', type=str,
                        help='File with allowed vocab (one per line)')
    parser.add_argument('--seed-model', default='qwen3:1.7b',
                        help='Model for generating variations')
    parser.add_argument('--resp-model', default='gemma2:9b', help='Model for responses')
    parser.add_argument('--judge-model', default='qwen3:1.7b', help='Model for validating pairs')
    parser.add_argument('--no-validate', action='store_true', help='Skip validation step')
    parser.add_argument('--debug', action='store_true', help='Show intermediate natural responses')
    parser.add_argument('--output', '-o', type=str, help='Append output to file (default: stdout)')
    args = parser.parse_args()

    # Load vocabulary
    vocab = DEFAULT_VOCAB
    if args.vocab_file:
        with open(args.vocab_file) as f:
            vocab = [line.strip() for line in f if line.strip()]

    # Open output file (append mode) or use stdout
    # Held open for the whole run and closed at the end; sys.stdout must not
    # be closed, so this cannot simply become a with-block.
    outfile = open(args.output, 'a') if args.output else sys.stdout  # noqa: SIM115

    print(f"# Generating {args.seeds} seeds with {args.variations} variations each",
          file=sys.stderr)
    print(f"# Variation model: {args.seed_model}, Response model: {args.resp_model}",
          file=sys.stderr)
    print(f"# Vocab size: {len(vocab)} words", file=sys.stderr)
    if args.output:
        print(f"# Output file: {args.output}", file=sys.stderr)

    # Generate seeds
    seeds = generate_seeds(args.seeds)
    print(f"# Seeds: {seeds}", file=sys.stderr)

    total = 0
    try:
        for i, seed in enumerate(seeds):
            print(f"# [{i+1}/{len(seeds)}] {seed}", file=sys.stderr)

            # Keep generating until we have enough valid pairs
            valid_count = 0
            attempts = 0
            max_attempts = args.variations * 3  # Give up after 3x tries

            while valid_count < args.variations and attempts < max_attempts:
                # Generate a batch of variations
                batch_size = min(args.variations - valid_count + 2, 5)  # Small batches
                variations = generate_variations(args.seed_model, seed, batch_size)
                attempts += 1

                for var in variations:
                    if valid_count >= args.variations:
                        break

                    # Generate constrained response
                    response = generate_response(args.resp_model, var, vocab, debug=args.debug)
                    if not response:
                        continue

                    # Enforce 8 char limit
                    if len(response) > 12:
                        print(f"#   TOO LONG: {var}|{response}", file=sys.stderr)
                        continue

                    # Validate the pair
                    if not args.no_validate and not validate_pair(
                        args.judge_model, var, response
                    ):
                        print(f"#   REJECTED: {var}|{response}", file=sys.stderr)
                        continue

                    print(f"{var}|{response}", file=outfile)
                    outfile.flush()
                    valid_count += 1
                    total += 1

    except KeyboardInterrupt:
        print("\n# Interrupted!", file=sys.stderr)

    if args.output:
        outfile.close()
    print(f"# Generated {total} training pairs", file=sys.stderr)


if __name__ == '__main__':
    main()
