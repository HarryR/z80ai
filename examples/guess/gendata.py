#!/usr/bin/env python3
"""
Generate 20-questions style training data.

Pick a secret topic, generate lots of yes/no questions about it.
The model learns to answer questions about ONE specific thing.

Usage:
    # Local Ollama (default)
    ./gendata.py --topic elephant -n 100 > training.txt
    ./gendata.py --topic elephant --model gemma3:12b -n 500

    # Claude API (set ANTHROPIC_API_KEY env var)
    ./gendata.py --topic elephant --claude -n 500 > training.txt
    ./gendata.py --topic elephant --claude --model claude-opus-4-20250514

    # MiniMax API (set MINIMAX_API_KEY env var)
    ./gendata.py --topic elephant --minimax -n 500 > training.txt
    ./gendata.py --topic elephant --minimax --model MiniMax-M2.7-highspeed

    # With distractors and nonsense
    ./gendata.py --topic elephant -d 20 --nonsense --claude

    # With paraphrases (5 per question, costs more but better coverage)
    ./gendata.py --topic chair --claude -p 5 -n 1000
"""

import argparse
import json
import os
import random
import re
import sys
import urllib.request

# Optional anthropic import
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

DEFAULT_MODEL = 'gemma2:9b'
DEFAULT_CLAUDE_MODEL = 'claude-sonnet-4-20250514'
DEFAULT_MINIMAX_MODEL = 'MiniMax-M2.7'

# Token tracking for cost estimation
total_input_tokens = 0
total_output_tokens = 0

# Pronoun variations - these all mean the same thing
PRONOUNS = ["it", "you", "this", "that", "the thing"]

# Off-topic / not playing the game -> IDK
NONSENSE = [
    # Greetings
    "hello", "hi", "hey", "hi there", "howdy",
    # Chat
    "how are you", "whats up", "sup", "how do you work",
    "who are you", "what are you", "are you ai",
    # Frustration
    "i give up", "this is dumb", "i dont know", "no idea",
    "tell me", "tell me the answer", "just tell me",
    "what is it", "whats the answer",
    # Random
    "test", "testing", "asdf", "qwerty", "lol", "lmao",
    "ok", "okay", "sure", "whatever", "idk",
    "help", "help me", "hint", "give me a hint",
    # Commands
    "quit", "exit", "stop", "restart", "reset",
]

# Question categories to steer LLM toward different question types
QUESTION_CATEGORIES = [
    # Physical properties
    "about its physical size, shape, or dimensions",
    "about its color, texture, or appearance",
    "about its weight or how heavy it is",
    "about what it's made of or its materials",
    "about whether it floats or sinks",
    "about its smell or taste",
    "about its temperature - hot or cold",
    "about whether it's soft or hard",
    "about whether it's transparent or opaque",
    "about whether it has a specific shape",
    # Location and environment
    "about where it's commonly found or located",
    "about whether you'd find it indoors or outdoors",
    "about what country or region it comes from",
    "about whether it's found in nature or cities",
    "about whether you'd find it in a home",
    "about whether you'd find it in an office or workplace",
    # Function and purpose
    "about its purpose, function, or use",
    "about how it moves, works, or behaves",
    "about whether it needs power, fuel, or energy",
    "about whether it has parts or components",
    "about whether it can be turned on or off",
    "about whether it requires batteries or electricity",
    # Living things
    "about whether it's alive, natural, or man-made",
    "about whether it breathes or needs air",
    "about whether it grows or changes over time",
    "about whether it reproduces or has babies",
    "about whether it has legs, wings, or fins",
    "about whether it's a plant or animal",
    # Human interaction
    "about whether children use or like it",
    "about whether you can hold it in your hand",
    "about whether you can buy it in a store",
    "about whether people wear it or carry it",
    "about whether it's used for entertainment",
    "about whether it's used for work or productivity",
    "about whether you'd find it in a kitchen",
    "about whether you'd find it in a bathroom",
    # Senses and perception
    "about whether it makes sounds or noise",
    "about whether it has a strong smell",
    "about whether you can eat it or drink it",
    "about whether it's pleasant or unpleasant",
    # Value and rarity
    "about its value, cost, or rarity",
    "about its age, history, or origin",
    "about whether it's common or rare",
    "about whether it's expensive or cheap",
    # Safety and danger
    "about whether it's dangerous or safe",
    "about whether it's fragile or durable",
    "about whether it can hurt you",
    "about whether it's sharp or pointy",
    # Comparisons
    "about its relationship to other things",
    "about whether it's bigger than a car",
    "about whether it fits in a pocket",
    "about whether it's heavier than a person",
    "about whether it's older than 100 years",
    # Binary splitters (50/50 questions)
    "about whether it's animal, vegetable, or mineral",
    "about whether it has ever been alive",
    "about whether it's larger than a breadbox",
    "about whether it's found in a typical household",
    "about whether you'd find it in a museum",
    "about whether it involves technology",
    "about whether you interact with it daily",
    "about whether you could see one within a mile",
    "about whether it's something you'd recognize by name",
    "about whether most people have seen one",
    "about whether it existed 100 years ago",
    "about whether it's a single object or a category",
    "about whether you need training to use it",
    "about whether it's associated with a profession",
    "about whether it moves on its own",
    "about whether it's usually found alone or in groups",
    "about whether it has a specific owner or is shared",
    "about whether you'd give it as a gift",
    "about whether it's seasonal or year-round",
    "about whether it's primarily used by one gender",
    # Taxonomic narrowing
    "about whether it's a mammal, bird, fish, or reptile",
    "about whether it's a tool, vehicle, or furniture",
    "about whether it's food, drink, or neither",
    "about whether it's clothing, accessory, or equipment",
    "about whether it's a machine, device, or appliance",
    "about whether it's a building, structure, or vehicle",
    "about what room of a house you would find it in",
    "about whether you can buy it at a grocery store",
    "about whether you can buy it at a hardware store",
    "about whether a doctor or nurse would use it",
    "about whether a teacher would use it",
    "about whether a chef would use it",
    # Actions and interactions
    "about whether you can sit on it",
    "about whether you can stand on it",
    "about whether you can throw it",
    "about whether you can stack them",
    "about whether you can fold it",
    "about whether you can open or close it",
    "about whether it can be locked",
    "about whether you can wash it",
    "about whether you can repair it yourself",
    # Social and cultural
    "about whether it's associated with a holiday",
    "about whether it's mentioned in songs or movies",
    "about whether it has religious significance",
    "about whether it's a status symbol",
    "about whether it's considered old-fashioned",
    "about whether it's a recent invention",
    # Physical state
    "about whether it's usually wet or dry",
    "about whether it can melt or freeze",
    "about whether it can break into pieces",
    "about whether it rusts or corrodes",
    "about whether it stretches or bends",
    "about whether it bounces",
    # Quantities and multiples
    "about whether people usually own more than one",
    "about whether they come in sets or pairs",
    "about whether they come in different sizes",
    "about whether they come in many colors",
    # Time and duration
    "about whether it lasts forever or wears out",
    "about whether you use it every day",
    "about whether you use it for just a few minutes",
    "about whether it needs maintenance",
    # Context and settings
    "about whether you'd find it at a party",
    "about whether you'd find it at school",
    "about whether you'd find it at a hospital",
    "about whether you'd find it in a car",
    "about whether you'd find it at a restaurant",
    "about whether you'd find it at a hotel",
    "about whether you'd find it in a garden",
    "about whether you'd find it at the beach",
    # More professions
    "about whether an artist would use it",
    "about whether a mechanic would use it",
    "about whether a musician would use it",
    "about whether an athlete would use it",
    # Emotional and subjective
    "about whether people get attached to it",
    "about whether it can be comforting",
    "about whether it can be boring",
    "about whether it can be exciting",
]


def ollama_json(model: str, prompt: str, max_tokens: int = 200, temperature: float = 0.75) -> dict | None:
    """Call Ollama API and return parsed JSON response."""
    global total_input_tokens, total_output_tokens

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
        }
    }

    try:
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read().decode('utf-8'))

            # Track tokens
            total_input_tokens += result.get("prompt_eval_count", 0)
            total_output_tokens += result.get("eval_count", 0)

            response_text = result.get("response", "").strip()
            try:
                return json.loads(response_text)
            except json.JSONDecodeError as e:
                print(f"# JSON parse error: {e}", file=sys.stderr)
                print(f"# Raw response: {response_text[:200]}", file=sys.stderr)
    except Exception as e:
        print(f"# Ollama error: {e}", file=sys.stderr)

    return None


def claude_json(model: str, prompt: str, max_tokens: int = 200) -> dict | None:
    """Call Claude API and return parsed JSON response."""
    global total_input_tokens, total_output_tokens

    if not HAS_ANTHROPIC:
        print("# Error: anthropic package not installed. Run: pip install anthropic", file=sys.stderr)
        return None

    try:
        client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt + "\nRespond with JSON only, no other text."}]
        )

        # Track tokens
        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens

        # Get text from first text block
        response_text = ""
        for block in response.content:
            text = getattr(block, 'text', None)
            if text:
                response_text = text.strip()
                break

        # Strip markdown code blocks if present
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            # Remove first line (```json) and last line (```)
            lines = [l for l in lines if not l.startswith("```")]
            response_text = "\n".join(lines).strip()

        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"# JSON parse error: {e}", file=sys.stderr)
            print(f"# Raw response: {response_text[:200]}", file=sys.stderr)
    except Exception as e:
        print(f"# Claude error: {e}", file=sys.stderr)

    return None


def minimax_json(model: str, prompt: str, max_tokens: int = 200, temperature: float = 1.0) -> dict | None:
    """Call MiniMax API (OpenAI-compatible) and return parsed JSON response."""
    global total_input_tokens, total_output_tokens

    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        print("# Error: MINIMAX_API_KEY environment variable not set", file=sys.stderr)
        return None

    base_url = os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io/v1")

    # MiniMax temperature must be in (0.0, 1.0], clamp if needed
    if temperature <= 0:
        temperature = 0.01
    elif temperature > 1.0:
        temperature = 1.0

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt + "\nRespond with JSON only, no other text."}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
        )
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read().decode('utf-8'))

            # Track tokens
            usage = result.get("usage", {})
            total_input_tokens += usage.get("prompt_tokens", 0)
            total_output_tokens += usage.get("completion_tokens", 0)

            # Get response text
            response_text = ""
            choices = result.get("choices", [])
            if choices:
                response_text = choices[0].get("message", {}).get("content", "").strip()

            # Strip <think>...</think> tags (MiniMax chain-of-thought)
            response_text = re.sub(r'<think>[\s\S]*?</think>', '', response_text).strip()

            # Strip markdown code blocks if present
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                lines = [l for l in lines if not l.startswith("```")]
                response_text = "\n".join(lines).strip()

            try:
                return json.loads(response_text)
            except json.JSONDecodeError as e:
                print(f"# JSON parse error: {e}", file=sys.stderr)
                print(f"# Raw response: {response_text[:200]}", file=sys.stderr)
    except Exception as e:
        print(f"# MiniMax error: {e}", file=sys.stderr)

    return None


# Active backend function (set in main)
api_json = ollama_json


def strip_prefixes(question: str) -> list[str]:
    """Generate bare versions without question prefixes."""
    prefixes = [
        "is it a ", "is it an ", "is it ", "is the ",
        "are you a ", "are you an ", "are you ",
        "does it ", "do you ", "can it ", "can you ",
        "has it ", "have you ", "is this ", "is that ",
    ]
    variations = []
    q = question.lower().rstrip("?")
    for prefix in prefixes:
        if q.startswith(prefix):
            bare = q[len(prefix):].strip()
            if bare and len(bare) > 2:
                variations.append(bare)
                variations.append(bare + "?")
    return variations


def expand_pronouns(question: str) -> list[str]:
    """Generate pronoun variations of a question."""
    variations = []
    for pronoun in PRONOUNS:
        # Replace 'it' with other pronouns, adjust verb if needed
        if "is it" in question:
            if pronoun == "you":
                var = question.replace("is it", "are you")
            else:
                var = question.replace("is it", f"is {pronoun}")
            variations.append(var)
        elif "does it" in question:
            if pronoun == "you":
                var = question.replace("does it", "do you")
            else:
                var = question.replace("does it", f"does {pronoun}")
            variations.append(var)
        elif "can it" in question:
            if pronoun == "you":
                var = question.replace("can it", "can you")
            else:
                var = question.replace("can it", f"can {pronoun}")
            variations.append(var)
        elif "has it" in question:
            if pronoun == "you":
                var = question.replace("has it", "have you")
            else:
                var = question.replace("has it", f"has {pronoun}")
            variations.append(var)
    return variations


def generate_paraphrases(model: str, question: str, n: int = 5) -> list[str]:
    """Generate paraphrases of a question."""
    prompt = f"""Generate {n} different ways to ask this yes/no question:
"{question}"

Keep them short (under 8 words). Same meaning, different words.
Return JSON: {{"paraphrases": ["way1", "way2", ...]}}"""

    parsed = api_json(model, prompt, max_tokens=200)
    if not parsed:
        return []

    paraphrases = []
    for item in parsed.get("paraphrases", []):
        if isinstance(item, str):
            p = item.strip().lower()
            p = ''.join(c for c in p if ord(c) < 128)
            if p and 3 < len(p) < 60 and p != question.lower():
                paraphrases.append(p)
    return paraphrases[:n]


def generate_win_guesses(model: str, topic: str, n: int = 20) -> list[str]:
    """Generate different ways someone might guess the topic."""

    prompt = f"""The secret answer in 20 questions is "{topic}".
Generate {n} different ways someone might GUESS that the answer is "{topic}".
Include:
- Direct guesses: "chair", "is it a chair", "its a chair"
- Confident guesses: "must be a chair", "definitely a chair"
- Tentative guesses: "i think its a chair", "maybe a chair"
- Synonyms and types: "armchair", "seat", "office chair"
Keep them short and natural.
Return JSON: {{"guesses": ["chair", "is it a chair", ...]}}"""

    parsed = api_json(model, prompt, max_tokens=500)
    if not parsed:
        return []

    guesses = []
    for item in parsed.get("guesses", []):
        if isinstance(item, str):
            g = item.strip().lower()
            g = ''.join(c for c in g if ord(c) < 128)
            if g and 1 < len(g) < 60:
                guesses.append(g)

    return guesses


def generate_yes_questions(model: str, topic: str, n: int = 10) -> list[str]:
    """Generate questions that would be answered YES for this topic."""

    category = random.choice(QUESTION_CATEGORIES)
    prompt = f"""The secret answer is "{topic}".
Generate {n} SHORT yes/no questions {category} that would be answered YES for {topic}.
Only questions where the answer is definitely YES.
Keep questions under 8 words.
Return JSON: {{"questions": ["does it have legs", "can you sit on it", ...]}}"""

    parsed = api_json(model, prompt, max_tokens=500)
    if not parsed:
        return []

    questions = []
    for item in parsed.get("questions", []):
        if isinstance(item, str):
            q = item.strip().lower()
            q = ''.join(c for c in q if ord(c) < 128)
            if q and 3 < len(q) < 60:
                questions.append(q)

    return questions


def generate_qa_batch(model: str, topic: str, n: int = 10) -> list[tuple[str, str]]:
    """Have LLM generate yes/no questions and answers about the topic."""

    category = random.choice(QUESTION_CATEGORIES)
    prompt = f"""You are playing 20 questions. The secret answer is "{topic}".
Generate {n} SHORT questions {category}, with correct answers.
Keep questions under 8 words. Simple like "is it big?" or "can it fly?"
Answer YES, NO, or MAYBE (for subjective/ambiguous questions).
Return JSON: {{"pairs": [{{"q": "is it alive", "a": "NO"}}, ...]}}"""

    parsed = api_json(model, prompt, max_tokens=500)
    if not parsed:
        print(f"# generate_qa_batch: api returned None", file=sys.stderr)
        return []

    raw_pairs = parsed.get("pairs", [])
    if not raw_pairs:
        print(f"# generate_qa_batch: no 'pairs' key, got: {list(parsed.keys())}", file=sys.stderr)
        return []

    pairs = []
    for item in raw_pairs:
        # Handle both dict {"q": ..., "a": ...} and other formats
        if isinstance(item, dict):
            q = str(item.get("q", item.get("question", ""))).strip().lower()
            a = str(item.get("a", item.get("answer", ""))).strip().upper()
        else:
            continue  # Skip malformed items
        # Filter garbage
        if '<' in q or '>' in q:
            continue
        # Keep only ASCII
        q = ''.join(c for c in q if ord(c) < 128)
        if q and a in ("YES", "NO", "MAYBE") and 3 < len(q) < 80:
            pairs.append((q, a))

    return pairs


def generate_distractors(model: str, topic: str, n: int = 10) -> list[str]:
    """Have LLM generate plausible wrong guesses for the topic."""

    prompt = f"""In 20 questions, the secret is "{topic}".
Generate {n} SPECIFIC WRONG GUESSES (not categories) someone might make.
These must be DIFFERENT OBJECTS, not categories that include {topic}.
Bad example for "chair": "furniture" (chair IS furniture)
Good example for "chair": "table", "stool", "couch", "bench"
Return JSON: {{"wrong": ["item1", "item2", ...]}}"""

    # More tokens needed for longer lists
    parsed = api_json(model, prompt, max_tokens=50 + n * 20)
    if not parsed:
        return []

    distractors = []
    for item in parsed.get("wrong", []):
        if isinstance(item, str):
            word = item.strip().lower()
            word = ''.join(c for c in word if ord(c) < 128)
            if word and 2 < len(word) < 30:
                distractors.append(word)
    return distractors[:n]


def generate_wrong_guesses(distractors: list[str]) -> list[tuple[str, str]]:
    """Generate wrong guess phrases for distractor topics."""
    pairs = []
    for wrong in distractors:
        wrong = wrong.strip().lower()
        if not wrong:
            continue
        # Same pattern as WIN phrases, but output NO
        wrong_phrases = [
            wrong,
            f"is it {wrong}",
            f"is it a {wrong}",
            f"is it an {wrong}",
            f"are you {wrong}",
            f"are you a {wrong}",
            f"its {wrong}",
            f"i think its {wrong}",
            f"i guess {wrong}",
            f"{wrong}?",
        ]
        for phrase in wrong_phrases:
            pairs.append((phrase.lower(), "NO"))
    return pairs


def main():
    global api_json

    parser = argparse.ArgumentParser(description='Generate 20 questions training data')
    parser.add_argument('--topic', '-t', required=True, help='The secret topic/thing')
    parser.add_argument('-n', '--count', type=int, default=100, help='Number of Q&A pairs to generate')
    parser.add_argument('--model', '-m', default=None, help='Model to use (default: gemma2:9b for ollama, claude-sonnet-4 for claude, MiniMax-M2.7 for minimax)')
    parser.add_argument('--claude', action='store_true', help='Use Claude API instead of Ollama')
    parser.add_argument('--minimax', action='store_true', help='Use MiniMax API instead of Ollama (set MINIMAX_API_KEY env var)')
    parser.add_argument('--batch', '-b', type=int, default=10, help='Questions per LLM call')
    parser.add_argument('--distractors', '-d', type=int, default=10, help='Number of wrong guesses to auto-generate (0 to disable)')
    parser.add_argument('--nonsense', action='store_true', help='Add nonsense/gibberish -> IDK training')
    parser.add_argument('--paraphrase', '-p', type=int, default=0, help='Generate N paraphrases per question (costs more tokens)')
    parser.add_argument('--yes-only', action='store_true', help='Only generate YES-answer questions (to rebalance data)')
    parser.add_argument('--win-only', action='store_true', help='Only generate WIN variations (different ways to guess the topic)')
    args = parser.parse_args()

    # Set up API backend
    if args.claude:
        if not HAS_ANTHROPIC:
            print("# Error: anthropic package not installed. Run: pip install anthropic", file=sys.stderr)
            sys.exit(1)
        api_json = lambda model, prompt, max_tokens=200: claude_json(model, prompt, max_tokens)
        if args.model is None:
            args.model = DEFAULT_CLAUDE_MODEL
    elif args.minimax:
        if not os.environ.get("MINIMAX_API_KEY"):
            print("# Error: MINIMAX_API_KEY environment variable not set", file=sys.stderr)
            sys.exit(1)
        # MiniMax uses chain-of-thought internally, so scale up max_tokens
        api_json = lambda model, prompt, max_tokens=200: minimax_json(model, prompt, max(max_tokens * 4, 2048))
        if args.model is None:
            args.model = DEFAULT_MINIMAX_MODEL
    else:
        api_json = lambda model, prompt, max_tokens=200, temp=0.75: ollama_json(model, prompt, max_tokens, temp)
        if args.model is None:
            args.model = DEFAULT_MODEL

    topic = args.topic.lower()
    backend = "claude" if args.claude else ("minimax" if args.minimax else "ollama")
    print(f"# Topic: {topic}", file=sys.stderr)
    print(f"# Backend: {backend} | Model: {args.model}", file=sys.stderr)
    print(f"# Generating {args.count} Q&A pairs...", file=sys.stderr)

    generated = 0
    questions_used = set()
    stale_batches = 0
    max_stale = 5  # Give up after 5 batches with no new questions
    interrupted = False

    # WIN-only mode: just generate guesses
    if args.win_only:
        try:
            while generated < args.count and not interrupted:
                guesses = generate_win_guesses(args.model, topic, args.batch)
                if not guesses:
                    print(f"# No WIN guesses returned, retrying...", file=sys.stderr)
                    stale_batches += 1
                    if stale_batches >= max_stale:
                        break
                    continue

                for g in guesses:
                    if generated >= args.count:
                        break
                    if g not in questions_used:
                        print(f"{g}|WIN", flush=True)
                        questions_used.add(g)
                        generated += 1

                stale_batches = 0
                print(f"# Progress: {generated}/{args.count} | tokens: {total_input_tokens:,}in/{total_output_tokens:,}out", file=sys.stderr, flush=True)
        except KeyboardInterrupt:
            print(f"\n# Interrupted!", file=sys.stderr)

        print(f"# Generated {generated} WIN-only pairs", file=sys.stderr)
        print(f"# Tokens - input: {total_input_tokens:,}, output: {total_output_tokens:,}, total: {total_input_tokens + total_output_tokens:,}", file=sys.stderr)
        return

    # Generate Q&A in batches
    try:
        while generated < args.count and not interrupted:
            # YES-only mode: generate only YES questions
            if args.yes_only:
                yes_questions = generate_yes_questions(args.model, topic, args.batch)
                if not yes_questions:
                    print(f"# No YES questions returned, retrying...", file=sys.stderr)
                    stale_batches += 1
                    if stale_batches >= max_stale:
                        print(f"# Giving up after {max_stale} failed batches", file=sys.stderr)
                        break
                    continue
                batch = [(q, "YES") for q in yes_questions]
            else:
                batch = generate_qa_batch(args.model, topic, args.batch)
                if not batch:
                    print(f"# No batch returned from {args.model}, retrying...", file=sys.stderr)
                    stale_batches += 1
                    if stale_batches >= max_stale:
                        print(f"# Giving up after {max_stale} failed batches", file=sys.stderr)
                        break
                    continue

            prev_generated = generated

            for q, a in batch:
                if generated >= args.count:
                    break

                # Output base question
                if q not in questions_used:
                    print(f"{q}|{a}", flush=True)
                    questions_used.add(q)
                    generated += 1

                # Add pronoun variations (free, no LLM needed)
                for var in expand_pronouns(q):
                    if generated >= args.count:
                        break
                    if var not in questions_used:
                        print(f"{var}|{a}", flush=True)
                        questions_used.add(var)
                        generated += 1

                # Add bare versions without prefixes (e.g., "big" from "is it big?")
                for var in strip_prefixes(q):
                    if generated >= args.count:
                        break
                    if var not in questions_used:
                        print(f"{var}|{a}", flush=True)
                        questions_used.add(var)
                        generated += 1

                # Add paraphrases (costs extra tokens)
                if args.paraphrase > 0:
                    for var in generate_paraphrases(args.model, q, args.paraphrase):
                        if generated >= args.count:
                            break
                        if var not in questions_used:
                            print(f"{var}|{a}", flush=True)
                            questions_used.add(var)
                            generated += 1

            # Track stale batches (all duplicates)
            new_this_batch = generated - prev_generated
            if new_this_batch == 0:
                stale_batches += 1
                print(f"# Stale batch (all duplicates), {stale_batches}/{max_stale}", file=sys.stderr)
                if stale_batches >= max_stale:
                    print(f"# Model exhausted - only generated {generated}/{args.count} unique pairs", file=sys.stderr)
                    break
            else:
                stale_batches = 0  # Reset on success

            print(f"# Progress: {generated}/{args.count} (+{new_this_batch}) | tokens: {total_input_tokens:,}in/{total_output_tokens:,}out", file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        print(f"\n# Interrupted! Generated {generated} pairs so far", file=sys.stderr)
        interrupted = True

    # Skip WIN/distractors/nonsense in yes-only mode
    if args.yes_only:
        print(f"# Generated {generated} YES-only pairs", file=sys.stderr)
        print(f"# Tokens - input: {total_input_tokens:,}, output: {total_output_tokens:,}, total: {total_input_tokens + total_output_tokens:,}", file=sys.stderr)
        return

    # Add winning guesses - variations of the actual answer
    win_phrases = [
        topic,
        f"is it {topic}",
        f"is it a {topic}",
        f"is it an {topic}",
        f"it must be a {topic}",
        f"it must be {topic}",
        f"it must be an {topic}",
        f"it has to be {topic}",
        f"it has to be a {topic}",
        f"it has to be an {topic}",
        f"are you {topic}",
        f"are you a {topic}",
        f"are you an {topic}",
        f"its {topic}",
        f"its a {topic}",
        f"its an {topic}",
        f"a {topic}",
        f"an {topic}",
        f"the {topic}",
        f"i think its {topic}",
        f"i think its a {topic}",
        f"i guess {topic}",
        f"my guess is {topic}",
        f"{topic}?",
        f"{topic}!",
    ]

    for phrase in win_phrases:
        phrase = phrase.lower()
        if phrase not in questions_used:
            print(f"{phrase}|WIN", flush=True)
            questions_used.add(phrase)
            generated += 1

    # Add wrong guesses (distractors) - these should all be NO
    if args.distractors > 0:
        print(f"# Generating {args.distractors} distractors...", file=sys.stderr)
        distractors = generate_distractors(args.model, topic, args.distractors)
        print(f"# Got distractors: {distractors}", file=sys.stderr)
        wrong_pairs = generate_wrong_guesses(distractors)
        added = 0
        for phrase, answer in wrong_pairs:
            if phrase not in questions_used:
                print(f"{phrase}|{answer}", flush=True)
                questions_used.add(phrase)
                generated += 1
                added += 1
        print(f"# Added {added} distractor phrases", file=sys.stderr)

    # Add nonsense/gibberish -> IDK training
    if args.nonsense:
        nonsense_count = 0
        for phrase in NONSENSE:
            if phrase not in questions_used:
                print(f"{phrase}|IDK", flush=True)
                questions_used.add(phrase)
                generated += 1
                nonsense_count += 1
        print(f"# Added {nonsense_count} nonsense phrases", file=sys.stderr)

    print(f"# Generated {generated} pairs", file=sys.stderr)
    print(f"# Tokens - input: {total_input_tokens:,}, output: {total_output_tokens:,}, total: {total_input_tokens + total_output_tokens:,}", file=sys.stderr)


if __name__ == '__main__':
    main()
