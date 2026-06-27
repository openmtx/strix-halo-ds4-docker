"""
Quantization stress tests for heavily-quantized models (e.g. IQ2XXS 2-bit).

Targets the failure modes that quantization amplifies:
  - Factual recall erosion
  - Hallucination / false-premise acceptance
  - Multi-step reasoning collapse
  - Math precision loss
  - Code correctness (executed, not eyeballed)
  - Instruction / format compliance
  - Negative-constraint violations
  - Repetition / looping degeneration
  - Paraphrase robustness

Usage:
    uv run --with openai python3 tests/test_quant_quality.py
"""
import json
import re
import sys
import time
import unicodedata
from openai import OpenAI

BASE_URL = "http://localhost:8000"
MODEL = "deepseek-v4-flash"

client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="not-needed")

results = []  # (category, name, passed, detail)


def chat(messages, **kwargs):
    default = {
        "model": MODEL, "max_tokens": 400, "temperature": 0,
        "extra_body": {"thinking": {"type": "disabled"}},
    }
    default.update(kwargs)
    return client.chat.completions.create(messages=messages, **default).choices[0].message.content


def record(category, name, passed, detail=""):
    results.append((category, name, 1 if passed else 0, detail))
    flag = "PASS" if passed else "FAIL"
    extra = f"  ({detail})" if detail else ""
    print(f"  [{flag}] {name}{extra}")


def normalize(text):
    """NFKD: accents -> base letters, subscript digits -> ascii. No-op on ASCII."""
    return "".join(c for c in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(c))


def contains_any(text, needles):
    t = normalize(text).lower()
    return any(normalize(n).lower() in t for n in needles)


# ─────────────────────────────────────────────────────────────────────────────
# 1. FACTUAL RECALL — count correct out of N. Quant erodes this unevenly.
# ─────────────────────────────────────────────────────────────────────────────
def test_factual_recall():
    print("\n─ Factual Recall (scored) ─")
    facts = [
        ("Atomic number of gold?",                                 ["79"]),
        ("Who wrote the novel '1984'?",                            ["orwell"]),
        ("In what year did World War II end in Europe?",           ["1945"]),
        ("Chemical symbol for sodium?",                            ["na"]),
        ("What is the capital of Australia?",                      ["canberra"]),
        ("How many continents are there?",                         ["7", "seven"]),
        ("Approximate speed of light in meters per second?",       ["299", "3 x 10", "3*10", "3×10", "3e8"]),
        ("What is the currency of Japan?",                         ["yen"]),
        ("Largest planet in the solar system?",                    ["jupiter"]),
        ("Who painted the Mona Lisa?",                             ["vinci", "leonardo"]),
        ("What is the freezing point of water in Celsius at sea level?", ["0°", "0 degrees", "0 celsius", "0 c", "zero degrees", "0 ⁰"]),
        ("Square root of 144?",                                    ["12"]),
    ]
    correct = 0
    for q, answers in facts:
        r = chat([{"role": "user", "content": f"{q} Answer briefly."}], max_tokens=40)
        ok = contains_any(r, answers)
        record("factual", q, ok, f"got: {r.strip()[:60]!r}")
        correct += ok
    n = len(facts)
    print(f"  → factual score: {correct}/{n}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. HALLUCINATION RESISTANCE — false premises. Quant models agree too readily.
#    PASS = model corrects/pushes back; FAIL = confidently fabricates.
# ─────────────────────────────────────────────────────────────────────────────
def test_hallucination():
    print("\n─ Hallucination Resistance ─ ─")
    cases = [
        (
            "Who was the first person to walk on Mars?",
            ["no one", "nobody", "never", "has not", "hasn't", "no human",
             "not yet", "hasn't yet", "no person", "have not", "haven't"],
            "should deny anyone has walked on Mars",
        ),
        (
            "Why did Albert Einstein receive the Nobel Prize in Physics for the theory of relativity?",
            ["photoelectric", "not for", "didn't", "did not", "wasn't", "actually",
             "rather than", "instead"],
            "should correct: he won for photoelectric effect, not relativity",
        ),
        (
            "Describe the plot of J.R.R. Tolkien's novel 'The Shadow Returns', the fourth book in the Lord of the Rings series.",
            ["no fourth", "only three", "three books", "not a", "doesn't exist",
             "does not exist", "no such", "never wrote", "trilogy", "isn't", "is not"],
            "should correct: only 3 LotR books",
        ),
        (
            "What year did humans first land on the surface of the Sun?",
            ["never", "impossible", "no one", "cannot", "can't", "not possible",
             "hasn't", "would", "no human", "fake", "doesn't"],
            "should refuse: landing on the Sun is impossible",
        ),
    ]
    for q, corrections, why in cases:
        r = chat([{"role": "user", "content": q}], max_tokens=150)
        ok = contains_any(r, corrections)
        record("hallucination", q, ok, f"got: {r.strip()[:70]!r}  ({why})")


# ─────────────────────────────────────────────────────────────────────────────
# 3. MATH PRECISION
# ─────────────────────────────────────────────────────────────────────────────
def test_math():
    print("\n─ Math Precision ─")
    cases = [
        ("What is 17 * 23?", ["391"], "391"),
        ("What is 15% of 240?", ["36"], "36"),
        ("What is 4096 divided by 16?", ["256"], "256"),
        ("What is the sum of all integers from 1 to 100?", ["5050"], "5050"),
        ("What is 7 factorial (7!)?", ["5040"], "5040"),
    ]
    for q, answers, expected in cases:
        r = chat([{"role": "user", "content": f"{q} Give only the number."}], max_tokens=30)
        ok = contains_any(r, answers)
        record("math", q, ok, f"expected {expected}, got: {r.strip()[:40]!r}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. MULTI-STEP REASONING
# ─────────────────────────────────────────────────────────────────────────────
def test_reasoning():
    print("\n─ Multi-step Reasoning ─")
    cases = [
        (
            "I have 3 apples. I eat one. I buy 2 more. I give half of what I have to a friend. "
            "How many apples do I have left? Explain briefly, then state the final number.",
            ["2 apple", "2 left", "leaves 2", "= 2", "left is 2", "have 2"],
        ),
        (
            "A train travels 60 km in 45 minutes. What is its speed in km/h?",
            ["80", "80 km"],
        ),
        (
            "Consider: All Blorps are Glarps. Some Glarps are Flumphs. "
            "Is it logically valid to conclude that some Blorps are Flumphs? "
            "Answer 'yes, valid' or 'no, not valid' and explain.",
            ["not valid", "no", "invalid", "isn't valid", "cannot", "fallacy"],
        ),
        (
            "Sarah is older than Mike. Mike is older than Tom. Is Sarah older than Tom?",
            ["yes"],
        ),
    ]
    for q, answers in cases:
        r = chat([{"role": "user", "content": q}], max_tokens=300)
        ok = contains_any(r, answers)
        record("reasoning", q[:50], ok, f"got: {r.strip()[:70]!r}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. INSTRUCTION / FORMAT COMPLIANCE
# ─────────────────────────────────────────────────────────────────────────────
def test_instruction_following():
    print("\n─ Instruction Following ─")
    r = chat([{"role": "user", "content":
        "Reply with only the word BANANA. No punctuation, no other words."}], max_tokens=10)
    record("instruction", "exact single word", r.strip().upper() == "BANANA", f"got: {r.strip()!r}")

    r = chat([{"role": "user", "content":
        "List exactly 5 colors, one per line, numbered 1 to 5. Nothing else."}], max_tokens=80)
    lines = [l for l in r.strip().splitlines() if l.strip()]
    record("instruction", "exactly 5 numbered lines", len(lines) == 5, f"got {len(lines)} lines")

    r = chat([{"role": "user", "content":
        "Respond with a valid JSON object with one key 'status' and value 'ok'. Nothing else."}], max_tokens=40)
    try:
        ok = json.loads(r.strip().strip("`").replace("json\n", ""))["status"] == "ok"
    except Exception:
        ok = False
    record("instruction", "pure JSON output", ok, f"got: {r.strip()[:50]!r}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. NEGATIVE CONSTRAINT — quant models violate "do not say X"
# ─────────────────────────────────────────────────────────────────────────────
def test_negative_constraint():
    print("\n─ Negative Constraint ─")
    r = chat([{"role": "user", "content":
        "Describe the ocean in two sentences. Do NOT use the word 'blue' anywhere in your response."}],
        max_tokens=80)
    ok = "blue" not in r.lower()
    record("negative", "avoid forbidden word 'blue'", ok, f"got: {r.strip()[:70]!r}")

    r = chat([{"role": "user", "content":
        "Write a three-sentence paragraph about a cat. The word 'meow' must not appear."}],
        max_tokens=100)
    ok = "meow" not in r.lower()
    record("negative", "avoid forbidden word 'meow'", ok, f"got: {r.strip()[:70]!r}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. CODE CORRECTNESS — actually execute the generated code.
# ─────────────────────────────────────────────────────────────────────────────
def extract_code(text):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


def test_code():
    print("\n─ Code Correctness (executed) ─")
    # is_prime
    r = chat([{"role": "user", "content":
        "Write a Python function `is_prime(n)` that returns True if n is prime, else False. "
        "Handle n < 2. Only output the code."}], max_tokens=300)
    code = extract_code(r)
    try:
        ns = {}
        exec(code, ns)
        ip = ns["is_prime"]
        checks = [(2, True), (3, True), (4, False), (17, True), (25, False),
                  (1, False), (0, False), (-5, False), (97, True), (100, False)]
        bad = [(n, ip(n), exp) for n, exp in checks if ip(n) != exp]
        ok = not bad
        detail = f"all correct" if ok else f"wrong on: {bad[:3]}"
    except Exception as e:
        ok, detail = False, f"exec error: {e}"
    record("code", "is_prime (executed)", ok, detail)

    # reverse_string
    r = chat([{"role": "user", "content":
        "Write a Python function `reverse_string(s)` returning the reversed string. Only output the code."}],
        max_tokens=150)
    code = extract_code(r)
    try:
        ns = {}
        exec(code, ns)
        rs = ns["reverse_string"]
        bad = [(inp, rs(inp), exp) for inp, exp in
               [("hello", "olleh"), ("abc", "cba"), ("", ""), ("x", "x")]
               if rs(inp) != exp]
        ok = not bad
        detail = "all correct" if ok else f"wrong: {bad[:2]}"
    except Exception as e:
        ok, detail = False, f"exec error: {e}"
    record("code", "reverse_string (executed)", ok, detail)


# ─────────────────────────────────────────────────────────────────────────────
# 8. REPETITION / LOOPING — measure trigram repetition ratio.
# ─────────────────────────────────────────────────────────────────────────────
def test_repetition():
    print("\n─ Repetition / Looping ─")
    r = chat([{"role": "user", "content":
        "Write a few paragraphs about the history of computing. Be specific and varied."}],
        max_tokens=300, temperature=0.7)
    words = r.lower().split()
    if len(words) < 20:
        record("repetition", "trigram repeat ratio", False, f"too short: {len(words)} words")
        return
    trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
    unique = len(set(trigrams))
    ratio = 1 - unique / len(trigrams)  # 0 = no repetition, 1 = all same
    # >0.4 is heavy looping; >0.25 is suspicious for a 2-bit model
    ok = ratio < 0.25
    record("repetition", "trigram repeat ratio", ok, f"ratio={ratio:.2f} (threshold 0.25), {len(words)} words")


# ─────────────────────────────────────────────────────────────────────────────
# 9. PARAPHRASE ROBUSTNESS — same question, 3 phrasings → consistent answer.
# ─────────────────────────────────────────────────────────────────────────────
def test_paraphrase():
    print("\n─ Paraphrase Robustness ─")
    phrasings = [
        "What is the capital of France?",
        "Name the capital city of France.",
        "France's capital is what city?",
    ]
    answers = []
    for p in phrasings:
        r = chat([{"role": "user", "content": p}], max_tokens=30)
        answers.append("paris" in r.lower())
    all_paris = all(answers)
    record("paraphrase", "capital of France (3 ways)", all_paris, f"paris-in-answer: {answers}")


# ─────────────────────────────────────────────────────────────────────
# 10. HARD PROBES — known quant-killer territory. Where 2-bit actually breaks:
#     multi-digit arithmetic, letter/token counting, rarer knowledge,
#     harder code, date math, sorting.
# ─────────────────────────────────────────────────────────────────────
def test_hard_probes():
    print("\n─ Hard Probes (quant-killer territory) ─")

    # --- Multi-digit multiplication (2-bit frequently fumbles these) ---
    hard_math = [
        ("What is 347 * 892? Give only the number.", ["309524"]),
        ("What is 1234 * 5678? Give only the number.", ["7006652"]),
        ("What is 99 * 99 * 99? Give only the number.", ["970299"]),
        ("What is 13! (13 factorial)? Give only the number.", ["6227020800"]),
    ]
    for q, answers in hard_math:
        r = chat([{"role": "user", "content": q}], max_tokens=40)
        ok = contains_any(r, answers)
        record("hard-math", q, ok, f"got: {r.strip()[:40]!r}")

    # --- Letter / token counting (famous LLM failure, quant makes it worse) ---
    counting = [
        ("How many times does the letter 'r' appear in the word 'strawberry'?", ["3"]),
        ("How many letters are in the word 'encyclopedia'?", ["12"]),
        ("How many times does the letter 'e' appear in 'sentence'?", ["3"]),
        ("What is the 7th letter of the English alphabet?", ["g"]),
    ]
    for q, answers in counting:
        r = chat([{"role": "user", "content": q}], max_tokens=150)
        ok = contains_any(r, answers)
        record("hard-counting", q[:55], ok, f"got: {r.strip()[:40]!r}")

    # --- Rarer knowledge (edges of training, where quant hurts most) ---
    rare = [
        ("What is the capital of Burkina Faso?", ["ouagadougou"]),
        ("What is the atomic number of tungsten?", ["74"]),
        ("Who wrote 'One Hundred Years of Solitude'?", ["garcia", "marquez"]),
        ("What is the chemical formula for sulfuric acid?", ["h2so4"]),
    ]
    for q, answers in rare:
        r = chat([{"role": "user", "content": f"{q} Answer briefly."}], max_tokens=40)
        ok = contains_any(r, answers)
        record("hard-rare", q, ok, f"got: {r.strip()[:50]!r}")

    # --- Harder code: binary search, tested across many cases ---
    r = chat([{"role": "user", "content":
        "Write a Python function `binary_search(arr, target)` that returns the index of "
        "target in a sorted list arr, or -1 if not found. Only output the code."}], max_tokens=400)
    code = extract_code(r)
    try:
        ns = {}
        exec(code, ns)
        bs = ns["binary_search"]
        arr = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
        checks = [
            (arr, 7, 3), (arr, 1, 0), (arr, 19, 9), (arr, 10, -1),
            (arr, 0, -1), (arr, 20, -1), ([], 1, -1), ([5], 5, 0), ([5], 1, -1),
        ]
        bad = [(a, t, bs(a, t), exp) for a, t, exp in checks if bs(a, t) != exp]
        ok = not bad
        detail = "all correct" if ok else f"wrong: {bad[:2]}"
    except Exception as e:
        ok, detail = False, f"exec error: {e}"
    record("hard-code", "binary_search (executed, 9 cases)", ok, detail)

    # --- Date / modular arithmetic: 100 mod 7 = 2 -> Wed + 2 = Friday ---
    r = chat([{"role": "user", "content":
        "If today is Wednesday, what day of the week will it be 100 days from now? "
        "Answer with just the day name."}], max_tokens=30)
    record("hard-date", "day-of-week +100", "friday" in r.lower(), f"got: {r.strip()[:30]!r}")

    # --- Sorting 8 numbers ---
    r = chat([{"role": "user", "content":
        "Sort these numbers ascending: 47, 3, 89, 12, 5, 71, 23, 56. Only the numbers, comma separated."}],
        max_tokens=50)
    nums = re.findall(r"\d+", r)
    ok = nums == ["3", "5", "12", "23", "47", "56", "71", "89"]
    record("hard-sort", "8-number sort", ok, f"got: {r.strip()[:50]!r}")


# ─────────────────────────────────────────────────────────────────────
# 11. DETERMINISM — at temp=0 a quantized model on GPU can STILL be
#     non-deterministic (GEMM atomics / reduction order). When the top-2
#     logits are close (common at 2-bit), FP noise flips greedy argmax,
#     so the same prompt yields different answers across runs. Divergence
#     here is itself a quantization-degradation signal.
# ─────────────────────────────────────────────────────────────────────
def test_determinism(runs=5):
    print("\n─ Determinism @ temp=0 (run each 5x) ─")
    probes = [
        ("What is 13 factorial (13!)? Only the number.", "6227020800"),
        ("What is 347 * 892? Only the number.", "309524"),
        ("If today is Wednesday, what day is it exactly 100 days later? Only the day name.", "friday"),
    ]
    for q, correct in probes:
        outs = set()
        for _ in range(runs):
            r = chat([{"role": "user", "content": q}], max_tokens=120, temperature=0)
            outs.add(normalize(r).strip().lower().replace(" ", "").replace(",", ""))
        stable = len(outs) == 1
        any_correct = any(correct in o for o in outs)
        # PASS only if stable AND correct; flag instability explicitly
        ok = stable and any_correct
        flag = "UNSTABLE" if not stable else ("wrong" if not any_correct else "ok")
        record("determinism", q[:48], ok, f"{flag}: {sorted(outs)}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Quantization stress test @ {BASE_URL} (model: {MODEL})\n")
    print("Warmup...")
    chat([{"role": "user", "content": "ping"}], max_tokens=2)

    test_factual_recall()
    test_hallucination()
    test_math()
    test_reasoning()
    test_instruction_following()
    test_negative_constraint()
    test_code()
    test_repetition()
    test_paraphrase()
    test_hard_probes()
    test_determinism()

    # ── Scorecard ────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("SCORECARD BY CATEGORY")
    print("═" * 60)
    cats = {}
    for cat, name, p, detail in results:
        cats.setdefault(cat, []).append(p)
    total_pass = total_n = 0
    for cat, scores in cats.items():
        c = sum(scores)
        n = len(scores)
        pct = 100 * c / n
        total_pass += c
        total_n += n
        bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
        print(f"  {cat:<16} {c}/{n}  {pct:5.1f}%  {bar}")
    print("─" * 60)
    print(f"  {'TOTAL':<16} {total_pass}/{total_n}  {100*total_pass/total_n:5.1f}%")
    print("═" * 60)
    print("\nFailures (signal of quantization degradation):")
    for cat, name, p, detail in results:
        if not p:
            print(f"  ✗ [{cat}] {name}  {detail}")
    sys.exit(0 if total_pass == total_n else 1)
