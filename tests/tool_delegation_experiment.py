"""
Tool-delegation experiment.

Question: does a 2-bit quant model reliably CHOOSE to delegate arithmetic to a
calculator tool, and does making a tool available actually improve reliability?

Design:
  - Battery of arithmetic prompts graded trivial/medium/hard + non-math distractors.
  - Two conditions: BASELINE (no tool) vs TOOL_AVAILABLE (calc tool, tool_choice=auto).
  - 3 trials each at temp=0 (within-session stable; trials surface the model's policy).
  - Per-trial outcome classified as:
        delegated -> correct / tool-error / wrong-final
        direct   -> correct / wrong
  - The "direct -> wrong" bucket WITH the tool available is the silent failure:
    the case where a delegation-only design would miss the error.

Run:
    uv run --with openai python3 tests/tool_delegation_experiment.py
"""
import json
import math
import re
import sys
import time

from openai import OpenAI

BASE_URL = "http://localhost:8000"
MODEL = "deepseek-v4-flash"
TRIALS = 3
client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="not-needed")

# ── Battery ──────────────────────────────────────────────────────────────────
# (difficulty, prompt, expected_value)
BATTERY = [
    # --- trivial: model should get these right directly ---
    ("trivial",     "What is 2 + 2?",                            4),
    ("trivial",     "What is 7 times 8?",                        56),
    ("trivial",     "What is 100 minus 37?",                     63),
    ("trivial",     "What is 144 divided by 12?",                12),
    # --- medium ---
    ("medium",      "What is 17 times 23?",                      391),
    ("medium",      "What is 25 times 25?",                      625),
    ("medium",      "What is 15 percent of 240?",                36),
    ("medium",      "What is 2 to the power of 10?",             1024),
    # --- hard: known/likely failure territory ---
    ("hard",        "What is 347 times 892?",                    309524),
    ("hard",        "What is 1234 times 5678?",                  7006652),
    ("hard",        "What is 99 to the power of 3?",             970299),
    ("hard",        "What is 13 factorial?",                     6227020800),  # known intermittent
    ("hard",        "What is 20 factorial?",                     2432902008176640000),
    ("hard",        "What is 783 times 459?",                    359397),
    ("hard",        "What is 56 times 56 times 56?",             175616),
    # --- distractors: non-math, should NOT trigger the calculator ---
    ("distractor",  "What is the capital of France?",            "paris"),
    ("distractor",  "Who wrote the play Hamlet?",                "shakespeare"),
]

# ── The calculator tool (harness-side; reliably correct) ─────────────────────
CALC_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": (
            "Evaluate a mathematical expression and return the exact numeric result. "
            "Use this for ANY arithmetic: addition, subtraction, multiplication, "
            "division, percentages, powers, and factorials. Pass the expression in "
            "Python-like syntax, e.g. '347 * 892', '2 ** 10', '0.15 * 240', '13!' "
            "(factorial), '(100 - 37)'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The expression to evaluate, e.g. '347 * 892' or '13!'",
                }
            },
            "required": ["expression"],
        },
    },
}


def tool_evaluate(expression):
    """Safe-ish arithmetic evaluator. Returns (ok, value_or_error)."""
    e = expression.strip()
    e = re.sub(r"(\d+)\s*!", r"math.factorial(\1)", e)      # 13! -> math.factorial(13)
    e = (e.replace("×", "*").replace("÷", "/")
           .replace("^", "**").replace("−", "-")
           .replace(",", ""))
    try:
        v = eval(e, {"__builtins__": {}}, {"math": math})
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        return True, v
    except Exception as ex:
        return False, f"ERROR: {ex}"


def answer_is_correct(text, expected):
    """Exact-number match: is the expected value among the numbers in the text?"""
    if isinstance(expected, str):
        return expected.lower() in text.lower()
    nums = re.findall(r"\d+", text.replace(",", ""))
    return str(expected) in nums


# ── Run one trial under a given condition ────────────────────────────────────
def run_baseline(prompt):
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200, temperature=0,
        extra_body={"thinking": {"type": "disabled"}},
    )
    msg = r.choices[0].message
    text = msg.content or ""
    return {"mode": "direct", "correct": answer_is_correct(text, _expected),
            "text": text.strip(), "expr": None, "tool_result": None}


def run_with_tool(prompt, expected):
    _ = expected
    r1 = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        tools=[CALC_TOOL],
        tool_choice="auto",
        max_tokens=300, temperature=0,
        extra_body={"thinking": {"type": "disabled"}},
    )
    msg = r1.choices[0].message

    # No tool call -> model answered directly.
    if not msg.tool_calls:
        text = msg.content or ""
        return {"mode": "direct", "correct": answer_is_correct(text, expected),
                "text": text.strip(), "expr": None, "tool_result": None}

    # Tool call -> evaluate, feed back, get final answer.
    tc = msg.tool_calls[0]
    try:
        args = json.loads(tc.function.arguments)
        expr = args.get("expression", "")
    except Exception:
        expr = tc.function.arguments
    ok, val = tool_evaluate(expr)

    # Second round: return tool result to the model.
    msgs = [
        {"role": "user", "content": prompt},
        msg.model_dump(exclude_none=True),
        {"role": "tool", "tool_call_id": tc.id,
         "content": json.dumps({"expression": expr, "result": str(val)})},
    ]
    r2 = client.chat.completions.create(
        model=MODEL,
        messages=msgs,
        tools=[CALC_TOOL],
        max_tokens=200, temperature=0,
        extra_body={"thinking": {"type": "disabled"}},
    )
    final = r2.choices[0].message.content or ""
    return {
        "mode": "delegated" if ok else "delegated-toolerror",
        "correct": answer_is_correct(final, expected) if ok else False,
        "text": final.strip(), "expr": expr, "tool_result": str(val),
    }


# ── Drive the experiment ─────────────────────────────────────────────────────
def main():
    global _expected
    print(f"Tool-delegation experiment @ {BASE_URL} (model: {MODEL})")
    print(f"Battery: {len(BATTERY)} prompts x {TRIALS} trials x 2 conditions\n")

    print("Warmup... ", end="", flush=True)
    client.chat.completions.create(
        model=MODEL, max_tokens=2, temperature=0,
        messages=[{"role": "user", "content": "ping"}],
        extra_body={"thinking": {"type": "disabled"}})
    print("ok\n")

    rows = []
    for diff, prompt, expected in BATTERY:
        _expected = expected
        row = {"difficulty": diff, "prompt": prompt, "expected": expected,
               "baseline": [], "with_tool": []}
        print(f"[{diff:>10}] {prompt}")
        for _ in range(TRIALS):
            row["baseline"].append(run_baseline(prompt))
        for _ in range(TRIALS):
            row["with_tool"].append(run_with_tool(prompt, expected))
        rows.append(row)
        # compact per-prompt summary
        b_corr = sum(t["correct"] for t in row["baseline"])
        deleg = sum(1 for t in row["with_tool"] if t["mode"] == "delegated")
        t_corr = sum(t["correct"] for t in row["with_tool"])
        print(f"    baseline {b_corr}/{TRIALS} correct | "
              f"with-tool: delegated {deleg}/{TRIALS}, net correct {t_corr}/{TRIALS}")
        print()

    # ── Analysis ─────────────────────────────────────────────────────────────
    print("=" * 78)
    print("SUMMARY BY DIFFICULTY")
    print("=" * 78)
    print(f"{'difficulty':<12} {'n':>3} {'base%':>6} {'deleg%':>7} "
          f"{'net%':>6} {'direct&wrong(tool)':>20}")
    overall = {"b": [0, 0], "t": [0, 0], "deleg": [0, 0], "silent": [0, 0]}
    for diff in ["trivial", "medium", "hard", "distractor"]:
        rs = [r for r in rows if r["difficulty"] == diff]
        if not rs:
            continue
        n_prompts = len(rs); n = n_prompts * TRIALS
        b_corr = sum(t["correct"] for r in rs for t in r["baseline"])
        deleg = sum(1 for r in rs for t in r["with_tool"] if t["mode"] == "delegated")
        t_corr = sum(t["correct"] for r in rs for t in r["with_tool"])
        # silent failures: tool available, model answered direct, AND wrong
        silent = sum(1 for r in rs for t in r["with_tool"]
                     if t["mode"] == "direct" and not t["correct"])
        overall["b"][0] += b_corr; overall["b"][1] += n
        overall["t"][0] += t_corr; overall["t"][1] += n
        overall["deleg"][0] += deleg; overall["deleg"][1] += n
        overall["silent"][0] += silent; overall["silent"][1] += n
        print(f"{diff:<12} {n:>3} {100*b_corr/n:>5.0f}% {100*deleg/n:>6.0f}% "
              f"{100*t_corr/n:>5.0f}% {silent:>20}")

    print("-" * 78)
    n = overall["b"][1]
    print(f"{'OVERALL':<12} {n:>3} {100*overall['b'][0]/n:>5.0f}% "
          f"{100*overall['deleg'][0]/n:>6.0f}% {100*overall['t'][0]/n:>5.0f}% "
          f"{overall['silent'][0]:>20}")
    print("=" * 78)
    print(f"\nBaseline accuracy:       {overall['b'][0]}/{n}  "
          f"({100*overall['b'][0]/n:.1f}%)")
    print(f"With-tool net accuracy:  {overall['t'][0]}/{n}  "
          f"({100*overall['t'][0]/n:.1f}%)")
    print(f"Delegation rate:         {overall['deleg'][0]}/{n}  "
          f"({100*overall['deleg'][0]/n:.1f}%)")
    print(f"SILENT failures (direct+wrong despite tool available): "
          f"{overall['silent'][0]}/{n}  ({100*overall['silent'][0]/n:.1f}%)")

    # ── Detail: every silent failure ─────────────────────────────────────────
    print("\n" + "=" * 78)
    print("SILENT FAILURES (tool was available, model answered directly & wrong)")
    print("=" * 78)
    any_silent = False
    for r in rows:
        for i, t in enumerate(r["with_tool"]):
            if t["mode"] == "direct" and not t["correct"]:
                any_silent = True
                print(f"  [{r['difficulty']}] {r['prompt']}  (expected {r['expected']})")
                print(f"      trial {i+1}: {t['text'][:80]!r}")
    if not any_silent:
        print("  (none)")

    # ── Detail: delegated calls — did extraction work? ───────────────────────
    print("\n" + "=" * 78)
    print("DELEGATED CALLS (expression the model passed -> tool result)")
    print("=" * 78)
    any_deleg = False
    for r in rows:
        for i, t in enumerate(r["with_tool"]):
            if t["mode"].startswith("delegated"):
                any_deleg = True
                print(f"  [{r['difficulty']}] {r['prompt']}")
                print(f"      expr: {t['expr']!r}  ->  {t['tool_result']}  "
                      f"| final correct: {t['correct']}")
    if not any_deleg:
        print("  (none — model never called the tool)")

    with open("tests/delegation_results.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print("\n(raw per-trial data -> tests/delegation_results.json)")


if __name__ == "__main__":
    main()
