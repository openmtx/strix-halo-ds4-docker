import time, urllib.request, json, sys, statistics

BASE = "http://localhost:8000"
TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
LABEL = sys.argv[2] if len(sys.argv) > 2 else "run"
MAX_TOK = int(sys.argv[3]) if len(sys.argv) > 3 else 2048

PROMPT = (
    "Write a long, detailed daily travel journal for a fictional two-week trip "
    "across Japan. For each day, describe the city, specific meals eaten, people "
    "met, weather, and personal reflections. Be vivid, specific, and continue at "
    "length without stopping early."
)

def gen(max_tokens, label=""):
    body = {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True},
        "thinking": {"type": "disabled"},
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time(); first_tok = None; usage = None
    for line in urllib.request.urlopen(req, timeout=600):
        if not line.startswith(b"data: "):
            continue
        d = line[6:].strip()
        if d == b"[DONE]":
            break
        try:
            obj = json.loads(d)
        except Exception:
            continue
        if obj.get("usage"):
            usage = obj["usage"]
        choices = obj.get("choices") or []
        ch = choices[0] if choices else {}
        if first_tok is None and ch.get("delta", {}).get("content"):
            first_tok = time.time()
    total = time.time() - t0
    gen_t = (time.time() - first_tok) if first_tok else total
    ctok = usage.get("completion_tokens") if usage else None
    r = {
        "total_s": total, "gen_s": gen_t,
        "completion_tokens": ctok,
        "decode_toks": (ctok / gen_t) if (ctok and gen_t) else 0,
        "e2e_toks": (ctok / total) if (ctok and total) else 0,
    }
    if label:
        print(f"  [{label}] {ctok} toks | decode {r['decode_toks']:.2f} tok/s | "
              f"e2e {r['e2e_toks']:.2f} tok/s | gen {gen_t:.1f}s")
    return r

print(f"[{LABEL}] === thermal warmup (~1000 tok) ===")
gen(1024)
time.sleep(3)

print(f"[{LABEL}] === {TRIALS} measured trials, {MAX_TOK} tok each ===")
results = []
for i in range(TRIALS):
    results.append(gen(MAX_TOK, f"{LABEL} t{i+1}"))
    time.sleep(5)

dec = sorted(r["decode_toks"] for r in results)
print(f"\n[{LABEL}] DECODE tok/s  min={min(dec):.2f}  median={statistics.median(dec):.2f}  "
      f"mean={statistics.mean(dec):.2f}  max={max(dec):.2f}")
e2e = sorted(r["e2e_toks"] for r in results)
print(f"[{LABEL}] E2E    tok/s  min={min(e2e):.2f}  median={statistics.median(e2e):.2f}  "
      f"mean={statistics.mean(e2e):.2f}  max={max(e2e):.2f}")
