"""Measure prefill (prompt-processing) throughput, plus reconfirm decode.

Prefill: send a large prompt with max_tokens=1 so almost all wall-clock is
prompt processing. prompt_tokens / total_time ~= prefill tok/s.
Decode: stream a long completion, measure tokens generated / generation time.
"""
import time, urllib.request, json

BASE = "http://localhost:8000"

def prefill(n_words):
    # build a prompt of roughly n_words, where the 'task' is trivial
    body = ("words " * (n_words // 2 + 1) +
            "\n\nReply to the above list with only the word 'done'.")
    payload = json.dumps({
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": body}],
        "max_tokens": 1,
        "temperature": 0,
        "stream": False,
        "stream_options": {"include_usage": True},
        "thinking": {"type": "disabled"},
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=300).read())
    dt = time.time() - t0
    ptok = r["usage"]["prompt_tokens"]
    print(f"  prompt {ptok:>7} tok | wall {dt:6.2f}s | prefill {ptok/dt:7.1f} tok/s")
    return ptok, dt

def decode():
    payload = json.dumps({
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user",
                      "content": "Write a long detailed travel journal. Continue at length."}],
        "max_tokens": 2048, "temperature": 0,
        "stream": True, "stream_options": {"include_usage": True},
        "thinking": {"type": "disabled"},
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; usage = None
    for line in urllib.request.urlopen(req, timeout=600):
        if line.startswith(b"data: "):
            d = line[6:].strip()
            if d == b"[DONE]": break
            obj = json.loads(d)
            if obj.get("usage"): usage = obj["usage"]
            if first is None and (obj.get("choices") or [{}])[0].get("delta", {}).get("content"):
                first = time.time()
    gen = time.time() - first
    ctok = usage["completion_tokens"]
    print(f"  decode {ctok} tok | gen {gen:6.2f}s | decode {ctok/gen:6.2f} tok/s")

print("PREFILL (max_tokens=1, so wall-clock ~= prompt processing):")
prefill(2000); prefill(8000); prefill(20000)
print("\nDECODE:")
decode()
