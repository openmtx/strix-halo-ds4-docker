import json
import sys
import time
from openai import OpenAI

BASE_URL = "http://localhost:8000"
MODEL = "deepseek-v4-flash"

client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="not-needed")

passed = 0
failed = 0

def test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  PASS  {name}")
        passed += 1
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        failed += 1

def check(condition, msg=""):
    if not condition:
        raise AssertionError(msg)

def chat(messages, **kwargs):
    default = {
        "model": MODEL, "max_tokens": 200, "temperature": 0,
        "extra_body": {"thinking": {"type": "disabled"}}
    }
    default.update(kwargs)
    return client.chat.completions.create(messages=messages, **default)

# ── Basic ────────────────────────────────────────────────────────────────────

def test_basic():
    r = chat([{"role": "user", "content": "What is 2+2? Answer with just the number."}], max_tokens=50)
    check("4" in r.choices[0].message.content, f"got: {r.choices[0].message.content}")

# ── Needle in a haystack ─────────────────────────────────────────────────────

def test_needle_haystack():
    haystack = (
        "We have many facts about the city. "
        "The mayor of Springfield is named Elizabeth Hopper. "
        "She was elected in 2021 on a platform of public transit reform. "
        "The city has three public libraries and a new aquatics centre. "
        "The population is approximately 287,000. "
        "Springfield was founded in 1847 by Samuel T. Caldwell. "
        "The city's main industries are manufacturing and healthcare. "
        "The annual budget is $1.2 billion. "
        "There are 14 public elementary schools. "
        "The city council meets every second Tuesday. "
        "Springfield's nickname is 'The River City'. "
        "There is a sister city relationship with Lyon, France. "
        "The local newspaper is called the Springfield Herald. "
        "The Springfield Symphony Orchestra performs at the Grand Theatre. "
        "The city has a hockey team called the Springfield Falcons. "
        "The airport code is SPF. "
        "Springfield was recently awarded the 'Green City' award. "
        "The public transport system includes 27 bus routes. "
        "There are over 200 acres of parkland. "
        "The oldest building in town is the Caldwell House from 1852."
    )
    r = chat([{"role": "user", "content": f"{haystack}\n\nBased on the text above, who is the mayor of Springfield?"}], max_tokens=100)
    content = r.choices[0].message.content
    check("Elizabeth" in content and "Hopper" in content, f"got: {content}")

# ── Complex tool calling ─────────────────────────────────────────────────────

def test_tool_calling():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                        "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["city"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "send_email",
                "description": "Send an email to a recipient",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string"},
                        "subject": {"type": "string"},
                        "body": {"type": "string"}
                    },
                    "required": ["to", "subject", "body"]
                }
            }
        }
    ]
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "What's the weather in Tokyo? Use celsius."}],
        tools=tools,
        tool_choice="auto",
        max_tokens=200,
        temperature=0,
        extra_body={"thinking": {"type": "disabled"}}
    )
    msg = r.choices[0].message
    check(msg.tool_calls is not None, "no tool call returned")
    call = msg.tool_calls[0]
    check(call.function.name == "get_weather", f"expected get_weather, got {call.function.name}")
    args = json.loads(call.function.arguments)
    check("tokyo" in args.get("city", "").lower(), f"unexpected city: {args}")
    check(args.get("units") == "celsius", f"unexpected units: {args}")

# ── Multi-turn conversation ──────────────────────────────────────────────────

def test_multi_turn():
    messages = [
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Nice to meet you, Alice!"},
        {"role": "user", "content": "What is my name?"},
    ]
    r = chat(messages, max_tokens=50)
    content = r.choices[0].message.content
    check("Alice" in content, f"got: {content}")

# ── Reasoning / logic ────────────────────────────────────────────────────────

def test_reasoning():
    r = chat([{"role": "user", "content": (
        "A farmer has 17 chickens, 9 cows, and 12 sheep. "
        "How many legs do all the animals have in total? "
        "Think step by step."
    )}], max_tokens=200)
    content = r.choices[0].message.content
    check("118" in content, f"expected 118 legs, got: {content}")

# ── Long output / code generation ────────────────────────────────────────────

def test_code_generation():
    r = chat([{"role": "user", "content": "Write a Python function that checks if a string is a palindrome. Return just the code."}], max_tokens=300)
    content = r.choices[0].message.content
    check("def " in content, "no function definition")
    check("palindrome" in content.lower(), "no palindrome mention")
    check("return" in content, "no return statement")

# ── System prompt adherence ──────────────────────────────────────────────────

def test_system_prompt():
    r = chat([
        {"role": "system", "content": "You only speak in French."},
        {"role": "user", "content": "Say hello."}
    ], max_tokens=50)
    content = r.choices[0].message.content
    check("bonjour" in content.lower(), f"expected French, got: {content}")

# ── Context length / recall ─────────────────────────────────────────────────

def test_needle_long_context():
    paragraphs = []
    for i in range(50):
        paragraphs.append(f"Fact {i+1}: The color of item {i+1} is {'red' if i == 37 else 'blue'}.")
    context = " ".join(paragraphs)
    r = chat([{"role": "user", "content": f"{context}\n\nWhat is the color of item 38?"}], max_tokens=50)
    check("red" in r.choices[0].message.content.lower(), f"got: {r.choices[0].message.content}")

# ── Run everything ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Testing ds4-server at {BASE_URL} (model: {MODEL})\n")

    print("Warmup...")
    chat([{"role": "user", "content": "ping"}], max_tokens=2)

    print(f"\n{'─'*50}\nRunning tests...\n")

    test("basic Q&A",                         test_basic)
    test("needle in a haystack",              test_needle_haystack)
    test("long context recall",               test_needle_long_context)
    test("complex tool calling",              test_tool_calling)
    test("multi-turn conversation",           test_multi_turn)
    test("reasoning / arithmetic",            test_reasoning)
    test("code generation",                   test_code_generation)
    test("system prompt adherence",           test_system_prompt)

    print(f"\n{'─'*50}")
    print(f"Results: {passed} passed, {failed} failed, {passed+failed} total")
    sys.exit(0 if failed == 0 else 1)
