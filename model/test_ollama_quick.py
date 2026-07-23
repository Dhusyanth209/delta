"""Quick test: directly call Ollama to confirm it responds."""
import urllib.request, json

payload = {
    "model": "llama3.2:latest",
    "messages": [
        {"role": "system", "content": "You are DELTA Copilot, an AI Project Manager assistant. Keep answers to 2-3 sentences."},
        {"role": "user", "content": "Why might a project with 65% employee cost ratio and 3 attrition events be at risk?"}
    ],
    "stream": False,
    "options": {"temperature": 0.3}
}

req = urllib.request.Request(
    "http://localhost:11434/api/chat",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST"
)

with urllib.request.urlopen(req, timeout=30) as resp:
    data = json.loads(resp.read().decode("utf-8"))
    text = data.get("message", {}).get("content", "")
    model = data.get("model", "unknown")
    print(f"Model: {model}")
    print(f"Response: {text}")
