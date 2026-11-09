"""Chat with a model in the terminal.

  python chat.py                     # the published Heretic model (downloads from Hugging Face)
  python chat.py original            # unmodified Qwen2.5-1.5B-Instruct, for comparison
  python chat.py ./path/to/model     # any local folder or Hugging Face repo id

Commands inside the chat: /reset clears history, /exit quits.
"""
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

MODELS = {
    "heretic": "aasis001/Qwen2.5-1.5B-Instruct-heretic-ara",
    "original": "Qwen/Qwen2.5-1.5B-Instruct",
}

name = sys.argv[1] if len(sys.argv) > 1 else "heretic"
path = MODELS.get(name, name)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading {path} on {device} ...")
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(device).eval()
streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)

history = []
print("Ready. /reset clears history, /exit quits.\n")
while True:
    try:
        user = input("you> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if user == "/exit":
        break
    if user == "/reset":
        history = []
        print("(history cleared)\n")
        continue
    if not user:
        continue
    history.append({"role": "user", "content": user})
    ids = tok.apply_chat_template(history, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(device)
    print("bot> ", end="", flush=True)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9, streamer=streamer)
    history.append({"role": "assistant", "content": tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)})
    print()
