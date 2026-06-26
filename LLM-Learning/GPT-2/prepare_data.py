import numpy as np
import tiktoken
from datasets import load_dataset

TARGET_TOKENS = 600_000_000

ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
enc = tiktoken.get_encoding("gpt2")
written = 0

with open("/Users/olivergreen/Documents/GitHub/OliverEGreen.github.io/LLM-Learning/GPT-2/fineweb_train.bin", "wb") as f:
    for doc in ds:
        ids = enc.encode(doc["text"])
        ids.append(enc.eot_token)
        arr = np.array(ids, dtype=np.uint16)
        f.write(arr.tobytes())  # write this doc's tokens straight to disk
        written += len(arr)
        if written >= TARGET_TOKENS:
            break
print("wrote", written, "tokens")
