# LoRA Adapter for Enhanced Structured Reasoning (3B)

This repository contains a LoRA fine-tuned adapter built to improve structured reasoning capabilities in a 3B parameter language model. The adapter was trained using supervised fine-tuning (SFT) on a distilled reasoning dataset, teaching the base model to produce step-by-step solutions, articulate intermediate thinking, and generate logically consistent answers to multi-step problems. By leveraging parameter-efficient fine-tuning (PEFT), this work achieves meaningful behavioral improvements while modifying only a small fraction of the model’s weights, demonstrating an efficient approach to enhancing reasoning performance without full model retraining.

---

# Demo URL
You can run inference on these reasoning capabilites here: https://huggingface.co/spaces/sprothia/finetuned-llama

## Base Model

**unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit**

⚠️ This adapter **must** be used with the base model above. Using a different base model will lead to incorrect outputs or loading errors.

---

# 📦 Installation

Install the required dependencies: pip install -U torch transformers peft accelerate safetensors


```bash
pip install -U torch transformers peft accelerate safetensors
```
# 📦 Usage
Clone this repository and attach the adapter to the base model using PEFT.

```bash
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL_ID = "PUT_YOUR_BASE_MODEL_ID_HERE"   # must match what you fine-tuned on
ADAPTER_PATH = "./finetuned-llama-001-3B"       # path to this repo folder

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer (your adapter repo includes tokenizer files)
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

# Load base model
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    device_map="auto" if device == "cuda" else None,
    torch_dtype=torch.float16 if device == "cuda" else None,
)

# Attach LoRA adapter
model = PeftModel.from_pretrained(base, ADAPTER_PATH)
model.eval()

prompt = "Solve: A bat and a ball cost $1.10 total. The bat costs $1 more. How much is the ball?"
inputs = tokenizer(prompt, return_tensors="pt")

if device == "cuda":
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=200)

print(tokenizer.decode(out[0], skip_special_tokens=True))
```

# ⚠️ Notes and Other Info

This repository contains only the adapter, not the full base model.
You must download the base model separately.
GPU is recommended for best inference performance.

---
base_model: unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit
- lora
- sft
- transformers
- trl
- unsloth
---


- PEFT 0.18.1
