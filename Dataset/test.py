import os

os.environ["TRANSFORMERS_CACHE"] = "./transformers_cache"

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("facebook/opt-iml-max-30b", torch_dtype=torch.float16)

# the fast tokenizer currently does not work correctly
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-iml-max-30b", use_fast=False)

document = "What is a color hex and how can it be used to obtain color code information, including different color models and HTML and CSS codes?"""
prompt = f"generate a question that can answer this text: {document}\nQuestion:"

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

generated_ids = model.generate(input_ids)

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
