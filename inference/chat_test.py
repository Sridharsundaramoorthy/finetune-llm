import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_PATH = "adapters/sample_phi3_lora"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

prompt = """
<|system|>
You are ABC Group of Companies, Goa.

If asked outside this scope, politely refuse and redirect.
<|user|>
How can i contact ABC company?
<|assistant|>
"""


inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.3,
        do_sample=True
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
