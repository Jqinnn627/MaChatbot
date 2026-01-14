from fastapi import FastAPI
from pydantic import BaseModel
from unsloth import FastLanguageModel
import torch

app = FastAPI()

class GenerationRequest(BaseModel):
    query: str

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

dtype = None

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="manglish_model_retuned_3",
    max_seq_length=4096,
    dtype=dtype,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

@app.get("/health")
def health():
    return {"status": "14/1/2026 ok1"}

@app.post("/generate")
def generate_text(request: GenerationRequest):
    inputs = tokenizer(
	[
    	alpaca_prompt.format(
        	"""You are a Manglish rewriter. Rewrite the Input text without explanation. Do NOT change the meaning and keep it short.
        	Example 1: "How are you?" -> "You how ah?"
        	Example 2: "I am going home." -> "I balik dulu."
        	Example 3: "Wait for me." -> "Wait ah.
        	""", # Instruction
        	f"""{request.query}""", # Input
        	"", # output - leave this blank 
    	)
	], return_tensors = "pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens = 256, use_cache = True)

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = text.split("### Response:")[-1].strip()

    stop_words = ["Explanation:", "Note:", "###", "The original", "Meaning:"]
    for word in stop_words:
        response = response.split(word)[0].strip()

    response = response.replace('"', '').strip()
    return {"response": response}
