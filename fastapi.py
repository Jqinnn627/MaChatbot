# !unzip -o '/content/manglish_model_retuned_3.zip' -d '/manglish_model_retuned_3/'

# %%capture
# import os, re
# if "COLAB_" not in "".join(os.environ.keys()):
#     !pip install unsloth
# else:
#     # Do this only in Colab notebooks! Otherwise use pip install unsloth
#     import torch; v = re.match(r"[0-9]{1,}\.[0-9]{1,}", str(torch.__version__)).group(0)
#     xformers = "xformers==" + ("0.0.33.post1" if v=="2.9" else "0.0.32.post2" if v=="2.8" else "0.0.29.post3")
#     !pip install --no-deps bitsandbytes accelerate {xformers} peft trl triton cut_cross_entropy unsloth_zoo
#     !pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer
#     !pip install --no-deps unsloth
# !pip install transformers==4.56.2
# !pip install --no-deps trl==0.22.2
# !pip install unsloth_zoo

# %%writefile main.py #(in colab)
from fastapi import FastAPI
from pydantic import BaseModel
from unsloth import FastLanguageModel

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
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

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
