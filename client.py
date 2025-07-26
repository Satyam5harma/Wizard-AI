import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Optional: set HF token in environment
# I cant share you my API key, Please generate your own API key
os.environ["HF_TOKEN"] = "Your-API-Key"

# Model details
model_id = "bosonai/higgs-audio-v2-generation-3B-base"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=os.environ["HF_TOKEN"])
model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=os.environ["HF_TOKEN"])

# Set up the pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Example prompt
prompt = "Who is the Prime Minister of India?"

# Generate response
result = generator(prompt, max_length=100, do_sample=True, top_k=50, num_return_sequences=1)

# Print the response
print(result[0]["generated_text"])
