import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

tokenizer = AutoTokenizer.from_pretrained("TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R")
model = AutoModelForCausalLM.from_pretrained("TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R")
model.half()  # Convert model parameters to float16
model.to('cuda')

def generate_response(prompt, max_length=200, num_return_sequences=1):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to('cuda')  # Move input_ids to GPU

    start_time = time.time()
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=0,
            top_p=0,
            temperature=0.95,
        )
    end_time = time.time()
    
    generation_time = end_time - start_time
    tokens_generated = sum(len(out) for out in output) - len(input_ids[0])
    tokens_per_second = tokens_generated / generation_time
    
    responses = [tokenizer.decode(out, skip_special_tokens=True) for out in output]
    return responses, tokens_per_second


prompt = "Once upon a time"
responses, tokens_per_second = generate_response(prompt)
for i, response in enumerate(responses):
    print(f"Response {i+1}: {response}")
print(f"Tokens per second: {tokens_per_second:.2f}")
