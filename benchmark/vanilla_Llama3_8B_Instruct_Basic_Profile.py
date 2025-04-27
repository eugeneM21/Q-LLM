import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import nvtx
import argparse
import csv
import ast

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="Parse the arguments for prompt and model configuration")

parser.add_argument('--prompt', type=str, help="Prompt to run inference on")
parser.add_argument('--record_csv', type=str, help="Flag to record output into CSV file")

args = parser.parse_args()

torch.cuda.empty_cache()

record_in_csv = False
if args.record_csv:
    if args.record_csv.lower() == "true":
        record_in_csv = True

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
model.eval()

# Open context prompt
with open(args.prompt, "r") as file:
    text = file.read()
    text = ast.literal_eval(text)

num_queries = len(text)

inputs = tokenizer(text, return_tensors="pt").to(device)


torch.cuda.reset_peak_memory_stats()

input_ids = inputs["input_ids"]

# Append assistant flag to prompt. We do this here to handle case with batched inference where an extra "assistant" gets generated because of different sequence lengths padded with EOT.
assistant_header_string = "<|start_header_id|>assistant<|end_header_id|>\n"
assistant_header_tokens = tokenizer(assistant_header_string, add_special_tokens=False, return_tensors="pt")["input_ids"]

batch_size = input_ids.shape[0]
assistant_header_ids = assistant_header_tokens.repeat(batch_size,1).to(device)
input_ids = torch.cat([input_ids, assistant_header_ids], dim=1)


# Model Generation
torch.cuda.synchronize()
nvtx.mark("model.generate")
start_time = time.time()
with torch.no_grad():
    #torch.cuda.synchronize()
    output = model.generate(input_ids, max_new_tokens=200)
end_time = time.time()
nvtx.mark()
torch.cuda.synchronize()

peak_mem = torch.cuda.max_memory_allocated(device)
length = input_ids.shape[1]

num_tokens_generated = len(output[0][length:])
throughput = num_tokens_generated / (end_time - start_time)


output = tokenizer.decode(output[0][length:], skip_special_tokens=True)

print(f"Number of input tokens: {length}")
print(f"Inference time: {end_time - start_time:.3f} seconds")
print(f"Peak memory used: {peak_mem / (1024**3):.2f} GB")
print(f"Number of tokens generated: {num_tokens_generated}")
print(f"Throughput: {throughput:.2f}")
print(f"Output: {[output]}")

if record_in_csv:
    csv_file = "data_collection_output/vanilla_outputs/vanilla_input_token_len_" + str(length) + ".csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "input_tokens", 
            "inference_time_sec", 
            "peak_memory_gb", 
            "tokens_generated", 
            "throughput_tokens_per_sec", 
            "output_text"
        ])
        writer.writerow([
            length, 
            round(end_time - start_time, 3), 
            round(peak_mem / (1024**3), 2), 
            num_tokens_generated, 
            round(throughput, 2), 
            output
        ])
    print("\nRecorded in CSV")