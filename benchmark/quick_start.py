import torch
from qllm.models import LlamaForCausalLM
from transformers import AutoTokenizer
import transformers
import time
import nvtx
from omegaconf import OmegaConf
from qllm.utils import patch_hf, GreedySearch, patch_model_center
from qllm.utils.extract_question import extract_question_id_quick_start
import argparse
import ast
import csv


device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="Parse the arguments for prompt and model configuration")

parser.add_argument('--prompt', type=str, help="Prompt to run inference on")
parser.add_argument('--yaml', type=str, help="Config to run QuickLlama on")
parser.add_argument('--non_query_aware', type=str, help="Using default non query aware mode")
parser.add_argument('--record_csv', type=str, help="Flag to record output into CSV file")


args = parser.parse_args()

conf = OmegaConf.load(args.yaml)
#conf = OmegaConf.load("config/llama3-inf-llm-repr4-l1k-bs128-topk8.yaml")
#conf = OmegaConf.load("config/Llama3-inf-llm-low.yaml")
model_path = conf.model.path
#model_path = "models/Meta-Llama-3-8B-Instruct"
if args.non_query_aware.lower() == "false":
    query_aware = True
else: 
    query_aware = False

record_in_csv = False
if args.record_csv:
    if args.record_csv.lower() == "true":
        record_in_csv = True

model = LlamaForCausalLM.from_pretrained(
	model_path,
	torch_dtype=torch.bfloat16,
	trust_remote_code=True
	).to("cuda:0")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, add_bos_token=True, add_eos_token=False)
tokenizer.pad_token = tokenizer.eos_token


model = patch_hf(model, "qllm", conf.model)
model = GreedySearch(model, tokenizer)


with open(args.prompt, "r") as file:
    text = file.read()
    text = ast.literal_eval(text)
# text = '''


num_queries = len(text)
inputs = tokenizer(
    text,
    return_tensors="pt",
    padding=True,
    truncation=True,
)


input_ids = inputs["input_ids"]

# Append assistant flag to prompt. We do this here to handle case with batched inference where an extra "assistant" gets generated because of different sequence lengths padded with EOT.
assistant_header_string = "<|start_header_id|>assistant<|end_header_id|>\n"
assistant_header_tokens = tokenizer(assistant_header_string, add_special_tokens=False, return_tensors="pt")["input_ids"]

batch_size = input_ids.shape[0]
assistant_header_ids = assistant_header_tokens.repeat(batch_size,1)  # (batch_size, header_len)
input_ids = torch.cat([input_ids, assistant_header_ids], dim=1)

torch.cuda.reset_peak_memory_stats()
    
# Model Generation
if not query_aware:
    print("Execute Non Query Aware")
    torch.cuda.synchronize()
    nvtx.mark("model.generate")
    start_time = time.time()
    with torch.no_grad():
        output, num_generated_tokens = model.generate(input_ids, max_length=200)
    end_time = time.time()
    nvtx.mark()
    torch.cuda.synchronize()
else:
    print("Execute Query Aware")
    if (input_ids.shape[0] > 1):
        print("Error: Batched inference unsupported for query aware mode. Set non_query_aware flag to true.")
        exit(2)
    kwargs = {}
    dataset = "AI_wikipedia"
    kwargs["question_ids"] = extract_question_id_quick_start(dataset, tokenizer, input_ids[0])
    torch.cuda.synchronize()
    nvtx.mark("model.generate")
    start_time = time.time()
    with torch.no_grad():
        output, num_generated_tokens = model.generate(input_ids, max_length=200, **kwargs)
    end_time = time.time()
    nvtx.mark()
    torch.cuda.synchronize()




peak_mem = torch.cuda.max_memory_allocated(device)
length = input_ids.shape[1]
total = length*batch_size

print(f"Number of input tokens: {length*batch_size}")
print(f"Inference time: {end_time - start_time:.3f} seconds")
print(f"Throughput: {num_generated_tokens/(end_time - start_time):.2f} Tokens/s")
print(f"Peak memory used: {peak_mem / (1024**3):.2f} GB")

for i in range(num_queries):
    print(f"Output {i}:")
    print(f"{output[0][i]}\n")

#csv_file = "data_collection_output/non_query_aware_1_query/QLLM-config-high/non_query_aware_input_token_len_" + str(length) + ".csv"
#csv_file = "data_collection_output/query_aware_1_query/QLLM-config-high/query_aware_input_token_len_" + str(length) + ".csv"

#csv_file = "data_collection_output/query_aware_1_query/QLLM-config-medium/query_aware_input_token_len_" + str(length) + ".csv"
#csv_file = "data_collection_output/non_query_aware_1_query/QLLM-config-medium/non_query_aware_input_token_len_" + str(length) + ".csv"

#csv_file = "data_collection_output/non_query_aware_1_query/QLLM-config-lowest/non_query_aware_input_token_len_" + str(length) + ".csv"
#csv_file = "data_collection_output/query_aware_1_query/QLLM-config-lowest/query_aware_input_token_len_" + str(length) + ".csv"


#csv_file = "data_collection_output/non_query_aware_2_query/QLLM-config-lowest/non_query_aware_input_token_len_" + str(total) + ".csv"
#csv_file = "data_collection_output/non_query_aware_2_query/QLLM-config-medium/non_query_aware_input_token_len_" + str(total) + ".csv"
#csv_file = "data_collection_output/non_query_aware_2_query/QLLM-config-highest/non_query_aware_input_token_len_" + str(total) + ".csv"

#csv_file = "data_collection_output/non_query_aware_4_query/QLLM-config-lowest/non_query_aware_input_token_len_" + str(total) + ".csv"
#csv_file = "data_collection_output/non_query_aware_4_query/QLLM-config-medium/non_query_aware_input_token_len_" + str(total) + ".csv"
#csv_file = "data_collection_output/non_query_aware_4_query/QLLM-config-highest/non_query_aware_input_token_len_" + str(total) + ".csv"

csv_file = "data_collection_output/testing.csv"
#csv_file = "testing.csv"
if record_in_csv:
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "input_tokens", 
            "inference_time_sec", 
            "peak_memory_GB", 
            "tokens_generated", 
            "throughput_tokens_per_sec"
        ])
        writer.writerow([
            length, 
            round(end_time - start_time, 3), 
            round(peak_mem / (1024**3), 2), 
            num_generated_tokens, 
            round(num_generated_tokens/(end_time - start_time), 2), 
        ])
        for i in range(num_queries):
            output_string = "Output " + str(i)
            writer.writerow([output_string,
            output[0][i]])
    print("\nRecorded in CSV")

model.clear()
