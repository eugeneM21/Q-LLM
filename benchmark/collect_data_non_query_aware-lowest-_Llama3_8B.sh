#!/bin/bash

files=(
  "AI_wikipedia_299_tokens.txt"
  "AI_wikipedia_624_tokens.txt"
  "AI_wikipedia_1111_tokens.txt"
  "AI_wikipedia_2037_tokens.txt"
  "AI_wikipedia_5646_tokens.txt"
  "AI_wikipedia_6253_tokens.txt"
  "AI_wikipedia_7094_tokens.txt"
  "AI_wikipedia_7441_tokens.txt"
  "AI_wikipedia_7531_tokens.txt"
  "AI_wikipedia_7623_tokens.txt"
  "AI_wikipedia_7663_tokens.txt"
  "AI_wikipedia_7725_tokens.txt"
  "AI_wikipedia_7856_tokens.txt"
  "AI_wikipedia_8018_tokens.txt"
  "AI_wikipedia_8284_tokens.txt"
  "AI_wikipedia_8494_tokens.txt"
  "AI_wikipedia_8577_tokens.txt"
  "AI_wikipedia_8838_tokens.txt"
  "AI_wikipedia_8945_tokens.txt"
  "AI_wikipedia_9230_tokens.txt"
  "AI_wikipedia_10096_tokens.txt"
  "AI_wikipedia_15275_tokens.txt"
  "AI_wikipedia_17011_tokens.txt"
)


for i in "${!files[@]}"
do
    echo "Running inference on ${files[$i]}..."
    python quick_start.py --prompt "prompts/non_query_aware/${files[$i]}" --yaml "config/Llama3-inf-llm-low-low.yaml" --non_query_aware "True" --record_csv true
    echo "Cooldown for 15 seconds..."
    sleep 15
done

echo "All runs completed."
