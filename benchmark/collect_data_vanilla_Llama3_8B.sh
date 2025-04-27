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
)

for i in "${!files[@]}"
do
    echo "Running inference on ${files[$i]}..."
    python vanilla_Llama3_8B_Instruct_Basic_Profile.py --prompt "prompts/non_query_aware/${files[$i]}" --record_csv true
    echo "Cooldown for 15 seconds..."
    sleep 15
done

echo "All runs completed."
