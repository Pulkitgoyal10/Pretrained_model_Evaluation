# generate_data.py
import time
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
import os

# 1. Setup
models = [
    "sshleifer/distilbart-cnn-12-6", # Lightweight
    "t5-small",                      # Fast/Small
    "facebook/bart-large-cnn",       # High Quality/Heavy
    "google/pegasus-xsum"            # Specialized
]

article = """
Artificial Intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. 
AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions 
that maximize its chance of achieving its goals. The term "artificial intelligence" had previously been used to describe machines that mimic and 
display "human" cognitive skills that are associated with the human mind, such as "learning" and "problem-solving". This definition has since been 
rejected by major AI researchers who now describe AI in terms of rationality and acting rationally, which does not limit how intelligence can be articulated.
"""
reference_summary = "AI is intelligence demonstrated by machines. It is the study of agents that perceive their environment and act to maximize goal achievement."

# 2. Initialize Metrics
rouge = evaluate.load('rouge')
results = []

print(f"{'Model':<30} | {'Status'}")
print("-" * 50)

for model_name in models:
    try:
        # Load Model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Calculate Size (in MB)
        param_size = sum(p.numel() for p in model.parameters())
        size_mb = param_size * 4 / (1024 * 1024) # Assuming float32 (4 bytes)

        # Calculate Inference Time
        inputs = tokenizer(article, return_tensors="pt", max_length=512, truncation=True)
        start_time = time.time()
        # Generate summary
        summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        end_time = time.time()
        inference_time = end_time - start_time
        
        # Calculate ROUGE
        prediction = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        scores = rouge.compute(predictions=[prediction], references=[reference_summary])
        
        results.append({
            "Model": model_name,
            "Rouge-Score": round(scores['rougeL'], 4),  # Benefit (+)
            "Inference-Time": round(inference_time, 4), # Cost (-)
            "Model-Size-MB": round(size_mb, 2)          # Cost (-)
        })
        print(f"{model_name:<30} | Done")
        
    except Exception as e:
        print(f"{model_name:<30} | Failed: {e}")

# 3. Save Data
df = pd.DataFrame(results)
df.to_csv("data.csv", index=False)
print("\nSuccess! 'data.csv' created.")
print(df)