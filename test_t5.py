from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, load_dataset, interleave_datasets, load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import time
import evaluate
import pandas as pd
import numpy as np
model_name = 't5-small'

tokenizer = AutoTokenizer.from_pretrained(model_name)

original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
original_model = original_model.to('cuda')

finetuned_model = AutoModelForSeq2SeqLM.from_pretrained("finetuned_model_2_epoch")
finetuned_model = finetuned_model.to('cuda')
data = pd.read_csv("text-to-sql_from_spider.csv")

question = data["question"][0] #dataset['test'][index]['question']
context = "CREATE TABLE table_name_11 (date VARCHAR, away_team VARCHAR)" #dataset['test'][index]['schema']
answer = data["sql"][0] #dataset['test'][index]['sql']

prompt = f"""Tables:
{context}

Question:
{question}

Answer:
"""

inputs = tokenizer(prompt, return_tensors='pt')
inputs = inputs.to('cuda')

output = tokenizer.decode(
    finetuned_model.generate(
        inputs["input_ids"],
        max_new_tokens=200,
    )[0],
    skip_special_tokens=True
)

dash_line = '-'*100
print(dash_line)
print(f'INPUT PROMPT:\n{prompt}')
print(dash_line)
print(f'BASELINE HUMAN ANSWER:\n{answer}\n')
print(dash_line)
print(f'MODEL GENERATION - ZERO SHOT:\n{output}')