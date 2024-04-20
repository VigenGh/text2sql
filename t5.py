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

data = pd.read_csv("text-to-sql_from_spider.csv")
# print(data)

dataset = load_dataset("csv", data_files="text-to-sql_from_spider.csv")
dataset = dataset["train"].train_test_split(test_size=0.4)
test_dataset = dataset["test"].train_test_split(test_size=0.5)
print(dataset["train"])
dataset = DatasetDict({"train": dataset["train"],
                       "test": test_dataset["test"],
                       "validation": test_dataset["train"]})


def tokenize_function(example):

#     print(len(example["question"]))
    start_prompt = "Tables:\n"
    middle_prompt = "\n\nQuestion:\n"
    end_prompt = "\n\nAnswer:\n"

    data_zip = zip(example['schema'], example['question'])
    prompt = [start_prompt + context + middle_prompt + question + end_prompt for context, question in data_zip]
    example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
    example['labels'] = tokenizer(example['sql'], padding="max_length", truncation=True, return_tensors="pt").input_ids
#     print(prompt[0])
#     print()

    return example

try:
    tokenized_datasets = load_from_disk("tokenized_datasets")
    print("Loaded Tokenized Dataset")
except:
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(['sql', 'question', 'schema'])

    tokenized_datasets.save_to_disk("tokenized_datasets")
    print("Tokenized and Saved Dataset")
# tokenized_datasets = dataset.map(tokenize_function, batched=True)

# print(tokenized_datasets["train"][0]["input_ids"])


try:
    finetuned_model = AutoModelForSeq2SeqLM.from_pretrained("finetuned_model_2_epoch")
    finetuned_model = finetuned_model.to('cuda')
    to_train = False

except:
    to_train = True
    finetuned_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    finetuned_model = finetuned_model.to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

if to_train:
    output_dir = f'./sql-training-{str(int(time.time()))}'

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=5e-3,
        num_train_epochs=2,
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=16,  # batch size for evaluation
        weight_decay=0.01,
        logging_steps=50,
        evaluation_strategy='steps',  # evaluation strategy to adopt during training
        eval_steps=500,  # number of steps between evaluation
    )

    trainer = Trainer(
        model=finetuned_model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
    )

    trainer.train()

    finetuned_model.save_pretrained("finetuned_model_2_epoch")

questions = dataset['test']['question']
contexts = dataset['test']['schema']
human_baseline_answers = dataset['test']['sql']

original_model_answers = []
finetuned_model_answers = []

for idx, question in enumerate(questions):
    prompt = f"""Tables:
{contexts[idx]}

Question:
{question}

Answer:
"""

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to('cuda')

    human_baseline_text_output = human_baseline_answers[idx]

    original_model_outputs = original_model.generate(input_ids=input_ids,
                                                     generation_config=GenerationConfig(max_new_tokens=300))
    original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)
    original_model_answers.append(original_model_text_output)

    finetuned_model_outputs = finetuned_model.generate(input_ids=input_ids,
                                                       generation_config=GenerationConfig(max_new_tokens=300))
    finetuned_model_text_output = tokenizer.decode(finetuned_model_outputs[0], skip_special_tokens=True)
    finetuned_model_answers.append(finetuned_model_text_output)

zipped_summaries = list(zip(human_baseline_answers, original_model_answers, finetuned_model_answers))

df = pd.DataFrame(zipped_summaries,
                  columns=['human_baseline_answers', 'original_model_answers', 'finetuned_model_answers'])

rouge = evaluate.load('rouge')

original_model_results = rouge.compute(
    predictions=original_model_answers,
    references=human_baseline_answers[0:len(original_model_answers)],
    use_aggregator=True,
    use_stemmer=True,
)
print('ORIGINAL MODEL:')
print(original_model_results)


finetuned_model_results = rouge.compute(
    predictions=finetuned_model_answers,
    references=human_baseline_answers[0:len(finetuned_model_answers)],
    use_aggregator=True,
    use_stemmer=True,
)
print('FINE-TUNED MODEL:')
print(finetuned_model_results)
