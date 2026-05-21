from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)

OUTPUT_DIR   = "./output"

import httpx
from huggingface_hub import set_client_factory

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")

def my_client_factory():
    return httpx.Client(
        verify=False, # Disable SSL verification if needed
        timeout=60.0
    )

set_client_factory(my_client_factory)

dataset = load_dataset("dany0407/eli5_category", split="train[:5000]")

eli5 = dataset.flatten()

def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["answers.text"]],truncation=True, max_length=1024)

block_size = 128

def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) if isinstance(examples[k][0], list) else examples[k] for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size

    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }

    result["labels"] = result["input_ids"].copy()
    return result

# Step 1: tokenize
tokenized_eli5 = eli5.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=eli5.column_names
)


# Step 2: group into blocks
tokenized_eli5 = tokenized_eli5.map(
    group_texts,
    batched=True,
    num_proc=4,
)

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="no",
    learning_rate=2e-5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_eli5,
    data_collator=data_collator,
    processing_class=tokenizer,
)

trainer.train()
