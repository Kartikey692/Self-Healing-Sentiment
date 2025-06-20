from dotenv import load_dotenv
import os
from huggingface_hub import login

load_dotenv()
login(os.environ["HF_TOKEN"])

import numpy as np
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "lora-distilbert-imdb"

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer

def load_and_prepare_data(tokenizer, n_samples=200):
    imdb = load_dataset("imdb")
    idx = np.random.randint(0, 24999, n_samples)
    idx = [int(i) for i in idx]  # Ensure all indices are Python ints
    x_train = [imdb['train'][i]['text'] for i in idx]
    y_train = [imdb['train'][i]['label'] for i in idx]
    x_test = [imdb['test'][i]['text'] for i in idx]
    y_test = [imdb['test'][i]['label'] for i in idx]
    dataset = DatasetDict({
        'train': Dataset.from_dict({'label': y_train, 'text': x_train}),
        'validation': Dataset.from_dict({'label': y_test, 'text': x_test})
    })

    def tokenize_fn(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)
    return dataset.map(tokenize_fn, batched=True)

def main():
    print("Loading tokenizer and data...")
    tokenizer = get_tokenizer()
    data = load_and_prepare_data(tokenizer)
    print("Loading base model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.resize_token_embeddings(len(tokenizer))
    print("Configuring LoRA...")
    lora_config = LoraConfig(task_type="SEQ_CLS", r=4, lora_alpha=32, lora_dropout=0.01, target_modules=["q_lin", "v_lin"])
    model = get_peft_model(model, lora_config)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        learning_rate=1e-3,
        eval_strategy="epoch",             
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=10,
        save_total_limit=1,
        report_to="none"
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    print("Starting fine-tuning...")
    trainer.train()
    print("Saving model and tokenizer...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Fine-tuned model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
