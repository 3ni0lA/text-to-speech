import os
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from datasets import load_dataset
import evaluate
import numpy as np
from transformers import pipeline
import torch
from huggingface_hub import login

from preprocessing import preprocess_function
from summarization import summarize_text
from tone_adjustment import conversational_tone, rule_based_tone

# Define constants
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device="cuda")

MODEL_NAME = "facebook/bart-large-cnn"

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Load dataset
dataset = load_dataset("billsum")

# Preprocess dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Create data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Load evaluation metric
rouge = evaluate.load("rouge")

# Compute metrics
def compute_metrics(eval_pred):
    predictions, references = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="my_summarizer_model",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=True,
)

# Initialize trainer with low CPU memory usage
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, low_cpu_mem_usage=True)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Push model to Hugging Face Hub
trainer.push_to_hub()

# Test the summarization function
text = """
Text summarization is the process of distilling the most important information from a source text.
It can help reduce the volume of information while retaining the core meaning.
Summarization is especially useful for news, documents, or lengthy reports.
"""
print("Summary:", summarize_text(text))

# Test the conversational tone function
sample_text = "I am here to help you understand how to create a conversational tone."
conversational_text = conversational_tone(sample_text, tone="friendly")
print("Conversational Tone:", conversational_text)

# Test the rule-based tone function
text = "I do not think this will work."
print("Rule-based Tone:", rule_based_tone(text))