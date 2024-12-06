import os
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
from datasets import load_dataset
import evaluate
import numpy as np
import torch
from huggingface_hub import login

# login("3niola", "hf")

# Define constants
MODEL_NAME = "facebook/bart-large-cnn"
SUMMARIZATION_TASK = "summarization"

# Function to preprocess data
def preprocess_function(examples):
    inputs = [f"{SUMMARIZATION_TASK}: {doc}" for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

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
    num_train_epochs=4,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=True,
)

# Initialize trainer
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

# Function to generate summary
def summarize_text(text, max_length=50):
    summarizer = pipeline(SUMMARIZATION_TASK, model="your-hub-model-name")
    summary = summarizer(text, max_length=max_length, min_length=25, do_sample=False)
    return summary[0]['summary_text']

# Test the summarization function
text = """
Text summarization is the process of distilling the most important information from a source text.
It can help reduce the volume of information while retaining the core meaning.
Summarization is especially useful for news, documents, or lengthy reports.
"""
print("Summary:", summarize_text(text))

# Function to add conversational tone
def conversational_tone(text, tone="neutral"):
    model_name = f"facebook/bart-large-cnn"
    prompt = f"Make this sound {tone}: {text}"
    result = pipeline("text-generation", model=model_name)(prompt, max_length=100, num_return_sequences=1)
    return result[0]['generated_text']

# Test the conversational tone function
sample_text = "I am here to help you understand how to create a conversational tone."
conversational_text = conversational_tone(sample_text, tone="friendly")
print("Conversational Tone:", conversational_text)

# Function to add rule-based contractions
contractions = {
    "do not": "don't",
    "will not": "won't",
    "cannot": "can't",
    "it is": "it's",
    "you are": "you're"
}

def rule_based_tone(text):
    for key, value in contractions.items():
        text = text.replace(key, value)
    return text

# Test the rule-based tone function
text = "I do not think this will work."
print("Rule-based Tone:", rule_based_tone(text))
