from transformers import pipeline

def summarize_text(text, model_name="your-hub-model-name", max_length=50):
    summarizer = pipeline("summarization", model=model_name)
    summary = summarizer(text, max_length=max_length, min_length=25, do_sample=False)
    return summary[0]['summary_text']
