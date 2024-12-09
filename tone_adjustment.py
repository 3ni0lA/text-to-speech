from transformers import pipeline

def conversational_tone(text, tone="neutral", model_name="facebook/bart-large-cnn"):
    prompt = f"Make this sound {tone}: {text}"
    result = pipeline("text-generation", model=model_name)(prompt, max_length=100, num_return_sequences=1)
    return result[0]['generated_text']

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
