import re

def add_natural_pauses(text):
    """
    Adds pauses by inserting additional spaces after punctuation marks.
    """
    # Add a slight pause after commas and periods
    text = re.sub(r'([,.!?])\s*', r'\1 ', text)
    return text

def simplify_language(text):
    """
    Simplifies formal language to make it sound more conversational.
    """
    # Replace formal words with casual counterparts
    text = re.sub(r'\bdo not\b', "don't", text, flags=re.IGNORECASE)
    text = re.sub(r'\bwill not\b', "won't", text, flags=re.IGNORECASE)
    text = re.sub(r'\bare not\b', "aren't", text, flags=re.IGNORECASE)
    text = re.sub(r'\bI am\b', "I'm", text, flags=re.IGNORECASE)
    text = re.sub(r'\bis not\b', "isn't", text, flags=re.IGNORECASE)
    text = re.sub(r'\bhave to\b', "need to", text, flags=re.IGNORECASE)
    text = re.sub(r'\bfor instance\b', "for example", text, flags=re.IGNORECASE)
    text = re.sub(r'\btherefore\b', "so", text, flags=re.IGNORECASE)
    
    # You can add more replacements as needed
    return text

def text_segmentation(text, max_length=100):
    """
    Splits the text into smaller chunks based on a maximum length.
    """
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    segmented_text = []
    chunk = ""
    
    for sentence in sentences:
        if len(chunk) + len(sentence) <= max_length:
            chunk += sentence + " "
        else:
             # Capitalize only the first letter of each chunk
            chunk = chunk.strip()
            if chunk:
                chunk = chunk[0].upper() + chunk[1:]  # Capitalize first letter only
            segmented_text.append(chunk)
            chunk = sentence + " "

    # Add the last chunk
    chunk = chunk.strip()
    if chunk:
        chunk = chunk[0].upper() + chunk[1:]  # Capitalize first letter only
        segmented_text.append(chunk)
    return segmented_text

# Main Function to Preprocess Text
def preprocess_text(text):
    # Step 1: Add Natural Pauses
    text = add_natural_pauses(text)
    
    # Step 2: Simplify Language
    text = simplify_language(text)
    
    # Step 3: Segment Text
    segmented_text = text_segmentation(text)
    
    return segmented_text

# Example usage:
sample_text = """Hello, I am here to demonstrate how natural language processing can help. 
                 Do not worry if this seems complicated. We will break it down step by step."""

processed_text = preprocess_text(sample_text)
print("Processed Text for TTS:")
for segment in processed_text:
    print(segment)
