import speech_recognition as sr
from transformers import pipeline, AutoTokenizer

try:
    summarizer = pipeline("summarization", model="google/pegasus-large")
    tokenizer = AutoTokenizer.from_pretrained("google/pegasus-large")
    print("google/pegasus-large model loaded successfully.")
except OSError as e:
    print(f"Error loading google/pegasus-large model: {e}. Please ensure you have an internet connection.")
    summarizer = None
    tokenizer = None

def summarize_text(text):
    """Generates a summary from the given text."""
    if summarizer is None:
        return "Summarization model not loaded."
    try:
        summary = summarizer(text, max_length=50, min_length=40, do_sample=True)[0]['summary_text']
        return summary
    except Exception as e:
        print(f"Error during summarization: {e}")
        return "Error generating summary."

def summarize_long_text(text, max_token_length=800):
    """Summarizes long text by splitting it into chunks based on tokens."""
    if summarizer is None or tokenizer is None:
        return "Summarization model or tokenizer not loaded."
    sentences = text.split(". ")
    chunks = []
    current_chunk = ""
    current_token_count = 0
    for sentence in sentences:
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        sentence_token_count = len(sentence_tokens)

        if current_token_count + sentence_token_count < max_token_length:
            current_chunk += sentence + ". "
            current_token_count += sentence_token_count + 1
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
            current_token_count = sentence_token_count + 1

    if current_chunk:
        chunks.append(current_chunk.strip())

    summaries = []
    for chunk in chunks:
        try:
            summaries.append(summarize_text(chunk))
        except Exception as e:
            print(f"Error summarizing chunk: {e}")
            summaries.append("Error summarizing chunk.")
    return " ".join(summaries)