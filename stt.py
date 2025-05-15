import os
import zipfile
import numpy as np
import sounddevice as sd
import traceback
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# === Configuration ===
MODEL_ZIP_PATH = r"O:\FinalYearProject\myenv\src\whisper_trained_model_complete.zip"
EXTRACTED_MODEL_PATH = r"O:\FinalYearProject\myenv\src\whisper_trained_model_complete"
SAMPLE_RATE = 16000


def extract_model(zip_path, extract_path):
    """Extracts the model from a ZIP file if it hasn't already been extracted."""
    if not os.path.exists(extract_path):
        print(f"Extracting model from {zip_path} to {extract_path}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            print("Model extracted successfully.")
            return True
        except FileNotFoundError:
            print(f"Error: Zip file not found at {zip_path}")
            return False
        except Exception as e:
            print(f"Error extracting model: {e}")
            return False
    else:
        print(f"Model already extracted at {extract_path}.")
        return True


def check_model_files(model_path):
    """Check if essential Whisper model files exist."""
    required_files = [
        "config.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "generation_config.json"
    ]
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            print(f"Missing model file: {file}")
            return False
    return True


def transcribe_from_microphone_whisper(record_duration=15):
    """
    Records audio from the microphone and transcribes it using the Whisper model.

    Args:
        record_duration (int): Duration to record audio in seconds.

    Returns:
        str | None: Transcribed text, or None if an error occurs.
    """
    if not extract_model(MODEL_ZIP_PATH, EXTRACTED_MODEL_PATH):
        return None

    if not check_model_files(EXTRACTED_MODEL_PATH):
        print("Model files are incomplete or corrupted.")
        return None

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        processor = WhisperProcessor.from_pretrained(EXTRACTED_MODEL_PATH)
        model = WhisperForConditionalGeneration.from_pretrained(EXTRACTED_MODEL_PATH).to(device)
        model.eval()

        print(f"Recording for {record_duration} seconds at {SAMPLE_RATE} Hz...")
        recorded_data = sd.rec(
            int(SAMPLE_RATE * record_duration),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='int16'
        )
        sd.wait()
        print("Finished recording.")

        # Normalize and inspect audio
        audio_array = recorded_data.flatten().astype(np.float32) / 32768.0
        print(f"Audio shape: {audio_array.shape}, dtype: {audio_array.dtype}")
        print(f"Min: {np.min(audio_array)}, Max: {np.max(audio_array)}")

        if np.max(np.abs(audio_array)) < 0.01:
            print("Warning: Audio may be too quiet or silent.")

        # Prepare inputs
        inputs = processor(
            audio_array,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            language="en",
            task="transcript"  # Forces translation to English
        )

        input_features = inputs.input_features.to(device)

        # Manually create attention_mask (all ones, since there's no padding)
        attention_mask = torch.ones(input_features.shape, dtype=torch.long).to(device)

        # Generate transcription
        predicted_ids = model.generate(input_features, attention_mask=attention_mask)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        final_text = transcription[0].strip()

        print("Transcription:", final_text)
        return final_text

    except Exception:
        print("Error during transcription:")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    transcript = transcribe_from_microphone_whisper(record_duration=15)
    if transcript:
        print("Final Transcript:", transcript)
