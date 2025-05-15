import speech_recognition as sr
from transformers import pipeline, AutoTokenizer

# trialstt.py
def record_and_transcribe(record_duration=300):
    """
    Records audio from the default microphone for a specified duration and transcribes it.

    :param record_duration: Duration of recording in seconds (default is 30 seconds).
    :return: The transcribed text as a string, or None if an error occurs.
    """
    r = sr.Recognizer()

    with sr.Microphone() as source:
        print(f"Recording for {record_duration} seconds...")
        try:
            audio = r.record(source, duration=record_duration)
            print("Finished recording.")

            print("Transcribing...")
            text = r.recognize_google(audio, language="en-gb")
            print("Transcription:")
            return text

        except sr.UnknownValueError:
            print("Google Web Speech API could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Could not request results from Google Web Speech API; {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None