import speech_recognition as sr
# The 'transformers' import is not used in this function,
# so it can be removed if this is the entire script for this specific test.
# from transformers import pipeline, AutoTokenizer

# trialstt.py
def record_and_transcribe(record_duration=30): # Reduced default duration for quicker testing
    """
    Records audio from the default microphone for a specified duration and transcribes it.

    :param record_duration: Duration of recording in seconds (default is 30 seconds).
    :return: The transcribed text as a string, or None if an error occurs.
    """
    r = sr.Recognizer()
    print("Recognizer initialized.") # Added for debugging

    # You can list microphones to see if your designated mic is detected
    try:
        print("Available microphone devices are: ")
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            print(f"Microphone with name \"{name}\" found for `Microphone(device_index={index})`")
    except Exception as e:
        print(f"Could not list microphones: {e}")
        # This might indicate an issue with PortAudio or ALSA setup

    with sr.Microphone() as source:
        print("Microphone source opened.") # Added for debugging
        # Optional: Adjust for ambient noise once, if needed, especially in noisy environments
        # print("Adjusting for ambient noise, please wait...")
        # try:
        #     r.adjust_for_ambient_noise(source, duration=1)
        #     print("Ambient noise adjustment complete.")
        # except Exception as e:
        #     print(f"Error adjusting for ambient noise: {e}")

        print(f"Recording for {record_duration} seconds...")
        try:
            audio = r.record(source, duration=record_duration)
            print("Finished recording.")

            print("Transcribing using Google Web Speech API...")
            # Ensure your Pi has a stable internet connection for this part
            text = r.recognize_google(audio, language="en-IN") # Changed to en-IN for Indian accent
            print("Transcription successful.")
            print("Transcription: ", text) # Print the text here
            return text

        except sr.WaitTimeoutError:
            print("No speech detected within the recording duration while using r.listen(). (Not applicable to r.record with duration)")
            return None
        except sr.UnknownValueError:
            print("Google Web Speech API could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Could not request results from Google Web Speech API; {e}")
            print("This often means there's an issue with your internet connection or API access.")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during recording or transcription: {e}")
            return None
    print("Exiting record_and_transcribe function.") # Should not be reached if return happens earlier
    return None # Fallback

# This is the crucial part:
if __name__ == "__main__":
    print("Script started.")
    transcribed_text = record_and_transcribe(record_duration=10) # Call with a shorter duration for testing
    if transcribed_text:
        print("\nFinal Transcribed Text:")
        print(transcribed_text)
    else:
        print("\nNo transcription received or an error occurred.")
    print("Script finished.")
