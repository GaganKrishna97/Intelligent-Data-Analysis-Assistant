# NOTE: Needs streamlit-webrtc or streamlit-audio-components for real use!
import speech_recognition as sr

def transcribe_wav(file_bytes):
    recognizer = sr.Recognizer()
    with sr.AudioFile(BytesIO(file_bytes)) as source:
        audio = recognizer.record(source)
    return recognizer.recognize_google(audio)
