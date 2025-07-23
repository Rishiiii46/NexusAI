import os
import speech_recognition as sr
import pygame
import google.generativeai as genai
from dotenv import load_dotenv
from gtts import gTTS
import nltk
import threading

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("First time setup: Downloading NLTK 'punkt' model...")
    nltk.download('punkt')
    print("Download complete.")

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("API key not found. Please set the GOOGLE_API_KEY environment variable.")
genai.configure(api_key=api_key)

def initialize_pygame():
    pygame.init()
    pygame.mixer.init()
    print("Pygame initialized successfully.")

def speak(text, stop_event):
    stop_event.clear()
    
    try:
        sentences = nltk.sent_tokenize(text)
    except Exception as e:
        print(f"NLTK sentence tokenization failed: {e}. Falling back to simple split.")
        sentences = text.split('.')

    for sentence in sentences:
        if stop_event.is_set():
            break
        
        sentence = sentence.strip()
        if not sentence:
            continue
            
        try:
            tts = gTTS(text=sentence, lang='en')
            speech_file = f'response_{threading.get_ident()}.mp3'
            tts.save(speech_file)
            
            pygame.mixer.music.load(speech_file)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                if stop_event.is_set():
                    pygame.mixer.music.stop()
                    break
                pygame.time.Clock().tick(10)
                
        except Exception as e:
            print(f"Error in text-to-speech for sentence: '{sentence}'. Error: {e}")
        finally:
            if os.path.exists(speech_file):
                if pygame.mixer.music.get_busy():
                     pygame.mixer.music.stop()
                pygame.mixer.music.unload()
                os.remove(speech_file)

def recognize_speech_from_microphone(recognizer, microphone):
    with microphone as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=0.2)
        print("Listening...")
        
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            return None

    print("Processing speech...")
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        print(f"API unavailable: {e}")
        return None

def main():
    initialize_pygame()

    model = genai.GenerativeModel('gemini-1.5-flash')
    chat = model.start_chat(history=[])

    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    stop_event = threading.Event()
    speak_thread = None

    speak("Hello! I am ready. How can I assist you?", stop_event)

    while True:
        user_text = recognize_speech_from_microphone(recognizer, microphone)

        if not user_text:
            continue

        print(f"You said: {user_text}")

        is_speaking = speak_thread and speak_thread.is_alive()
        
        exit_phrases = ("stop", "exit", "goodbye", "that's all")
        is_exit_command = any(phrase in user_text.lower() for phrase in exit_phrases)

        if is_exit_command:
            if is_speaking:
                print("--- Interrupting speech. ---")
                stop_event.set()
            else:
                print("--- Exit phrase detected. Shutting down. ---")
                break 
        else:
            
            if is_speaking:
                print("--- New command received, stopping current speech. ---")
                stop_event.set()
                speak_thread.join() 
            try:
                response = chat.send_message(user_text)
                ai_response = response.text
                print(f"Gemini says: {ai_response}")
                
                speak_thread = threading.Thread(target=speak, args=(ai_response, stop_event))
                speak_thread.start()
            except Exception as e:
                print(f"An error occurred with the AI service: {e}")
            
    if speak_thread and speak_thread.is_alive():
        stop_event.set()
        speak_thread.join()
        
    speak("Goodbye!", stop_event)
    pygame.quit()


if __name__ == "__main__":
    main()