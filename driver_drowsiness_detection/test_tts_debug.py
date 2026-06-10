import pyttsx3
import time

print("Initializing TTS Engine...")
try:
    engine = pyttsx3.init()
    print("Engine initialized.")
except Exception as e:
    print(f"Failed to init engine: {e}")
    exit(1)

print("Testing voice properties...")
try:
    voices = engine.getProperty('voices')
    for v in voices:
        print(f" - Found voice: {v.name}")
    rate = engine.getProperty('rate')
    print(f" - Rate: {rate}")
except Exception as e:
    print(f"Failed to get properties: {e}")

print("Attempting to speak...")
try:
    engine.say("Testing voice alert system based on python.")
    engine.runAndWait()
    print("Speech command completed.")
except Exception as e:
    print(f"Failed to speak: {e}")

print("Test complete.")
