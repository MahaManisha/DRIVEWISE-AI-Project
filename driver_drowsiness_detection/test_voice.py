
import pyttsx3
import time

def test_voice():
    print("Initializing TTS Engine...")
    try:
        engine = pyttsx3.init()
        print("TTS Initialized.")
    except Exception as e:
        print(f"Failed to initialize TTS: {e}")
        return

    try:
        voices = engine.getProperty('voices')
        print(f"Found {len(voices)} voices.")
        for voice in voices:
            print(f" - {voice.name}")
    except Exception as e:
        print(f"Could not list voices: {e}")

    print("Testing speech...")
    try:
        engine.say("Testing voice alert system. If you can hear this, the voice module is working.")
        engine.runAndWait()
        print("Speech command finished.")
    except Exception as e:
        print(f"Speech failed: {e}")

if __name__ == "__main__":
    test_voice()
