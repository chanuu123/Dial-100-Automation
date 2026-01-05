import os
import time
import datetime
import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper
import pygame
from gtts import gTTS
import ollama

# Ensure recordings folder exists
os.makedirs("recordings", exist_ok=True)

# Initialize pygame mixer once
pygame.mixer.init()

# Loading Whisper model
asr_model = whisper.load_model("small")



# Speak function for text using gTTS + pygame
def speak(text, lang="en"):
    filename = f"reply_{int(time.time()*1000)}.mp3"
    tts = gTTS(text=text, lang=lang)
    tts.save(filename)

    sound = pygame.mixer.Sound(filename)
    channel = sound.play()
    while channel.get_busy():
        pygame.time.Clock().tick(10)

    channel.stop()
    sound.stop()
    try:
        os.remove(filename)
    except PermissionError:
        time.sleep(0.5)
        os.remove(filename)

# Recording audio until silence
def record_audio_silence(samplerate=32000, chunk=1024, silence_threshold=0.01, max_silence_sec=1.0):
    print("🎤 Listening... (start speaking)")
    audio_frames = []
    silent_chunks = 0
    max_silent_chunks = int(max_silence_sec * samplerate / chunk)

    with sd.InputStream(channels=1, samplerate=samplerate, dtype='float32') as stream:
        while True:
            data, _ = stream.read(chunk)
            audio_frames.append(data)
            volume = np.abs(data).mean()
            if volume < silence_threshold:
                silent_chunks += 1
            else:
                silent_chunks = 0
            if silent_chunks > max_silent_chunks:
                break

    audio_np = np.concatenate(audio_frames, axis=0)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"recordings/user_{timestamp}.wav"
    sf.write(filepath, np.squeeze(audio_np), samplerate)
    print(f"💾 Saved recording: {filepath}")
    return filepath


# Transcribing audio using Whisper
def transcribe_audio(audio_file):
    audio, sr = sf.read(audio_file, dtype='float32')  
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)  
    result = asr_model.transcribe(audio, fp16=False)
    text = result["text"].strip()
    lang = result.get("language", "en")
    return text, lang


# Ask Ollama
def ask_ollama(conversation):
    system_prompt = '''
    You are an Emergency Response Assistant for a Public Safety Answering Point (PSAP).

    Your job:
    - Greet the caller politely and ask what happened.
    - Ask follow-up questions step by step to collect:
    1. Location of the incident
    2. Type of incident
    3. Number of injured people
    4. Number of dead people
    5. Whether there is fire
    - Be conversational and natural, like a real call taker.
    - Respond in the same language the caller is using (Hindi or English).
    - Keep replies short (1–2 sentences).
    - Once you have all details, summarize them back to the caller and say that help is on the way.
    - Do not give rescue/medical advice.
    '''

    # Format messages for Ollama
    messages = [{"role": "system", "content": system_prompt}] + conversation

    response = ollama.chat(model="gemma3:12b", messages=messages)
    return response["message"]["content"]


# Save report to a separate file per incident
def save_report(report):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    reports_dir = "incident_reports"
    os.makedirs(reports_dir, exist_ok=True)
    filename = os.path.join(reports_dir, f"incident_report_{ts.replace(':','-')}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"--- Emergency Call Report {ts} ---\n\n")
        for msg in report["conversation"]:
            role = msg["role"].title()
            content = msg["content"]
            f.write(f"{role}: {content}\n")
    print(f"💾 Conversation saved to {filename}")


def emergency_bot():
    # Initializing conversation report
    report = {
        "conversation": []  # storing full conversation
    }

    speak("Hello, this is the emergency helpline. Please tell me what happened.", "en")

    while True:
        # Recording caller audio
        audio_file = record_audio_silence()
        user_text, lang = transcribe_audio(audio_file)
        if not user_text.strip():
            speak("I did not catch that. Please repeat.", lang)
            continue

        print(f"👤 Caller: {user_text}")
        # Appending user message to conversation
        report["conversation"].append({"role": "user", "content": user_text})

        # Generating bot reply
        bot_reply = ask_ollama(report["conversation"])
        
        print(f"🤖 Assistant: {bot_reply}")
        speak(bot_reply, "hi" if lang == "hi" else "en")

        # Appending bot message to conversation
        report["conversation"].append({"role": "assistant", "content": bot_reply})

        # Checking if bot indicates end of conversation
        if "help is on the way" in bot_reply.lower():
            break

    # Final closing statement
    speak("Thank you for calling. Help is on the way.", "en")

    # Save conversation to file
    save_report(report)

    return report





if __name__ == "__main__":
    emergency_bot()
