import re
import time
import librosa
import threading
import webbrowser
import subprocess
import sounddevice as sd
from queue import Queue
from playsound import playsound
from melo.api import TTS
from stt.VoiceActivityDetection import VADDetector
from mlx_lm import load, generate
from pydantic import BaseModel

# Note keep this at the bottom to avoid errors. Or fix it and submit a PR
from stt.whisper.transcribe import FastTranscriber

master = """You are a helpful assistant designed to run offline with decent latency, you are open source. Answer the following input from the user in no more than three sentences. Address them as Sir at all times. Only respond with the dialogue, nothing else.
If the user asks you to open maps, include <action>open_maps</action> in your response.
If the user asks you to open news, include <action>open_news</action> in your response.
If the user asks you to build a code project or write code, include the bash commands to create the project inside a <bash>...</bash> block in your response."""


class ChatMLMessage(BaseModel):
    role: str
    content: str


class Client:
    def __init__(self, startListening=True, history: list[ChatMLMessage] = []):
        self.greet()
        self.listening = False
        self.history = history
        self.vad = VADDetector(lambda: None, self.onSpeechEnd, sensitivity=0.3)
        self.vad_data = Queue()
        self.tts = TTS(language="EN_NEWEST", device="mps")
        self.stt = FastTranscriber("mlx-community/whisper-large-v3-mlx-4bit")
        self.model, self.tokenizer = load(
            "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
        )  # want lower ltency? use mlx-community/Phi-3-mini-4k-instruct-8bit

        if startListening:
            self.toggleListening()
            self.startListening()
            t = threading.Thread(target=self.transcription_loop)
            t.start()

    def greet(self):
        print()
        print(
            "\033[36mWelcome to JARVIS-MLX\n\nFollow @huwprosser_ on X for updates\033[0m"
        )
        print()

    def startListening(self):
        t = threading.Thread(target=self.vad.startListening)
        t.start()

    def toggleListening(self):
        if not self.listening:
            print()
            playsound("beep.mp3")
            print("\033[36mListening...\033[0m")

        while not self.vad_data.empty():
            self.vad_data.get()

        self.listening = not self.listening

    def onSpeechEnd(self, data):
        if data.any():
            self.vad_data.put(data)

    def addToHistory(self, content: str, role: str):
        if role == "user":
            print(f"\033[32m{content}\033[0m")
        else:
            print(f"\033[33m{content}\033[0m")

        if role == "user":
            content = f"""{master}\n\n{content}"""
        self.history.append(ChatMLMessage(content=content, role=role))

    def getHistoryAsString(self):
        final_str = ""
        for message in self.history:
            final_str += f"<|{message.role}|>{message.content}<|end|>\n"

        return final_str

    def transcription_loop(self):
        while True:
            if not self.vad_data.empty():
                data = self.vad_data.get()
                if self.listening and len(data) > 12000:
                    self.toggleListening()
                    transcribed = self.stt.transcribe(data, language="en")
                    self.addToHistory(transcribed["text"], "user")

                    history = self.getHistoryAsString()
                    response = generate(
                        self.model,
                        self.tokenizer,
                        prompt=history + "\n<|assistant|>",
                        verbose=False,
                    )
                    response = (
                        response.split("<|assistant|>")[0].split("<|end|>")[0].strip()
                    )
                    self.addToHistory(response, "assistant")

                    cleaned_response = self.parse_and_execute(response)

                    if cleaned_response:
                        self.speak(cleaned_response)
                    else:
                        self.toggleListening()

    def parse_and_execute(self, response: str) -> str:
        # Extract and execute actions
        actions = re.findall(r'<action>(.*?)</action>', response)
        for action in actions:
            if action == 'open_maps':
                print("\033[35mExecuting action: open_maps\033[0m")
                webbrowser.open("https://maps.google.com")
            elif action == 'open_news':
                print("\033[35mExecuting action: open_news\033[0m")
                webbrowser.open("https://news.google.com")

        # Extract and execute bash blocks
        bash_blocks = re.findall(r'<bash>(.*?)</bash>', response, re.DOTALL)
        for bash_script in bash_blocks:
            print(f"\033[35mI would like to run the following bash script:\n{bash_script}\033[0m")
            user_input = input("Should I proceed? (y/n): ")
            if user_input.lower().strip() == 'y':
                try:
                    subprocess.Popen(bash_script, shell=True)
                except Exception as e:
                    print(f"\033[31mError executing bash script: {e}\033[0m")
            else:
                print("\033[33mCommand cancelled by user.\033[0m")

        # Clean the response to be spoken
        cleaned_response = re.sub(r'<action>.*?</action>', '', response)
        cleaned_response = re.sub(r'<bash>.*?</bash>', '', cleaned_response, flags=re.DOTALL)
        return cleaned_response.strip()

    def speak(self, text):
        data = self.tts.tts_to_file(
            text,
            self.tts.hps.data.spk2id["EN-Newest"],
            speed=0.95,
            quiet=True,
            sdp_ratio=0.5,
        )
        trimmed_audio, _ = librosa.effects.trim(data, top_db=20)
        sd.play(trimmed_audio, 44100, blocking=True)
        time.sleep(1)

        self.toggleListening()


if __name__ == "__main__":
    jc = Client(startListening=True, history=[])
