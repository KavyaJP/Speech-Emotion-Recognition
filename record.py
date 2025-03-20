import pyaudio
import wave
import tkinter as tk
from tkinter import messagebox


def on_mic_pressed():
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 5
    OUTPUT_FILENAME = "output.wav"

    audio = pyaudio.PyAudio()

    stream = audio.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )

    messagebox.showinfo("Mic Button", "Recording Started")

    frames = []

    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    messagebox.showinfo("Mic Button", "Recording finished")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open(OUTPUT_FILENAME, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()


root = tk.Tk()
root.title("Mic Button")
root.geometry("200x200")
root.configure(background="black")

micbutton = tk.PhotoImage(file="mic_icon.png")
mic_button_label = tk.Button(root, image=micbutton, command=on_mic_pressed)
mic_button_label.pack()

root.mainloop()
