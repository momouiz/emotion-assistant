import cv2
import tkinter as tk
from PIL import Image, ImageTk
from fer import FER
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
import threading
import sounddevice as sd
import numpy as np
import soundfile as sf
import speech_recognition as sr
import pyttsx3
import random

# --- video emotion stuff ---
emotion_detector = FER(mtcnn=False)
cap = cv2.VideoCapture(0)
frame_count = 0
last_results = []
latest_video_emotion = "neutral"

def smooth_box(old_box, new_box, alpha=0.5):
    if old_box is None:
        return new_box
    smoothed = []
    for o, n in zip(old_box, new_box):
        smoothed.append(int(alpha * n + (1 - alpha) * o))
    return smoothed

# --- main window ---
window = tk.Tk()
window.title("Mental Health Support Bot")
window.configure(bg="#121212")

video_label = tk.Label(window, bg="#121212")
video_label.pack(padx=10, pady=10)

chat_frame = tk.Frame(window, bg="#121212")
chat_frame.pack(pady=(0, 10))

chat_log = tk.Text(chat_frame, width=60, height=15, bg="#1e1e1e", fg="white", state="disabled", wrap="word")
chat_log.pack(side="left")

scrollbar = tk.Scrollbar(chat_frame, command=chat_log.yview)
scrollbar.pack(side="right", fill="y")
chat_log.config(yscrollcommand=scrollbar.set)

audio_label = tk.Label(window, text="Audio Emotion: None", font=("Helvetica", 14), fg="white", bg="#121212")
audio_label.pack(pady=(0, 10))

voice_button = tk.Button(window, text="🎤 Speak", font=("Helvetica", 14))
voice_button.pack(pady=(0, 10))

robot_label = tk.Label(window, text="🤖", font=("Helvetica", 60), bg="#121212")
robot_label.pack()

# --- confidence graph ---
fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
fig.patch.set_facecolor("#121212")
ax.set_facecolor("#1e1e1e")
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_color("white")

emotion_history = deque(maxlen=20)
time_steps = list(range(-19, 1))
line, = ax.plot(time_steps, [0] * 20, color="cyan")
ax.set_ylim(0, 1)
ax.set_ylabel("Confidence", color="white")
ax.set_xlabel("Frame", color="white")

canvas = FigureCanvasTkAgg(fig, master=window)
canvas.get_tk_widget().pack()

# --- speech stuff ---
recognizer = sr.Recognizer()
mic = sr.Microphone()
tts_engine = pyttsx3.init()

conversation_history = deque(maxlen=50)
last_bot_replies = deque(maxlen=3)

replies = {
    "happy": [
        "That's wonderful! What made you feel this way?",
        "I love hearing that! Can you share more?",
        "Awesome! How can you make it last?",
    ],
    "sad": [
        "Feeling down can be tough. Want to talk about it?",
        "I'm here for you. What's making you feel sad?",
        "It's okay to feel sad. Do you want to share why?",
    ],
    "angry": [
        "Anger is valid. What triggered it?",
        "Take a deep breath. Can you tell me why you're angry?",
        "I hear you. Want to vent?",
    ],
    "neutral": [
        "Tell me more about your day.",
        "How are you feeling overall?",
        "I'm listening. What's on your mind?",
    ],
    "surprise": [
        "Wow, that sounds surprising! Tell me more.",
        "That's unexpected! How do you feel about it?",
        "Interesting! What happened next?",
    ],
    "fear": [
        "It's okay to feel scared. Want to talk about it?",
        "I understand. What's worrying you?",
        "I'm here. Can you describe what's frightening?",
    ],
}

# --- video loop ---
def update_graph(emotion_val):
    emotion_history.append(emotion_val)
    data = list(emotion_history) + [0] * (20 - len(emotion_history))
    line.set_ydata(data)
    canvas.draw()

def update_video():
    global frame_count, last_results, latest_video_emotion

    if not window.winfo_exists():
        return

    ret, frame = cap.read()
    if not ret:
        window.after(10, update_video)
        return

    frame = cv2.resize(frame, (640, 480))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if frame_count % 2 == 0:
        results = emotion_detector.detect_emotions(rgb_frame)
        if results:
            smoothed = []
            for i, r in enumerate(results):
                old_box = last_results[i]["box"] if i < len(last_results) else None
                r["box"] = smooth_box(old_box, r["box"])
                smoothed.append(r)
            last_results = smoothed

    frame_count += 1

    top_emotion = "neutral"
    confidence = 0

    for r in last_results:
        x, y, w, h = r["box"]
        emotions = r["emotions"]
        top_emotion = max(emotions, key=emotions.get)
        confidence = emotions[top_emotion]
        latest_video_emotion = top_emotion

        cv2.rectangle(rgb_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            rgb_frame,
            f"{top_emotion} ({confidence:.2f})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    update_graph(confidence)

    img_pil = Image.fromarray(rgb_frame)
    img_tk = ImageTk.PhotoImage(image=img_pil)
    video_label.imgtk = img_tk
    video_label.configure(image=img_tk)

    window.after(30, update_video)

# --- bot logic ---
def bot_reply(user_text, video_emotion, audio_emotion):
    emotion = audio_emotion.lower() if audio_emotion != "neutral" else video_emotion.lower()

    # check if user said something that gives it away
    keywords = ["sad", "happy", "angry", "stressed", "lonely", "fear", "surprise"]
    for word in keywords:
        if word in user_text.lower():
            emotion = word
            break

    if emotion not in replies:
        emotion = "neutral"

    # avoid repeating the same reply 3 times in a row
    options = [r for r in replies[emotion] if r not in last_bot_replies]
    if not options:
        options = replies[emotion]

    reply = random.choice(options)
    last_bot_replies.append(reply)

    conversation_history.append(("User", user_text))
    conversation_history.append(("Bot", reply))

    return reply

def send_message(user_text, video_emotion, audio_emotion):
    chat_log.config(state="normal")
    chat_log.insert("end", f"You: {user_text}\n")

    reply = bot_reply(user_text, video_emotion, audio_emotion)
    chat_log.insert("end", f"Bot: {reply}\n\n")

    chat_log.config(state="disabled")
    chat_log.see("end")

    # speak the reply in a background thread so the UI doesn't freeze
    def speak():
        tts_engine.say(reply)
        tts_engine.runAndWait()

    threading.Thread(target=speak, daemon=True).start()

# --- audio emotion (very basic placeholder) ---
def predict_emotion_audio(audio_data, fs=16000):
    mean_val = np.mean(audio_data)
    if mean_val > 0.01:
        return "happy"
    elif mean_val < -0.01:
        return "sad"
    else:
        return "neutral"

def listen_and_reply():
    global latest_video_emotion

    voice_button.config(text="🎤 Listening...", fg="red")

    chat_log.config(state="normal")
    chat_log.insert("end", "Listening...\n", "info")
    chat_log.tag_config("info", foreground="yellow")
    chat_log.config(state="disabled")
    chat_log.see("end")

    user_text = "(no speech detected)"
    audio_emotion = "neutral"

    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
            try:
                user_text = recognizer.recognize_google(audio)
            except Exception:
                user_text = "(no speech detected)"

        # record a quick clip for audio emotion prediction
        audio_data = sd.rec(int(2 * 16000), samplerate=16000, channels=1)
        sd.wait()
        audio_data = audio_data[:, 0].astype(np.float32)
        audio_emotion = predict_emotion_audio(audio_data)

    except Exception:
        audio_emotion = "neutral"

    audio_label.config(text=f"Audio Emotion: {audio_emotion}")
    send_message(user_text, latest_video_emotion, audio_emotion)
    voice_button.config(text="🎤 Speak", fg="white")

voice_button.config(
    command=lambda: threading.Thread(target=listen_and_reply, daemon=True).start()
)

# --- cleanup on close ---
def on_close():
    cap.release()
    window.destroy()

window.protocol("WM_DELETE_WINDOW", on_close)
update_video()
window.mainloop()
