# Emotion Assistant

A Python desktop app that uses your webcam to detect your facial emotion. You can also talk to it, and it replies with a supportive message out loud.

## What it does

- Reads your webcam with **OpenCV** and detects faces and emotions with **FER** (happy, sad, angry, surprise, fear, neutral).
- Draws a box around your face with the emotion and confidence on it.
- Shows a live graph of the confidence with matplotlib inside the Tkinter window.
- When you click **Speak**, it listens using **SpeechRecognition**, picks a reply based on your emotion and what you said, and reads it out with **pyttsx3**.
- Avoids saying the same reply over and over.
- Runs the speech parts in threads so the window doesn't freeze.

## Why I built it

I wanted to try computer vision and speech recognition together in one project, and make something that responds to how you're feeling instead of just what you type.

## How to run it

You need a webcam and a microphone.

```
pip install -r requirements.txt
python emotion_assistant.py
```

Notes:
- `PyAudio` is needed for the microphone. On Windows `pip install pyaudio` usually works.
- Speech-to-text uses Google's free recognizer, so you need an internet connection. That means your audio is sent to Google.
- The first run takes a while because TensorFlow and the FER model have to load.

## Example

```
[webcam window with a green box: "happy (0.87)"]
You: I had a really good day today
Bot: That's wonderful! What made you feel this way?
Audio Emotion: neutral
```

## What I learned

- Using OpenCV to capture frames and convert them between BGR and RGB so they show properly in Tkinter.
- Running detection every frame was slow, so I only run it every 2nd frame and smooth the face box so it doesn't jitter.
- Threading. Speech recognition and text to speech block, so without threads the whole UI froze.
- The audio emotion part is just a placeholder: it only looks at the average volume, so it isn't very accurate. Real audio emotion detection would need a trained model.

## TODO

- Replace the placeholder audio emotion with a proper model.
- Let you type as well as speak.
- Use an offline speech recogniser (like Vosk) so audio doesn't leave the computer.
