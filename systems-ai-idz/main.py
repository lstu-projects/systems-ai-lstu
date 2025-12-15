import whisper
import librosa
import noisereduce as nr
import numpy as np

print("Загрузка аудиофайла...")
y, sr = librosa.load("I_Hate_Everything_About_You_-_Three_Days_Grace.mp3", sr=16000, mono=True)


print("Снижение шума...")
y = nr.reduce_noise(y=y, sr=sr)

print("Загрузка модели large")
model = whisper.load_model("large")

print("Распознавание текста...")
result = model.transcribe(
    y.astype(np.float32),
    language="en",
    temperature=0.0,
    beam_size=5,
    verbose=True
)

print("\n=== РЕЗУЛЬТАТ ===")
print(result['text'])

with open("recognized_text.txt", "w", encoding="utf-8") as f:
    f.write(result['text'])
