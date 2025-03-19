import soundfile as sf
import torch
import sounddevice as sd
import pickle

from speech import extract_feature, CNNBiGRU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

observed_emotions = ['calm', 'happy', 'fearful', 'disgust', 'sad', 'angry', 'surprised', 'neutral']

def record_and_predict(seconds:int):
    print(f"Recording for {seconds} seconds...")
    recording = sd.rec(int(seconds * 44100), samplerate=44100, channels=1)
    sd.wait()
    sf.write('realtime_audio.wav', recording, 44100)

    feature = extract_feature('realtime_audio.wav')
    feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    model = CNNBiGRU(num_classes=8)
    model.load_state_dict(torch.load("emotion_model.pth", weights_only=True))
    model.to(device)
    model.eval()

    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    with torch.no_grad():
        output = model(feature)
        predicted = torch.argmax(output, dim=1)
        emotion = le.inverse_transform(predicted.cpu().numpy())[0]
        print(f"Predicted Emotion: {emotion}")

record_and_predict(5)