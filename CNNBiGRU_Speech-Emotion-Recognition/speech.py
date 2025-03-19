import librosa
import soundfile as sf
import os, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

def extract_feature(file_name, max_pad_len=100):
    with sf.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        
        mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
        mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]

    return mfcc

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

def load_data(test_size=0.2):
    X, y = [], []
    dataset_path = r"C:\Users\ranta\OneDrive\Desktop\Python\py_nb Programs\sklearn\project\Dataset\Actor_*"

    for file in glob.glob(os.path.join(dataset_path, "*.wav")):
        file_name = os.path.basename(file)
        emotion = emotions.get(file_name.split("-")[2])

        if emotion not in observed_emotions:
            continue

        feature = extract_feature(file)
        X.append(feature)
        y.append(emotion)

    le = LabelEncoder()
    y = le.fit_transform(y)

    return train_test_split(np.array(X), np.array(y), test_size=test_size, random_state=123), le

class EmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class CNNBiGRU(nn.Module):
    def __init__(self, num_classes):
        super(CNNBiGRU, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.dropout = nn.Dropout(0.3)
        self.bigru = nn.GRU(512, 128, bidirectional=True, batch_first=True)
        self.attention_layer = nn.Linear(256, 1)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def attention(self, gru_output):
        weights = F.softmax(self.attention_layer(gru_output).squeeze(-1), dim=1)
        return torch.sum(gru_output * weights.unsqueeze(-1), dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.dropout(x)

        batch, channel, height, width = x.size()
        x = x.view(batch, width, channel * height)
        gru_out, _ = self.bigru(x)
        attn_out = self.attention(gru_out)
        x = self.fc1(attn_out)
        x = self.fc2(x)
        return self.fc3(x)

def main():
    (X_train, X_test, y_train, y_test), le = load_data(test_size=0.2)

    train_dataset = EmotionDataset(X_train, y_train)
    test_dataset = EmotionDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNBiGRU(num_classes=len(np.unique(y_train))).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0004, weight_decay=0.00008)

    losses = []
    accuracies = []
    epochs = 45

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

        avg_loss = running_loss / len(train_loader)
        acc = correct / total
        losses.append(avg_loss)
        accuracies.append(acc)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {100*acc:.2f}%")

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(y_batch.numpy())
            y_pred.extend(predicted.cpu().numpy())

    test_accuracy = accuracy_score(y_true, y_pred)
    print("Testing Accuracy: {:.2f}%".format(test_accuracy * 100))

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs+1), losses, label="Loss", marker='o', linestyle='-')
    plt.plot(range(1, epochs+1), accuracies, label="Accuracy", marker='s', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Values')
    plt.title('Loss & Accuracy vs Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png", dpi=400, bbox_inches='tight')
    plt.show()

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f"Confusion Matrix | Testing accuracy:{test_accuracy*100:.2f}")
    plt.savefig("Confusion_Matrix.png", dpi=400, bbox_inches='tight')
    plt.show()

    torch.save(model.state_dict(), "emotion_model.pth")

    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

if __name__ == "__main__":
    main()