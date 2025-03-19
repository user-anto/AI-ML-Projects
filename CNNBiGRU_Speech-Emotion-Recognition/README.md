# CNN-BiGRU for Speech Emotion Recognition

## Overview
This repository contains an implementation of a **CNN-BiGRU model** for **Speech Emotion Recognition**, based on the research paper _"CNN-BiGRU Speech Emotion Recognition Based on Attention Mechanism"_. The model leverages convolutional neural networks (CNNs) for feature extraction and bidirectional gated recurrent units (BiGRUs) for capturing sequential dependencies in speech data.

## Features
- MFCC based feature extraction and converting the speech signal into 2D data, in the **extract_feature** function.
- Implements a **CNN-BiGRU** architecture for speech emotion classification, where **CNN layers** perform feature extraction and **BiGRU** performs temporal modelling. The convolution layers used in this model are 2D blocks as suggested in the paper _"Speech Emotion Recognition using Convolutional and Recurrent Neural Networks"_, except MFCC based feature extraction was observed to be more efficient than short-time fourier transform.
- Uses an **attention mechanism** for weighted feature importance.
- Trained and evaluated on the RAVDESS Dataset. This dataset may be used only for research purposes. For real-life applications a more diverse dataset is necessary. Please consider contributing to a new dataset [DECS](https://docs.google.com/forms/d/e/1FAIpQLSeX_uB4qKX7kWhard0iXRlClv7XfucTdFCupmxVSrC3hP6RDA/viewform?usp=dialog) that will enable this model to provide more accurate results.
- Provides performance metrics such as **confusion matrix** and **loss curve**.

## Dataset
This implementation can be trained on publicly available **speech emotion datasets** such as:
- **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **CREMA-D** (Crowd-sourced Emotional Multimodal Actors Dataset)
- **IEMOCAP** (Interactive Emotional Dyadic Motion Capture)

## Installation
### Prerequisites
Ensure you have the following dependencies installed:
```bash
pip install librosa soundfile glob numpy torch scikit-learn seaborn matplotlib pickle
```

## Usage
### Training the Model
To train the model on the RAVDESS dataset and check its performance on the 20% of the dataset as test data, run:
```bash
python speech.py
```
### Evaluating the Model
To test the model or real time voice execute:
```bash
python testing.py
```
When the program outputs "Recording for 5 seconds...", speak into the microphone and the program will evaluate the emotion detected in the voice recording.

## Explanation of Key Components
### Feature Extraction: `extract_feature`
```python
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
```
#### Explanation:
- **Reads the audio file** using `SoundFile`.
- **Extracts 40 MFCC features** using `librosa.feature.mfcc`.
- **Normalizes the MFCCs** to ensure zero mean and unit variance.
- **Pads or truncates** the MFCC matrix to a fixed length of `max_pad_len=100`.

### CNN-BiGRU Model
```python
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
```
#### Explanation:
- **CNN Layers (`conv1` to `conv5`)** extract high-level spatial features from the MFCC input.
- **BiGRU Layer (`bigru`)** captures temporal dependencies in the feature sequence.
- **Attention Mechanism (`attention`)** assigns weights to GRU outputs to focus on the most relevant information.
- **Fully Connected Layers (`fc1`, `fc2`, `fc3`)** classify emotions based on the extracted and processed features.

## Results
- The **confusion matrix** and **loss curve** are recorded in Confusion_Matrix.png and loss_curve.png.

## References
- **Paper:** [CNN-BiGRU Speech Emotion Recognition Based on Attention Mechanism] and [Speech Emotion Recognition using Convolutional and Recurrent Neural Networks]
- **Dataset:** [RAVDESS](https://zenodo.org/record/1188976)

## Conclusion
This implementation of a **CNN-BiGRU with Attention Mechanism** provides an effective way to classify speech emotions. The **combination of CNN for feature extraction, BiGRU for sequential modeling, and attention for focusing on important features** makes it a good approach approach for **speech emotion recognition**. Although, there is a lot of scope for improvement, training on more diverse data.
