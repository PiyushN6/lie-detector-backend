import torch
import torch.nn as nn
import numpy as np
import librosa

class AudioModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

    def extract_features(self, audio_path):
        y, sr = librosa.load(audio_path, sr=None, duration=5.0)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        return mfcc.mean(axis=1)

    def predict_deception(self, audio_path):
        self.eval()
        feat = torch.tensor(self.extract_features(audio_path)).float().unsqueeze(0)
        with torch.no_grad():
            score = self.forward(feat)
        return torch.sigmoid(score).item()
