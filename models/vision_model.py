import torch
import torch.nn as nn
import torchvision.transforms as T

class VisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, 2)
        self.fc1 = nn.Linear(32 * 6 * 6, 100)
        self.fc2 = nn.Linear(100, 1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((6, 6))
        self.transform = T.Compose([
            T.Resize((48, 48)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

    def predict_deception(self, image):
        self.eval()
        img_tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            score = self.forward(img_tensor)
        return torch.sigmoid(score).item()
