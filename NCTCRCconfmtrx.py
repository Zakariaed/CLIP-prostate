from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import argparse
import itertools
import os
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from models.clip import clip
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import models
from nct_crc_dataset import NCTCRC  # <-- Updated Dataset class

# -------------------
# Argument Parser
# -------------------
parser = argparse.ArgumentParser(description='PyTorch CLIP + CustomNet Testing on NCT-CRC-HE-100K')
parser.add_argument('--model', type=str, default='Ourmodel', help='CNN architecture')
parser.add_argument('--mode', type=int, default=1, help='Feature mode (0=Image only, 1=Image+Text)')
parser.add_argument('--bs', default=64, type=int, help='Batch size')
parser.add_argument('--lr', default=0.003, type=float, help='Learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--file_path', type=str, default='nct_crc_he_100k.csv', help='CSV file path')
parser.add_argument('--checkpoint', type=str, default='best_model.pth', help='Path to checkpoint')
opt = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# -------------------
# Transforms
# -------------------
transforms_valid = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
])

# -------------------
# Class Names
# -------------------
class_names = [
    "ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"
]
class_labels = list(range(len(class_names)))

# -------------------
# Confusion Matrix Plot
# -------------------
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()

# -------------------
# Custom Model
# -------------------
class CustomNet(nn.Module):
    def __init__(self, num_classes, feature_dim):
        super(CustomNet, self).__init__()
        self.fc1 = nn.Linear(feature_dim, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# -------------------
# Load CLIP
# -------------------
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model = clip_model.float()
clip_model.eval()

# -------------------
# Load CustomNet
# -------------------
num_classes = len(class_names)
if opt.mode == 0:
    net = CustomNet(num_classes=num_classes, feature_dim=512)
elif opt.mode == 1:
    net = CustomNet(num_classes=num_classes, feature_dim=384)
else:
    net = nn.Sequential(nn.ReLU(), nn.Linear(512, num_classes))

net.load_state_dict(torch.load(opt.checkpoint, map_location=device)['net'])
net.to(device)
net.eval()

# -------------------
# Load Dataset
# -------------------
testset = NCTCRC(split='Testing', transform=transforms_valid, file_path=opt.file_path)
testloader = torch.utils.data.DataLoader(testset, batch_size=opt.bs, shuffle=False, num_workers=opt.workers)

# -------------------
# Evaluation
# -------------------
correct = 0
total = 0
overall_conf_matrix = np.zeros((num_classes, num_classes))
avg_pool = nn.AvgPool1d(kernel_size=2)

all_predicted = []
all_targets = []

for images, captions, labels in testloader:
    images, captions, labels = images.to(device), captions.to(device), labels.to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(images)
        image_features = image_features.unsqueeze(1)
        image_features = avg_pool(image_features).squeeze(1)

        if opt.mode == 1:
            text_features = clip_model.encode_text(captions)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            features = torch.cat((image_features, text_features), dim=1)
        else:
            features = image_features

        pooling_layer = nn.AvgPool1d(kernel_size=2, stride=2)
        features = pooling_layer(features)
        features = features.view(features.size(0), -1).float()

        outputs = net(features)

    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += predicted.eq(labels.data).cpu().sum().item()

    conf_matrix = confusion_matrix(labels.cpu().numpy(), predicted.cpu().numpy(), labels=class_labels)
    overall_conf_matrix += conf_matrix

    all_predicted.extend(predicted.cpu().numpy())
    all_targets.extend(labels.cpu().numpy())

# -------------------
# Metrics
# -------------------
acc = 100. * correct / total
print(f"Test Accuracy: {acc:.3f}%")
print('Classification Report:\n', classification_report(all_targets, all_predicted, target_names=class_names))

plt.figure(figsize=(10, 8))
plot_confusion_matrix(overall_conf_matrix, classes=class_names, normalize=False,
                      title=f'Confusion Matrix (Accuracy: {acc:.3f}%)')
plt.savefig('confusion_matrix_nct_crc.png')
plt.close()
