from __future__ import print_function
import os
import time
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
from models.clip import clip
from nct_crc_dataset import NCTCRC      # <-- your CSV-based dataset
import utils

# -------------------
# Argparse
# -------------------
parser = argparse.ArgumentParser(description='PyTorch CLIP Training on NCT-CRC-HE-100K')
parser.add_argument('--model', type=str, default='Ourmodel', help='CNN architecture')
parser.add_argument('--mode', type=int, default=1, help='Feature mode (0=image, 1=image+text)')
parser.add_argument('--dataset', type=str, default='NCTCRC_Ourmodel', help='dataset folder prefix')
parser.add_argument('--fold', default=1, type=int, help='(ignored for this dataset)')
parser.add_argument('--bs', default=64, type=int, help='batch_size')
parser.add_argument('--lr', default=0.003, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--file_path', type=str, default='nct_crc_he_100k.csv', help='CSV describing dataset')
parser.add_argument('--checkpoint', type=str, default='', help='optional checkpoint path to load')
opt = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = "cuda:0" if use_cuda else "cpu"

# -------------------
# Classes
# -------------------
class_names = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
num_classes = len(class_names)
path = os.path.join(opt.dataset, str(opt.fold))

# -------------------
# Transforms
# -------------------
transforms_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(30),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
])

transforms_valid = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
])

# -------------------
# Custom network & helpers
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

class EarlyStopping:
    def __init__(self, patience=7, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, accuracy):
        score = accuracy
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

# -------------------
# Load CLIP and infer feature dim
# -------------------
print("Loading CLIP model...")
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model = clip_model.float()
clip_model.eval()

def infer_feature_dim(clip_model, device, mode):
    """
    Run a tiny forward pass with dummy inputs to infer flattened feature size
    after the avgpool operations used in train/test.
    """
    clip_model.eval()
    with torch.no_grad():
        dummy_image = torch.randn(1, 3, 224, 224).to(device)
        dummy_text = clip.tokenize(["dummy"]).to(device)
        image_features = clip_model.encode_image(dummy_image)           # (1, D)
        image_features = image_features.unsqueeze(1)                    # (1,1,D)
        image_features = nn.AvgPool1d(kernel_size=2)(image_features).squeeze(1)  # (1, D')
        if mode == 1:
            text_features = clip_model.encode_text(dummy_text)         # (1, D_text)
            text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-10)
            features = torch.cat((image_features, text_features), dim=1)  # (1, D_sum)
        else:
            features = image_features
        features = nn.AvgPool1d(kernel_size=2, stride=2)(features)     # 1D pooling
        flat = features.view(features.size(0), -1)
        return flat.size(1)

feature_dim = infer_feature_dim(clip_model, device, opt.mode)
print(f"Inferred flattened feature dim: {feature_dim}")

# -------------------
# Instantiate network
# -------------------
if opt.mode == 0:
    net = CustomNet(num_classes=num_classes, feature_dim=feature_dim)
elif opt.mode == 1:
    net = CustomNet(num_classes=num_classes, feature_dim=feature_dim)
else:
    net = nn.Sequential(nn.ReLU(), nn.Linear(feature_dim, num_classes))

net = net.to(device)

# -------------------
# DataLoaders
# -------------------
print("Preparing data loaders...")
trainset = NCTCRC(split='Training', transform=transforms_train, file_path=opt.file_path)
testset  = NCTCRC(split='Testing',  transform=transforms_valid, file_path=opt.file_path)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.bs, shuffle=True, num_workers=opt.workers)
testloader  = torch.utils.data.DataLoader(testset,  batch_size=opt.bs, shuffle=False, num_workers=opt.workers)

print(f"Train samples: {len(trainset)}, Test samples: {len(testset)}")

# -------------------
# Criterion & Optimizer
# -------------------
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

# optionally resume
best_Test_acc = 0.0
start_epoch = 0
if opt.resume and opt.checkpoint:
    print("Loading checkpoint:", opt.checkpoint)
    ck = torch.load(opt.checkpoint, map_location=device)
    net.load_state_dict(ck['net'])
    best_Test_acc = ck.get('best_Test_acc', 0.0)
    start_epoch = ck.get('best_Test_acc_epoch', 0) + 1

# -------------------
# Training & Testing functions
# -------------------
early_stopping = EarlyStopping(patience=10, delta=0.001)
total_processing_time_train = 0.0
total_processing_time_test = 0.0

def train(epoch):
    global total_processing_time_train
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0
    avg_pool = nn.AvgPool1d(kernel_size=2)

    all_labels = []
    all_predictions = []

    start_time = time.monotonic()
    for idx, (images, captions, labels) in enumerate(trainloader):
        batch_start = time.time()
        images = images.to(device)
        captions = captions.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(images)
        image_features = image_features.unsqueeze(1)
        image_features = avg_pool(image_features).squeeze(1)

        if opt.mode == 1:
            with torch.no_grad():
                text_features = clip_model.encode_text(captions)
                text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-10)
            features = torch.cat((image_features, text_features), dim=1)
        else:
            features = image_features

        features = nn.AvgPool1d(kernel_size=2, stride=2)(features)
        features = features.view(features.size(0), -1).float()

        optimizer.zero_grad()
        outputs = net(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_end = time.time()
        total_processing_time_train += (batch_end - batch_start)

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum().item()

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

        utils.progress_bar(idx, len(trainloader),
                           'TrainLoss: %.4f | TrainAcc: %.3f%% (%d/%d)' %
                           (running_loss / (idx + 1), 100. * correct / total, correct, total))

    epoch_time_elapsed = time.monotonic() - start_time
    train_acc = 100. * correct / total if total > 0 else 0.0
    train_loss = running_loss / (len(trainloader) if len(trainloader) > 0 else 1)
    train_f1 = f1_score(all_labels, all_predictions, average='weighted') if all_labels else 0.0

    print(f'\nEpoch {epoch} TRAIN => Loss: {train_loss:.4f} Acc: {train_acc:.3f}% F1: {train_f1:.4f} Time: {epoch_time_elapsed:.1f}s')

def test(epoch):
    global total_processing_time_test, best_Test_acc
    net.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    avg_pool = nn.AvgPool1d(kernel_size=2)

    all_labels = []
    all_predictions = []

    start_time = time.monotonic()
    with torch.no_grad():
        for idx, (images, captions, labels) in enumerate(testloader):
            batch_start = time.time()
            images = images.to(device)
            captions = captions.to(device)
            labels = labels.to(device)

            image_features = clip_model.encode_image(images)
            image_features = image_features.unsqueeze(1)
            image_features = avg_pool(image_features).squeeze(1)

            if opt.mode == 1:
                text_features = clip_model.encode_text(captions)
                text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-10)
                features = torch.cat((image_features, text_features), dim=1)
            else:
                features = image_features

            features = nn.AvgPool1d(kernel_size=2, stride=2)(features)
            features = features.view(features.size(0), -1).float()

            outputs = net(features)
            loss = criterion(outputs, labels)

            batch_end = time.time()
            total_processing_time_test += (batch_end - batch_start)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            utils.progress_bar(idx, len(testloader),
                               'TestLoss: %.4f | TestAcc: %.3f%% (%d/%d)' %
                               (running_loss / (idx + 1), 100. * correct / total, correct, total))

    epoch_time_elapsed = time.monotonic() - start_time
    test_acc = 100. * correct / total if total > 0 else 0.0
    test_loss = running_loss / (len(testloader) if len(testloader) > 0 else 1)
    test_f1 = f1_score(all_labels, all_predictions, average='weighted') if all_labels else 0.0

    print(f'\nEpoch {epoch} TEST  => Loss: {test_loss:.4f} Acc: {test_acc:.3f}% F1: {test_f1:.4f} Time: {epoch_time_elapsed:.1f}s')
    # Early stopping
    early_stopping(test_acc)
    if early_stopping.early_stop:
        print("Early stopping triggered.")
        return True

    # Save checkpoint if improved
    if test_acc > best_Test_acc:
        print("Saving best checkpoint..")
        state = {
            'net': net.state_dict(),
            'best_Test_acc': test_acc,
            'best_Test_acc_epoch': epoch
        }
        os.makedirs(path, exist_ok=True)
        torch.save(state, os.path.join(path, 'best_textonly.t7'))
        best_Test_acc = test_acc

    return False

# -------------------
# Training loop
# -------------------
total_start_time = time.monotonic()
for epoch in range(start_epoch, opt.epochs):
    train(epoch)
    should_stop = test(epoch)
    if should_stop:
        break
total_end_time = time.monotonic()

total_hours = int((total_end_time - total_start_time) // 3600)
total_mins = int(((total_end_time - total_start_time) % 3600) // 60)
total_secs = int((total_end_time - total_start_time) % 60)
print(f"Total training time: {total_hours}h {total_mins}m {total_secs}s")
print(f"Best test acc: {best_Test_acc:.3f}%")
