from __future__ import print_function
from PIL import Image
import torch
import clip
import torch.utils.data as data
import pandas as pd
from sklearn.model_selection import train_test_split

# Load CLIP model and preprocessing
clip_model, preprocess = clip.load("ViT-B/32", device="cpu")

class NCTCRC(data.Dataset):
    def __init__(self, split='Training', transform=None, file_path=None, test_size=0.2, random_state=42):
        self.transform = transform
        self.split = split
        self.file_path = file_path

        # Read CSV (images, text, labels)
        self.data = pd.read_csv(self.file_path)

        # Stratified train/test split
        train_df, test_df = train_test_split(
            self.data,
            test_size=test_size,
            stratify=self.data['labels'],
            random_state=random_state
        )

        if self.split == 'Training':
            self.dataset = train_df.reset_index(drop=True)
        else:
            self.dataset = test_df.reset_index(drop=True)

        print(f"{self.split} set size: {len(self.dataset)}")

    def __getitem__(self, index):
        row = self.dataset.iloc[index]
        image_path = row['images']
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        text = row['text']
        caption = clip.tokenize([text])  # Tokenized text for CLIP
        label = torch.tensor(row['labels'], dtype=torch.long)

        return image, caption.squeeze(0), label

    def __len__(self):
        return len(self.dataset)
