
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import cv2
import numpy as np

class VideoCaptioningDataset(Dataset):
    def __init__(self, data):
        self.input_texts = data['input_text']
        self.labels = data['target_caption']
        self.video_paths = data['video_path']

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        input_text = self.input_texts[idx]
        label = self.labels[idx]
        video_path = self.video_paths[idx]

        # Load and preprocess video frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        cap.release()
        pixel_values = np.array(frames)

        return {
            'input_ids': torch.tensor(input_text),
            'attention_mask': torch.tensor([1] * len(input_text)),
            'pixel_values': torch.tensor(pixel_values).permute(0, 3, 1, 2),
            'labels': torch.tensor(label)
        }

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_video_data(dataset, tokenizer, max_length):
    tokenized_inputs = tokenizer(
        dataset['input_text'].tolist(),
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    dataset['input_ids'] = tokenized_inputs['input_ids']
    dataset['attention_mask'] = tokenized_inputs['attention_mask']
    dataset['labels'] = tokenizer(
        dataset['target_caption'].tolist(),
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )['input_ids']
    return dataset
