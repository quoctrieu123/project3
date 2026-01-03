import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

# Expose an embedder and label map for external scripts (e.g., test.py)
embedder = SentenceTransformer("intfloat/multilingual-e5-large")

# simple label mapping used by the classifier
label_map = {"documents": 0, "extract_laws": 1}

def map_label(label):
    return label_map[label]

def embedding_sentence(sentence):
    embedding = embedder.encode([sentence])
    return embedding[0]

class PathClassifier(nn.Module):
    def __init__(self,input_size = 1024, num_classes = 2):
        super(PathClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # return raw logits (no activation) so CrossEntropyLoss can be applied
        x = self.fc2(x)
        return x

def train_model(model, x_train, y_train, x_test, y_test, epochs=20, learning_rate=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        print(f"----------- Epoch {epoch+1}/{epochs} -----------")
        model.train()
        output = model(x_train)
        loss = loss_fn(output, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            test_output = model(x_test)
            test_loss = loss_fn(test_output, y_test)
            _, predicted = torch.max(test_output, 1)
            accuracy = (predicted == y_test).sum().item() / len(y_test)
        print(f"Train Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f} | Test Accuracy: {accuracy*100:.2f}%")


if __name__ == "__main__":
    # Training entrypoint. This block runs only when the file is executed directly.
    # It was purposely moved out of top-level import to avoid heavy processing on import.
    data  = pd.read_csv("dataset/dataset_for_path_classifier_training.csv")
    data  = data.sample(frac=1, random_state = 7).reset_index(drop=True)

    data["embedding"] = data["query"].apply(embedding_sentence)
    data["label"] = data["label"].apply(map_label)
    train_size = int(0.8*len(data))
    test_size = len(data) - train_size

    train_data, test_data = train_test_split(data, test_size=test_size, stratify=data['label'], random_state=7)
    y_train = torch.tensor(train_data["label"].to_numpy(), dtype=torch.long)
    x_train = torch.tensor(np.stack(train_data["embedding"].to_numpy()), dtype=torch.float32)
    y_test = torch.tensor(test_data["label"].to_numpy(), dtype=torch.long)
    x_test = torch.tensor(np.stack(test_data["embedding"].to_numpy()), dtype=torch.float32)

    model = PathClassifier()
    train_model(model, x_train, y_train, x_test, y_test, epochs=20, learning_rate=0.0001)
    torch.save(model.state_dict(), "path_classifier_model.pth")