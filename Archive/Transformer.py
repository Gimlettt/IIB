import pandas as pd
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

#the model
class TimeSeriesTransformer(nn.Module):
    # num_layers is how many encoder block
    # embedding_dim is the dimension of the embedding for one token, note we either use 1 feature(z-axis coordinate) or all features, this projection from feature_dim to embedding_dim is through a linear layer
    # each time step is like a word(token) with feature_dim features, and turned to an embedding
    
    def __init__(self, feature_dim, embedding_dim=64, nhead=4, num_layers=2, num_classes=2):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, embedding_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 9, embedding_dim))  # position encoding, for each token(now with size embedding_dim), we add a position embedding of size embedding_dim

        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # after encoder, we have a MLP for the output
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x) + self.pos_embedding  # Add position encoding
        x = self.encoder(x)  # Shape: (B, 9, embedding_dim)
        x = x.mean(dim=1)    # Pool across sequence
        return self.classifier(x)






#the dataset loader
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

data = np.load("dataset/transformer_dataset_zaxis.npz")
X, y = data["X"], data["y"]                # X: (N, 9, F), y: (N,)

# Normalize each feature channel
#    – flatten over samples & time → compute mean/std per feature
X_flat = X.reshape(-1, X.shape[2])         # shape (N*9, F)
mean = X_flat.mean(axis=0, keepdims=True)  # shape (1, F)
std  = X_flat.std(axis=0, keepdims=True)
std[std == 0] = 1                          # avoid div-by-zero

# Apply normalization: broadcasting over N and 9
X = (X - mean[None, None, :]) / std[None, None, :]
X = X.squeeze(0)# there is a extra dimension, remove it




X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y       # comment out if you don't need stratified split
)

print("Train samples:", X_train.shape[0])
print("Test  samples:", X_test.shape[0])

# Wrap in your Dataset & DataLoader
train_ds = TimeSeriesDataset(X_train, y_train)
test_ds  = TimeSeriesDataset(X_test,  y_test)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False)


# set up the model, and define the hyperparameters
model = TimeSeriesTransformer(feature_dim=X.shape[2]).to("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = 100




# training
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1} - Loss: {running_loss:.4f} - Accuracy: {acc:.4f}")
    
    
    
    
# testing
model.eval()
test_loss = 0.0
correct   = 0
total     = 0

all_preds   = []
all_targets = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)                  # (B, num_classes)
        loss    = criterion(outputs, batch_y)
        
        # accumulate loss
        test_loss += loss.item() * batch_X.size(0)
        
        # get predicted classes
        preds = outputs.argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total   += batch_y.size(0)

        # collect for detailed metrics
        all_preds.append(preds.cpu())
        all_targets.append(batch_y.cpu())

# compute averages
avg_loss = test_loss / total
accuracy = correct / total

print(f"\nTest set: Average loss: {avg_loss:.4f}, "
      f"Accuracy: {correct}/{total} ({100.*accuracy:.2f}%)")


from sklearn.metrics import classification_report, confusion_matrix


all_preds   = torch.cat(all_preds).numpy()
all_targets = torch.cat(all_targets).numpy()
print("\nClassification Report:")
print(classification_report(all_targets, all_preds, digits=4))
print("Confusion Matrix:")
print(confusion_matrix(all_targets, all_preds))