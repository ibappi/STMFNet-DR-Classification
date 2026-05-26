import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from pretrainedmodels import inceptionv4  # For InceptionV4
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Set device for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data Directory
dr_images = '../data/RFundus_images/'

# Define Image Preprocessing and Augmentation
data_transforms = transforms.Compose([
    transforms.Resize((299, 299)),  # Required input size for InceptionV4 is 299x299
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create a dataset from the directory
full_dataset = datasets.ImageFolder(dr_images, transform=data_transforms)

# Prepare your dataset and labels
X = full_dataset.imgs  # List of image paths
y = full_dataset.targets  # Corresponding labels

# Split the dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Further split the temp set into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Create PyTorch datasets from the splits
train_dataset = datasets.ImageFolder(dr_images, transform=data_transforms)
val_dataset = datasets.ImageFolder(dr_images, transform=data_transforms)
test_dataset = datasets.ImageFolder(dr_images, transform=data_transforms)

# Set up DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load InceptionV4 Pretrained Model
class InceptionV4ForClassification(nn.Module):
    def __init__(self, out_channels=11):  # Assuming 11 classes
        super(InceptionV4ForClassification, self).__init__()
        self.inceptionv4 = inceptionv4(pretrained='imagenet')  # Load pretrained InceptionV4

        # Replace the final fully connected layer for classification
        num_ftrs = self.inceptionv4.last_linear.in_features
        self.inceptionv4.last_linear = nn.Linear(num_ftrs, out_channels)

    def forward(self, x):
        return self.inceptionv4(x)

# Instantiate Model and Optimizer
model = InceptionV4ForClassification().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.00001)

# Training Loop with tqdm
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}') as pbar:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                pbar.update(1)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f'\nEpoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')

    return train_losses, val_losses

# Training the model
train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer)

# Plotting Training and Validation Loss
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model on the test set
def evaluate_model_with_metrics(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []  # For AUC-ROC
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            probs = torch.softmax(outputs, dim=1)  # Get probabilities
            all_probs.append(probs.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'ROC AUC Score: {roc_auc:.4f}')

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", cm)

# Evaluate the model on the test set
evaluate_model_with_metrics(model, test_loader)

# Accuracy: 0.9254
# Precision: 0.9264
# Recall: 0.9254
# F1 Score: 0.9256
# ROC AUC Score: 0.9397