import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torchvision import datasets, models
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

# Set device for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data Directory
dr_images = './data/RFundus_images/'

# Define Image Preprocessing and Augmentation
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
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


# Texture Attention Module
class TextureAttentionModule(nn.Module):
    def __init__(self):
        super(TextureAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Global Average Pooling after convolution layers
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Reduces spatial dimensions to (1, 1)
        self.fc = nn.Linear(32, 512)  # Adjusted to match reduced feature size

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        # Apply Global Average Pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output

        x = torch.relu(self.fc(x))
        return x


# Spatial Attention Module
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels=1792, out_channels=1, kernel_size=1)

    def forward(self, x):
        attention_map = torch.sigmoid(self.conv(x))  # Generate attention map
        x = x * attention_map  # Apply attention to input
        return x


# Final Model with EfficientNet, Texture Attention, and Spatial Attention
class DRClassificationModel(nn.Module):
    def __init__(self):
        super(DRClassificationModel, self).__init__()
        # Pretrained EfficientNet
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        self.efficientnet.classifier = nn.Identity()  # Remove original classifier

        # Texture Attention Module
        self.texture_attention = TextureAttentionModule()

        # Spatial Attention
        self.spatial_attention = SpatialAttentionModule()

        # Fully Connected Network
        self.fc1 = nn.Linear(1280 + 512, 256)  # EfficientNet output size (1280) + texture output size (512)
        self.fc2 = nn.Linear(256, 11)  # 11 classes

    def forward(self, x):
        # print(f"Input shape: {x.shape}")  # Input shape from the DataLoader

        # EfficientNet feature extraction
        efficientnet_features = self.efficientnet(x)
        # print(f"EfficientNet output shape: {efficientnet_features.shape}")  # Shape after EfficientNet

        # Texture Attention Module
        texture_features = self.texture_attention(x)
        # print(f"Texture Attention output shape: {texture_features.shape}")  # Shape after Texture Attention

        # Concatenate features
        combined_features = torch.cat((efficientnet_features, texture_features), dim=1)
        # print(f"Combined features shape (EfficientNet + Texture): {combined_features.shape}")  # Combined shape

        # Apply Spatial Attention
        combined_features = combined_features.unsqueeze(2).unsqueeze(3)  # Shape becomes (batch_size, 1792, 1, 1)
        attended_features = self.spatial_attention(combined_features)
        # print(f"Attended features shape (after Spatial Attention): {attended_features.shape}")  # Shape after Spatial Attention

        # Flatten attended features
        x = attended_features.view(attended_features.size(0), -1)
        # print(f"Flattened features shape: {x.shape}")  # Shape after flattening

        # Fully Connected Network
        x = torch.relu(self.fc1(x))
        # print(f"Shape after first fully connected layer (fc1): {x.shape}")  # Shape after FC1
        x = self.fc2(x)
        # print(f"Final output shape (after fc2): {x.shape}")  # Shape after final FC layer

        return x


# Instantiate Model and Optimizer
model = DRClassificationModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.00001)


# Training Loop with tqdm
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100):
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

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')

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


def plot_actual_vs_predicted(all_labels, all_preds):
    plt.figure(figsize=(10, 6))
    unique_classes = np.unique(all_labels)
    plt.scatter(all_labels, all_preds, alpha=0.6)

    # Plotting the diagonal line
    min_val = min(np.min(all_labels), np.min(all_preds))
    max_val = max(np.max(all_labels), np.max(all_preds))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    plt.title('Actual vs. Predicted')
    plt.xlabel('Actual Labels')
    plt.ylabel('Predicted Labels')
    plt.xticks(unique_classes)
    plt.yticks(unique_classes)
    plt.grid()
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

    # Plotting Confusion Matrix with Value Labels
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(full_dataset.classes))
    plt.xticks(tick_marks, full_dataset.classes, rotation=45)
    plt.yticks(tick_marks, full_dataset.classes)

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, f'{cm[i, j]}', horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

    # Plot Actual vs Predicted
    plot_actual_vs_predicted(all_labels, all_preds)


# Evaluate the model on the test set
evaluate_model_with_metrics(model, test_loader)


# Epoch [100/100], Train Loss: 0.0184, Val Loss: 0.0065
# Accuracy: 0.9772
# Precision: 0.9972
# Recall: 0.9972
# F1 Score: 0.9872
# ROC AUC Score: 1.0000


