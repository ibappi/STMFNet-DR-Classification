import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets, models
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, precision_score, \
    recall_score, f1_score

# Set device for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data Directories
train_dir = './data/RFundus_images/train'
val_dir = './data/RFundus_images/val'
test_dir = './data/RFundus_images/test'

# Define Image Preprocessing and Augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Data Loaders
train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms['val'])
test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms['test'])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Texture Attention Module (Placeholder example, should be enhanced for actual texture-based attention)
class TextureAttentionModule(nn.Module):
    def __init__(self):
        super(TextureAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=3, padding=1)  # Adjusted to output 24 channels
        self.conv2 = nn.Conv2d(24, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 56 * 56, 512)  # Adjusted based on final output dimensions

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        return x


# Spatial Attention Module
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        # Modify the convolutional layers to accept 2D input
        self.conv = nn.Conv2d(in_channels=1792, out_channels=1, kernel_size=1)

    def forward(self, x):
        # x should already be in shape (batch_size, channels, height, width)
        attention_map = torch.sigmoid(self.conv(x))  # Generate attention map
        return x * attention_map  # Apply attention to input


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
        self.fc1 = nn.Linear(1792, 512)  # Updated to match the combined features from EfficientNet and Texture
        self.fc2 = nn.Linear(512, 11)  # 11 classes

    def forward(self, x):
        print(f"Input shape: {x.shape}")  # Print input shape

        # Extract features from both EfficientNet and Texture Attention
        efficientnet_features = self.efficientnet(x)
        print(f"EfficientNet features shape: {efficientnet_features.shape}")  # Print shape after EfficientNet

        texture_features = self.texture_attention(x)
        print(f"Texture features shape: {texture_features.shape}")  # Print shape after Texture Attention

        # Concatenate features
        combined_features = torch.cat((efficientnet_features, texture_features), dim=1)
        print(f"Combined features shape: {combined_features.shape}")  # Print shape after concatenation

        # Here we do not unsqueeze for spatial attention directly as we want (batch_size, 1792, 1, 1)
        combined_features = combined_features.unsqueeze(2).unsqueeze(3)  # Shape becomes (batch_size, 1792, 1, 1)
        attended_features = self.spatial_attention(combined_features)
        print(f"Attended features shape: {attended_features.shape}")  # Print shape after spatial attention

        # Flatten attended features before passing to fully connected layers
        x = attended_features.view(attended_features.size(0), -1)  # Flatten to (batch_size, features)
        print(f"Flattened features shape: {x.shape}")  # Print shape after flattening

        # Fully Connected Network
        x = torch.relu(self.fc1(x))  # Adjusted to use flattened input
        print(f"Output of fc1 shape: {x.shape}")  # Print shape after first FC
        x = self.fc2(x)
        print(f"Final output shape: {x.shape}")  # Print final output shape

        return x


# Instantiate Model and Optimizer
model = DRClassificationModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# Training Loop with tqdm
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100):
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}') as pbar:
            for inputs, labels in train_loader:
                print('inputs shape', inputs[0])
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


# Evaluation on Test Set with additional metrics
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

    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')  # For multiclass
    recall = recall_score(all_labels, all_preds, average='macro')  # For multiclass
    f1 = f1_score(all_labels, all_preds, average='macro')  # For multiclass
    roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovo')  # One-vs-One AUC

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(classification_report(all_labels, all_preds))

    return cm, all_labels, all_preds


# Call the modified evaluation function
cm, labels, preds = evaluate_model_with_metrics(model, test_loader)

# Plot Confusion Matrix
plt.figure()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.show()


# Prediction vs Actual Plot
plt.figure(figsize=(10, 6))
plt.scatter(labels, preds, alpha=0.5)
plt.title('Prediction vs Actual')
plt.xlabel('Actual Classes')
plt.ylabel('Predicted Classes')
plt.xlim(0, max(np.max(labels), np.max(preds)) + 1)
plt.ylim(0, max(np.max(labels), np.max(preds)) + 1)
plt.plot([0, max(np.max(labels), np.max(preds))], [0, max(np.max(labels), np.max(preds))], 'r--')  # Diagonal line
plt.grid()
plt.xticks(range(11))  # Assuming you have 11 classes (0 to 10)
plt.yticks(range(11))  # Assuming you have 11 classes (0 to 10)
plt.show()
