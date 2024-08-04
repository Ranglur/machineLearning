import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MNIST_model(nn.Module):
    def __init__(self) -> None:
        super(MNIST_model, self).__init__()
        
        # Define layers
        self.layer1 = nn.Linear(784, 128)  # First hidden layer, 128 units
        self.layer2 = nn.Linear(128, 64)   # Second hidden layer, 64 units
        self.layer3 = nn.Linear(64, 10)    # Output layer, 10 units 
        
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))  # Apply first layer and ReLU
        x = self.relu(self.layer2(x))  # Apply second layer and ReLU
        x = self.layer3(x)  # Apply third layer 
        
        return x

# Load MNIST dataset with tensor flattening in the transformation pipeline
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # Flatten the tensor
])

# Download and load the training dataset
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Split dataset into training (80%) and validation (20%) sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Initialize the model
model = MNIST_model().to(device)

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Number of epochs
num_epochs = 20

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = loss_fn(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# Save the model
torch.save(model.state_dict(), 'mnist_model.pth')

# Testing the ML model with the remaining 20% of the dataset
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in val_loader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

accuracy = 100 * correct / total
print(f'Validation Accuracy: {accuracy:.2f}%')
