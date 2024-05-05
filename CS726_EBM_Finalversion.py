import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
from torch import nn, optim
import os


def load_data(data_dir):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to fit the model
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
    ])
    
    # Load training and testing datasets
    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    return train_loader, test_loader


def create_model():
    # Load a pre-trained VGG16 model
    model = models.vgg16(pretrained=True)

    # Freeze all the layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace the classifier with a new one that ends with a single output neuron for energy scoring
    model.classifier[6] = nn.Sequential(
        nn.Linear(4096, 1),  # Change the output features to 1
        nn.Flatten()  # Flatten the output
    )

    return model


def custom_loss_function(outputs, labels):
    # Example: MSE loss that expects drones to have lower energies
    target_energies = torch.where(labels == 0, torch.zeros_like(outputs), torch.ones_like(outputs) * 1)  # drones: 0, birds: 1
    loss = torch.mean((outputs - target_energies) ** 2)
    return loss

def train_model(model, train_loader, device, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = custom_loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Average Loss: {total_loss / len(train_loader)}')


def evaluate_model(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            predicted = outputs < 0.5  # Threshold for classifying as drone
            correct += (predicted.flatten() == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy}%')


def main():
    data_dir = r'/home/user/CS726_OOD/dataset_ebm1'  # Update this path to your dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, test_loader = load_data(data_dir)
    model = create_model()
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    
    train_model(model, train_loader, device, optimizer, epochs=25)
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()


def collect_scores(model, data_loader, device):
    model.eval()
    scores = []
    labels = []
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            outputs = model(data)
            scores.extend(outputs.cpu().numpy().flatten())  # Flatten and convert to list
            labels.extend(target.cpu().numpy())  # Store labels to differentiate the groups later

    return scores, labels


import matplotlib.pyplot as plt
def plot_histograms(scores, labels):
    drone_scores = [score for score, label in zip(scores, labels) if label == 0]
    bird_scores = [score for score, label in zip(scores, labels) if label == 1]

    plt.hist(drone_scores, bins=30, alpha=0.5, label='Drones (ID)', color='blue')
    plt.hist(bird_scores, bins=30, alpha=0.5, label='Birds (OOD)', color='red')
    plt.xlabel('Predicted Energy Scores')
    plt.ylabel('Frequency')
    plt.title('Histogram of Predicted Energy Scores')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.save()


def main():
    data_dir = r'/home/user/CS726_OOD/dataset_ebm1'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    _, test_loader = load_data(data_dir)
    model = create_model().to(device)
    
    # Assuming the model is already trained and loaded
    scores, labels = collect_scores(model, test_loader, device)
    plot_histograms(scores, labels)

if __name__ == "__main__":
    main()

