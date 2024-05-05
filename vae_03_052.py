import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# VAE Model
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256,40)
        )
        self.decoder = nn.Sequential(
            nn.Linear(20, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = x.view(-1, 784)
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

# Loss Function
def loss_function(recon_x, x, mu, logvar, beta=1.0):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD * beta

# Training Function
def train(model, device, train_loader, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        print(f'Epoch {epoch+1}, Average Loss: {total_loss / len(train_loader.dataset)}')

# Testing Function for OOD Detection
def test(model, device, data_loader):
    model.eval()
    losses = []
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            losses.append(loss.item() / data.size(0))  # Normalized loss
    return losses

# Plotting Histogram of OOD and In-distribution scores
def plot_histogram(in_losses, ood_losses):
    plt.hist(in_losses, bins=50, alpha=0.85, color='navy',label='In-Distribution (Drones)')
    plt.hist(ood_losses, bins=50, alpha=0.85, color='darkorange', label='Out-of-Distribution (Birds)')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

## Main Execution
if __name__ == "__main__":
    device = torch.device('cpu')
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Resize as per your model input requirements
        transforms.ToTensor(),
        #transforms.Normalize((0.5,), (0.5,))
    ])

    # Load data
    drone_dataset = CustomDataset('/home/user/CS726_OOD/dataset_ebm1/train/drones', transform=transform)
    bird_dataset = CustomDataset('/home/user/CS726_OOD/dataset_ebm1/train/birds', transform=transform)
    drone_loader = DataLoader(drone_dataset, batch_size=128, shuffle=True)
    bird_loader = DataLoader(bird_dataset, batch_size=128, shuffle=False)

    # Model, Optimizer
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training the model on drone images (in-distribution)
    print("Starting training...")
    train(model, device, drone_loader, optimizer, epochs=50)

    # Testing the model on drone images (in-distribution)
    print("Testing on in-distribution data...")
    in_losses = test(model, device, drone_loader)

    # Testing the model on bird images (out-of-distribution)
    print("Testing on out-of-distribution data...")
    ood_losses = test(model, device, bird_loader)

    in_distribution_errors = test(model, device, drone_loader)  # Drone data
    out_of_distribution_errors = test(model, device, bird_loader)  # Bird data

	# Plotting with Seaborn
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.kdeplot(in_distribution_errors, bw_adjust=2, fill=True, color='blue', label='In-Distribution (Drones)')
    sns.kdeplot(out_of_distribution_errors, bw_adjust=2, fill=True, color='orange', label='Out-of-Distribution (Birds)')
    plt.title('Comparison of Reconstruction Errors')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    # Plotting the histogram for OOD detection
    print("Plotting histogram...")
    plot_histogram(in_losses, ood_losses)

