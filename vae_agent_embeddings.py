import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import json
import numpy as np
import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Configuration
EMBEDDING_DIM = 384  # Original embedding dimension
LATENT_DIM = 346
HIDDEN_DIMS = [256, 128, 64]  # Hidden layer dimensions
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
RANDOM_SEED = 42
DATA_PATH = 'data'

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Custom dataset for agent embeddings
class AgentEmbeddingDataset(Dataset):
    def __init__(self, embeddings):
        self.embeddings = embeddings
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx], dtype=torch.float32)

# Variational Autoencoder
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims):
        super(VAE, self).__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(nn.LeakyReLU())
            prev_dim = h_dim
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        decoder_hidden_dims = hidden_dims[::-1]  # Reverse for decoder
        
        self.decoder_input = nn.Linear(latent_dim, decoder_hidden_dims[0])
        
        prev_dim = decoder_hidden_dims[0]
        for idx, h_dim in enumerate(decoder_hidden_dims[1:]):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.BatchNorm1d(h_dim))
            decoder_layers.append(nn.LeakyReLU())
            prev_dim = h_dim
            
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x):
        """Encode the input and return latent parameters."""
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """Reparameterization trick to sample from latent space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """Decode the latent representation."""
        z = self.decoder_input(z)
        z = F.leaky_relu(z)
        x_hat = self.decoder(z)
        return x_hat
    
    def forward(self, x):
        """Forward pass through the network."""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var

# Loss function for VAE
def vae_loss(recon_x, x, mu, log_var):
    """
    VAE loss function combining reconstruction loss and KL divergence
    """
    # Reconstruction loss (mean squared error)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return recon_loss + kld_loss

def load_embeddings(faiss_path, json_path):
    """Load embeddings from FAISS index and metadata from JSON."""
    # Load the index
    index = faiss.read_index(faiss_path)
    
    # Get the number of vectors
    num_vectors = index.ntotal
    print(f"Index contains {num_vectors} vectors of dimension {index.d}")
    
    # Extract vectors from the index
    vectors = np.zeros((num_vectors, index.d), dtype=np.float32)
    for i in range(num_vectors):
        vector = index.reconstruct(i)
        vectors[i] = vector
    
    # Load metadata for additional information
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    
    return vectors, metadata

def train_vae(model, train_loader, optimizer, epochs, device):
    """Train the VAE model."""
    model.train()
    train_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            recon_batch, mu, log_var = model(data)
            
            # Compute loss
            loss = vae_loss(recon_batch, data, mu, log_var)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item()/len(data):.4f}")
        
        avg_epoch_loss = epoch_loss / len(train_loader.dataset)
        train_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_epoch_loss:.4f}")
    
    return train_losses

def plot_latent_space(model, data_loader, device):
    """Plot latent space visualization."""
    model.eval()
    with torch.no_grad():
        all_latent_points = []
        all_colors = []
        
        for batch_idx, data in enumerate(data_loader):
            data = data.to(device)
            mu, _ = model.encode(data)
            all_latent_points.append(mu.cpu().numpy())
            
            # Using batch index for coloring (could be replaced with agent type later)
            all_colors.extend([batch_idx % 10] * len(data))
        
        latent_space = np.vstack(all_latent_points)
        
        # Plotting
        if LATENT_DIM == 2:
            plt.figure(figsize=(10, 8))
            plt.scatter(latent_space[:, 0], latent_space[:, 1], c=all_colors, alpha=0.6, cmap='viridis')
            plt.colorbar(label='Batch index')
            plt.title(f'VAE Latent Space (2D)')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.savefig(os.path.join(DATA_PATH, 'vae_latent_space_2d.png'))
            plt.close()
        
        elif LATENT_DIM == 3:
            # 3D plot
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(
                latent_space[:, 0], 
                latent_space[:, 1], 
                latent_space[:, 2],
                c=all_colors, 
                alpha=0.6, 
                cmap='viridis'
            )
            plt.colorbar(scatter, label='Batch index')
            ax.set_title(f'VAE Latent Space (3D)')
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.set_zlabel('Dimension 3')
            plt.savefig(os.path.join(DATA_PATH, 'vae_latent_space_3d.png'))
            plt.close()
        
        return latent_space, all_colors

def apply_pca_and_tsne(embeddings, perplexity=30, n_iter=1000):
    """Apply PCA and t-SNE to the embeddings and return the results."""
    # Apply PCA for 2D and 3D visualization
    pca_2d = PCA(n_components=2, random_state=RANDOM_SEED)
    pca_3d = PCA(n_components=3, random_state=RANDOM_SEED)
    
    pca_result_2d = pca_2d.fit_transform(embeddings)
    pca_result_3d = pca_3d.fit_transform(embeddings)
    
    # Apply t-SNE for 2D and 3D visualization
    tsne_2d = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter, random_state=RANDOM_SEED)
    tsne_3d = TSNE(n_components=3, perplexity=perplexity, max_iter=n_iter, random_state=RANDOM_SEED)
    
    tsne_result_2d = tsne_2d.fit_transform(embeddings)
    tsne_result_3d = tsne_3d.fit_transform(embeddings)
    
    return {
        'pca_2d': pca_result_2d,
        'pca_3d': pca_result_3d,
        'tsne_2d': tsne_result_2d,
        'tsne_3d': tsne_result_3d
    }

def plot_reduction_results(results, colors, method_name):
    """Plot the results of dimensionality reduction."""
    # 2D plot
    plt.figure(figsize=(10, 8))
    plt.scatter(results[f'{method_name}_2d'][:, 0], results[f'{method_name}_2d'][:, 1], 
                c=colors, alpha=0.6, cmap='viridis')
    plt.colorbar(label='Batch index')
    plt.title(f'{method_name.upper()} 2D Visualization of VAE-encoded data')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig(os.path.join(DATA_PATH, f'vae_{method_name}_2d.png'))
    plt.close()
    
    # 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        results[f'{method_name}_3d'][:, 0], 
        results[f'{method_name}_3d'][:, 1], 
        results[f'{method_name}_3d'][:, 2],
        c=colors, 
        alpha=0.6, 
        cmap='viridis'
    )
    plt.colorbar(scatter, label='Batch index')
    ax.set_title(f'{method_name.upper()} 3D Visualization of VAE-encoded data')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    plt.savefig(os.path.join(DATA_PATH, f'vae_{method_name}_3d.png'))
    plt.close()

def main():
    """Main function to train VAE and reduce dimensions."""
    # Load embeddings
    faiss_path = os.path.join(DATA_PATH, 'faiss_index.faiss')
    json_path = os.path.join(DATA_PATH, 'faiss_index.json')
    
    print("Loading embeddings from FAISS index...")
    embeddings, metadata = load_embeddings(faiss_path, json_path)
    
    # Normalize embeddings (since they use cosine similarity)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms
    
    # Split data
    train_data, test_data = train_test_split(
        normalized_embeddings, test_size=0.2, random_state=RANDOM_SEED
    )
    
    print(f"Training data: {train_data.shape}, Test data: {test_data.shape}")
    
    # Create datasets and data loaders
    train_dataset = AgentEmbeddingDataset(train_data)
    test_dataset = AgentEmbeddingDataset(test_data)
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )
    
    # Initialize model
    model = VAE(
        input_dim=EMBEDDING_DIM,
        latent_dim=LATENT_DIM,
        hidden_dims=HIDDEN_DIMS
    ).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train model
    print("Training VAE model...")
    train_losses = train_vae(model, train_loader, optimizer, EPOCHS, device)
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS+1), train_losses)
    plt.title('VAE Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(DATA_PATH, 'vae_training_loss.png'))
    plt.close()
    
    # Get latent space
    print("Getting latent space for visualization...")
    latent_space, colors = plot_latent_space(model, test_loader, device)
    
    # Apply PCA and t-SNE to latent space
    print("Applying PCA and t-SNE to latent space...")
    reduction_results = apply_pca_and_tsne(latent_space, perplexity=30, n_iter=1000)
    
    # Plot PCA and t-SNE results
    print("Plotting PCA results...")
    plot_reduction_results(reduction_results, colors, 'pca')
    
    print("Plotting t-SNE results...")
    plot_reduction_results(reduction_results, colors, 'tsne')
    
    # Save model
    torch.save(model.state_dict(), os.path.join(DATA_PATH, f'vae_model_{LATENT_DIM}d.pt'))
    print(f"Model saved to {os.path.join(DATA_PATH, f'vae_model_{LATENT_DIM}d.pt')}")
    
    # Generate reduced embeddings for all data
    model.eval()
    all_data_loader = DataLoader(
        AgentEmbeddingDataset(normalized_embeddings),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    reduced_embeddings = []
    with torch.no_grad():
        for data in all_data_loader:
            data = data.to(device)
            mu, _ = model.encode(data)
            reduced_embeddings.append(mu.cpu().numpy())
    
    reduced_embeddings = np.vstack(reduced_embeddings)
    
    # Apply PCA and t-SNE to all reduced embeddings
    print("Applying PCA and t-SNE to all VAE compressed embeddings...")
    all_colors = np.arange(len(reduced_embeddings)) % 10  # Simple coloring scheme
    all_reduction_results = apply_pca_and_tsne(reduced_embeddings, perplexity=50, n_iter=2000)
    
    # Plot results for all data
    print("Plotting final PCA and t-SNE visualizations...")
    plot_reduction_results(all_reduction_results, all_colors, 'pca')
    plot_reduction_results(all_reduction_results, all_colors, 'tsne')
    
    # Save reduced embeddings
    np.save(os.path.join(DATA_PATH, f'agent_embeddings_{LATENT_DIM}d.npy'), reduced_embeddings)
    print(f"Reduced embeddings saved to {os.path.join(DATA_PATH, f'agent_embeddings_{LATENT_DIM}d.npy')}")
    
    print("Done!")

if __name__ == "__main__":
    main() 