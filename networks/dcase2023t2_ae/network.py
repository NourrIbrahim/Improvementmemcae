import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import math

class AENet(nn.Module):
    def __init__(self, input_dim, block_size=64):
        super(AENet, self).__init__()
        self.input_dim = input_dim
        
        # Parameters for domain adaptation (keeping from original)
        self.cov_source = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)
        self.cov_target = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)
        
        # Latent dimension
        self.latent_dim = 128
        
        # Encoder with fully connected layers for 2D input
        self.encoder_block1 = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        
        self.encoder_block2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        self.encoder_block3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        self.encoder_output = nn.Sequential(
            nn.Linear(128, self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            nn.ReLU(),
        )
        
        # Memory module
        self.mem_size = 2000  # As mentioned in the paper
        self.memory = nn.Parameter(torch.Tensor(self.mem_size, self.latent_dim))
        self.temperature = 0.1  # Temperature for softmax
        
        # Decoder with fully connected layers
        self.decoder_block1 = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        self.decoder_block2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        self.decoder_block3 = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        
        self.decoder_output = nn.Linear(512, self.input_dim)
        
        # Domain classifier (keeping from original)
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Initialize memory and weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        # Initialize memory with orthogonal weights
        init.orthogonal_(self.memory)
        
        # Initialize other weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
    
    def memory_addressing(self, z):
        """
        Memory addressing mechanism as described in the paper
        
        Args:
            z: encoded representation [batch_size, latent_dim]
            
        Returns:
            z_hat: retrieved memory representation
            addressing_weights: softmax weights for memory slots
        """
        # Compute dot product between z and each memory slot
        # Equation (4) in the paper
        similarity = torch.matmul(z, self.memory.t()) / self.temperature
        
        # Apply softmax to get addressing weights w
        addressing_weights = F.softmax(similarity, dim=1)
        
        # Weighted sum of memory items to get z_hat
        # Equation (3) in the paper
        z_hat = torch.matmul(addressing_weights, self.memory)
        
        return z_hat, addressing_weights
    
    def encode(self, x):
        """
        Encode input to latent representation
        
        Args:
            x: input features [batch_size, input_dim]
            
        Returns:
            z: latent representation [batch_size, latent_dim]
        """
        # Pass through encoder blocks
        h1 = self.encoder_block1(x)
        h2 = self.encoder_block2(h1)
        h3 = self.encoder_block3(h2)
        z = self.encoder_output(h3)
        
        return z
    
    def decode(self, z):
        """
        Decode latent representation to reconstruction
        
        Args:
            z: latent representation [batch_size, latent_dim]
            
        Returns:
            x_hat: reconstructed features [batch_size, input_dim]
        """
        # Pass through decoder blocks
        h1 = self.decoder_block1(z)
        h2 = self.decoder_block2(h1)
        h3 = self.decoder_block3(h2)
        x_hat = self.decoder_output(h3)
        
        return x_hat
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: input features [batch_size, input_dim]
            
        Returns:
            x_hat: reconstructed features
            z: original latent representation
        """
        # Encode input
        z = self.encode(x)
        
        # Memory addressing
        z_hat, _ = self.memory_addressing(z)
        
        # Decode with memory-enhanced representation
        x_hat = self.decode(z_hat)
        
        return x_hat, z
    
    def get_reconstruction_error(self, x):
        """
        Calculate reconstruction error for anomaly detection
        
        Args:
            x: input features [batch_size, input_dim]
            
        Returns:
            error: reconstruction error for each sample [batch_size]
        """
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            x_hat, _ = self.forward(x)
            # Mean squared error as reconstruction error
            error = F.mse_loss(x_hat, x, reduction='none')
            # Sum over all dimensions except batch
            error = error.view(error.size(0), -1).mean(dim=1)
        return error
    
    def domain_prediction(self, z, lambda_param=1.0):
        """
        Domain prediction for domain adaptation (keeping from original)
        
        Args:
            z: latent representation [batch_size, latent_dim]
            lambda_param: gradient reversal parameter
            
        Returns:
            domain_pred: domain prediction [batch_size, 1]
        """
        # Gradient reversal for adversarial training
        reverse_feature = GradientReversalFunction.apply(z, lambda_param)
        domain_pred = self.domain_classifier(reverse_feature)
        return domain_pred


# Gradient Reversal Layer for domain adaptation (keeping from original)
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_param):
        ctx.lambda_param = lambda_param
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_param, None


def train_model(model, train_loader, val_loader=None, num_epochs=100, device='cuda'):
    """
    Training function for AENet
    
    Args:
        model: AENet model
        train_loader: DataLoader for training data (normal samples only)
        val_loader: Optional DataLoader for validation
        num_epochs: Number of training epochs
        device: Device to train on ('cuda' or 'cpu')
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            # Forward pass
            recon, z = model(data)
            
            # Reconstruction loss
            recon_loss = F.mse_loss(recon, data)
            
            # Memory addressing for entropy loss
            _, addressing_weights = model.memory_addressing(z)
            
            # Memory entropy loss (to encourage diverse memory usage)
            entropy_loss = -torch.mean(torch.sum(addressing_weights * 
                                               torch.log(addressing_weights + 1e-12), dim=1))
            
            # Total loss
            loss = recon_loss - 0.0002 * entropy_loss  # Small weight for entropy loss
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'====> Epoch: {epoch} Average loss: {avg_loss:.6f}')
        
        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data, _ in val_loader:
                    data = data.to(device)
                    recon, _ = model(data)
                    val_loss += F.mse_loss(recon, data).item()
            
            avg_val_loss = val_loss / len(val_loader)
            print(f'====> Validation loss: {avg_val_loss:.6f}')
            
            # Update learning rate based on validation loss
            scheduler.step(avg_val_loss)
    
    return model