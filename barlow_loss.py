import torch

def barlow_twins_loss(z1, z2, lambd=5e-3):
    """
    Compute Barlow Twins loss for contrastive learning
    
    Args:
        z1: First batch of latent representations
        z2: Second batch of latent representations
        lambd: Weight for the off-diagonal terms
    
    Returns:
        Barlow Twins loss
    """
    # Normalize the representations along the batch dimension
    N, D = z1.size()
    z1_norm = (z1 - z1.mean(0)) / (z1.std(0) + 1e-6)  # Added epsilon for numerical stability
    z2_norm = (z2 - z2.mean(0)) / (z2.std(0) + 1e-6)
    
    # Cross-correlation matrix
    c = torch.mm(z1_norm.T, z2_norm) / N
    
    # Loss computation
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()
    
    return on_diag + lambd * off_diag

def off_diagonal(x):
    """
    Return a flattened view of the off-diagonal elements of a square matrix
    """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()