import numpy as np
import torch
import numpy as np
import random

def generate_split(total_samples, train_size=0.8, val_size=0.1, test_size=0.1, random_seed=42):
    """
    Generates train, validation, and test splits based on total_samples.
    
    Parameters:
        total_samples (int): The total number of samples (or indexes).
        train_size (float): Proportion of data used for training.
        val_size (float): Proportion of data used for validation.
        test_size (float): Proportion of data used for testing.
        random_seed (int): Random seed for reproducibility.
    
    Returns:
        tuple: train_indices, val_indices, test_indices
    """
    # Ensure the sum of sizes is 1
    assert train_size + val_size + test_size == 1, "Sizes must sum to 1."
    
    # Create an array of indices
    indices = np.arange(total_samples)
    
    # Shuffle the indices to randomize the split
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    
    # Calculate split sizes
    train_end = int(train_size * total_samples)
    val_end = int((train_size + val_size) * total_samples)
    
    # Generate splits
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    return train_indices, val_indices, test_indices


def set_seed(seed):
    random.seed(seed)
    #os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def to_standard(X):
    """Standardize a dataset without affecting relative scaling.
        return (X - mean) / std
    std is the largest standard deviation and mean is the mean
    of the features.

    Args:
        X (array, tensor): array or tensor of shape (n points, dim)

    Returns: tuple (X, mean, std)
        X (array, tensor): array or tensor of shape (n points, dim)
        mean (array, tensor): array or tensor of shape (dim, )
        std (float): standard deviation
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X-mean)/std
    return X, mean, std

def from_standard(X, mean, std):
    """Un-standardize a dataset without affecting relative scaling.
        return X * std + mean
    This inverts the to_standard() function

    Args:
        X (array, tensor): array or tensor of shape (n points, dim)
        mean (array, tensor): array or tensor of shape (dim, )
        std (float): standard deviation

    Returns:
        (array, tensor): array or tensor of shape (n points, dim)
    """
    return X * std + mean