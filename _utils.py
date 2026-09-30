import torch
import os
import random
import numpy as np

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_indices(dataset, ratio=0.8, seed=42):
    num_data = len(dataset)
    indices = torch.randperm(num_data, generator=torch.Generator().manual_seed(seed)).tolist()
    train_size = int(num_data * ratio)
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    return train_indices, val_indices

def get_random_sample(labels, sample_per_class, seed=42):
    if isinstance(labels, dict):
        labels = labels["labels"]

    class_to_indices = {}
    
    for i, label in enumerate(labels):
        if torch.is_tensor(label) and label.ndim == 0:
            key = label.item()
            class_to_indices.setdefault(key, []).append(i)
        else:
            if torch.is_tensor(label):
                if label.ndim == 1:
                    label_list = label.tolist()
                else:
                    label_list = torch.unique(label).tolist()
            else: 
                label_list = label
            
            for key in set(label_list): 
                class_to_indices.setdefault(key, []).append(i)
    
    selected_indices = []
    rng = random.Random(seed)

    for class_label, indices in class_to_indices.items():
        if len(indices) < sample_per_class:
            raise ValueError(f"Class {class_label} has only {len(indices)} samples, but SAMPLE_PER_CLASS={sample_per_class}")

        selected_indices.extend(rng.sample(indices, sample_per_class))
    
    return sorted(selected_indices)