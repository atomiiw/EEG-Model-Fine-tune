"""
Training utilities for EEG model fine-tuning

Key improvements implemented to fix learning issues:
1. Euclidean Alignment (EA) consistency: Compute EA matrix on training data once,
   then apply the same transformation to validation data to avoid distribution mismatch.
2. Per-trial z-score normalization: Normalize each trial before EA for stable learning.
3. Discriminative learning rates: Use higher LR (2x) for classifier head, lower (0.1x)
   for backbone to allow the new head to learn while preserving pretrained features.
4. Label smoothing (0.1): Prevents overconfident predictions and improves generalization.
5. Proper scheduler usage: LR decay applied per-epoch (not per-batch) to prevent
   premature learning rate collapse.
6. Improved logging: Clear train vs val accuracy tracking to diagnose learning issues.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from scipy.linalg import fractional_matrix_power
from scipy.spatial.distance import cdist
from utils.channel_list import *
import random
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
import torch.nn as nn
from dataset import EEGDataset
from model.mlm import mlm_mask
import os


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pad_missing_channels_diff(x, target_channels, actual_channels):
    B, C, T = x.shape
    num_target = len(target_channels)
    
    existing_pos = np.array([channel_positions[ch] for ch in actual_channels])

    target_pos = np.array([channel_positions[ch] for ch in target_channels])
    
    W = np.zeros((num_target, C))
    for i, (target_ch, pos) in enumerate(zip(target_channels, target_pos)):
        if target_ch in actual_channels:
            src_idx = actual_channels.index(target_ch)
            W[i, src_idx] = 1.0
        else:
            dist = cdist([pos], existing_pos)[0]
            weights = 1 / (dist + 1e-6)  
            weights /= weights.sum()     
            W[i] = weights
    
    padded = np.zeros((B, num_target, T))
    for b in range(B):
        padded[b] = W @ x[b]  
    
    return padded


def EA(x, return_matrix=False, W=None):
    """
    Euclidean Alignment with optional matrix reuse

    Parameters
    ----------
    x : numpy array
        data of shape (num_samples, num_channels, num_time_samples)
    return_matrix : bool
        if True, return both aligned data and transformation matrix
    W : numpy array or None
        pre-computed transformation matrix to apply (if provided, skips computation)

    Returns
    ----------
    XEA : numpy array
        data of shape (num_samples, num_channels, num_time_samples)
    sqrtRefEA : numpy array (optional)
        transformation matrix, returned only if return_matrix=True
    """
    if W is not None:
        # Apply pre-computed transformation matrix
        XEA = np.zeros(x.shape)
        for i in range(x.shape[0]):
            XEA[i] = np.dot(W, x[i])
        return XEA

    # Compute transformation matrix from data
    cov = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        cov[i] = np.cov(x[i])
    refEA = np.mean(cov, 0)
    sqrtRefEA = fractional_matrix_power(refEA, -0.5)
    XEA = np.zeros(x.shape)
    for i in range(x.shape[0]):
        XEA[i] = np.dot(sqrtRefEA, x[i])

    if return_matrix:
        return XEA, sqrtRefEA
    return XEA


def process_and_replace_loader(loader, ischangechn, dataset, return_ea=False, W=None):
    """
    Process data loader with EA and optional normalization

    Parameters
    ----------
    loader : DataLoader
        input data loader
    ischangechn : bool
        whether to pad/interpolate missing channels
    dataset : str
        dataset name for channel configuration
    return_ea : bool
        if True, return both loader and EA transformation matrix
    W : numpy array or None
        pre-computed EA transformation matrix to apply

    Returns
    ----------
    new_loader : DataLoader
        processed data loader
    ea_matrix : numpy array (optional)
        EA transformation matrix, returned only if return_ea=True
    """
    all_data = []
    all_labels = []
    for i in range(len(loader.dataset)):
        data, label = loader.dataset[i]
        all_data.append(data.numpy())
        all_labels.append(label)

    data_np = np.stack(all_data, axis=0)
    labels_tensor = torch.stack(all_labels)

    # Per-trial z-score normalization (critical for stable learning)
    data_np = (data_np - data_np.mean(axis=2, keepdims=True)) / (data_np.std(axis=2, keepdims=True) + 1e-6)

    # Apply Euclidean Alignment
    if W is not None:
        # Use pre-computed transformation from training set
        processed_data = EA(data_np, W=W).astype(np.float32)
        ea_matrix = None
    elif return_ea:
        # Compute and return transformation matrix
        processed_data, ea_matrix = EA(data_np, return_matrix=True)
        processed_data = processed_data.astype(np.float32)
    else:
        # Standard EA without returning matrix
        processed_data = EA(data_np).astype(np.float32)
        ea_matrix = None

    if ischangechn:
        print("before processed：", processed_data.shape)
        if dataset == 'BNCI2014001':
            channels_names = BNCI2014001_chn_names
        elif dataset == 'BNCI2014004':
            channels_names = BNCI2014004_chn_names
        elif dataset == 'BNCI2014001-4':
            channels_names = BNCI2014001_chn_names
        elif dataset == 'AlexMI':
            channels_names = AlexMI_chn_names
        elif dataset =='BNCI2015001':
            channels_names = BNCI2015001_chn_names
        else:
            channels_names = use_channels_names
        processed_data = pad_missing_channels_diff(processed_data, use_channels_names, channels_names)
        print("after processed：", processed_data.shape)

    new_dataset = TensorDataset(
        torch.from_numpy(processed_data).float(),
        labels_tensor
    )

    loader_args = {
        'batch_size': loader.batch_size,
        'num_workers': loader.num_workers,
        'pin_memory': loader.pin_memory,
        'drop_last': loader.drop_last,
        'shuffle': isinstance(loader.sampler, torch.utils.data.RandomSampler)
    }

    new_loader = torch.utils.data.DataLoader(new_dataset, **loader_args)

    if return_ea:
        return new_loader, ea_matrix
    return new_loader


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)

        _, outputs = model(data)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    current_lr = optimizer.param_groups[0]['lr']

    epoch_loss = running_loss / len(train_loader)
    accuracy = correct / total * 100

    return epoch_loss, accuracy, current_lr

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    ### Added ### 
    all_preds = []
    all_labels = []
    #############

    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            _, outputs = model(data)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            ### Added ### 
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            #############

    epoch_loss = running_loss / len(val_loader)
    accuracy = correct / total * 100

    ### Added ### 

    # Concatenate results for the full val set
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # ✅ Print first few predictions vs labels for inspection
    print("Predicted:", all_preds[:32])
    print("Actual:   ", all_labels[:32])

    # Report accuracies
    print(f"Got {correct} out of {total} correct. Accuracy: {accuracy}.")

    #############

    return epoch_loss, accuracy

def run_experiment(args, log_file):
    """Run complete experiment pipeline with configurable hyperparameters"""
    # Set up experiment tracking
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"./result/log/{args.dataset_name}_{args.model_name}_{now}_log.txt"
    csv_filename = f"./result/acc/{args.dataset_name}_{args.model_name}_{now}_results.csv"
    
    # Create log file handler
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    file_handler = open(log_filename, 'w')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []

    # Dataset configuration
    dataset_subjects = {
        'BNCI2014001': 9,
        'BNCI2015001': 12,
        'BNCI2014004': 9,
        'BNCI2014001-4': 9,
        'AlexMI': 8
    }
    args.sub_num = dataset_subjects.get(args.dataset_name, 0)
    if args.sub_num == 0:  # custom dataset fallback
        args.sub_num = 1
    
    # Seed iteration
    for seed_offset in range(args.num_exp):
        seed = seed_offset + 666
        set_seed(seed)
        subject_results = {}

        # Subject iteration
        for subject in range(args.sub_num):
            log_message = f"Starting Subject {subject}: Seed {seed}\n"
            log_file.write(log_message)
            file_handler.write(log_message)
            val_acc = train_subject(args, subject, seed, device, log_file)
            subject_results[subject] = val_acc

        results.append([seed] + list(subject_results.values()))
    
    save_results(results, args.sub_num, csv_filename)
    file_handler.close()

def train_subject(args, subject, seed, device, log_file):
    """Train and validate model for single subject with configurable hyperparams"""
    # Prepare dataset
    args.sub = [subject]
    dataset = EEGDataset(args=args)
    train_data, val_data = train_test_split(dataset, test_size=args.val_split, random_state=seed)
    
    # Configure data loaders
    train_loader = DataLoader(
        train_data, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_data, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    # Preprocess data
    # train_loader = process_and_replace_loader(
    #     train_loader, 
    #     ischangechn=True, 
    #     dataset=args.dataset_name
    # )
    # val_loader = process_and_replace_loader(
    #     val_loader, 
    #     ischangechn=True, 
    #     dataset=args.dataset_name
    # )

    train_loader, ea_matrix = process_and_replace_loader(train_loader, ischangechn=False,
                                                    dataset=args.dataset_name, return_ea=True)
    val_loader = process_and_replace_loader(val_loader, ischangechn=False,
                                        dataset=args.dataset_name, W=ea_matrix)

    
    # Initialize model
    model = mlm_mask(
        emb_size=args.emb_size,
        depth=args.depth,
        n_classes=args.num_classes,
        pretrainmode=False,
        pretrain=args.pretrain_path
    ).to(device)

    # Set up training components with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Use discriminative learning rates: higher LR for classifier head, lower for backbone
    head_params = list(model.classifier.parameters())
    backbone_params = [p for n, p in model.named_parameters() if 'classifier' not in n]

    if args.optimizer == 'adam':
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': args.lr * 0.1, 'weight_decay': args.weight_decay},
            {'params': head_params, 'lr': args.lr * 2.0, 'weight_decay': args.weight_decay}
        ])
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD([
            {'params': backbone_params, 'lr': args.lr * 0.1, 'momentum': args.momentum, 'weight_decay': args.weight_decay},
            {'params': head_params, 'lr': args.lr * 2.0, 'momentum': args.momentum, 'weight_decay': args.weight_decay}
        ])
    else:
        # Fallback to AdamW if optimizer not specified
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': args.lr * 0.1, 'weight_decay': args.weight_decay},
            {'params': head_params, 'lr': args.lr * 2.0, 'weight_decay': args.weight_decay}
        ])
    
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=args.epochs
        )
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.gamma
        )
    else:
        scheduler = None
    
    final_val_acc = 0.0
    print(f"\n{'='*70}")
    print(f"Training Subject {subject} | Seed {seed}")
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")
    print(f"{'='*70}\n")

    # Training loop
    for epoch in range(args.epochs):
        # Training phase
        train_loss, train_acc, curr_lr = train(
            model, train_loader, criterion,
            optimizer, device
        )

        # Validation phase
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )
        final_val_acc = val_acc

        if scheduler is not None:
            scheduler.step()

        # Improved console logging - clearly show train vs val
        print(f"Epoch {epoch+1:03d}/{args.epochs} | "
              f"Train: {train_acc:5.2f}% | Val: {val_acc:5.2f}% | "
              f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
              f"LR: {curr_lr:.5g}")

        # File logging
        log_file.write(
            f"Seed: {seed}, Subject: {subject}, Epoch: {epoch+1}\n"
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
            f"LR: {curr_lr:.6f}\n"
        )
    
    ### Added ###

    # Save the fine-tuned model weights for later inference
    torch.save(model.state_dict(), f"./weight/{args.dataset_name}_{args.model_name}_finetuned.pth")
    print(f"Saved fine-tuned weights to ./weight/{args.dataset_name}_{args.model_name}_finetuned.pth")

    #############
    
    return final_val_acc

def save_results(results, subject_count, filename):
    """Save experiment results to CSV file"""
    columns = ["Seed"] + [f"Subject_{i}_Acc" for i in range(subject_count)]
    results_df = pd.DataFrame(results, columns=columns)
    
    # Calculate summary statistics
    results_df['Seed_Avg_Acc'] = results_df.iloc[:, 1:].mean(axis=1)
    subject_avg = results_df.iloc[:, 1:-1].mean(axis=0)
    seed_avg = results_df['Seed_Avg_Acc'].mean()
    
    # Add summary row
    summary_row = ['Average'] + subject_avg.tolist() + [seed_avg]
    results_df.loc[len(results_df)] = summary_row
    
    # Save to CSV
    results_df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
