"""
Training utilities and visualization functions for facial keypoints detection.

This module contains helper functions and utilities extracted from the notebook
to keep the notebook clean and focused on model architecture and training logic.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime


class WeightCheckpoint:
    """
    Manages weight checkpointing for training resumption after power loss or interruptions.
    Saves checkpoints both per-epoch and every 20 minutes of training.
    """
    
    def __init__(self, checkpoint_dir='checkpoints', model_name='facial_keypoints', checkpoint_interval_minutes=20):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
            model_name: Base name for checkpoint files
            checkpoint_interval_minutes: Time interval in minutes for time-based checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.checkpoint_interval_minutes = checkpoint_interval_minutes
        self.last_checkpoint_time = datetime.now()
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, model, optimizer, epoch, loss, metrics=None, force=False):
        """
        Save model checkpoint with metadata.
        Saves on every epoch and additionally every 20 minutes.
        
        Args:
            model: PyTorch model to save
            optimizer: Optimizer with current state
            epoch: Current epoch number
            loss: Current loss value
            metrics: Optional dict of metrics to save
            force: If True, force save regardless of time interval
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics or {}
        }
        
        # Save latest checkpoint (overwrite every epoch)
        latest_path = os.path.join(self.checkpoint_dir, f'{self.model_name}_latest.pt')
        torch.save(checkpoint, latest_path)
        
        # Save epoch-specific checkpoint every 5 epochs
        if epoch % 5 == 0:
            epoch_path = os.path.join(self.checkpoint_dir, f'{self.model_name}_epoch_{epoch:04d}.pt')
            torch.save(checkpoint, epoch_path)
            print(f"✓ Epoch checkpoint saved: {epoch_path}")
        
        # Check if time-based checkpoint is needed
        current_time = datetime.now()
        time_elapsed = (current_time - self.last_checkpoint_time).total_seconds() / 60  # minutes
        
        if force or time_elapsed >= self.checkpoint_interval_minutes:
            time_checkpoint_path = os.path.join(self.checkpoint_dir, 
                f'{self.model_name}_time_{current_time.strftime("%Y%m%d_%H%M%S")}.pt')
            torch.save(checkpoint, time_checkpoint_path)
            self.last_checkpoint_time = current_time
            print(f"✓ Time-based checkpoint saved (every {self.checkpoint_interval_minutes} mins): {time_checkpoint_path}")
        
        return latest_path
    
    def load_checkpoint(self, model, optimizer=None, device='cpu'):
        """
        Load latest checkpoint for resuming training.
        
        Args:
            model: PyTorch model to load weights into
            optimizer: Optional optimizer to load state into
            device: Device to load to
        
        Returns:
            dict with checkpoint info (epoch, loss, metrics) or None if no checkpoint found
        """
        latest_path = os.path.join(self.checkpoint_dir, f'{self.model_name}_latest.pt')
        
        if not os.path.exists(latest_path):
            return None
        
        try:
            checkpoint = torch.load(latest_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            print(f"✓ Checkpoint loaded from epoch {checkpoint['epoch']}")
            print(f"  Loss: {checkpoint['loss']:.6f}")
            print(f"  Timestamp: {checkpoint['timestamp']}")
            
            return {
                'epoch': checkpoint['epoch'],
                'loss': checkpoint['loss'],
                'metrics': checkpoint.get('metrics', {})
            }
        except Exception as e:
            print(f"✗ Failed to load checkpoint: {str(e)}")
            return None
    
    def get_best_checkpoint(self):
        """Get path to best checkpoint if it exists."""
        best_path = os.path.join(self.checkpoint_dir, f'{self.model_name}_best.pt')
        return best_path if os.path.exists(best_path) else None


def show_all_keypoints(image, predicted_key_pts, gt_pts=None):
    """
    Visualize image with predicted and ground truth keypoints.
    
    Args:
        image: Grayscale image to display
        predicted_key_pts: Predicted keypoint coordinates
        gt_pts: Optional ground truth keypoint coordinates
    """
    plt.imshow(image, cmap='gray')
    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], 
                s=20, marker='.', c='m', label='Predicted')
    if gt_pts is not None:
        plt.scatter(gt_pts[:, 0], gt_pts[:, 1], 
                    s=20, marker='.', c='g', label='Ground Truth')


def visualize_output(test_images, test_outputs, gt_pts=None, batch_size=10):
    """
    Visualize model predictions on test images.
    
    Args:
        test_images: Batch of test images
        test_outputs: Model predictions for keypoints
        gt_pts: Optional ground truth keypoints
        batch_size: Number of images to visualize
    """
    for i in range(batch_size):
        plt.figure(figsize=(20, 10))
        ax = plt.subplot(1, batch_size, i + 1)
        
        # Un-transform image data
        image = test_images[i].data
        image = image.cpu()
        image = image.numpy()
        image = np.transpose(image, (1, 2, 0))
        
        # Un-transform predicted keypoints
        predicted_key_pts = test_outputs[i].data
        predicted_key_pts = predicted_key_pts.cpu()
        predicted_key_pts = predicted_key_pts.numpy()
        # Undo normalization of keypoints (inverse of: (x - 49.07) / 13.00)
        predicted_key_pts = predicted_key_pts * 13.00 + 49.07
        
        # Prepare ground truth points if provided
        ground_truth_pts = None
        if gt_pts is not None:
            ground_truth_pts = gt_pts[i].data.cpu().numpy()
            ground_truth_pts = ground_truth_pts * 13.00 + 49.07
        
        show_all_keypoints(np.squeeze(image), predicted_key_pts, ground_truth_pts)
        plt.axis('off')
    
    plt.show()


def compute_mse_loss(predictions, targets):
    """Compute mean squared error loss."""
    criterion = nn.MSELoss()
    return criterion(predictions, targets).item()


def compute_mae_loss(predictions, targets):
    """Compute mean absolute error loss."""
    criterion = nn.L1Loss()
    return criterion(predictions, targets).item()


def log_training_metrics(epoch, loss, lr=None, batch_time=None):
    """
    Log training metrics in a formatted way.
    
    Args:
        epoch: Current epoch
        loss: Loss value
        lr: Optional learning rate
        batch_time: Optional batch processing time
    """
    log_str = f"Epoch {epoch:3d} | Loss: {loss:.6f}"
    if lr is not None:
        log_str += f" | LR: {lr:.6f}"
    if batch_time is not None:
        log_str += f" | Time: {batch_time:.3f}s"
    print(log_str)
