"""
Helper functions and utilities for facial keypoint detection.
Includes transforms, visualization, and analysis functions.
"""

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from collections import OrderedDict


class ConvLayerVisualizer:
    """Visualize and analyze convolutional layer activations."""
    
    @staticmethod
    def get_conv_output(model, images, layer_name='conv3'):
        """Extract output from a specific convolutional layer.
        
        Args:
            model: Neural network model
            images: Input tensor [batch_size, 1, 224, 224]
            layer_name: Name of conv layer to extract from (e.g., 'conv1', 'conv2', 'conv3')
            
        Returns:
            Tensor of shape [batch_size, num_filters, height, width]
        """
        activation = {}
        
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        
        # Find and hook the layer
        layer_found = False
        for name, layer in model.named_modules():
            if layer_name in name and isinstance(layer, torch.nn.Conv2d):
                layer.register_forward_hook(get_activation(layer_name))
                layer_found = True
                break
        
        if not layer_found:
            raise ValueError(f"Layer '{layer_name}' not found in model")
        
        # Forward pass
        with torch.no_grad():
            _ = model(images)
        
        return activation.get(layer_name)
    
    @staticmethod
    def visualize_filters(conv_output, num_filters=16, figsize=(16, 8)):
        """Visualize multiple filters from conv layer output.
        
        Args:
            conv_output: Tensor of shape [batch_size, num_filters, height, width]
            num_filters: Number of filters to display
            figsize: Size of output figure
        """
        # Get first image from batch
        activations = conv_output[0].cpu().numpy()
        num_output_filters = activations.shape[0]
        
        # Limit to available filters
        num_filters = min(num_filters, num_output_filters)
        
        fig, axes = plt.subplots(4, 4, figsize=figsize)
        axes = axes.flatten()
        
        for i in range(num_filters):
            activation_map = activations[i]
            
            # Normalize to [0, 1] for display
            activation_min = activation_map.min()
            activation_max = activation_map.max()
            if activation_max > activation_min:
                activation_map = (activation_map - activation_min) / (activation_max - activation_min)
            
            axes[i].imshow(activation_map, cmap='hot')
            axes[i].set_title(f'Filter {i}')
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(num_filters, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def project_to_image(conv_output, original_image, layer_index=0, figsize=(12, 4)):
        """Project a single filter's activation back onto the original image.
        
        Args:
            conv_output: Tensor of shape [batch_size, num_filters, height, width]
            original_image: Original input image tensor [batch_size, 1, height, width]
            layer_index: Which filter to project (0-indexed)
            figsize: Figure size
        """
        activation_map = conv_output[0, layer_index].cpu().numpy()
        original_img = original_image[0, 0].cpu().numpy()
        
        # Normalize activation to [0, 1]
        activation_min = activation_map.min()
        activation_max = activation_map.max()
        if activation_max > activation_min:
            activation_map = (activation_map - activation_min) / (activation_max - activation_min)
        
        # Upscale activation to match original image size
        original_size = original_img.shape
        activation_upscaled = cv2.resize(
            activation_map, 
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Create heatmap overlay
        heatmap = cv2.applyColorMap(
            (activation_upscaled * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Blend original image with heatmap
        original_bgr = cv2.cvtColor(
            (original_img * 255).astype(np.uint8),
            cv2.COLOR_GRAY2BGR
        )
        blended = cv2.addWeighted(original_bgr, 0.6, heatmap, 0.4, 0)
        
        # Display
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        axes[0].imshow(original_img, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(activation_upscaled, cmap='hot')
        axes[1].set_title(f'Activation Map (Filter {layer_index})')
        axes[1].axis('off')
        
        axes[2].imshow(blended)
        axes[2].set_title('Blended')
        axes[2].axis('off')
        
        plt.tight_layout()
        return fig


def calculate_output_variance(model, data_loader, device, num_batches=5):
    """Calculate variance in model predictions across different images.
    
    Args:
        model: Neural network model
        data_loader: DataLoader with images
        device: torch device
        num_batches: Number of batches to sample
        
    Returns:
        dict with variance statistics
    """
    model.eval()
    outputs_list = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
            
            images = batch['image'].to(device).float()
            outputs = model(images)
            outputs_list.append(outputs.cpu())
    
    all_outputs = torch.cat(outputs_list, dim=0)
    
    return {
        'mean': all_outputs.mean().item(),
        'std': all_outputs.std().item(),
        'variance': all_outputs.var().item(),
        'per_output_variance': all_outputs.var(dim=0).mean().item(),
        'min': all_outputs.min().item(),
        'max': all_outputs.max().item(),
        'shape': all_outputs.shape
    }


def build_conv_layers(conv_configs, input_channels=1):
    """Build convolutional layers dynamically.
    
    Args:
        conv_configs: List of dicts with keys:
            - 'out_channels': number of output filters
            - 'kernel_size': kernel size (default: 3)
            - 'pool_size': max pool size (default: 2)
            - 'activation': activation function (default: F.relu)
            
        input_channels: Number of input channels (default: 1 for grayscale)
        
    Returns:
        nn.Sequential with conv layers
        list of output channel counts for FC layer calculation
    """
    layers = OrderedDict()
    in_channels = input_channels
    channel_counts = []
    
    for i, config in enumerate(conv_configs):
        out_channels = config.get('out_channels', 32)
        kernel_size = config.get('kernel_size', 3)
        pool_size = config.get('pool_size', 2)
        
        # Conv layer
        layers[f'conv{i+1}'] = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0
        )
        
        # Pool layer
        layers[f'pool{i+1}'] = nn.MaxPool2d(pool_size)
        
        channel_counts.append(out_channels)
        in_channels = out_channels
    
    return nn.Sequential(layers), channel_counts


def calculate_conv_output_dim(input_dim, conv_configs):
    """Calculate output dimension after convolutional layers.
    
    Args:
        input_dim: Input image dimension (e.g., 224)
        conv_configs: List of conv layer configs
        
    Returns:
        Output spatial dimension
    """
    dim = input_dim
    
    for config in conv_configs:
        kernel_size = config.get('kernel_size', 3)
        pool_size = config.get('pool_size', 2)
        
        # After convolution
        dim = dim - kernel_size + 1
        
        # After pooling
        dim = dim // pool_size
        
        if dim <= 0:
            raise ValueError(f"Dimension became invalid: {dim} after config {config}")
    
    return dim


class GridSearchLogger:
    """Log grid search results to file."""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.results = []
    
    def log_result(self, config, metrics):
        """Log a single result.
        
        Args:
            config: Model configuration dict
            metrics: Metrics dict with keys like 'loss', 'accuracy', etc.
        """
        result = {
            'config': config,
            'metrics': metrics
        }
        self.results.append(result)
        
        # Write to file immediately
        self._write_single_result(result)
    
    def _write_single_result(self, result):
        """Append a single result to file."""
        with open(self.filepath, 'a') as f:
            f.write(f"\n{'='*70}\n")
            f.write(f"CONFIG: {result['config']}\n")
            f.write(f"METRICS:\n")
            for key, value in result['metrics'].items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.6f}\n")
                else:
                    f.write(f"  {key}: {value}\n")
    
    def get_results(self):
        """Get all results as list."""
        return self.results


def get_gpu_memory_usage(device):
    """Get current GPU memory usage in MB.
    
    Args:
        device: torch device
        
    Returns:
        dict with allocated and reserved memory
    """
    if device.type != 'cuda':
        return {'allocated_mb': 0, 'reserved_mb': 0}
    
    return {
        'allocated_mb': torch.cuda.memory_allocated(device) / 1024 / 1024,
        'reserved_mb': torch.cuda.memory_reserved(device) / 1024 / 1024,
        'max_allocated_mb': torch.cuda.max_memory_allocated(device) / 1024 / 1024
    }


def gpu_memory_limit_warning(device, limit_percentage=95):
    """Check if GPU memory usage exceeds limit.
    
    Args:
        device: torch device
        limit_percentage: Warning threshold (0-100)
        
    Returns:
        bool: True if over limit
    """
    if device.type != 'cuda':
        return False
    
    props = torch.cuda.get_device_properties(0)
    total_memory = props.total_memory / 1024 / 1024  # Convert to MB
    allocated = torch.cuda.memory_allocated(device) / 1024 / 1024
    
    usage_percentage = (allocated / total_memory) * 100
    
    if usage_percentage > limit_percentage:
        print(f"⚠️  GPU Memory Warning: {usage_percentage:.1f}% of {total_memory:.0f}MB used")
        return True
    
    return False
