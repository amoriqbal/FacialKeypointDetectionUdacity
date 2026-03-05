## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
from collections import OrderedDict


class NetFlexible(nn.Module):
    """Flexible CNN with arbitrary number of conv and FC layers.
    
    Example:
        # With flexible conv layers
        conv_configs = [
            {'out_channels': 32, 'kernel_size': 3, 'pool_size': 2},
            {'out_channels': 64, 'kernel_size': 5, 'pool_size': 2},
            {'out_channels': 128, 'kernel_size': 5, 'pool_size': 2},
        ]
        fc_sizes = (512, 256)  # Before output layer
        model = NetFlexible(conv_configs=conv_configs, fc_sizes=fc_sizes)
    """
    
    def __init__(self, conv_configs=None, fc_sizes=(512, 256), activation_func=F.relu, 
                 dropout=0.2, input_dim=224):
        """
        Args:
            conv_configs: List of dicts with keys 'out_channels', 'kernel_size', 'pool_size'
                         If None, uses default single conv layer
            fc_sizes: Tuple of FC layer sizes (final output is always 136 keypoints)
            activation_func: Activation function (default: F.relu)
            dropout: Dropout rate (default: 0.2)
            input_dim: Input image dimension (default: 224)
        """
        super(NetFlexible, self).__init__()
        
        if conv_configs is None:
            conv_configs = [
                {'out_channels': 32, 'kernel_size': 3, 'pool_size': 2},
                {'out_channels': 64, 'kernel_size': 5, 'pool_size': 2},
            ]
        
        self.conv_configs = conv_configs
        self.fc_sizes = fc_sizes
        self.activation_func = activation_func
        self.dropout = dropout
        self.input_dim = input_dim
        
        # Build convolutional layers
        self.conv_layers, self.channel_counts = self._build_conv_layers(conv_configs)
        
        # Calculate output dimension after conv layers
        self.conv_output_dim = self._calculate_conv_output_dim(input_dim, conv_configs)
        
        # Calculate flatten size
        final_channels = conv_configs[-1]['out_channels']
        flatten_size = final_channels * self.conv_output_dim * self.conv_output_dim
        
        assert flatten_size > 0, f"Flatten size invalid: {flatten_size}"
        
        # Build FC layers
        self.fc_layers = self._build_fc_layers(flatten_size, fc_sizes, dropout)
        
        # Conv dropout
        self.conv_dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def _build_conv_layers(self, conv_configs):
        """Build sequential conv layers from configs."""
        layers = OrderedDict()
        channel_counts = []
        in_channels = 1  # Grayscale input
        
        for i, config in enumerate(conv_configs):
            out_channels = config.get('out_channels', 32)
            kernel_size = config.get('kernel_size', 3)
            pool_size = config.get('pool_size', 2)
            
            # Conv layer
            layers[f'conv{i+1}'] = nn.Conv2d(
                in_channels, out_channels, kernel_size, padding=0
            )
            
            # Pool layer
            layers[f'pool{i+1}'] = nn.MaxPool2d(pool_size)
            
            channel_counts.append(out_channels)
            in_channels = out_channels
        
        return nn.Sequential(layers), channel_counts
    
    def _build_fc_layers(self, input_size, fc_sizes, dropout):
        """Build sequential FC layers."""
        layers = OrderedDict()
        
        # First FC layer
        layers['fc0'] = nn.Linear(input_size, fc_sizes[0])
        if dropout > 0:
            layers['dropout0'] = nn.Dropout(dropout)
        
        # Remaining FC layers
        for i in range(1, len(fc_sizes)):
            layers[f'fc{i}'] = nn.Linear(fc_sizes[i-1], fc_sizes[i])
            if dropout > 0 and i < len(fc_sizes) - 1:
                layers[f'dropout{i}'] = nn.Dropout(dropout)
        
        # Output layer
        layers['fc_output'] = nn.Linear(fc_sizes[-1], 136)
        
        return nn.Sequential(layers)
    
    def _calculate_conv_output_dim(self, input_dim, conv_configs):
        """Calculate output spatial dimension after conv layers."""
        dim = input_dim
        
        for config in conv_configs:
            kernel_size = config.get('kernel_size', 3)
            pool_size = config.get('pool_size', 2)
            
            # After convolution
            dim = dim - kernel_size + 1
            assert dim > 0, f"After conv: dimension {dim} <= 0"
            
            # After pooling
            dim = dim // pool_size
            assert dim > 0, f"After pool: dimension {dim} <= 0"
        
        return dim
    
    def forward(self, x, debug=False):
        """Forward pass with optional debug output."""
        if debug:
            print(f"Input: {x.shape}")
        
        # Conv layers with activation and dropout
        for i, layer in enumerate(self.conv_layers):
            x = layer(x)
            
            # Apply activation after conv (but not pool/dropout)
            if 'conv' in str(type(layer).__name__).lower() or 'Conv' in str(layer):
                # Check next layer to apply activation appropriately
                pass
            
            if debug and ('conv' in str(layer) or isinstance(layer, nn.Conv2d)):
                print(f"After layer {i}: {x.shape}")
        
        # Manual iteration for proper activation placement
        i = 0
        while i < len(self.conv_layers):
            layer = self.conv_layers[i]
            
            if isinstance(layer, nn.Conv2d):
                x = layer(x)
                x = self.activation_func(x)
                if debug:
                    print(f"After conv{i//2 + 1}: {x.shape}")
            elif isinstance(layer, nn.MaxPool2d):
                x = layer(x)
                if self.conv_dropout:
                    x = self.conv_dropout(x)
                if debug:
                    print(f"After pool{i//2 + 1}: {x.shape}")
            
            i += 1
        
        # Flatten
        x = x.view(x.size(0), -1)
        if debug:
            print(f"After flatten: {x.shape}")
        
        # FC layers with activation handling
        linear_count = 0
        total_linear = sum(1 for layer in self.fc_layers if isinstance(layer, nn.Linear))
        
        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):
                linear_count += 1
                x = layer(x)
                if linear_count < total_linear:
                    x = self.activation_func(x)
            else:
                x = layer(x)
        
        return x

class Net(nn.Module):

    def __init__(self, activation_func=F.relu, c1_layers=48, c2_layers=96, c3_layers=192, c4_layers=384, 
                 c1_ksize=3, c2_ksize=3, c3_ksize=3, c4_ksize=3, 
                 maxp1_size=3, maxp2_size=3, maxp3_size=3, maxp4_size=3,
                 fc_sizes=(1000, 1000, 1000, 100), dropout=0.4):
        super(Net, self).__init__()
        
        self.c1_layers = c1_layers
        self.c2_layers = c2_layers
        self.c3_layers = c3_layers
        self.c4_layers = c4_layers
        
        self.c1_ksize = c1_ksize
        self.maxp1_size = maxp1_size
        self.n_layers_last_conv = c1_layers
        
        self.activation_func = activation_func
        self.input_dim = 224
        self.dropout = dropout
        self.fc_sizes = fc_sizes
        
        # Build convolutional layers using nn.Sequential with OrderedDict
        conv_layers = OrderedDict()
        conv_layers['conv1'] = nn.Conv2d(1, self.c1_layers, self.c1_ksize)  # 1x224x224 input
        conv_layers['pool1'] = nn.MaxPool2d(self.maxp1_size)
        
        if c2_layers >= 1:
            self.maxp2_size = maxp2_size
            self.c2_ksize = c2_ksize
            conv_layers['conv2'] = nn.Conv2d(self.c1_layers, self.c2_layers, self.c2_ksize)
            conv_layers['pool2'] = nn.MaxPool2d(self.maxp2_size)
            self.n_layers_last_conv = c2_layers
        else:
            self.maxp2_size = 1
            self.c2_ksize = 1
            
        if c3_layers >= 1:
            self.maxp3_size = maxp3_size
            self.c3_ksize = c3_ksize
            conv_layers['conv3'] = nn.Conv2d(self.c2_layers, self.c3_layers, self.c3_ksize)
            conv_layers['pool3'] = nn.MaxPool2d(self.maxp3_size)
            self.n_layers_last_conv = c3_layers
        else:
            self.maxp3_size = 1
            self.c3_ksize = 1
            
        if c4_layers >= 1:
            self.maxp4_size = maxp4_size
            self.c4_ksize = c4_ksize
            conv_layers['conv4'] = nn.Conv2d(self.c3_layers, self.c4_layers, self.c4_ksize)
            conv_layers['pool4'] = nn.MaxPool2d(self.maxp4_size)
            self.n_layers_last_conv = c4_layers
        else:
            self.maxp4_size = 1
            self.c4_ksize = 1
        
        self.conv_layers = nn.Sequential(conv_layers)
        
        # Calculate flattened dimension step-by-step with assertions
        dim = self.input_dim
        
        # After conv1
        dim = dim - self.c1_ksize + 1
        assert dim > 0, f"After conv1: dimension {dim} <= 0"
        # After pool1
        dim = dim // self.maxp1_size
        assert dim > 0, f"After pool1: dimension {dim} <= 0"
        
        # After conv2 (if exists)
        if self.c2_layers > 0:
            dim = dim - self.c2_ksize + 1
            assert dim > 0, f"After conv2: dimension {dim} <= 0"
            dim = dim // self.maxp2_size
            assert dim > 0, f"After pool2: dimension {dim} <= 0"
        
        # After conv3 (if exists)
        if self.c3_layers > 0:
            dim = dim - self.c3_ksize + 1
            assert dim > 0, f"After conv3: dimension {dim} <= 0"
            dim = dim // self.maxp3_size
            assert dim > 0, f"After pool3: dimension {dim} <= 0"
        
        # After conv4 (if exists)
        if self.c4_layers > 0:
            dim = dim - self.c4_ksize + 1
            assert dim > 0, f"After conv4: dimension {dim} <= 0"
            dim = dim // self.maxp4_size
            assert dim > 0, f"After pool4: dimension {dim} <= 0"
        
        self.conv_dim = dim
        assert self.conv_dim > 0, f"Final conv_dim {self.conv_dim} <= 0"
        
        flatten_size = self.n_layers_last_conv * self.conv_dim * self.conv_dim
        
        # Build fully connected layers dynamically using nn.Sequential with OrderedDict
        fc_layers = OrderedDict()
        
        # First FC layer
        fc_layers['fc0'] = nn.Linear(flatten_size, fc_sizes[0])
        if self.dropout > 0:
            fc_layers['dropout0'] = nn.Dropout(self.dropout)
        
        # Remaining FC layers from fc_sizes
        for i in range(1, len(fc_sizes)):
            fc_layers[f'fc{i}'] = nn.Linear(fc_sizes[i-1], fc_sizes[i])
            # Add dropout after each layer except the last hidden layer
            if self.dropout > 0 and i < len(fc_sizes) - 1:
                fc_layers[f'dropout{i}'] = nn.Dropout(self.dropout)
        
        # Final output layer: always maps to 136 keypoints
        fc_layers[f'fc_output'] = nn.Linear(fc_sizes[-1], 136)
        
        self.fc_layers = nn.Sequential(fc_layers)
        
        # Dropout layers for convolutional output
        self.dropout_conv = nn.Dropout(self.dropout) if dropout > 0 else None
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x, debug=False):
        """Define the feedforward behavior of this model
        
        Args:
            x: Input tensor of shape [batch_size, 1, 224, 224]
            debug: If True, print shape information at each layer
        """
        
        if debug:
            print(f"Input shape: {x.shape}")
        
        # Pass through all convolutional layers
        # Each conv layer is followed by activation, then pooling, then optional dropout
        conv_blocks = [
            (0, 1, self.c1_layers > 0),    # (conv_idx, pool_idx, should_exist)
            (2, 3, self.c2_layers > 0),
            (4, 5, self.c3_layers > 0),
            (6, 7, self.c4_layers > 0),
        ]
        
        for block_num, (conv_idx, pool_idx, should_exist) in enumerate(conv_blocks, 1):
            if should_exist and conv_idx < len(self.conv_layers):
                # Apply convolution
                x = self.conv_layers[conv_idx](x)
                if debug:
                    print(f"After conv{block_num}: {x.shape}")
                
                x = self.activation_func(x)
                
                # Apply pooling
                if pool_idx < len(self.conv_layers):
                    x = self.conv_layers[pool_idx](x)
                    if debug:
                        print(f"After pool{block_num}: {x.shape}")
                
                # Apply dropout after pooling if enabled
                if self.dropout > 0:
                    x = self.dropout_conv(x)
        
        # Flatten for fully connected layers
        x_flat = x.view(x.size(0), -1)
        if debug:
            print(f"After flatten: {x_flat.shape}, expected: ({x.size(0)}, {self.n_layers_last_conv * self.conv_dim * self.conv_dim})")
        
        x = x_flat
        
        # Process through FC layers
        # Count total linear layers to identify the final output layer
        linear_count = 0
        total_linear = sum(1 for layer in self.fc_layers if isinstance(layer, nn.Linear))
        
        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):
                linear_count += 1
                x = layer(x)
                # Apply activation to all layers EXCEPT the final output layer
                if linear_count < total_linear:
                    x = self.activation_func(x)
            else:
                # Dropout or other layer
                x = layer(x)
        
        return x


class ResNet34(nn.Module):
    """ResNet34 adapted for facial keypoints detection from grayscale images.
    
    This model takes grayscale facial images (224x224) and outputs 136 facial keypoints
    (68 keypoints x 2 coordinates each).
    
    The model:
    - Converts grayscale input to handled by adapted first conv layer
    - Uses ResNet34 backbone with pretrained weights removed (train from scratch)
    - Replaces the classification head with a regression head for keypoint detection
    - Outputs 136 values (136 keypoint coordinates normalized to [-1, 1])
    """
    
    def __init__(self, pretrained=False, num_keypoints=136):
        """
        Args:
            pretrained: If True, load ImageNet pretrained weights. If False, train from scratch.
            num_keypoints: Number of keypoint coordinates to output (default: 136 for 68 keypoints)
        """
        super(ResNet34, self).__init__()
        
        import torchvision.models as models
        
        # Load ResNet34 architecture
        if pretrained:
            resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet34(weights=None)
        
        # Adapt first conv layer to accept grayscale input (1 channel -> 3 channels)
        # Method: Replicate the single grayscale channel 3 times
        original_conv1 = resnet.conv1
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Initialize the new conv1 layer properly
        # Copy weights from original conv1 by averaging across 3 channels
        if pretrained:
            with torch.no_grad():
                # Average the pretrained weights across RGB channels
                resnet.conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)
        else:
            # Initialize with proper kaiming initialization
            nn.init.kaiming_normal_(resnet.conv1.weight, mode='fan_out', nonlinearity='relu')
        
        # Remove the original classification head
        resnet.fc = nn.Identity()
        
        # Store the feature extraction part
        self.backbone = resnet
        
        # Add custom regression head for keypoint detection
        # ResNet34 outputs 512 features before the classification layer
        self.feature_dim = 512
        
        # Regression head: 512 features -> 1024 -> 512 -> 136 keypoints
        self.fc_head = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_keypoints)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, 1, 224, 224] (grayscale images)
            
        Returns:
            Tensor of shape [batch_size, 136] containing keypoint coordinates
        """
        # x shape: [batch_size, 1, 224, 224]
        
        # Pass through ResNet backbone
        # ResNet expects: [batch_size, 3, 224, 224] but we're passing [batch_size, 1, 224, 224]
        # The first conv layer is adapted to accept 1 channel
        x = self.backbone.conv1(x)  # [batch_size, 64, 112, 112]
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)  # [batch_size, 64, 56, 56]
        
        # Residual blocks
        x = self.backbone.layer1(x)   # [batch_size, 64, 56, 56]
        x = self.backbone.layer2(x)   # [batch_size, 128, 28, 28]
        x = self.backbone.layer3(x)   # [batch_size, 256, 14, 14]
        x = self.backbone.layer4(x)   # [batch_size, 512, 7, 7]
        
        # Global average pooling
        x = self.backbone.avgpool(x)  # [batch_size, 512, 1, 1]
        
        # Flatten
        x = torch.flatten(x, 1)  # [batch_size, 512]
        
        # Apply regression head
        x = self.fc_head(x)  # [batch_size, 136]
        
        return x