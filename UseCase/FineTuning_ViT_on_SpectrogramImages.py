import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from PIL import Image, ImageDraw, ImageOps, ImageFont
import torch.nn as nn
from transformers import ViTModel
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import time
from scipy.stats import pearsonr
from torch.amp import GradScaler
from torch.cuda.amp import autocast
from multiprocessing import freeze_support
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Configuration
DATA_DIR = r"C:\Users\Nooshin\myenv\Lib\site-packages\Seizure_TimeFreqROI_Annotator\assets\sample_spectrograms"
LABELS_PATH = r"C:\Users\Nooshin\myenv\Lib\site-packages\Seizure_TimeFreqROI_Annotator\assets\labels\default_annotations.xlsx"
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 100
PATIENCE = 7  
LR = 2e-5  
WARMUP_EPOCHS = 5  
MIN_LR = 1e-6  
WEIGHT_DECAY = 0.01  

# Spectrogram parameters (from label generation code)
FREQ_RANGE = (1, 60)  # 1 to 60 Hz
TIME_RANGE = (0, 60)   # 0 to 60 seconds

# Create output directories
os.makedirs(os.path.join(DATA_DIR, "attention_maps"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "latent_representations"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "roi_overlays"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "latent_visualizations"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "head_attention_maps"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "prediction_results"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "attention_decompositions"), exist_ok=True)

# Global variables to store sample images and latent vectors
SAMPLE_IMAGES = None
SAMPLE_INDICES = None
ALL_LATENT_VECTORS = []
ALL_LABELS = []
ALL_IMAGE_FILES = []

def remove_white_border(image):
    """Clean the image by removing white borders and keeping only the spectrogram"""
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image.copy()
    
    # Convert to binary image (assuming white background)
    threshold = 240  # Adjust based on your images
    binary = img < threshold
        
    # Find rows and columns that contain non-white pixels
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
        
    # Find the bounding box of non-white regions
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
        
    # Crop the image to the bounding box
    img = img[rmin:rmax+1, cmin:cmax+1]
    
    if isinstance(image, Image.Image):
        return Image.fromarray(img)
    return img

def convert_to_grayscale(image):
    """Convert image to grayscale"""
    if isinstance(image, Image.Image):
        return image.convert('L')
    else:
        return Image.fromarray(image).convert('L')

def time_to_pixel(time_val):
    """Convert time value (seconds) to pixel coordinate"""
    return int((time_val - TIME_RANGE[0]) / (TIME_RANGE[1] - TIME_RANGE[0]) * IMAGE_SIZE)

def freq_to_pixel(freq_val):
    """Convert frequency value (Hz) to pixel coordinate (note: y-axis is inverted)"""
    return int(IMAGE_SIZE - (freq_val - FREQ_RANGE[0]) / (FREQ_RANGE[1] - FREQ_RANGE[0]) * IMAGE_SIZE)

def pixel_to_time(x_pixel):
    """Convert pixel coordinate to time value (seconds)"""
    return TIME_RANGE[0] + (x_pixel / IMAGE_SIZE) * (TIME_RANGE[1] - TIME_RANGE[0])

def pixel_to_freq(y_pixel):
    """Convert pixel coordinate to frequency value (Hz) (note: y-axis is inverted)"""
    return FREQ_RANGE[0] + ((IMAGE_SIZE - y_pixel) / IMAGE_SIZE) * (FREQ_RANGE[1] - FREQ_RANGE[0])

# Load and preprocess labels
print("Loading and preprocessing labels...")
labels_df = pd.read_excel(LABELS_PATH)

# Filter out images that don't exist
existing_images = []
for img_file in labels_df['image_file']:
    img_path = os.path.join(DATA_DIR, img_file)
    if os.path.exists(img_path):
        existing_images.append(img_file)
labels_df = labels_df[labels_df['image_file'].isin(existing_images)].reset_index(drop=True)

# Prepare labels
labels_df['has_roi'] = labels_df['has_roi'].astype(float)
coord_cols = ['start_time', 'end_time', 'start_freq', 'end_freq']
labels_df[coord_cols] = labels_df[coord_cols].fillna(0)

# Normalize coordinates based on spectrogram parameters
time_mean, time_std = (TIME_RANGE[1] + TIME_RANGE[0])/2, (TIME_RANGE[1] - TIME_RANGE[0])/2
freq_mean, freq_std = (FREQ_RANGE[1] + FREQ_RANGE[0])/2, (FREQ_RANGE[1] - FREQ_RANGE[0])/2

labels_df['start_time'] = (labels_df['start_time'] - time_mean) / time_std
labels_df['end_time'] = (labels_df['end_time'] - time_mean) / time_std
labels_df['start_freq'] = (labels_df['start_freq'] - freq_mean) / freq_std
labels_df['end_freq'] = (labels_df['end_freq'] - freq_mean) / freq_std

# Custom Dataset class
class SpectrogramROIDataset(Dataset):
    def __init__(self, data_dir, df, transform=None):
        self.data_dir = data_dir
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.df.iloc[idx]['image_file'])
        img = Image.open(img_path).convert('RGB')
        
        # Remove white borders before resizing
        img = remove_white_border(img)
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        
        img = np.array(img) / 255.0
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        
        has_roi = torch.tensor(self.df.iloc[idx]['has_roi'], dtype=torch.float32)
        coords = torch.tensor(self.df.iloc[idx][coord_cols].values.astype(float), dtype=torch.float32)
        reg_mask = torch.tensor(1.0 if has_roi == 1 else 0.0, dtype=torch.float32)
        
        return img, has_roi, coords, reg_mask, self.df.iloc[idx]['image_file']

# Split data
print("Splitting data into train/test sets...")
train_df, test_df = train_test_split(labels_df, test_size=0.2, random_state=42, stratify=labels_df['has_roi'])

# Create datasets and dataloaders
print("Creating datasets and dataloaders...")
train_dataset = SpectrogramROIDataset(DATA_DIR, train_df)
test_dataset = SpectrogramROIDataset(DATA_DIR, test_df)

# Select fixed sample images for visualization
def select_sample_images(dataset):
    # List of specific images we want to track
    target_images = [
        '18210000_4.0_ch2_augment8_spectrogram.png',
        '18308005_86.4_ch2_spectrogram.png',
        '18307010_90.0_ch2_augment4_spectrogram.png'
    ]
    
    sample_images = []
    sample_indices = []
    
    # Find these images in the dataset
    for idx in range(len(dataset)):
        img_file = dataset.df.iloc[idx]['image_file']
        if img_file in target_images:
            img, has_roi, coords, reg_mask, img_file = dataset[idx]
            sample_images.append((img, has_roi, coords, reg_mask, img_file))
            sample_indices.append(idx)
            
            # Stop when we've found all three
            if len(sample_images) == 3:
                break
    
    # If we didn't find all three, add some random ones
    if len(sample_images) < 3:
        remaining = 3 - len(sample_images)
        for idx in range(remaining):
            img, has_roi, coords, reg_mask, img_file = dataset[idx]
            sample_images.append((img, has_roi, coords, reg_mask, img_file))
            sample_indices.append(idx)
    
    return sample_images, sample_indices

# Select sample images from test set
SAMPLE_IMAGES, SAMPLE_INDICES = select_sample_images(test_dataset)
print(f"Selected sample images: {[img_file for _, _, _, _, img_file in SAMPLE_IMAGES]}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# Enhanced Model definition with separate heads for each coordinate
class ViTForROIDetection(nn.Module):
    def __init__(self, pretrained_model_name="google/vit-base-patch16-224"):
        super(ViTForROIDetection, self).__init__()
        self.vit = ViTModel.from_pretrained(pretrained_model_name, output_attentions=True)
        
        # Enhanced LoRA configuration
        lora_config = LoraConfig(
            r=16,  # Increased rank for better adaptation
            lora_alpha=32,
            target_modules=["query", "value", "key", "dense"],
            lora_dropout=0.1,
            bias="lora_only"
        )

        self.vit = get_peft_model(self.vit, lora_config)
        
        # Classification head with dropout and layer norm
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.vit.config.hidden_size),
            nn.Linear(self.vit.config.hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )
        
        # Shared feature extractor for regression
        self.regression_feature = nn.Sequential(
            nn.LayerNorm(self.vit.config.hidden_size),
            nn.Linear(self.vit.config.hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Separate regression heads for each coordinate with shared features
        self.start_time_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
        self.end_time_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
        self.start_freq_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
        self.end_freq_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
        # Attention heads for each coordinate with shared features
        self.coord_attention = nn.ModuleDict({
            'start_time': nn.Sequential(
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Linear(128, 1)
            ),
            'end_time': nn.Sequential(
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Linear(128, 1)
            ),
            'start_freq': nn.Sequential(
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Linear(128, 1)
            ),
            'end_freq': nn.Sequential(
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Linear(128, 1)
            )
        })
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better convergence"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x):
        outputs = self.vit(pixel_values=x, output_attentions=True)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        patch_output = outputs.last_hidden_state[:, 1:, :]  # Patch tokens
        
        # Store attention maps and hidden states for visualization
        self.attentions = outputs.attentions
        self.hidden_states = outputs.last_hidden_state
        
        # Classification output
        roi_prob = self.classifier(cls_output)
        
        # Shared regression features
        reg_features = self.regression_feature(cls_output)
        
        # Coordinate predictions
        start_time = self.start_time_head(reg_features)
        end_time = self.end_time_head(reg_features)
        start_freq = self.start_freq_head(reg_features)
        end_freq = self.end_freq_head(reg_features)
        
        # Combine coordinates
        coords = torch.cat([start_time, end_time, start_freq, end_freq], dim=1)
        
        # Compute attention maps for each coordinate
        self.coord_attention_maps = {}
        patch_features = self.regression_feature(patch_output)  # Shared features for patches
        
        for coord_name, attn_head in self.coord_attention.items():
            # Compute attention scores for each patch
            attn_scores = attn_head(patch_features).squeeze(-1)  # [batch_size, num_patches]
            attn_weights = torch.softmax(attn_scores, dim=-1)
            self.coord_attention_maps[coord_name] = attn_weights
        
        return roi_prob, coords

# Initialize model
print("Initializing model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTForROIDetection().to(device)

# Enhanced loss functions with adaptive weighting
class AdaptiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_criterion = nn.BCEWithLogitsLoss()
        self.reg_criterion = nn.SmoothL1Loss()  # More robust than MSE
        self.cls_weight = 1.0
        self.reg_weight = 1.0
        self.adaptive_weights = True
    
    def forward(self, cls_pred, cls_target, reg_pred, reg_target, reg_mask, compute_gradients=True):
        cls_loss = self.cls_criterion(cls_pred.squeeze(), cls_target)
        
        # Calculate regression loss for each coordinate separately
        reg_loss = 0
        valid_samples = reg_mask.sum()
        
        if valid_samples > 0:
            for i in range(4):  # For each coordinate
                reg_loss += self.reg_criterion(
                    reg_pred[:, i] * reg_mask,
                    reg_target[:, i] * reg_mask
                ) / valid_samples
        
        # Adaptive weighting - only compute gradients during training
        if self.adaptive_weights and compute_gradients:
            # Need to ensure we're not in no_grad context
            with torch.enable_grad():
                cls_grad = torch.autograd.grad(cls_loss, cls_pred, retain_graph=True)[0].norm(2)
                reg_grad = torch.autograd.grad(reg_loss, reg_pred, retain_graph=True)[0].norm(2)
                
                if reg_grad > 0 and cls_grad > 0:
                    grad_ratio = cls_grad / (reg_grad + 1e-8)
                    self.reg_weight = min(max(grad_ratio.item(), 0.1), 10.0)
                    self.cls_weight = 1.0
        
        total_loss = self.cls_weight * cls_loss + self.reg_weight * reg_loss
        return total_loss, cls_loss, reg_loss

# Optimizer with weight decay
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# Combined learning rate scheduler
scheduler_plateau = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
scheduler_cosine = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=MIN_LR)

# Gradient scaler for mixed precision
scaler = GradScaler(device='cuda')

# Loss function
criterion = AdaptiveLoss()

def save_latent_visualization(latent_vectors, labels, image_files, output_dir, epoch):
    """Visualize latent space in 2D using t-SNE and PCA"""
    # Convert to numpy
    latent_np = np.array(latent_vectors)  # Using [CLS] token
    labels_np = np.array(labels)
    
    # Reduce dimensionality with t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latent_np)-1))
    latent_tsne = tsne.fit_transform(latent_np)
    
    # Reduce dimensionality with PCA
    pca = PCA(n_components=2)
    latent_pca = pca.fit_transform(latent_np)
    
    # Create plots
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=labels_np, cmap='viridis')
    plt.colorbar(scatter, label='ROI Presence')
    plt.title(f't-SNE of Latent Space (Epoch {epoch})')
    
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(latent_pca[:, 0], latent_pca[:, 1], c=labels_np, cmap='viridis')
    plt.colorbar(scatter, label='ROI Presence')
    plt.title(f'PCA of Latent Space (Epoch {epoch})')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'latent_space_epoch_{epoch}.png'))
    plt.close()

def decompose_attention_maps(attention_maps, n_clusters=3):
    """Decompose attention maps into clusters using K-means for each layer and head"""
    all_attention = []
    
    for layer_idx, layer_attention in enumerate(attention_maps):
        # layer_attention shape: [batch_size, num_heads, seq_len, seq_len]
        if len(layer_attention.shape) == 4:
            layer_attention = layer_attention[0]  # Take first (and only) batch item
        
        num_heads = layer_attention.shape[0]
        
        for head_idx in range(num_heads):
            # Get CLS token attention to patches (excluding CLS token self-attention)
            cls_attention = layer_attention[head_idx, 0, 1:].cpu().numpy()  # [seq_len-1]
            all_attention.append(cls_attention)
    
    if not all_attention:
        return None
    
    # Stack all attention maps (n_samples, n_features)
    all_attention = np.vstack(all_attention)  # Shape: (num_layers*num_heads, seq_len-1)
    
    # Apply K-means clustering
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(all_attention)
        
        # Get cluster centers (representative attention patterns)
        cluster_centers = kmeans.cluster_centers_
        
        return cluster_labels, cluster_centers
    except Exception as e:
        print(f"Error in KMeans clustering: {str(e)}")
        return None

def visualize_attention_decomposition(attention_maps, image_file, original_image, epoch, output_dir, n_clusters=3):
    """Visualize decomposed attention maps for each layer and head"""
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_file))[0]
    
    # Convert original image to grayscale
    gray_img = convert_to_grayscale(original_image)
    gray_img_np = np.array(gray_img)
    
    # Decompose attention maps
    decomposition = decompose_attention_maps(attention_maps, n_clusters)
    if decomposition is None:
        print(f"Warning: Could not decompose attention maps for {image_file}")
        return
    
    cluster_labels, cluster_centers = decomposition
    
    # Calculate grid size (assuming square patches)
    num_patches = cluster_centers.shape[1]
    grid_size = int(np.sqrt(num_patches))
    
    # Verify we can reshape the attention patterns
    if grid_size * grid_size != num_patches:
        print(f"Warning: Cannot reshape {num_patches} patches to square grid for {image_file}")
        return
    
    # Visualize each cluster center
    for cluster_idx in range(n_clusters):
        # Get the cluster center attention pattern
        cluster_center = cluster_centers[cluster_idx]
        
        try:
            # Reshape to grid
            attention_grid = cluster_center.reshape(grid_size, grid_size)
            
            # Resize to match original image dimensions
            attention_resized = np.array(Image.fromarray(attention_grid).resize(
                (gray_img_np.shape[1], gray_img_np.shape[0]), Image.Resampling.BILINEAR))
            
            plt.figure(figsize=(12, 6))
            
            # Plot grayscale image
            plt.subplot(1, 2, 1)
            plt.imshow(gray_img_np, cmap='gray')
            plt.title("Grayscale Spectrogram")
            
            # Plot attention overlay
            plt.subplot(1, 2, 2)
            plt.imshow(gray_img_np, cmap='gray')
            plt.imshow(attention_resized, cmap='viridis', alpha=0.5)
            plt.colorbar()
            plt.title(f"Cluster {cluster_idx} Attention (Epoch {epoch})")
            
            plt.tight_layout()
            
            # Save the figure
            cluster_dir = os.path.join(output_dir, f"cluster_{cluster_idx}")
            os.makedirs(cluster_dir, exist_ok=True)
            
            plt.savefig(os.path.join(cluster_dir, f"{base_name}_cluster_{cluster_idx}_epoch_{epoch}.png"))
            plt.close()
        except Exception as e:
            print(f"Error visualizing cluster {cluster_idx} for {image_file}: {str(e)}")

def save_sample_attention_maps(attentions, image_file, output_dir, original_image, coord_attention_maps=None, epoch=None):
    """Save attention maps only for the sample images"""
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_file))[0]
    
    # Convert original image to grayscale
    gray_img = convert_to_grayscale(original_image)
    gray_img_np = np.array(gray_img)
    
    # Save standard attention maps for each layer and each head
    for layer_idx, layer_attention in enumerate(attentions):
        # layer_attention shape: [batch_size, num_heads, seq_len, seq_len]
        if len(layer_attention.shape) == 4:  # Already has batch dimension
            layer_attention = layer_attention[0]  # Take first (and only) batch item
            
        num_heads = layer_attention.shape[0]
        
        for head_idx in range(num_heads):
            # Get attention for this head (CLS token attention to patches)
            head_attention = layer_attention[head_idx, 0, 1:]  # [seq_len-1]
            
            # Calculate expected grid size (for ViT-Base/16, this should be 14x14)
            seq_len_minus_1 = head_attention.shape[-1]
            grid_size = int(np.sqrt(seq_len_minus_1))
            
            # Verify we can reshape to square
            if grid_size * grid_size != seq_len_minus_1:
                print(f"Warning: Cannot reshape attention of size {seq_len_minus_1} to square grid")
                continue
                
            # Reshape attention to grid
            attention_grid = head_attention.reshape(grid_size, grid_size).cpu().numpy()
            
            # Resize attention to match image dimensions
            attention_resized = np.array(Image.fromarray(attention_grid).resize(
                (gray_img_np.shape[1], gray_img_np.shape[0]), Image.Resampling.BILINEAR))
            
            plt.figure(figsize=(12, 6))
            
            # Plot grayscale image
            plt.subplot(1, 2, 1)
            plt.imshow(gray_img_np, cmap='gray')
            plt.title("Grayscale Spectrogram")
            
            # Plot attention overlay
            plt.subplot(1, 2, 2)
            plt.imshow(gray_img_np, cmap='gray')
            plt.imshow(attention_resized, cmap='viridis', alpha=0.5)
            plt.colorbar()
            plt.title(f"Attention Layer {layer_idx} Head {head_idx} (Epoch {epoch})")
            
            plt.tight_layout()
            
            # Create subdirectory for this layer
            layer_dir = os.path.join(output_dir, f"layer_{layer_idx}")
            os.makedirs(layer_dir, exist_ok=True)
            
            # Save the figure
            plt.savefig(os.path.join(layer_dir, f"{base_name}_layer_{layer_idx}_head_{head_idx}_epoch_{epoch}.png"))
            plt.close()
    
    # Save coordinate-specific attention maps if available
    if coord_attention_maps is not None:
        coord_attn_dir = os.path.join(DATA_DIR, "head_attention_maps", f"epoch_{epoch}")
        os.makedirs(coord_attn_dir, exist_ok=True)
        
        for coord_name, attn_weights in coord_attention_maps.items():
            # Get attention weights (first sample in batch)
            attn = attn_weights.cpu().numpy()  # [num_patches]
            
            # Calculate expected grid size
            num_patches = attn.shape[-1]
            grid_size = int(np.sqrt(num_patches))
            
            # Verify we can reshape to square
            if grid_size * grid_size != num_patches:
                print(f"Warning: Cannot reshape {coord_name} attention of size {num_patches} to square grid")
                continue
                
            # Reshape to grid
            attention_grid = attn.reshape(grid_size, grid_size)
            
            # Resize attention to match image dimensions
            attention_resized = np.array(Image.fromarray(attention_grid).resize(
                (gray_img_np.shape[1], gray_img_np.shape[0]), Image.Resampling.BILINEAR))
            
            plt.figure(figsize=(12, 6))
            
            # Plot grayscale image
            plt.subplot(1, 2, 1)
            plt.imshow(gray_img_np, cmap='gray')
            plt.title("Grayscale Spectrogram")
            
            # Plot attention overlay
            plt.subplot(1, 2, 2)
            plt.imshow(gray_img_np, cmap='gray')
            plt.imshow(attention_resized, cmap='viridis', alpha=0.5)
            plt.colorbar()
            plt.title(f"{coord_name.replace('_', ' ').title()} Attention (Epoch {epoch})")
            
            plt.tight_layout()
            
            # Save in coordinate-specific subdirectory
            coord_dir = os.path.join(coord_attn_dir, coord_name)
            os.makedirs(coord_dir, exist_ok=True)
            
            plt.savefig(os.path.join(coord_dir, f"{base_name}_{coord_name}_attention_epoch_{epoch}.png"))
            plt.close()

def save_sample_roi_overlay(image_file, pred_coords, true_coords, has_roi, output_dir, epoch):
    """Save ROI predictions overlaid on the preprocessed grayscale image for sample images"""
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_file))[0]
    
    # Load and preprocess original image (remove white borders)
    img_path = os.path.join(DATA_DIR, image_file)
    img = Image.open(img_path).convert('RGB')
    img = remove_white_border(img)
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    
    # Convert to grayscale
    gray_img = convert_to_grayscale(img)
    gray_img = gray_img.convert('RGB')  # Convert back to RGB for drawing
    
    draw = ImageDraw.Draw(gray_img)
    
    if has_roi == 1:
        # Convert predicted coordinates to pixel space
        x1 = time_to_pixel(pred_coords[0])
        y1 = freq_to_pixel(pred_coords[2])
        x2 = time_to_pixel(pred_coords[1])
        y2 = freq_to_pixel(pred_coords[3])
        
        # Ensure valid rectangle coordinates
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Draw predicted rectangle
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        
        # Add predicted frequency and time as legend
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        # Create separate lines for time and frequency
        time_text = f"Time: {pred_coords[0]:.1f}-{pred_coords[1]:.1f}s"
        freq_text = f"Freq: {pred_coords[2]:.1f}-{pred_coords[3]:.1f}Hz"
        
        # Draw text on separate lines
        draw.text((10, 10), "Predicted:", fill="red", font=font)
        draw.text((10, 25), time_text, fill="red", font=font)
        draw.text((10, 40), freq_text, fill="red", font=font)
    
    # Save the image with epoch in filename
    overlay_path = os.path.join(output_dir, f"{base_name}_overlay_epoch_{epoch}.png")
    gray_img.save(overlay_path)

def process_sample_images(model, device, epoch):
    """Process and visualize the sample images for the current epoch"""
    model.eval()
    
    # Create epoch-specific directories
    epoch_dir = f"epoch_{epoch}"
    attention_dir = os.path.join(DATA_DIR, "attention_maps", epoch_dir)
    latent_dir = os.path.join(DATA_DIR, "latent_representations", epoch_dir)
    overlay_dir = os.path.join(DATA_DIR, "roi_overlays", epoch_dir)
    latent_viz_dir = os.path.join(DATA_DIR, "latent_visualizations", epoch_dir)
    decomposition_dir = os.path.join(DATA_DIR, "attention_decompositions", epoch_dir)
    
    os.makedirs(attention_dir, exist_ok=True)
    os.makedirs(latent_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(latent_viz_dir, exist_ok=True)
    os.makedirs(decomposition_dir, exist_ok=True)
    
    with torch.no_grad():
        for sample in SAMPLE_IMAGES:
            img, has_roi, coords, reg_mask, img_file = sample
            img = img.unsqueeze(0).to(device)  # Add batch dimension
            
            # Get model predictions
            roi_prob, pred_coords = model(img)
            
            # Get the original preprocessed image for visualization
            img_path = os.path.join(DATA_DIR, img_file)
            original_img = Image.open(img_path).convert('RGB')
            original_img = remove_white_border(original_img)
            original_img = original_img.resize((IMAGE_SIZE, IMAGE_SIZE))
            
            # Save attention maps for this sample
            save_sample_attention_maps(
                [attn.cpu() for attn in model.attentions],  
                img_file,
                attention_dir,
                original_img,
                {k: v[0].cpu() for k, v in model.coord_attention_maps.items()},
                epoch
            )
            
            # Save latent representation
            latent = model.hidden_states[0].cpu().numpy()
            np.save(
                os.path.join(latent_dir, 
                           f"{os.path.splitext(os.path.basename(img_file))[0]}.npy"),
                latent
            )
            
            # Denormalize predicted coordinates
            pred_denorm = pred_coords[0].cpu().numpy().copy()
            pred_denorm[:2] = pred_denorm[:2] * time_std + time_mean
            pred_denorm[2:] = pred_denorm[2:] * freq_std + freq_mean
            
            # Clip predicted coordinates to valid range
            pred_denorm[:2] = np.clip(pred_denorm[:2], TIME_RANGE[0], TIME_RANGE[1])
            pred_denorm[2:] = np.clip(pred_denorm[2:], FREQ_RANGE[0], FREQ_RANGE[1])
            
            # Denormalize true coordinates
            true_denorm = coords.cpu().numpy().copy()
            if has_roi == 1:
                true_denorm[:2] = true_denorm[:2] * time_std + time_mean
                true_denorm[2:] = true_denorm[2:] * freq_std + freq_mean
            else:
                true_denorm[:] = 0
            
            # Save ROI overlay
            save_sample_roi_overlay(
                img_file,
                pred_denorm,
                true_denorm if has_roi == 1 else None,
                has_roi,
                overlay_dir,
                epoch
            )
            
            # Visualize attention decomposition
            visualize_attention_decomposition(
                [attn.cpu() for attn in model.attentions],  
                img_file,
                original_img,
                epoch,
                decomposition_dir
            )
    
    # Collect latent vectors from test set for visualization
    test_latent = []
    test_labels = []
    test_image_files = []
    
    with torch.no_grad():
        for images, has_roi, _, _, img_files in test_loader:
            images = images.to(device)
            outputs = model(images)
            latent = model.hidden_states[:, 0, :].cpu().numpy()  # [CLS] token
            
            test_latent.append(latent)
            test_labels.extend(has_roi.cpu().numpy())
            test_image_files.extend(img_files)
    
    # Save latent space visualization
    if test_latent:
        test_latent = np.vstack(test_latent)
        save_latent_visualization(test_latent, test_labels, test_image_files, latent_viz_dir, epoch)

def train(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_reg_loss = 0
    
    for batch_idx, (images, has_roi, coords, reg_mask, _) in enumerate(dataloader):
        images = images.to(device)
        has_roi = has_roi.to(device)
        coords = coords.to(device)
        reg_mask = reg_mask.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            # Get model predictions
            roi_prob, pred_coords = model(images)
            
            # Compute loss
            loss, cls_loss, reg_loss = criterion(
                roi_prob, has_roi, pred_coords, coords, reg_mask, compute_gradients=True
            )
        
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_reg_loss += reg_loss.item() if reg_loss > 0 else 0
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx+1}/{len(dataloader)}, "
                  f"Loss: {loss.item():.4f}, "
                  f"Cls: {cls_loss.item():.4f}, "
                  f"Reg: {reg_loss.item() if reg_loss > 0 else 0:.4f}, "
                  f"ClsW: {criterion.cls_weight:.2f}, "
                  f"RegW: {criterion.reg_weight:.2f}")
    
    # Learning rate scheduling
    if epoch < WARMUP_EPOCHS:
        # Linear warmup
        lr_scale = min(1.0, float(epoch + 1) / WARMUP_EPOCHS)
        for param_group in optimizer.param_groups:
            param_group['lr'] = LR * lr_scale
    else:
        # Cosine annealing
        scheduler_cosine.step()
    
    return total_loss / len(dataloader), total_cls_loss / len(dataloader), total_reg_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_cls_loss = 0
    total_reg_loss = 0
    
    all_roi_probs = []
    all_has_roi = []
    all_pred_coords = []
    all_true_coords = []
    
    with torch.no_grad():
        for images, has_roi, coords, reg_mask, _ in dataloader:
            images = images.to(device)
            has_roi = has_roi.to(device)
            coords = coords.to(device)
            reg_mask = reg_mask.to(device)
            
            roi_prob, pred_coords = model(images)
            
            loss, cls_loss, reg_loss = criterion(
                roi_prob, has_roi, pred_coords, coords, reg_mask, compute_gradients=False
            )
            
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_reg_loss += reg_loss.item() if reg_loss > 0 else 0
            
            all_roi_probs.append(torch.sigmoid(roi_prob).cpu().numpy())
            all_has_roi.append(has_roi.cpu().numpy())
            all_pred_coords.append(pred_coords.cpu().numpy())
            all_true_coords.append(coords.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    avg_cls_loss = total_cls_loss / len(dataloader)
    avg_reg_loss = total_reg_loss / len(dataloader)
    
    roi_probs = np.concatenate(all_roi_probs)
    has_roi = np.concatenate(all_has_roi)
    
    auroc = roc_auc_score(has_roi, roi_probs)
    auprc = average_precision_score(has_roi, roi_probs)
    
    pred_coords = np.concatenate(all_pred_coords)
    true_coords = np.concatenate(all_true_coords)
    roi_mask = (has_roi == 1)
    
    if roi_mask.sum() > 0:
        coord_errors = np.abs(pred_coords[roi_mask] - true_coords[roi_mask])
        mean_coord_error = coord_errors.mean(axis=0)
        time_error = mean_coord_error[:2] * time_std
        freq_error = mean_coord_error[2:] * freq_std
    else:
        time_error = np.array([0, 0])
        freq_error = np.array([0, 0])
    
    return (avg_loss, avg_cls_loss, avg_reg_loss, auroc, auprc, 
            time_error, freq_error)

def main():
    best_auroc = 0
    best_loss = float('inf')
    epochs_without_improvement = 0
    
    # Create directory for model checkpoints
    os.makedirs(os.path.join(DATA_DIR, "checkpoints"), exist_ok=True)
    
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("Training...")
        train_loss, train_cls_loss, train_reg_loss = train(model, train_loader, optimizer, device, epoch)
        
        print("Evaluating...")
        (test_loss, test_cls_loss, test_reg_loss, test_auroc, test_auprc, 
         time_error, freq_error) = evaluate(model, test_loader, device)
        
        # Process and visualize sample images for this epoch
        print("Processing sample images...")
        process_sample_images(model, device, epoch)
        
        # Update plateau scheduler based on validation loss
        scheduler_plateau.step(test_loss)
        
        print(f"\nTrain Loss: {train_loss:.4f} (Cls: {train_cls_loss:.4f}, Reg: {train_reg_loss:.4f})")
        print(f"Test Loss: {test_loss:.4f} (Cls: {test_cls_loss:.4f}, Reg: {test_reg_loss:.4f})")
        print(f"Test AUROC: {test_auroc:.4f}, AUPRC: {test_auprc:.4f}")
        print(f"Time Error (start, end): {time_error[0]:.2f}s, {time_error[1]:.2f}s")
        print(f"Freq Error (start, end): {freq_error[0]:.2f}Hz, {freq_error[1]:.2f}Hz")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), 
                      os.path.join(DATA_DIR, "checkpoints", f"model_epoch_{epoch+1}.pth"))
            print(f"Saved model checkpoint at epoch {epoch+1}")
        
        # Early stopping based on both AUROC and loss
        if test_auroc > best_auroc or test_loss < best_loss:
            if test_auroc > best_auroc:
                best_auroc = test_auroc
            if test_loss < best_loss:
                best_loss = test_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(DATA_DIR, "best_vit_roi_model.pth"))
            print("Saved new best model.")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping triggered after {PATIENCE} epochs without improvement.")
                break
    
    print("\nFinal Evaluation:")
    (test_loss, test_cls_loss, test_reg_loss, test_auroc, test_auprc, 
     time_error, freq_error) = evaluate(model, test_loader, device)
    
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test AUROC: {test_auroc:.4f}")
    print(f"Final Test AUPRC: {test_auprc:.4f}")
    print(f"Final Time Error (start, end): {time_error[0]:.2f}s, {time_error[1]:.2f}s")
    print(f"Final Freq Error (start, end): {freq_error[0]:.2f}Hz, {freq_error[1]:.2f}Hz")
    
    metrics = {
        'test_loss': test_loss,
        'test_auroc': test_auroc,
        'test_auprc': test_auprc,
        'time_error_start': time_error[0],
        'time_error_end': time_error[1],
        'freq_error_start': freq_error[0],
        'freq_error_end': freq_error[1]
    }
    
    with open(os.path.join(DATA_DIR, "roi_metrics.txt"), 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

if __name__ == '__main__':
    freeze_support()
    main()