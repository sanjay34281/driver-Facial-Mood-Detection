import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights

class DriverStateClassifier(nn.Module):
    """
    EfficientNet-B0 CNN architecture adapted for driver state classification
    (Normal, Sleepy, Yawning) using transfer learning.
    """
    def __init__(self, num_classes=3, pretrained=True, dropout=0.35):
        super(DriverStateClassifier, self).__init__()
        
        # Load pre-trained EfficientNet-B0 backbone
        if pretrained:
            weights = EfficientNet_B0_Weights.DEFAULT
            self.backbone = models.efficientnet_b0(weights=weights)
        else:
            self.backbone = models.efficientnet_b0(weights=None)
            
        # Get input features from original classifier head (1280 for EfficientNet-B0)
        in_features = self.backbone.classifier[1].in_features
        
        # Replace classification head with custom head for driver states
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features=in_features, out_features=256),
            nn.SiLU(),  # Swish activation standard in EfficientNet
            nn.Dropout(p=dropout * 0.7, inplace=True),
            nn.Linear(in_features=256, out_features=num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)
        
    def freeze_feature_extractor(self):
        """Freeze backbone features for initial transfer learning stage."""
        for name, param in self.backbone.features.named_parameters():
            param.requires_grad = False
            
    def unfreeze_feature_extractor(self):
        """Unfreeze all backbone features for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


def get_classifier(num_classes=3, weights_path=None, device="cpu", pretrained=True):
    """
    Factory function to initialize and load the DriverStateClassifier.
    """
    model = DriverStateClassifier(num_classes=num_classes, pretrained=pretrained)
    
    if weights_path and os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        # Check if saved object is state_dict or full dict with 'model_state_dict'
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"[Model] Successfully loaded weights from: {weights_path}")
    else:
        if weights_path:
            print(f"[Model] Warning: Weights file not found at {weights_path}. Using base initialized model.")
            
    model = model.to(device)
    model.eval()
    return model
