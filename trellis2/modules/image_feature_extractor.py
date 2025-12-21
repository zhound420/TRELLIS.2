from typing import *
import math
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import DINOv3ConvNextModel, DINOv3ViTModel
import numpy as np
from PIL import Image


class DinoV2FeatureExtractor:
    """
    Feature extractor for DINOv2 models.
    """
    def __init__(self, model_name: str, image_size: int = 518):
        self.model_name = model_name
        self.image_size = image_size
        self.model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def to(self, device):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()
    
    @torch.no_grad()
    def __call__(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        Extract features from the image.
        
        Args:
            image: A batch of images as a tensor of shape (B, C, H, W) or a list of PIL images.
        
        Returns:
            A tensor of shape (B, N, D) where N is the number of patches and D is the feature dimension.
        """
        target_size = int(math.ceil(self.image_size / 14) * 14)
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
            if image.shape[-2:] != (target_size, target_size):
                image = F.interpolate(
                    image,
                    size=(target_size, target_size),
                    mode="bilinear",
                    align_corners=False,
                )
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((target_size, target_size), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).cuda()
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        image = self.transform(image).cuda()
        features = self.model(image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
    

class DinoV3FeatureExtractor:
    """
    Feature extractor for DINOv3 models.
    """
    def __init__(self, model_name: str, image_size=512):
        self.model_name = model_name
        self.model = DINOv3ViTModel.from_pretrained(model_name)
        self.model.eval()
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def to(self, device):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()

    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(self.model.embeddings.patch_embeddings.weight.dtype)
        hidden_states = self.model.embeddings(image, bool_masked_pos=None)
        position_embeddings = self.model.rope_embeddings(image)

        for i, layer_module in enumerate(self.model.layer):
            hidden_states = layer_module(
                hidden_states,
                position_embeddings=position_embeddings,
            )

        return F.layer_norm(hidden_states, hidden_states.shape[-1:])
        
    @torch.no_grad()
    def __call__(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        Extract features from the image.
        
        Args:
            image: A batch of images as a tensor of shape (B, C, H, W) or a list of PIL images.
        
        Returns:
            A tensor of shape (B, N, D) where N is the number of patches and D is the feature dimension.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((self.image_size, self.image_size), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).cuda()
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        image = self.transform(image).cuda()
        features = self.extract_features(image)
        return features


class DinoV3ConvNextFeatureExtractor:
    """
    Feature extractor for DINOv3 ConvNeXt models.
    """
    def __init__(self, model_name: str, image_size: int = 512, output_dim: int = 1024):
        self.model_name = model_name
        self.image_size = image_size
        self.model = DINOv3ConvNextModel.from_pretrained(model_name)
        self.model.eval()
        self.output_dim = output_dim
        hidden_dim = self.model.config.hidden_sizes[-1]
        self.proj = None
        if hidden_dim != output_dim:
            self.proj = torch.nn.Linear(hidden_dim, output_dim, bias=False)
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def to(self, device):
        self.model.to(device)
        if self.proj is not None:
            self.proj.to(device)

    def cuda(self):
        self.model.cuda()
        if self.proj is not None:
            self.proj.cuda()

    def cpu(self):
        self.model.cpu()
        if self.proj is not None:
            self.proj.cpu()

    @torch.no_grad()
    def __call__(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
            if image.shape[-2:] != (self.image_size, self.image_size):
                image = F.interpolate(
                    image,
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                )
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((self.image_size, self.image_size), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).cuda()
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")

        image = self.transform(image).cuda()
        features = self.model(image).last_hidden_state
        if self.proj is not None:
            features = self.proj(features)
        return F.layer_norm(features, features.shape[-1:])
