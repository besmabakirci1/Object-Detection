import torch
import torchvision.transforms as T
import cv2
import numpy as np
from typing import Optional, Tuple

class CarModelClassifier:
    """
    Lightweight wrapper around a Torch image classifier for car make/model.
    Expects a classification model that returns logits over label set.
    """
    def __init__(self, weights_path: Optional[str] = None, labels: Optional[list] = None, device: Optional[str] = None):
        self.device = device or ( 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu') )
        self.model = None
        self.labels = labels or []
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if weights_path:
            self.load(weights_path)

    def load(self, weights_path: str):
        """Load a Torch classification model (expects scripted or state_dict with architecture baked)"""
        try:
            self.model = torch.jit.load(weights_path, map_location=self.device)
        except Exception:
            # fallback to torch.load for non-scripted checkpoints if they contain full model
            self.model = torch.load(weights_path, map_location=self.device)
        self.model.eval().to(self.device)

    def predict(self, bgr_crop: np.ndarray) -> Tuple[str, float]:
        if self.model is None:
            return ("unknown", 0.0)
        # convert BGR (cv2) to RGB
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            probs = torch.softmax(logits, dim=1)
            conf, idx = torch.max(probs, dim=1)
            idx = idx.item()
            conf = float(conf.item())
        label = self.labels[idx] if self.labels and 0 <= idx < len(self.labels) else str(idx)
        return (label, conf)
