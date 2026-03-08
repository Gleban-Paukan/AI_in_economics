import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, in_features: int, n_classes: int, in_channels: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(400, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.dropout(x)
        x = self.fc2(x)

        return x

class Predictor:
    def __init__(self, model_path: str, in_features: int = 28 * 28, n_classes: int = 10, in_channels: int = 1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Classifier(in_features=in_features, n_classes=n_classes, in_channels=in_channels)
        
        # Load weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, data: torch.Tensor):
        """
        Predicts the class and confidence for the given input data.
        
        Args:
            data (torch.Tensor): A batch of images, shape [B, 1, 28, 28] with floated values
            
        Returns:
            predicted_class (np.ndarray): Predicted class indices
            confidence (np.ndarray): Confidence scores (probabilities)
            all_probs (np.ndarray): Probabilities for all 10 classes
        """
        data = data.to(self.device)
        with torch.no_grad():
            logits = self.model(data)
            probs = torch.softmax(logits, dim=1)
            confidence, predicted_class = torch.max(probs, dim=1)
        
        return predicted_class.cpu().numpy(), confidence.cpu().numpy(), probs.cpu().numpy()

    def get_saliency_map(self, data: torch.Tensor, target_class: int):
        """
        Generates a saliency map for the given input data and target class 
        using gradient w.r.t the input image to highlight important pixels.
        """
        self.model.eval()
        data = data.to(self.device)
        data.requires_grad_()
        
        logits = self.model(data)
        
        # Get the score for the target class
        score = logits[0, target_class]
        
        # Backward passing to get gradients
        score.backward()
        
        # Get max magnitude of gradients across color channels (only 1 channel here, but standard practice)
        saliency, _ = torch.max(data.grad.data.abs(), dim=1)
        return saliency.cpu().numpy()[0]
