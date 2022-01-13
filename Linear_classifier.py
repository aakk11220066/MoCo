import torch


class Classifier(torch.nn.Module):
    def __init__(self, feature_extractor: torch.nn.Module, num_classes=10):
        super(Classifier, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_classes = num_classes
        self.feature_extractor = feature_extractor.to(device=device)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=2048, out_features=self.num_classes, device=device),
            torch.nn.Softmax(dim=1)
        ).to(device)

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extractor(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
