import torch


class Classifier(torch.nn.Module):
    def __init__(self, feature_extractor: torch.nn.Module, loss_func=None, optimizer_type=None):
        super(Classifier, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.feature_extractor = feature_extractor.to(device=device)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=1000, out_features=10),
            torch.nn.Softmax(dim=1)
        )
        # ftr_exctr_final_layer = self.feature_extractor.fc[-1]
        # self.fc = torch.nn.Linear(in_features=ftr_exctr_final_layer.out_features,
        #                           out_features=ftr_exctr_final_layer.out_features, device=device)

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extractor(x)  # Only fine-tune classifier head
        x = self.fc(x)
        return x
