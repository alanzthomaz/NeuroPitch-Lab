# ann_model/model.py
import torch
import torch.nn as nn

class FootballANN(nn.Module):
    def __init__(self, input_dim):
        super(FootballANN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # Output logits for 3 classes: 0=Home Win, 1=Draw, 2=Away Win
        )
        
    def forward(self, x):
        return self.net(x)

    def get_layer_activations(self, x):
        """
        Executes a forward pass and captures intermediate activations at each layer.
        """
        activations = []
        current_x = x.clone().detach().unsqueeze(0) # Batch size 1
        
        activations.append({
            "name": "Input",
            "type": "input",
            "values": current_x.squeeze(0).tolist()
        })
        
        hidden_idx = 0
        for layer in self.net:
            current_x = layer(current_x)
            if isinstance(layer, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh, nn.ELU, nn.SELU)):
                activations.append({
                    "name": f"Hidden_{hidden_idx}",
                    "type": "hidden",
                    "values": current_x.squeeze(0).tolist()
                })
                hidden_idx += 1
                
        activations.append({
            "name": "Output",
            "type": "output",
            "values": current_x.squeeze(0).tolist()
        })
        return activations
