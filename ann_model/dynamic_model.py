# ann_model/dynamic_model.py
import torch
import torch.nn as nn

class DynamicFootballANN(nn.Module):
    def __init__(self, input_dim, hidden_layers, activation_name, dropout_rate=0.0, use_batch_norm=False, init_name="xavier_uniform"):
        super(DynamicFootballANN, self).__init__()
        
        layers = []
        
        # Activation mapping
        def get_activation_layer(name):
            n = name.lower()
            if n == "relu": return nn.ReLU()
            elif n == "leaky_relu": return nn.LeakyReLU()
            elif n == "sigmoid": return nn.Sigmoid()
            elif n == "tanh": return nn.Tanh()
            elif n == "elu": return nn.ELU()
            elif n == "selu": return nn.SELU()
            return nn.ReLU()
        
        prev_dim = input_dim
        for i, h_dim in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(get_activation_layer(activation_name))
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
            
        # Final prediction layer: 3 classes (0: Home Win, 1: Draw, 2: Away Win)
        layers.append(nn.Linear(prev_dim, 3))
        
        self.net = nn.Sequential(*layers)
        self._initialize_weights(init_name)

    def _initialize_weights(self, init_name):
        init_name = init_name.lower()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_name == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight)
                elif init_name == "xavier_normal":
                    nn.init.xavier_normal_(m.weight)
                elif init_name == "kaiming_uniform":
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif init_name == "kaiming_normal":
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                elif init_name == "orthogonal":
                    nn.init.orthogonal_(m.weight)
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.net(x)

    def get_layer_activations(self, x):
        """
        Executes a forward pass and captures intermediate activations at each layer.
        Returns a list of dicts with layer type and output values.
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

