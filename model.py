import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, hidden_size=128, dropout_rate=0.5):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, 10)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class FlexibleNN(nn.Module):
    def __init__(self, layer_specs=None):
        super(FlexibleNN, self).__init__()
        if layer_specs is None:
            # Initialize with a default architecture
            self.layer_specs = [
                ('Linear', {'in_features': 28*28, 'out_features': 128}),
                ('ReLU', {}),
                ('Dropout', {'p': 0.5}),
                ('Linear', {'in_features': 128, 'out_features': 10}),
            ]
        else:
            self.layer_specs = layer_specs
        # Build the layers
        self.build_layers()
        # Verify layer dimensions
        self.verify_dimensions()

    def build_layers(self):
        self.layer_list = nn.ModuleList()
        for layer_type, params in self.layer_specs:
            if layer_type == 'Linear':
                layer = nn.Linear(**params)
            elif layer_type == 'ReLU':
                layer = nn.ReLU()
            elif layer_type == 'Dropout':
                layer = nn.Dropout(**params)
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
            self.layer_list.append(layer)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layer_list:
            x = layer(x)
        return x

    def verify_dimensions(self):
        # Ensure that the in_features and out_features of Linear layers match
        previous_out_features = None
        for idx, (layer_type, params) in enumerate(self.layer_specs):
            if layer_type == 'Linear':
                in_features = params['in_features']
                out_features = params['out_features']
                if previous_out_features is not None and in_features != previous_out_features:
                    raise ValueError(f"Inconsistent dimensions at layer {idx}: "
                                    f"in_features {in_features} does not match previous out_features {previous_out_features}")
                previous_out_features = out_features
            elif layer_type == 'Conv2d':
                # Include checks for convolutional layers if added
                pass