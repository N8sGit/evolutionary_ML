import torch.nn as nn

class FlexibleNN(nn.Module):
    def __init__(self, layer_specs=None, input_size=28*28):
        """
        Flexible Neural Network that dynamically builds architecture based on provided layer_specs.
        Args:
        - layer_specs (list): A list of tuples where each tuple contains the layer type and its parameters.
        - input_size (int): The size of the input features (e.g., 28*28 for MNIST, 32*32*3 for CIFAR-10).
        """
        super(FlexibleNN, self).__init__()
        self.input_size = input_size  # Allow input size customization
        if layer_specs is None:
            raise ValueError("layer_specs cannot be None. Use create_random_architecture to generate layer_specs.")
        
        self.layer_specs = layer_specs
        self.build_layers()

    def build_layers(self):
        """
        Build layers dynamically based on the layer_specs.
        Supported layers include: 'Linear', 'ReLU', 'Dropout'.
        """
        self.layer_list = nn.ModuleList()
        for layer_type, params in self.layer_specs:
            if layer_type == 'Linear':
                # Adjust input layer to match the dataset input size
                if 'in_features' in params and params['in_features'] == 28*28:
                    params['in_features'] = self.input_size
                layer = nn.Linear(**params)
            elif layer_type == 'ReLU':
                layer = nn.ReLU()
            elif layer_type == 'Dropout':
                layer = nn.Dropout(**params)
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
            self.layer_list.append(layer)

    def forward(self, x):
        """
        Forward pass through the network.
        """
        # Flatten input if necessary (for fully connected layers)
        x = x.view(x.size(0), -1)  # Flatten input
        for layer in self.layer_list:
            x = layer(x)
        return x

    def verify_dimensions(self):
        """
        Verifies that the in_features and out_features of the 'Linear' layers match correctly.
        """
        previous_out_features = None
        for idx, (layer_type, params) in enumerate(self.layer_specs):
            if layer_type == 'Linear':
                in_features = params['in_features']
                out_features = params['out_features']
                if previous_out_features is not None and in_features != previous_out_features:
                    raise ValueError(f"Inconsistent dimensions at layer {idx}: "
                                    f"in_features {in_features} does not match previous out_features {previous_out_features}")
                previous_out_features = out_features