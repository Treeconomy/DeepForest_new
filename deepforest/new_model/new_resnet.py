import torch
import torchvision.models as models
import pickle

def initialize_model(model_path, num_input_channels=4):
    # Define your custom ResNet model for object detection
    class ResNetForObjectDetection(torch.nn.Module):
        def __init__(self):
            super(ResNetForObjectDetection, self).__init__()
            
        
            # Example:
            self.resnet = models.resnet50()  # Load ResNet-50 architecture

            # Change the first convolutional layer to accept 4 input channels
            num_input_channels = 4  # Change this to 4 for 4 bands input
            self.resnet.conv1 = torch.nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

            # Replace the final fully connected layer for object detection
            num_classes = 1  # Example: 10 classes for object detection
            num_features = self.resnet.fc.in_features
            self.resnet.fc = torch.nn.Linear(num_features, num_classes)

        def forward(self, x):
            # Forward pass of your custom model
            x = self.resnet(x)
            return x

    # Initialize your custom ResNet model
    model = ResNetForObjectDetection()


    def initialize_parameters(module):
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(module.weight, 1)
            torch.nn.init.constant_(module.bias, 0)


    # Load the weights from NEON_weights.pkl
    with open(model_path, 'rb') as f:
        state_dict = pickle.load(f)

    # Modify the keys in the state dictionary to replace "backbone.body" with "resnet"
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("backbone.body", "resnet")
        new_state_dict[new_key] = value

    # Manipulate the state dictionary to replace the resnet.conv1.weight with the randomly initialized one
    pretrained_resnet = models.resnet50(pretrained=True)
    pretrained_resnet.conv1.weight.data.normal_(mean=0.0, std=0.01)
    # Manipulate the state dictionary to replace the resnet.conv1.weight with the randomly initialized one

    expanded_conv1_weight = torch.cat([pretrained_resnet.conv1.weight, torch.zeros(64, 1, 7, 7)], dim=1)
    new_state_dict['resnet.conv1.weight'] = expanded_conv1_weight

    # Load the manipulated state dictionary into your model
    # Initialize missing parameters randomly and ignore unexpected keys
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

    # Initialize missing parameters randomly
    for missing_key in missing_keys:
     print("Initializing randomly:", missing_key)
     initialize_parameters(model.state_dict()[missing_key])
    # Print used keys
    used_keys = set(new_state_dict.keys()) - set(unexpected_keys)
    for used_key in used_keys:
        print("Used key:", used_key)
    # Print unused keys
    for  unused_key in unexpected_keys:
        print("Unused key:", unused_key)

    return model

# Example usage:
model_path = '/home/nadja/DeepForest_new/deepforest/data/NEON_weights.pkl'
model = initialize_model(model_path)
