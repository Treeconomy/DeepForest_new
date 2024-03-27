import torchvision
from torchvision.models.detection.retinanet import RetinaNet
from torchvision.models.detection.retinanet import RetinaNet_ResNet50_FPN_Weights
import torch
import pickle


def create_backbone(num_input_channels=4):
    # Load RetinaNet model with ResNet-50 backbone
    resnet = torchvision.models.detection.retinanet_resnet50_fpn(
            weights=RetinaNet_ResNet50_FPN_Weights.COCO_V1) # You can adjust num_classes as needed

    # Modify the first convolutional layer to accept a different number of input channels
    resnet.backbone.body.conv1 = torch.nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    backbone = resnet.backbone

    # Instantiate the RetinaNet model with modified backbone
    model = RetinaNet(backbone=backbone, num_classes=1)

    # Set additional parameters for the RetinaNet model
    model.nms_thresh = 0.4
    model.score_thresh = 0.5

    return model

def initialize_parameters(module):
    if isinstance(module, torch.nn.Conv2d):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(module.weight, 1)
        torch.nn.init.constant_(module.bias, 0)

def create_model_with_weights(model_path, num_input_channels=4):
    # Create the backbone with modified first convolutional layer
    backbone = create_backbone(num_input_channels)

    # Load the weights from NEON_weights.pkl
    with open(model_path, 'rb') as f:
        state_dict = pickle.load(f)

    # Manipulate the state dictionary to replace the backbone.body.conv1.weight with the randomly initialized one
    # expanded_conv1_weight = torch.cat([backbone.backbone.body.conv1.weight, torch.zeros(64, num_input_channels-4 , 7, 7)], dim=1)
    # state_dict['backbone.body.conv1.weight'] = expanded_conv1_weight
    # #backbone = backbone.backbone
    # Extract the weights for conv1 from the original state dictionary
    original_conv1_weight = state_dict['backbone.body.conv1.weight']

    # Create a tensor with zeros for the fourth channel
    zero_channel = torch.zeros_like(original_conv1_weight[:, :1, :, :])

    # Concatenate the original weights with the zero-filled tensor for the fourth channel
    expanded_conv1_weight = torch.cat([original_conv1_weight[:, :3, :, :], zero_channel], dim=1)

    # Update the state dictionary with the modified weights
    state_dict['backbone.body.conv1.weight'] = expanded_conv1_weight

    # Instantiate the RetinaNet model with modified backbone
    model = RetinaNet(backbone=backbone.backbone, num_classes=1)

    # Load the manipulated state dictionary into the model
    # Initialize missing parameters randomly and ignore unexpected keys
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    # Initialize missing parameters randomly
    for missing_key in missing_keys:
        print("Initializing randomly:", missing_key)
        initialize_parameters(model.state_dict()[missing_key])

    # Print used keys
    used_keys = set(state_dict.keys()) - set(unexpected_keys)
    for used_key in used_keys:
        print("Used key:", used_key)

    # Print unused keys
    for unused_key in unexpected_keys:
        print("Unused key:", unused_key)

    return model
    
