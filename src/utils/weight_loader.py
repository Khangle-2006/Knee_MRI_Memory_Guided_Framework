"""
Smart weight loader for transferring a contrastive-pretrained ResNet18
backbone into the HybridBackbone wrapper.

Handles:
- 'encoder.' prefix stripping
- key renaming (conv1 -> features_2d.0, layer1 -> features_2d.4, ...)
- 1-channel -> 3-channel conv1 expansion (replicate + divide by 3)
"""
import torch


def load_backbone_weights(model, weight_path):
    print(f"--> [Smart Weight Loading] {weight_path}")
    try:
        checkpoint = torch.load(weight_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items()}

        key_mapping = {
            'conv1.': 'features_2d.0.', 'bn1.': 'features_2d.1.',
            'layer1.': 'features_2d.4.', 'layer2.': 'features_2d.5.',
            'layer3.': 'features_2d.6.', 'layer4.': 'features_2d.7.'
        }

        model_state = model.backbone.state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            for old, new in key_mapping.items():
                if k.startswith(old):
                    new_key = k.replace(old, new)
                    break
            if new_key in model_state:
                target_shape = model_state[new_key].shape
                loaded_shape = v.shape
                if target_shape == loaded_shape:
                    new_state_dict[new_key] = v
                elif (len(target_shape) == 4 and len(loaded_shape) == 4
                      and target_shape[2:] == loaded_shape[2:]):
                    if loaded_shape[1] == 1 and target_shape[1] == 3:
                        v = torch.cat([v, v, v], dim=1) / 3.0
                        new_state_dict[new_key] = v
                else:
                    continue
        model.backbone.load_state_dict(new_state_dict, strict=False)
        print("--> Loaded backbone successfully!")
    except Exception as e:
        print(f"!!! Error loading weights: {e}")
    return model
