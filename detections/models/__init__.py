import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn as FasterRCNN
from torchvision.models.detection import fcos_resnet50_fpn as FCOS


__all__ = {
    'FasterRCNN': FasterRCNN,
    'FCOS': FCOS
}


def build_detector(cfg):
    assert isinstance(cfg, dict) and "TYPE" in cfg
    model_type = cfg.get("TYPE")
    pretrained = cfg.get("PRETRAINED", False)
    model = __all__[model_type](weights=None)
    if pretrained:
        model = load_state_dict(model, cfg.get("CKPT_PATH", ""))
    return model


def load_state_dict(model, ckpt_path):
    assert isinstance(ckpt_path, str) and ckpt_path != ''
    print('Load weights {}.'.format(ckpt_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = model.state_dict()
    loaded_dict = torch.load(ckpt_path, map_location=device)
    update_dict = {k: v for k, v in loaded_dict.items() if np.shape(state_dict[k]) == np.shape(v)}
    state_dict.update(update_dict)
    model.load_state_dict(state_dict)
    return model


# model = __all__['FasterRCNN'](pretrained=True)