import torch
import torchvision.transforms as T

def collate_fn(batch):
    return tuple(zip(*batch))

def get_transform(train, cfg):
    transforms = []

    if 'img_size' in cfg:
        transforms.append(T.Resize((cfg['img_size'], cfg['img_size'])))

    if train:
        if cfg.get('fliplr', 0) > 0:
            transforms.append(T.RandomHorizontalFlip(p=cfg['fliplr']))

        transforms.append(T.ColorJitter(
            brightness=cfg.get('hsv_v', 0),
            contrast=cfg.get('contrast', 0),
            saturation=cfg.get('hsv_s', 0),
            hue=cfg.get('hsv_h', 0)
        ))

        transforms.append(T.RandomAffine(
            degrees=0,
            translate=(cfg.get('translate', 0), cfg.get('translate', 0)),
            scale=(1 - cfg.get('mosaic', 0), 1 + cfg.get('mosaic', 0)),
            shear=cfg.get('shear', 0)
        ))

    transforms.append(T.ToTensor())

    # ⬇️ NAJWAŻNIEJSZE DODANIE: NORMALIZACJA DO ImageNet
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))

    return T.Compose(transforms)