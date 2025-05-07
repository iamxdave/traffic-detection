# yolo_model.py
import torch
import torch.nn as nn

class YOLO(nn.Module):
    def __init__(self, config):
        super(YOLO, self).__init__()
        self.config = config

        self.num_classes = config['num_classes']
        self.num_anchors = config['num_anchors']

        self.output_channels = self.num_anchors * (5 + self.num_classes)

        self.backbone = self.build_layers(config['layers'])

        # Assuming the last layer of the backbone is a convolutional layer
        last_out = config['layers'][-1]['out_channels']
        self.head = nn.Conv2d(last_out, self.output_channels, kernel_size=1)

    def build_layers(self, layer_configs):
        layers = []
        for layer in layer_configs:
            if layer['type'] == 'convolutional':
                layers.append(nn.Conv2d(
                    in_channels=layer['in_channels'],
                    out_channels=layer['out_channels'],
                    kernel_size=layer['kernel_size'],
                    stride=layer.get('stride', 1),
                    padding=layer.get('padding', 0)
                ))
                if layer.get('batch_norm'):
                    layers.append(nn.BatchNorm2d(layer['out_channels']))
                act = layer.get('activation')
                if act == 'relu':
                    layers.append(nn.ReLU())
                elif act == 'leaky':
                    layers.append(nn.LeakyReLU(0.1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)           # [B, C, H, W]
        x = self.head(x)               # [B, A*(5+C), H, W]

        B, _, H, W = x.shape
        A = self.num_anchors
        C = self.num_classes

        # Transform to [B, A, 5+C, H, W] → [B, H, W, A, 5+C]
        x = x.view(B, A, 5 + C, H, W)
        x = x.permute(0, 3, 4, 1, 2).contiguous()

        return x  # [B, H, W, A, 5 + num_classes]
    
    def compute_loss(self, preds, targets):
        """
        Loss: BCE + SmoothL1 + CrossEntropy
        """
        device = preds.device
        B, H, W, A, D = preds.shape
        C = D - 5
        box_loss_fn = torch.nn.SmoothL1Loss()
        obj_loss_fn = torch.nn.BCEWithLogitsLoss()
        cls_loss_fn = torch.nn.CrossEntropyLoss()

        box_loss = 0.0
        obj_loss = 0.0
        cls_loss = 0.0

        for b in range(B):
            target = targets[b].to(device)  # [N, 5]
            for t in target:
                cls_id, x, y, w, h = t
                gx, gy = int(x * W), int(y * H)
                if gx >= W or gy >= H:
                    continue
                pred = preds[b, gy, gx, 0]  # 1 anchor

                pred_box = pred[0:4]
                pred_obj = pred[4]
                pred_cls = pred[5:]

                gt_box = torch.tensor([x, y, w, h], device=device)
                box_loss += box_loss_fn(pred_box, gt_box)
                obj_loss += obj_loss_fn(pred_obj, torch.tensor(1.0, device=device))
                cls_loss += cls_loss_fn(pred_cls.unsqueeze(0), torch.tensor([int(cls_id)], device=device))

        return box_loss + obj_loss + cls_loss
