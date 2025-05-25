import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import time
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import numpy as np
from torchvision.transforms.functional import to_pil_image
from PIL import ImageDraw

def train_and_validate(model, cfg, data_cfg, train_loader, val_loader, save_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['step_size'], gamma=cfg['gamma'])
    scaler = torch.cuda.amp.GradScaler()

    results = []
    start_time = time.time()
    best_score = 0.0
    patience_counter = 0
    patience = cfg.get('patience', 10)
    conf_threshold = cfg.get('conf_threshold', 0.05)

    for epoch in range(cfg['epochs']):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{cfg['epochs']}] Training")
        for images, targets in pbar:
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

            with torch.cuda.amp.autocast():
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        scheduler.step()

        model.eval()
        coco_results = []
        coco_ground_truth = []
        image_id = 0

        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"[Epoch {epoch+1}] Validation"):
                images = [img.to(device, non_blocking=True) for img in images]
                outputs = model(images)

                for img, pred, target in zip(images, outputs, targets):
                    boxes = pred['boxes'].cpu().numpy()
                    scores = pred['scores'].cpu().numpy()
                    labels = pred['labels'].cpu().numpy()
                    keep = scores >= conf_threshold
                    for box, score, label in zip(boxes[keep], scores[keep], labels[keep]):
                        x_min, y_min, x_max, y_max = box
                        width, height = x_max - x_min, y_max - y_min
                        if width > 0 and height > 0:
                            coco_results.append({
                                "image_id": image_id,
                                "category_id": int(label),
                                "bbox": [float(x_min), float(y_min), float(width), float(height)],
                                "score": float(score)
                            })
                    for box, label in zip(target['boxes'].cpu().numpy(), target['labels'].cpu().numpy()):
                        x_min, y_min, x_max, y_max = box
                        width, height = x_max - x_min, y_max - y_min
                        coco_ground_truth.append({
                            "image_id": image_id,
                            "category_id": int(label),
                            "bbox": [float(x_min), float(y_min), float(width), float(height)],
                            "area": float(width * height),
                            "iscrowd": 0,
                            "id": len(coco_ground_truth) + 1
                        })
                    image_id += 1

        gt_path = save_dir / "_coco_gt.json"
        dt_path = save_dir / f"_coco_dt_epoch{epoch+1}.json"

        try:
            with open(gt_path, 'w') as f:
                json.dump({
                    "images": [{"id": i} for i in range(image_id)],
                    "annotations": coco_ground_truth,
                    "categories": [{"id": i, "name": name} for i, name in data_cfg['names'].items()]
                }, f)
            with open(dt_path, 'w') as f:
                json.dump(coco_results, f)
        except Exception:
            coco_results = []

        try:
            if coco_results:
                coco_gt = COCO(gt_path)
                coco_dt = coco_gt.loadRes(dt_path)
                coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                mAP50 = coco_eval.stats[1]
                mAP50_95 = coco_eval.stats[0]
            else:
                raise ValueError("Empty or invalid COCO results.")
        except Exception:
            mAP50 = 0.0
            mAP50_95 = 0.0

        results.append({
            "epoch": epoch+1,
            "time": round(time.time() - start_time, 2),
            "train/loss": epoch_loss / len(train_loader),
            "metrics/mAP50(B)": mAP50,
            "metrics/mAP50-95(B)": mAP50_95
        })

        torch.save(model.state_dict(), save_dir / "weights" / f"epoch{epoch+1}.pth")

        print(f"[Epoch {epoch+1}] Loss: {epoch_loss:.2f}, mAP50: {mAP50:.4f}, mAP50-95: {mAP50_95:.4f}")

        if mAP50_95 > best_score:
            best_score = mAP50_95
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} due to no improvement in mAP50_95")
                break

        del images, targets, outputs
        torch.cuda.empty_cache()

    pd.DataFrame(results).to_csv(save_dir / "results.csv", index=False)
    print("Training complete. Results saved in:", save_dir)

def prepare_save_dir(save_dir, cfg):
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "weights").mkdir(exist_ok=True)
    with open(save_dir / "args.yaml", 'w') as f:
        yaml.dump(cfg, f)
    return save_dir
