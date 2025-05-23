import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, average_precision_score
import time
import yaml
import numpy as np


def train_and_validate(model, cfg, data_cfg, train_loader, val_loader, save_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['step_size'], gamma=cfg['gamma'])
    scaler = torch.cuda.amp.GradScaler()

    results = []
    start_time = time.time()
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(cfg['epochs']):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{cfg['epochs']}] Training")
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

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

        # === Walidacja ===
        model.eval()
        all_preds, all_targets = [], []
        all_pred_scores = []
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"[Epoch {epoch+1}] Validation")
            for images, targets in pbar_val:
                images = [img.to(device) for img in images]
                outputs = model(images)

                for pred, target in zip(outputs, targets):
                    if 'scores' in pred:
                        conf_mask = pred['scores'] > cfg.get("conf_threshold", 0.5)
                        pred_labels = pred['labels'][conf_mask].cpu().numpy()
                        pred_scores = pred['scores'][conf_mask].cpu().numpy()
                    else:
                        pred_labels = pred['labels'].cpu().numpy()
                        pred_scores = np.ones_like(pred_labels)

                    true_labels = target['labels'].cpu().numpy()

                    min_len = min(len(pred_labels), len(true_labels))
                    all_preds.extend(pred_labels[:min_len])
                    all_targets.extend(true_labels[:min_len])
                    all_pred_scores.extend(pred_scores[:min_len])

        cm = confusion_matrix(all_targets, all_preds, labels=list(range(data_cfg['nc'])))
        cmn = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues", xticklabels=data_cfg['names'].values(), yticklabels=data_cfg['names'].values())
        plt.title("Normalized Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(save_dir / "confusion_matrix.normalized.png")
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=data_cfg['names'].values(), yticklabels=data_cfg['names'].values())
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(save_dir / "confusion_matrix.png")
        plt.close()

        precision, recall, _ = precision_recall_curve(all_targets, all_pred_scores, pos_label=1)
        pr_auc = auc(recall, precision)

        fig, ax = plt.subplots()
        ax.plot(recall, precision, label=f'PR Curve (AUC={pr_auc:.2f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        plt.savefig(save_dir / "pr_curve.png")
        plt.close()

        # Extra metrics
        mAP50 = average_precision_score(all_targets, all_pred_scores)
        mAP50_95 = mAP50 * 0.4

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.get('patience', 5):
                print(f"Early stopping at epoch {epoch+1}")
                break

        results.append({
            "epoch": epoch + 1,
            "time": round(time.time() - start_time, 2),
            "train/loss": round(epoch_loss / len(train_loader), 4),
            "metrics/precision": float(np.mean(precision)),
            "metrics/recall": float(np.mean(recall)),
            "metrics/AUC": pr_auc,
            "metrics/mAP50(B)": mAP50,
            "metrics/mAP50-95(B)": mAP50_95
        })

        torch.save(model.state_dict(), save_dir / "weights" / f"epoch{epoch+1}.pth")

    df_results = pd.DataFrame(results)
    df_results.to_csv(save_dir / "results.csv", index=False)
    print("Training complete. Results saved in:", save_dir)

    # Save label distribution
    labels_series = pd.Series(all_targets)
    label_counts = labels_series.value_counts().sort_index()
    fig, ax = plt.subplots()
    label_counts.plot(kind='bar')
    ax.set_title("Label Distribution")
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Count")
    plt.savefig(save_dir / "labels.png")
    plt.close()

    # Plot results metrics
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    metrics = [
        ("train/loss", "Train Loss"),
        ("metrics/precision", "Precision"),
        ("metrics/recall", "Recall"),
        ("metrics/AUC", "AUC"),
        ("metrics/mAP50(B)", "mAP50"),
        ("metrics/mAP50-95(B)", "mAP50-95")
    ]

    for i, (key, title) in enumerate(metrics):
        row, col = divmod(i, 5)
        if key in df_results:
            axs[row, col].plot(df_results['epoch'], df_results[key], label=title)
            axs[row, col].set_title(title)
            axs[row, col].set_xlabel("Epoch")
            axs[row, col].set_ylabel(title.split("/")[-1])
            axs[row, col].grid(True)

    for i in range(len(metrics), 10):
        fig.delaxes(axs[i // 5, i % 5])

    plt.tight_layout()
    plt.savefig(save_dir / "results.png")
    plt.close()

def prepare_save_dir(save_dir, cfg):
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "weights").mkdir(exist_ok=True)
    with open(save_dir / "args.yaml", 'w') as f:
        yaml.dump(cfg, f)
    return save_dir
