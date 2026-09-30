import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
from torchvision.ops import box_iou

import gc
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent

from src._dataset import ImageNet, PascalVOC, DistilledDataset
from _utils import seed_everything
seed_everything(seed=42)

# ============================================================
# Test Configuration
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LR = 1e-4
EPOCHS = 50
T_MAX = EPOCHS
ETA_MIN = 1e-6

# ============================================================
# Memory Cleanup
# ============================================================
def _clear_memory():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# ============================================================
# Custom Collate Fn
# ============================================================
def _det_collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def _seg_collate_fn(batch):
    images, masks = zip(*batch)
    return torch.stack(images), torch.stack(masks)

# ============================================================
# Classification: Train / Evaluate
# ============================================================
def _train_cls(model, loader, device, lr, epochs):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=ETA_MIN)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            loss = loss_fn(model(data), target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            del loss, data, target

        scheduler.step()

        train_loss /= len(loader)
        print(f"[Cls] Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f}")

    del optimizer, scheduler, loss_fn
    _clear_memory()

    return model

@torch.no_grad()
def _evaluate_cls(model, loader, device):
    model.to(device)
    model.eval()

    correct, total = 0, 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        output = model(data)
        pred = output.argmax(dim=-1)

        correct += (pred == target).sum().item()
        total += target.size(0)

        del output, pred, data, target

    accuracy = correct / max(total, 1)
    print(f"[Cls] Test Accuracy: {accuracy:.4f}")

    return {"accuracy": accuracy}

# ============================================================
# Detection: Train / Evaluate
# ============================================================
def _train_det(model, loader, device, lr, epochs):
    model.to(device)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=ETA_MIN)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for images, targets in loader:
            images = [image.to(device) for image in images]
            targets = [
                {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()}
                for t in targets
            ]
            optimizer.zero_grad()

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            del images, targets, loss_dict, loss

        scheduler.step()

        train_loss /= len(loader)
        print(f"[Det] Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f}")

    del optimizer, scheduler
    _clear_memory()

    return model

def _average_precision(recall, precision):
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    indices = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])

    return float(ap)

@torch.no_grad()
def _evaluate_det(model, loader, device, num_classes, iou_threshold=0.5, score_threshold=0.05):
    model.to(device)
    model.eval()

    detections = {c: [] for c in range(1, num_classes)}
    num_gt = {c: 0 for c in range(1, num_classes)}

    for images, targets in loader:
        images = [image.to(device) for image in images]
        predictions = model(images)

        for target, prediction in zip(targets, predictions):
            gt_boxes = target["boxes"].to(device)
            gt_labels = target["labels"].to(device)

            pred_boxes = prediction["boxes"]
            pred_labels = prediction["labels"]
            pred_scores = prediction["scores"]

            for c in range(1, num_classes):
                num_gt[c] += int((gt_labels == c).sum().item())

            keep = pred_scores >= score_threshold

            pred_boxes = pred_boxes[keep]
            pred_labels = pred_labels[keep]
            pred_scores = pred_scores[keep]

            order = torch.argsort(pred_scores, descending=True)
            matched_gt = set()

            for idx in order.tolist():
                label = int(pred_labels[idx].item())
                score = float(pred_scores[idx].item())

                if label not in detections:
                    continue

                class_mask = (gt_labels == label)

                if class_mask.sum() == 0:
                    detections[label].append((score, 0))
                    continue

                ious = box_iou(pred_boxes[idx].unsqueeze(0), gt_boxes[class_mask]).squeeze(0)
                best_iou, best_idx = ious.max(dim=0)

                gt_indices = class_mask.nonzero(as_tuple=True)[0]
                best_gt = int(gt_indices[best_idx].item())

                if best_iou.item() >= iou_threshold and (label, best_gt) not in matched_gt:
                    detections[label].append((score, 1))
                    matched_gt.add((label, best_gt))
                else:
                    detections[label].append((score, 0))

            del gt_boxes, gt_labels, pred_boxes, pred_labels
            del pred_scores
            del prediction

        del images, predictions

    aps = []

    for c in range(1, num_classes):
        if num_gt[c] == 0:
            continue

        entries = sorted(detections[c], key=lambda x: x[0], reverse=True)
        tp = np.array([e[1] for e in entries], dtype=np.float64)
        fp = 1.0 - tp

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)

        recall = tp_cum / max(num_gt[c], 1)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-8)

        aps.append(_average_precision(recall, precision))

    mAP = float(np.mean(aps)) if aps else 0.0

    print(f"[Det] mAP@{iou_threshold:.2f}: {mAP:.4f}")

    del detections, num_gt, aps

    return {"mAP": mAP}

# ============================================================
# Segmentation: Train / Evaluate
# ============================================================
def _train_seg(model, loader, device, lr, epochs):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=ETA_MIN)
    loss_fn = nn.CrossEntropyLoss(ignore_index=255)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            loss = loss_fn(model(data)["out"], target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            del data, target, loss

        scheduler.step()

        train_loss /= len(loader)
        print(f"[Seg] Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f}")

    del optimizer, scheduler, loss_fn
    _clear_memory()

    return model

@torch.no_grad()
def _evaluate_seg(model, loader, device, num_classes, ignore_index=255):
    model.to(device)
    model.eval()

    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        pred = model(data)["out"].argmax(dim=1)
        valid = (target != ignore_index)

        pred = pred[valid].cpu()
        target = target[valid].cpu()
        indices = target * num_classes + pred
        confusion += torch.bincount(indices, minlength=num_classes ** 2).reshape(num_classes, num_classes)

        del pred, target, data, indices, valid

    intersection = torch.diag(confusion).float()
    union = confusion.sum(dim=1).float() + confusion.sum(dim=0).float() - intersection
    valid_class = union > 0
    iou = intersection[valid_class] / union[valid_class].clamp(min=1)
    miou = iou.mean().item() if valid_class.any() else 0.0

    print(f"[Seg] Mean IoU: {miou:.4f}")

    del confusion, intersection, union, valid_class, iou

    return {"mIoU": miou}

# ============================================================
# Test
# ============================================================
def test(task):
    device = DEVICE

    if task == "cls":
        print("--- Test Classification ---")
        print("Load Stage1 Result")
        cls_s1_result = torch.load(ROOT / "src/stage1_result/cls.pt", map_location="cpu", weights_only=False)
        cls_num_classes = cls_s1_result["num_classes"]
        cls_data_default = cls_s1_result["data_default"]
        cls_cfg = cls_s1_result["config"]

        print("Load Stage2 Result")
        cls_s2_result = torch.load(ROOT / "src/stage2_result/cls.pt", map_location="cpu", weights_only=False)
        cls_paths = cls_s2_result["paths"]
        cls_labels = cls_s2_result["labels"]

        print("Load Validation Dataset")
        cls_val = ImageNet(root=ROOT, relative_path="src/org_data/ImageNet/imagenette2-160", split="val")
        val_cls_loader = cls_val.build_dataloader(batch_size=32, shuffle=False)

        print("Load Distilled Train Dataset")
        cls_train = DistilledDataset(task="cls", root=ROOT, relative_paths=cls_paths, labels=cls_labels, transforms=cls_data_default["TRANSFORMS"])
        train_cls_loader = cls_train.build_dataloader(batch_size=32, shuffle=True)

        print("Load Test Model ResNet18")
        cls_model = resnet18()
        cls_model.fc = nn.Linear(in_features=cls_model.fc.in_features, out_features=cls_num_classes)

        print("\nTrain Test Model on Distilled Data")
        cls_model = _train_cls(cls_model, train_cls_loader, device=device, lr=LR, epochs=EPOCHS)

        print("\nEvaluate on Real Validation Set")
        result = _evaluate_cls(cls_model, val_cls_loader, device=device)

        del cls_model, train_cls_loader, cls_train, val_cls_loader, cls_val
        del cls_s1_result, cls_s2_result
        del cls_paths, cls_labels
        del cls_num_classes, cls_data_default, cls_cfg
        _clear_memory()

    elif task == "det":
        print("--- Test Detection ---")
        print("Load Stage1 Result")
        det_s1_result = torch.load(ROOT / "src/stage1_result/det.pt", map_location="cpu", weights_only=False)
        det_num_classes = det_s1_result["num_classes"]
        det_data_default = det_s1_result["data_default"]
        det_cfg = det_s1_result["config"]

        print("Load Stage2 Result")
        det_s2_result = torch.load(ROOT / "src/stage2_result/det.pt", map_location="cpu", weights_only=False)
        det_paths = det_s2_result["paths"]
        det_labels = det_s2_result["labels"]

        print("Load Validation Dataset")
        val_det = PascalVOC(task="det", root=ROOT, relative_path="src/org_data/PascalVOC/VOC2012", split="val")
        val_det_loader = val_det.build_dataloader(batch_size=32, shuffle=False)

        print("Load Distilled Train Dataset")
        det_train = DistilledDataset(task="det", root=ROOT, relative_paths=det_paths, labels=det_labels, transforms=det_data_default["TRANSFORMS"])
        train_det_loader = DataLoader(det_train, batch_size=8, shuffle=True, collate_fn=_det_collate_fn)

        print("Load Test Model Faster R-CNN ResNet50 FPN")
        det_model = fasterrcnn_resnet50_fpn(pretrained=False)
        det_model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=det_model.roi_heads.box_predictor.cls_score.in_features, num_classes=det_num_classes)

        print("\nTrain Test Model on Distilled Data")
        det_model = _train_det(det_model, train_det_loader, device=device, lr=LR, epochs=EPOCHS)

        print("\nEvaluate on Real Validation Set")
        result = _evaluate_det(det_model, val_det_loader, device=device, num_classes=det_num_classes)

        del det_model, train_det_loader, det_train, val_det_loader, val_det
        del det_s1_result, det_s2_result
        del det_paths, det_labels
        del det_num_classes, det_data_default, det_cfg
        _clear_memory()

    elif task == "seg":
        print("--- Test Segmentation ---")
        print("Load Stage1 Result")
        seg_s1_result = torch.load(ROOT / "src/stage1_result/seg.pt", map_location="cpu", weights_only=False)
        seg_num_classes = seg_s1_result["num_classes"]
        seg_data_default = seg_s1_result["data_default"]
        seg_cfg = seg_s1_result["config"]

        print("Load Stage2 Result")
        seg_s2_result = torch.load(ROOT / "src/stage2_result/seg.pt", map_location="cpu", weights_only=False)
        seg_paths = seg_s2_result["paths"]
        seg_labels = seg_s2_result["labels"]

        print("Load Validation Dataset")
        val_seg = PascalVOC(task="seg", root=ROOT, relative_path="src/org_data/PascalVOC/VOC2012", split="val")
        val_seg_loader = val_seg.build_dataloader(batch_size=32, shuffle=False)

        print("Load Distilled Train Dataset")
        seg_train = DistilledDataset(task="seg", root=ROOT, relative_paths=seg_paths, labels=seg_labels, transforms=seg_data_default["TRANSFORMS"])
        train_seg_loader = DataLoader(seg_train, batch_size=16, shuffle=True, collate_fn=_seg_collate_fn)

        print("Load Test Model LR-ASPP MobileNetV3 Large")
        seg_model = lraspp_mobilenet_v3_large(pretrained=False)
        seg_model.classifier.low_classifier = nn.Conv2d(seg_model.classifier.low_classifier.in_channels, seg_num_classes, kernel_size=1)
        seg_model.classifier.high_classifier = nn.Conv2d(seg_model.classifier.high_classifier.in_channels, seg_num_classes, kernel_size=1)

        print("\nTrain Test Model on Distilled Data")
        seg_model = _train_seg(seg_model, train_seg_loader, device=device, lr=LR, epochs=EPOCHS)

        print("\nEvaluate on Real Validation Set")
        result = _evaluate_seg(seg_model, val_seg_loader, device=device, num_classes=seg_num_classes)

        del seg_model, train_seg_loader, seg_train, val_seg_loader, val_seg
        del seg_s1_result, seg_s2_result
        del seg_paths, seg_labels
        del seg_num_classes, seg_data_default, seg_cfg
        _clear_memory()

    else:
        raise ValueError(f"{task} not found | task: ['cls', 'det', 'seg']")

    print(f"\n===== Test Finish! ({task}) =====\n")

    return result

if __name__ == "__main__":
    print("=" * 60)
    print("Test Configuration")
    print(f"Device    : {DEVICE}")
    print(f"LR        : {LR}")
    print(f"Epochs    : {EPOCHS}")
    print(f"T_max     : {T_MAX}")
    print(f"Eta min   : {ETA_MIN}")
    print("=" * 60)

    results = {}

    for task in ["cls", "det", "seg"]:
        results[task] = test(task)
        _clear_memory()

    print("\n===== Summary =====")

    for task, result in results.items():
        print(f"{task}: {result}")

    del results
    _clear_memory()