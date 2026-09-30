import logging
logging.disable(logging.CRITICAL)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.segmentation import lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights

from pathlib import Path
ROOT = Path(__file__).resolve().parent

# ------------------------------------------------------------
# TaskSpecificProxy
# ------------------------------------------------------------
class TaskSpecificProxy(nn.Module):
    """
    Task Specific Proxy Model Class
    -> Capture and Retain Essential Dataset Information
    """
    def __init__(self, task, num_classes, data_default, config):
        super().__init__()

        self.task = task
        self.num_classes = num_classes
        self.transform = data_default["TRANSFORMS"]
        self.mean = torch.tensor(data_default["MEAN"], dtype=torch.float32).view(1, -1, 1, 1)
        self.std = torch.tensor(data_default["STD"], dtype=torch.float32).view(1, -1, 1, 1)

        self.device = config.DEVICE
        self.image_size = config.IMAGE_SIZE[task]
        self.task_threshold = config.TASK_THRESHOLD[task]

        self.lr = config.LR
        self.epochs = config.EPOCHS
        self.save_path = ROOT / f"tsp_model/tsp_{config.NAME}.pt"

        self.model = self._build_model(task, num_classes)
        self.loss_fn = self._loss_fn(task)

    def forward(self, x, target=None):
        if self.task == "det":
            if target is not None:
                return self.model(x, target)
        
        return self.model(x)
    
    @torch.no_grad()
    def predict(self, x):
        self.eval()

        return self.forward(x)
    
    # ------------------------------------------------------------
    # Method
    # ------------------------------------------------------------
    def load_model(self, load_checkpoint):
        state_dict = load_checkpoint if isinstance(load_checkpoint, dict) \
                else torch.load(load_checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict=state_dict)

        return self
    
    def train_model(self, train_loader, val_loader):
        print("Train Model")
        
        self.to(self.device)
        optimizer = AdamW(params=self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=self.epochs)
        best_val_loss = float("inf")

        for epoch in range(self.epochs):
            self.train()
            train_loss = 0.0

            for data, target in train_loader:
                data, target = self._to_device(data, target)
                optimizer.zero_grad()

                loss = self._calculate_loss(data, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            if val_loader is None:
                print(f"Epoch {epoch + 1}/{self.epochs} | Train Loss: {train_loss:.4f}")

            else:
                val_loss = 0.0
                if self.task == "cls" or self.task == "seg":
                    self.eval()

                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = self._to_device(data, target)

                        loss = self._calculate_loss(data, target)
                        val_loss += loss.item()

                val_loss /= len(val_loader)
                print(f"Epoch {epoch + 1}/{self.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), self.save_path)
                    print(f"Best model saved! Val Loss: {val_loss:.4f}")

            scheduler.step()

        if val_loader is None:
            torch.save(self.model.state_dict(), self.save_path)
        self.model.load_state_dict(torch.load(self.save_path, map_location=self.device, weights_only=True))

        return self
    
    def proxy_task_loss(self, tensor, target):
            tensor, target = self._to_device(tensor, target)

            if self.task == "det":
                data = [self._normalize(tensor[0])[0]]
                self.train()
                for p in self.model.parameters():
                    p.requires_grad_(False)
            else:
                data = self._normalize(tensor)
                self.eval()

            return self._calculate_loss(data, target)

    @torch.no_grad()
    def filter_and_relabel(self, images):
        self.to(self.device)
        self.eval()
        kept_images, kept_labels, kept_scores = [], [], []

        for image in images:
            data = self.transform(image).to(self.device)
            x = [data] if self.task == "det" else data.unsqueeze(0)
            pred = self.predict(x)
            score = self._score(pred)

            if score >= self.task_threshold:
                kept_images.append(image)
                kept_labels.append(self._relabel(pred))
                kept_scores.append(score)

        discard_rate = 1.0 - (len(kept_images) / max(len(images), 1))
        print(f"Refinement: {len(kept_images)}/{len(images)}\n"
              f"Discard_Rate: {discard_rate:.1%}")

        return {"images": kept_images, "labels": kept_labels, "scores": kept_scores}
    
    # ------------------------------------------------------------
    # Helper Method
    # ------------------------------------------------------------
    def _build_model(self, task, num_classes):
        if task == "cls":
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
            model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)

        elif task == "det":
            model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
            model.roi_heads.box_predictor = FastRCNNPredictor(
                in_channels=model.roi_heads.box_predictor.cls_score.in_features,
                num_classes=num_classes
            )

        elif task == "seg":
            model = lraspp_mobilenet_v3_large(weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT)
            model.classifier.low_classifier = nn.Conv2d(
                in_channels=model.classifier.low_classifier.in_channels,
                out_channels=num_classes, kernel_size=1
            )
            model.classifier.high_classifier = nn.Conv2d(
                in_channels=model.classifier.high_classifier.in_channels,
                out_channels=num_classes, kernel_size=1
            )

        else:
            raise ValueError(f"{task} not found")

        return model

    def _loss_fn(self, task):
        if task == "cls":
            return nn.CrossEntropyLoss()

        elif task == "det":
            return None

        elif task == "seg":
            return nn.CrossEntropyLoss(ignore_index=255)

        else:
            raise ValueError(f"{task} not found")
    
    def _to_device(self, datas, targets):
        if self.task == "det":
            data = [d.to(self.device) for d in datas]
            target = [{
                key: (value.to(self.device) if torch.is_tensor(value) else value)
                for key, value in t.items()} 
                for t in targets]
        else:
            data = datas.to(self.device)
            target = targets.to(self.device)
            if self.task == "cls" and target.dim() == 0:
                target = target.unsqueeze(0)

        return data, target
    
    def _calculate_loss(self, data, target):
        if self.task == "cls":
            return self.loss_fn(self(data), target)

        elif self.task == "det":
            return sum(self(data, target).values())

        elif self.task == "seg":
            if target.ndim == 2:
                target = target.unsqueeze(0)
            return self.loss_fn(self(data)["out"], target)

        else:
            raise ValueError(f"{self.task} not found")  
    
    def _normalize(self, image):
        if image.dim() == 3:
            image = image.unsqueeze(0)

        if image.shape[-2:] != self.image_size:
            image = F.interpolate(image, size=self.image_size, mode="bilinear", align_corners=False)

        mean = self.mean.to(device=image.device, dtype=image.dtype)
        std = self.std.to(device=image.device, dtype=image.dtype)

        return (image - mean) / std

    def _score(self, pred):
        if self.task == "cls":
            probs = torch.softmax(pred, dim=-1)
            return probs.max(dim=-1).values.mean().item()
        
        elif self.task == "det":
            scores = pred[0]["scores"]
            return scores.max().item() if scores.numel() > 0 else 0.0
        
        elif self.task == "seg":
            probs = torch.softmax(pred["out"], dim=1)
            return probs.max(dim=1).values.mean().item()
        
        else:
            raise ValueError(f"{self.task} not found")

    def _relabel(self, pred):
        if self.task == "cls":
            return torch.argmax(pred, dim=-1).squeeze(0).item()
        
        elif self.task == "det":
            return {
                "boxes": pred[0]["boxes"].detach().cpu(),
                "labels": pred[0]["labels"].detach().cpu(),
                "scores": pred[0]["scores"].detach().cpu(),
                }
        
        elif self.task == "seg":
            return torch.argmax(pred["out"], dim=1).squeeze(0).detach().cpu()
        
        else:
            raise ValueError(f"{self.task} not found")  

# ------------------------------------------------------------
# TaskAlignedContextualPrompt
# ------------------------------------------------------------ 
class TaskAlignedContextualPrompt:
    """
    TaskAlignedContextualPrompt VLLM Class
    -> Generate Contextual Information
    """
    def __init__(self, task, config):
        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor

        self.task = task
        self.vllm = self._build_vllm(LLM, config.MODEL_NAME)
        self.processor = AutoProcessor.from_pretrained(config.MODEL_NAME, use_fast=True)
        self.sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=77)
        self.prompt = self._build_prompt(task)

    # ------------------------------------------------------------
    # Method
    # ------------------------------------------------------------
    def generate_context(self, dataloader):
        print("Generate Contextual Prompt")

        labels, contexts = [], []
        total_batches, next_progress = len(dataloader), 0.1

        for batch_idx, (images, targets) in enumerate(dataloader, start=1):
            requests = []

            for image in images:
                messages = [{
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": self.prompt}
                            ]
                        }]
                prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                requests.append({"prompt": prompt, "multi_modal_data": {"image": image}})

            outputs = self.vllm.generate(requests, self.sampling_params, use_tqdm=False)

            for target, output in zip(targets, outputs):
                labels.append(self._to_cpu(target))
                contexts.append(output.outputs[0].text.strip())

            progress = batch_idx / total_batches
            if progress >= next_progress:
                print(f"Progress: {next_progress:.0%}")
                next_progress += 0.1
        
        if not labels:
            raise RuntimeError("Labels is empty")

        if not contexts:
            raise RuntimeError("Contexts is empty")

        return labels, contexts

    # ------------------------------------------------------------
    # Helper Method
    # ------------------------------------------------------------
    def _build_vllm(self, LLM, model_name):
        return LLM(model=model_name, limit_mm_per_prompt={"image": 1}, tensor_parallel_size=1,
                   gpu_memory_utilization=0.85, dtype="float16", max_model_len=4096, enforce_eager=True)

    def _build_prompt(self, task):
        base = \
        """
        Your role is to assist a dataset-distillation pipeline that uses a diffusion model to synthesize training images. 
        Describe the given image in a single concise sentence that can be directly used as a text-to-image prompt.
        """

        if task == "cls":
            prompt = base + \
            " Focus on the main object, its class-defining visual characteristics, and relevant surrounding context." + \
            " The main object is one of [tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute]"

        elif task == "det":
            prompt = base + \
            " Focus on all distinct object classes, their locations and spatial relationships, and their interactions with one another and the background." + \
            " The main object is one of [aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor]"

        elif task == "seg":
            prompt = base + \
            " Focus on semantic regions, object boundaries, materials and textures, and the spatial relationships between neighboring regions and the background." + \
            " The main object is one of [aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor]"

        else:
            raise ValueError(f'{task} not found | task: ["cls", "det", "seg"]')

        return prompt
    
    def _to_cpu(self, target):
        if self.task == "det":
            return {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in target.items()}

        else:
            return target.detach().cpu()