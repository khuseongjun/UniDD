import torch
from pathlib import Path
ROOT = Path(__file__).resolve().parent

from src._dataset import ImageNet, PascalVOC
from src._stage1 import TaskSpecificProxy as TSP, TaskAlignedContextualPrompt as VLLM
from _utils import seed_everything, get_indices
from config import stage1_config as s1_cfg
seed_everything(seed=42)

def stage1(task, train_tsp=True, prompt_context=True):
    if train_tsp:
        print("\n===== Load Dataset =====")
        if task == "cls":
            dataset = ImageNet(root=ROOT, relative_path=s1_cfg.DATASET_PATH[task], split="train")
            train_indices, val_indices = get_indices(dataset, ratio=s1_cfg.TRAIN_VAL_RATIO)
            del dataset

            train = ImageNet(root=ROOT, relative_path=s1_cfg.DATASET_PATH[task], split="train", indices=train_indices, transforms=s1_cfg.TRAIN_TRANSFORM)
            train_loader = train.build_dataloader(batch_size=s1_cfg.TRAIN_BATCH_SIZE, shuffle=True)
            val = ImageNet(root=ROOT, relative_path=s1_cfg.DATASET_PATH[task], split="train", indices=val_indices)
            val_loader = val.build_dataloader(batch_size=s1_cfg.VAL_BATCH_SIZE, shuffle=False)

        elif task == "det":
            dataset = PascalVOC(task="det", root=ROOT, relative_path=s1_cfg.DATASET_PATH[task], split="train")
            train_indices, val_indices = get_indices(dataset, ratio=s1_cfg.TRAIN_VAL_RATIO)
            del dataset

            train = PascalVOC(task="det", root=ROOT, relative_path=s1_cfg.DATASET_PATH[task], split="train", indices=train_indices, transforms=s1_cfg.TRAIN_TRANSFORM)
            train_loader = train.build_dataloader(batch_size=s1_cfg.TRAIN_BATCH_SIZE, shuffle=True)
            val = PascalVOC(task="det", root=ROOT, relative_path=s1_cfg.DATASET_PATH[task], split="train", indices=val_indices)
            val_loader = val.build_dataloader(batch_size=s1_cfg.VAL_BATCH_SIZE, shuffle=False)

        elif task == "seg":
            dataset = PascalVOC(task="seg", root=ROOT, relative_path=s1_cfg.DATASET_PATH[task], split="train")
            train_indices, val_indices = get_indices(dataset, ratio=s1_cfg.TRAIN_VAL_RATIO)
            del dataset

            train = PascalVOC(task="seg", root=ROOT, relative_path=s1_cfg.DATASET_PATH[task], split="train", indices=train_indices, transforms=s1_cfg.TRAIN_TRANSFORM)
            train_loader = train.build_dataloader(batch_size=s1_cfg.TRAIN_BATCH_SIZE, shuffle=True)
            val = PascalVOC(task="seg", root=ROOT, relative_path=s1_cfg.DATASET_PATH[task], split="train", indices=val_indices, transforms=s1_cfg.TRAIN_TRANSFORM)
            val_loader = val.build_dataloader(batch_size=s1_cfg.VAL_BATCH_SIZE, shuffle=False)

        else: 
            raise ValueError(f"{task} not found")
        
        print("\n===== Build TSP Model =====")
        tsp = TSP(task=task, num_classes=train.num_classes, data_default=train.DEFAULT, config=s1_cfg)
        tsp.train_model(train_loader, val_loader)

        del train, train_loader, val, val_loader, tsp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if prompt_context:
        print("\n===== Load Dataset =====")
        if task == "cls":
            train = ImageNet(root=ROOT, relative_path=s1_cfg.DATASET_PATH[task], split="train")
            pil = ImageNet(root=ROOT, relative_path=s1_cfg.DATASET_PATH[task], split="train", return_image=True)
            pil_loader = pil.build_dataloader(batch_size=s1_cfg.PIL_BATCH_SIZE, shuffle=False)

        elif task == "det":
            train = PascalVOC(task="det", root=ROOT, relative_path=s1_cfg.DATASET_PATH[task], split="train")
            pil = PascalVOC(task="det", root=ROOT, relative_path=s1_cfg.DATASET_PATH[task], split="train", return_image=True)
            pil_loader = pil.build_dataloader(batch_size=s1_cfg.PIL_BATCH_SIZE, shuffle=False)

        elif task == "seg":
            train = PascalVOC(task="seg", root=ROOT, relative_path=s1_cfg.DATASET_PATH[task], split="train")
            pil = PascalVOC(task="seg", root=ROOT, relative_path=s1_cfg.DATASET_PATH[task], split="train", return_image=True)
            pil_loader = pil.build_dataloader(batch_size=s1_cfg.PIL_BATCH_SIZE, shuffle=False)

        else:
            raise ValueError(f"{task} not found")
        
        print("\n===== Build VLLM =====")
        vllm = VLLM(task=task, config=s1_cfg)
        labels, contexts = vllm.generate_context(pil_loader)

        s1_results = {
            "num_classes": train.num_classes,
            "data_default": train.DEFAULT,
            "config": s1_cfg,
            "labels": labels,
            "contexts": contexts
        }

        del train, pil, pil_loader, vllm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("\n===== Save S1 Result =====")
        torch.save(s1_results, s1_cfg.S1_RESULT_PATH)
    else:
        if s1_cfg.S1_RESULT_PATH.exists():
            s1_results = torch.load(s1_cfg.S1_RESULT_PATH, map_location="cpu", weights_only=False)
        else:
            s1_results = {}

    print("\n===== Stage1 Finish! =====\n")

    return s1_results