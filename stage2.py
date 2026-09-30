import torch
import gc
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parent

from src._stage1 import TaskSpecificProxy as TSP
from src._stage2 import DiffusionModel as DM
from _utils import seed_everything, get_random_sample
from config import stage2_config as s2_cfg
seed_everything(seed=42)

def stage2(task, generate_sample=True, random_sample=False):
    if generate_sample:
        print("\n===== Load S1 Result =====")
        print("Load Labels and Contexts")
        s1_results = torch.load(s2_cfg.S1_RESULT_PATH, map_location="cpu", weights_only=False)
        num_classes, data_default, cfg = s1_results["num_classes"], s1_results["data_default"], s1_results["config"]
        labels, contexts = s1_results["labels"], s1_results["contexts"]

        print("Load TSP Model")    
        tsp = TSP(task=task, num_classes=num_classes, data_default=data_default, config=cfg)
        tsp.load_model(load_checkpoint=s2_cfg.TSP_CHECKPOINT)

        if random_sample:
            selected_indices = get_random_sample(labels, s2_cfg.SAMPLE_PER_CLASS[task])
        else:
            selected_indices = [i for i in range(len(labels))]
        
        del s1_results, num_classes, data_default, cfg

        print("\n===== Build Diffusion Model =====")
        dm = DM(task=task, tsp_model=tsp, config=s2_cfg)

        print("\n===== Generating Distilled Images =====")
        total_samples, start_time, next_progress = len(selected_indices), time.perf_counter(), 0.1
        images = []

        for batch_idx, i in enumerate(selected_indices, start=1):
            label = labels[i]
            context = contexts[i]

            if task == "det":
                label = [label]

            image = dm.generate_sample(label, context)
            images.append(image)

            if batch_idx % 20 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
            progress = batch_idx / total_samples
            if progress >= next_progress:
                elapsed = time.perf_counter() - start_time
                print(f"Progress: {next_progress:.0%} | time: {elapsed:.2f}s")
                start_time, next_progress = time.perf_counter(), next_progress + 0.1
        
        del dm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
        print("\n===== Refine Distilled Images =====")
        s2_results = tsp.filter_and_relabel(images)

        print("\n===== Save S2 Result =====")
        image_dir = ROOT / "src/distill_data" / f"{s2_cfg.NAME}"
        image_dir.mkdir(parents=True, exist_ok=True)
        image_paths = []

        if task == "cls":
            for idx, (img, label) in enumerate(zip(s2_results["images"], s2_results["labels"])):
                if torch.is_tensor(label):
                    label = label.item()

                label_dir = image_dir / str(label)
                label_dir.mkdir(parents=True, exist_ok=True)

                image_path = label_dir / f"{idx:05d}.jpg"
                img.save(image_path)
                image_paths.append(str(image_path.relative_to(ROOT)))
        
        else:
            for idx, img in enumerate(s2_results["images"]):
                image_path = image_dir / f"{idx:05d}.jpg"
                img.save(image_path)
                image_paths.append(str(image_path.relative_to(ROOT)))            

        s2_results["paths"] = image_paths
        del s2_results["images"]
        
        torch.save(s2_results, s2_cfg.S2_RESULT_PATH)

        print(f"Saved {len(s2_results['paths'])} distilled images to {image_dir}")
    else:
        if s2_cfg.S2_RESULT_PATH.exists():
            s2_results = torch.load(s2_cfg.S2_RESULT_PATH, map_location="cpu", weights_only=False)
        else:
            s2_results = {}

    print("\n===== Stage2 Finish! =====\n")
    
    return s2_results