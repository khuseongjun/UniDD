import torch
from pathlib import Path
ROOT = Path(__file__).resolve().parent

class config:
    # --- Load & Save
    NAME = "det"
    TASK = "det"
    # --- Load
    DATASET_PATH = {
        "cls": "src/org_data/ImageNet/imagenette2-160",
        "det": "src/org_data/PascalVOC/VOC2012",
        "seg": "src/org_data/PascalVOC/VOC2012"
    }
    S1_RESULT_PATH = ROOT / f"src/stage1_result/{NAME}.pt"
    S2_RESULT_PATH = ROOT / f"src/stage2_result/{NAME}.pt"
    TSP_CHECKPOINT = ROOT / f"src/tsp_model/tsp_{NAME}.pt"
    # --- Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
    # --- Hyperparameters
    IMAGE_SIZE = {"cls": 256, "det": 512, "seg": 512}
    TASK_THRESHOLD = {"cls": 0.9, "det": 0.85, "seg": 0.75}

class stage1_config(config):
    # --- Load Dataset
    TRAIN_VAL_RATIO = 0.8
    TRAIN_TRANSFORM = None
    TRAIN_BATCH_SIZE = 32
    VAL_BATCH_SIZE = 32
    PIL_BATCH_SIZE = 8
    # --- TaskSpecificProxy Model: Hyperparameters
    LR = 1e-4
    EPOCHS = 50
    # --- TaskAlignedContextualPrompting
    MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

class stage2_config(config):
    # --- DiffusionModel
    MODEL_NAME = "runwayml/stable-diffusion-v1-5"
    # --- DiffusionModel: Hyperparameters
    SAMPLE_PER_CLASS = {"cls": 100, "det": 50, "seg": 5}
    SAMPLING_STEPS = 250
    GUIDANCE_SCALE = 7.5
    LAMBDA_S = 100

class test_config(config):
    # --- TestModel: Hyperparameters
    LR = 1e-4
    EPOCHS = 100