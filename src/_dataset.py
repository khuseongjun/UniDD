import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import xml.etree.ElementTree as ET

# ------------------------------------------------------------
# DistilledDataset
# ------------------------------------------------------------
class DistilledDataset(Dataset):
    """
    Distilled Dataset Class
    -> Load s2_results: (paths, labels)
    """
    def __init__(self, task, root, relative_paths, labels, transforms):
        super().__init__()
        
        if len(relative_paths) != len(labels):
            raise ValueError(f"len(paths): {len(relative_paths)} != len(labels): {len(labels)}")
        
        self.task = task
        self.root = root
        self.relative_paths = relative_paths
        self.labels = labels
        self.transforms = transforms
    
    def __len__(self):
        return len(self.relative_paths)
    
    def __getitem__(self, index):
        image_path = self.root / self.relative_paths[index]
        label = self.labels[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
        tensor = self.transforms(image)

        return tensor, label
    
    # ------------------------------------------------------------
    # Method
    # ------------------------------------------------------------
    def build_dataloader(self, batch_size=32, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self._collate_fn)
    
    # ------------------------------------------------------------
    # Helper Method
    # ------------------------------------------------------------
    def _collate_fn(self, batch):
        datas, labels = zip(*batch)
        datas, labels = torch.stack(datas), torch.tensor(labels)

        return datas, labels
    
# ------------------------------------------------------------
# ImageNet
# ------------------------------------------------------------
class ImageNet(Dataset):
    """
    ImageNet Dataset Class
    -> Load ImageNet in: /org_data/ImageNet
    """
    DEFAULT = \
    {
        "IMAGE_SIZE": 256, "MEAN": [0.485, 0.456, 0.406], "STD": [0.229, 0.224, 0.225],
        "TRANSFORMS": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    def __init__(self, root, relative_path, split, 
                image_size=DEFAULT["IMAGE_SIZE"], transforms=DEFAULT["TRANSFORMS"], 
                indices=None, return_image=False):
        super().__init__()

        self.root = root
        self.relative_path = relative_path
        self.split = split
        self.image_size = image_size
        self.transforms = transforms if transforms is not None else self.DEFAULT["TRANSFORMS"]
        self.indices = indices
        self.return_image = return_image

        self.data_dir = self.root / self.relative_path / self.split
        if self.data_dir.exists():
            self.classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
            self.num_classes = len(self.classes)
        else:
            raise FileNotFoundError(f"Dir not found: {self.data_dir}")
        if self.classes:
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
            self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        else:
            raise FileNotFoundError(f"Class dir not found in: {self.data_dir}")
        
        self.dataset = self._build_dataset()

        if indices is not None:
            self.dataset = [self.dataset[i] for i in indices]
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image_path, label = self.dataset[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
        
        if self.return_image:
            return image, label
        
        else:
            tensor = self.transforms(image)
            return tensor, label

    # ------------------------------------------------------------
    # Method
    # ------------------------------------------------------------
    def build_dataloader(self, batch_size=32, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self._collate_fn)

    # ------------------------------------------------------------
    # Helper Method
    # ------------------------------------------------------------
    def _build_dataset(self):
        dataset = []

        for cls_name in self.classes:
            cls_dir = self.data_dir / cls_name
            label = self.class_to_idx[cls_name]

            for image_path in sorted(cls_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in [".jpeg", ".jpg"]:
                    dataset.append((image_path, label))
        
        if not dataset:
            raise FileNotFoundError(f"No images found in: {self.data_dir}")

        return dataset
    
    def _collate_fn(self, batch):
        datas, labels = zip(*batch)
        
        if self.return_image:
            datas = list(datas)
        else:
            datas = torch.stack(datas)
        labels = torch.tensor(labels)

        return datas, labels

# ------------------------------------------------------------
# PascalVOC
# ------------------------------------------------------------
class PascalVOC(Dataset):
    """
    PascalVOC Dataset Class
    -> Load PascalVOC in: /org_data/PascalVOC
    """
    DEFAULT = \
    {
        "IMAGE_SIZE": 512, "MEAN": [0.485, 0.456, 0.406], "STD": [0.229, 0.224, 0.225],
        "TRANSFORMS": transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "CLASSES": ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", 
                     "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    }
    def __init__(self, task, root, relative_path, split, 
                image_size=DEFAULT["IMAGE_SIZE"], transforms=DEFAULT["TRANSFORMS"], 
                indices=None, return_image=False):
        super().__init__()
        
        if task not in ["det", "seg", "inst"]:
            raise ValueError(f"{task} not found | task: ['det', 'seg', 'inst']")
        else:
            self.task = task
            self.root = root
            self.relative_path = relative_path
            self.split = split
            self.image_size = image_size
            self.transforms = transforms if transforms is not None else self.DEFAULT["TRANSFORMS"]
            self.indices = indices
            self.return_image = return_image
        
        self.data_dir = self.root / relative_path
        if self.data_dir.exists():
            self.annotation_dir = self.data_dir / "Annotations"
            self.imagesets_dir = self.data_dir / "ImageSets"
            self.image_dir = self.data_dir / "JPEGImages"
            self.seg_class_dir = self.data_dir / "SegmentationClass"
            self.seg_object_dir = self.data_dir / "SegmentationObject"
        else:
            raise FileNotFoundError(f"Dir not found: {self.data_dir}")
        if self.imagesets_dir.exists():
            self.classes = self.DEFAULT["CLASSES"]
            self.num_classes = len(self.classes)
            self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
            self.idx_to_class = {idx: class_name for class_name, idx in self.class_to_idx.items()}
        else:
            raise FileNotFoundError(f"ImageSets dir not found in: {self.data_dir}")

        self.dataset = self._build_dataset()

        if indices is not None:
            self.dataset = [self.dataset[i] for i in indices]
    

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]

        if self.task == "det":
            image_path, annotation_path = data

            with Image.open(image_path) as image:
                image = image.convert("RGB")
            target = self._parse_annotation(annotation_path)

            if self.return_image:
                return image, target
            
            else:
                tensor = self.transforms(image)
                return tensor, target
        
        elif self.task == "seg" or self.task == "inst":
            image_path, mask_path = data

            with Image.open(image_path) as image:
                image = image.convert("RGB")
            with Image.open(mask_path) as mask:
                mask = mask.copy()

            mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
            mask = torch.as_tensor(list(mask.getdata()), dtype=torch.long).reshape(self.image_size, self.image_size)
            
            if self.return_image:
                return image, mask
            
            else:
                tensor = self.transforms(image)
                return tensor, mask
        
        else:
            raise ValueError(f"{self.task} not found")

    # ------------------------------------------------------------
    # Method
    # ------------------------------------------------------------
    def build_dataloader(self, batch_size=32, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self._collate_fn)
    
    # ------------------------------------------------------------
    # Helper Method
    # ------------------------------------------------------------
    def _build_dataset(self):
        if self.task == "det":
            split_file = self.imagesets_dir / "Main" / f"{self.split}.txt"
            target_dir = self.annotation_dir
        elif self.task == "seg" or self.task == "inst":
            split_file = self.imagesets_dir / "Segmentation" / f"{self.split}.txt"
            target_dir = self.seg_class_dir if self.task == "seg" else self.seg_object_dir
        else:
            raise ValueError(f"{self.task} not found")
        
        dataset = []

        if split_file.exists():
            with open(split_file,"r") as f:
                image_ids = [line.strip() for line in f if line.strip()]

            for image_id in image_ids:
                image_path = self.image_dir / f"{image_id}.jpg"
                target_path = target_dir / (f"{image_id}.xml" if self.task == "det" else f"{image_id}.png")

                if image_path.exists() and target_path.exists():
                    dataset.append((image_path, target_path))
        else:
            raise FileNotFoundError(f"Split file not found: {split_file}")

        if not dataset:
            raise FileNotFoundError(f"No images found under: {self.data_dir}")

        return dataset

    def _parse_annotation(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall("object"):
            class_name = obj.find("name").text
            label = self.class_to_idx[class_name]

            bndbox = obj.find("bndbox")
            xmin, ymin = float(bndbox.find("xmin").text), float(bndbox.find("ymin").text)
            xmax, ymax = float(bndbox.find("xmax").text), float(bndbox.find("ymax").text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long)
            }

        return target
    
    def _collate_fn(self, batch):
        if self.task == "det":
            datas, targets = zip(*batch)

            if self.return_image:
                datas = list(datas)
            else:
                datas = torch.stack(datas)
            targets = list(targets)

            return datas, targets

        elif self.task == "seg" or self.task == "inst":
            datas, masks = zip(*batch)

            if self.return_image:
                datas, masks = list(datas), list(masks)
            else:
                datas, masks = torch.stack(datas), torch.stack(masks)

            return datas, masks

        else:
            raise ValueError(f"{self.task} not found")