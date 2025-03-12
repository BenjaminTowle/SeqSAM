import gdown
import nibabel as nib
import numpy as np
import os
import pickle
import random
import torch
import torch.nn.functional as F

from abc import ABC
from typing import Union, List
from datasets import Dataset
from os.path import join
from PIL import Image
from torch.utils.data import Subset


def get_bounding_box(ground_truth_map, add_perturbation=False):
    # get bounding box from mask
    y_indices, x_indices = np.where(ground_truth_map > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        # Return default value
        return [0, 0, ground_truth_map.shape[1], ground_truth_map.shape[0]]
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    if add_perturbation:
        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
    bbox = [x_min, y_min, x_max, y_max]

    return bbox


class MultiLabelDataset(Dataset, ABC):

    images = []
    labels = []

    def __init__(self, processor, data_dir: str) -> None:
        
        s_path = os.path.join(data_dir, self.file_name)
        if not os.path.exists(s_path):
            gdown.download(f"https://drive.google.com/uc?id={self.file_id}", s_path, quiet=False)

        self.processor = processor
        self.data_dir = data_dir

    def __getitem__(self, indices: Union[int, List[int]]):
        # Supports both regular and fancy indexing
        if type(indices) == int:
            index = indices
            image = np.expand_dims(self.images[index], axis=0)
            label = self.labels[index][random.randint(0, self.num_labels-1)].astype(float)

            image = np.repeat(image.transpose(1, 2, 0), 3, axis=2)
            inputs = self.processor(image, do_rescale=False, return_tensors="pt")
            # remove batch dimension which the processor adds by default
            inputs = {k:v.squeeze(0) for k,v in inputs.items()}
            inputs["original_sizes"] = torch.tensor([256, 256]).to(inputs["pixel_values"].device)

            inputs["labels"] = F.interpolate(
                torch.tensor(label).unsqueeze(0).unsqueeze(0),  
                size=(256, 256), 
                mode="nearest"
            ).bool().squeeze()
        else:
            bsz = len(indices)
            image = np.stack([self.images[index] for index in indices], axis=0)
            label = np.stack([
                self.labels[index] for index in indices]).astype(float)
            
            # Prepare image and prompt for the model
            image = np.repeat(np.expand_dims(image, axis=-1), 3, axis=-1)  # 3 channels
            input_boxes = []
            for l in label:
                while True:
                    idx = random.randint(0, self.num_labels - 1)
                    if l[idx].sum() > 0:
                        break
                input_boxes.append([get_bounding_box(l[idx], add_perturbation=False)])  
            
            inputs = self.processor(image, input_boxes=input_boxes, do_rescale=False, return_tensors="pt")

            inputs["original_sizes"] = torch.tensor([256, 256]).to(inputs["pixel_values"].device).unsqueeze(0).expand(bsz, -1)
            
            inputs["labels"] = F.interpolate(
                torch.tensor(label), 
                size=(256, 256), 
                mode="nearest"
            ).bool().squeeze(1)

            # Create a mask for labels that are blank
            inputs["label_mask"] = torch.sum(inputs["labels"], dim=(-1, -2)) > 0

        return inputs

    def __len__(self):
        return len(self.images)

    @property
    def num_labels(self):
        return len(self.labels[0])
        

class QUBIQ(MultiLabelDataset):

    file_reader = "nii.gz"
    file_id = "1XQVKBeNhOqm62nwnA2AlqZplyaMaXbrs"
    file_name = "qubiq.zip"

    def __init__(self, processor, data_dir: str, split="train") -> None:
        super().__init__(processor, data_dir)

        z_path = os.path.join(self.data_dir, self.file_name)
        f_path = os.path.splitext(z_path)[0]
        if not os.path.exists(f_path): 
            from zipfile import ZipFile 
        
            with ZipFile(z_path) as zObject:  
                zObject.extractall(path=self.data_dir) 

        self.i = 0
        self.images, self.labels = self.preprocess(split)

    def read_file(self, path: str):
        image = nib.load(path).get_fdata()

        image += np.abs(image.min())
        image = (256 * (image / image.max())).astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        
        return image
    
    def read_label(self, path: str):
        label = nib.load(path).get_fdata()
        label = np.array(Image.fromarray(label).resize((256, 256)))
        label = label > 0
        
        return label

    def dfs(self, paths: list, path: str):
        for item in os.listdir(path):
            new_path = join(path, item)
            if "kidney" not in new_path:
                continue

            if item.startswith("case"):
                paths.append(new_path)
            elif os.path.isdir(new_path):
                self.dfs(paths, new_path)
            else:
                continue
        return paths

    def get_images_labels(self, path: str, single_label: bool = False) -> dict:
        image = []
        label = []
        cases = self.dfs([], path)

        for case in cases:
            label_set = []
            for file in os.listdir(case):
                if not file.endswith(self.file_reader):
                    continue
                if file.startswith("image"):
                    image.append(join(case, file))
                else:
                    label_set.append(join(case, file))

            if single_label:
                label.append(label_set[self.i % len(label_set)])
                self.i += 1
            else:
                label.append(label_set)
        
        return {
            "image": image,
            "label": label
        }

    def preprocess(self, split="train"):
        dataset_path = os.path.join(self.data_dir, "qubiq")

        if split == "train":
            path = join(dataset_path, "training_data_v2")

        else:
            path = join(dataset_path, "validation_data_v2")
            
        dict = self.get_images_labels(path)

        images = [self.read_file(img) for img in dict["image"]]
        labels = [[self.read_label(l) for l in lbl] for lbl in dict["label"]]
            
        return images, labels
        

class LIDC_IDRI(MultiLabelDataset):

    file_id = "1QAtsh6qUgopFx1LJs20gOO9v5NP6eBgI"
    file_name = "data_lidc.pickle"

    series_uid = []

    def __init__(self, data_dir, processor=None):
        
        super().__init__(processor, data_dir)
        self.processor = processor
        
        max_bytes = 2**31 - 1
        data = {}
        for file in os.listdir(data_dir):
            filename = os.fsdecode(file)
            if '.pickle' in filename:
                print("Loading file", filename)
                file_path = os.path.join(data_dir, filename)
                bytes_in = bytearray(0)
                input_size = os.path.getsize(file_path)
                with open(file_path, 'rb') as f_in:
                    for _ in range(0, input_size, max_bytes):
                        bytes_in += f_in.read(max_bytes)
                new_data = pickle.loads(bytes_in)
                data.update(new_data)
        
        for i, (key, value) in enumerate(data.items()):
            self.images.append(value['image'].astype(float))
            self.labels.append(value['masks'])
            self.series_uid.append(value['series_uid'])


        assert (len(self.images) == len(self.labels) == len(self.series_uid))

        for img in self.images:
            assert np.max(img) <= 1 and np.min(img) >= 0
        for label in self.labels:
            assert np.max(label) <= 1 and np.min(label) >= 0

        del new_data
        del data

def load_lidc(processor, cfg):
    dataset = LIDC_IDRI(processor=processor, data_dir=cfg.data.path)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    TRAIN_SIZE = 0.9
    split = int(np.floor((1.0 - TRAIN_SIZE) * dataset_size))
    train_indices, valid_indices, test_indices = indices[2*split:], indices[:split], indices[split:split*2]

    eval_indices = valid_indices if cfg.mode == "train" else test_indices

    dataset = {"train": Subset(dataset, train_indices), "eval": Subset(dataset, eval_indices)}
    
    return dataset

def load_qubic(processor, cfg):

    def map_fn(split: str):
        dataset = QUBIQ(processor=processor, data_dir=cfg.data.path, split=split)
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        dataset = Subset(dataset, indices)

        return dataset

    splits = ["train", "eval"]

    return {split: map_fn(split) for split in splits}

    
def get_dataset_dict(processor, cfg):
    if not os.path.exists(cfg.data.path):
        os.makedirs(cfg.data.path)
    
    dataset2loader = {
        "lidc": load_lidc,
        "qubic": load_qubic
    }
    
    return dataset2loader[cfg.data.dataset](processor, cfg)
