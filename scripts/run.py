import cv2
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))  # add root folder to sys.path

from datasets import set_caching_enabled    
from dataclasses import dataclass, field
from torch.utils.data import Subset
from transformers import (
    HfArgumentParser, 
    SamProcessor,
    Trainer,
    TrainingArguments,
)
from functools import partial
from transformers.utils import logging
import os
import torch

from src.corpora import get_dataset_dict
from src.metrics import compute_metrics
from src.modeling import SamBaseline, SamAR
from src.utils import set_seed


set_seed()
logger = logging.get_logger()
logging.set_verbosity_info()
set_caching_enabled(False)
os.environ["WANDB_DISABLED"] = "true"

MEDSAM = "wanglab/medsam-vit-base"
SAM = "facebook/sam-vit-base"

SLIP_PATH = "data/lidc_slip"
MCL_PATH = "data/lidc_mcl"
AR_PATH = "data/lidc_ar_unet"
AR_LONG_PATH = "data/lidc_ar_long"


@dataclass
class ModelArguments:
    model_load_path: str = field(
        default=SAM,
        metadata={"help": "Path to the pretrained model or model identifier from huggingface.co/models"}
    )

    results_write_path: str = field(
        default=None,
        metadata={"help": "Path to save results"}
    )

    processor_load_path: str = field(
        default=SAM,
        metadata={"help": "Path to the pretrained model or model identifier from huggingface.co/models"}
    )

    teacher_load_path: str = field(
        default="facebook/sam-vit-base",
        metadata={"help": "Path to the pretrained model or model identifier from huggingface.co/models"}
    )

    model_save_path: str = field(
        default="data/lidc_unet",
        metadata={"help": "Path to the pretrained model or model identifier from huggingface.co/models"}
    )

    dataset: str = field(
        default="lidc",
        metadata={"help": "Path to the dataset or dataset identifier from huggingface.co/datasets",
                    "choices": ["lidc", "qubiq"]}
    )

    model_type: str = field(
        default="ar",
        metadata={"help": "Model type", "choices": ["mcl", "det", "ar"]}
    )

    ablation: str = field(
        default="none",
        metadata={"help": "Ablation study", "choices": ["none", "random", "sequential", "no_ha", "sg", "one"]}
    )

    learning_rate: float = field(
        default=1e-4,
        metadata={"help": "Learning rate"}
    )

    num_train_epochs: int = field(
        default=1,
        metadata={"help": "Number of training epochs"}
    )

    use_bounding_box: bool = field(
        default=True,
        metadata={"help": "Whether to use bounding boxes"}
    )

    use_input_masks: bool = field(
        default=False,
        metadata={"help": "Whether to use bounding boxes"}
    )

    mode: str = field(
        default="eval",
        metadata={"help": "Mode", "choices": ["train", "eval", "vis"]}
    )

from PIL import Image
import numpy as np


@torch.no_grad()
def _visualise(model, dataset, model_type: str):
    
    if not os.path.isdir("data/images"):
        os.makedirs("data/images")

    #indices = range(10)
    #indices = range(100)
    indices = [69]
    
    for i in indices:
        batch = dataset[i:i+1]
        img = Image.fromarray(batch["image"].squeeze()[:, :, 0] * 255).convert("L")
        
        #img = Image.fromarray(dataset.dataset.images[i] * 255).convert("L")

        #x1, y1, x2, y2 = (dataset.dataset[[i]]["input_boxes"] / 8).numpy().tolist()[0][0]
        #img = np.array(img)
        #cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
        #img = Image.fromarray(img)
        img.save(f"data/images/img_{i}.png")

        #pred = Image.fromarray((predictions[i, 0] > 0).astype(np.uint8))
        #pred.save(f"data/images/pred_{i}.png")
        for j in range(4):
            #label = Image.fromarray(dataset.dataset.labels[i][j] * 255).convert("L")
            label = Image.fromarray((batch["labels"].squeeze()[j].cpu().numpy() * 255).astype(np.uint8))
            label.save(f"data/images/label_{i}_{j}.png")

        preds = model(
            pixel_values=batch["pixel_values"].cuda(),
            input_boxes=batch["input_boxes"].cuda(),
            labels=batch["labels"].cuda(),
            label_mask=batch["label_mask"].cuda(),
        ).pred_masks.squeeze()

        for j, pred in enumerate(preds):
            pred = Image.fromarray(((pred > 0.0).cpu().numpy() * 255).astype(np.uint8))
            pred.save(f"data/images/pred_{i}_{model_type}_{j}.png")

        print(preds[0].shape)


def _main(args):
    # Load dataset
    processor = SamProcessor.from_pretrained(args.processor_load_path) if args.model_type != "unet" else None

    # Load model
    if args.model_type == "det":
        model = SamBaseline.from_pretrained(
            args.model_load_path,
            processor=processor,
            multimask_output=False
        )

    elif args.model_type == "mcl":
        model = SamBaseline.from_pretrained(
            args.model_load_path,
            processor=processor,
            multimask_output=True
        )

    elif args.model_type == "ar":
        model = SamAR.from_pretrained(
            args.model_load_path,
            processor=processor,
            num_samples=args.num_samples,
            ablation=args.ablation
        )

    else:
        raise ValueError(f"Model type {args.model_type} not supported")

    
    dataset = get_dataset_dict(args.dataset, processor, args)

    # Print number of parameters
    print(f"Number of parameters: {model.num_parameters()}")
 
    # Make sure we only compute gradients for mask decoder
    for name, param in model.named_parameters():
        if name.startswith("sam.vision_encoder"):
            param.requires_grad_(False)
    
    # Set up trainer
    training_args = TrainingArguments(
        output_dir=args.model_save_path,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        dataloader_drop_last=False,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        #evaluation_strategy="epoch",
        evaluation_strategy="steps",
        eval_steps=200,
        save_strategy="epoch",
        #fp16=True,
        #save_total_limit=1,
        learning_rate=args.learning_rate,
    )

    _compute_metrics = partial(compute_metrics, write_path=args.results_write_path, dataset=dataset["test"])

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"] if args.mode == "train" else dataset["test"],
        compute_metrics=_compute_metrics,
    )

    if args.mode == "eval":
        results = trainer.evaluate()
        print(results)
        exit()

    elif args.mode == "vis":
        _visualise(model, dataset["test"], args.model_type)
        exit()

    trainer.evaluate()
    trainer.train()

    if args.model_type == "theta":
        model.env = None

    if args.model_type == "unet":
        torch.save(model.state_dict(), args.model_save_path + ".pth")
    else:
        model.save_pretrained(args.model_save_path)


def main():
    parser = HfArgumentParser((ModelArguments,))
    args, = parser.parse_args_into_dataclasses()
    _main(args)


if __name__ == "__main__":
    main()
