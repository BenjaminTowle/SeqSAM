import monai
import torch
import random


from abc import ABC

from dataclasses import dataclass
from enum import Enum, auto
from torch import nn
from transformers.models.sam.modeling_sam import (
    SamModel, 
    SamPreTrainedModel,
    SamImageSegmentationOutput
)
from typing import Optional, Callable
from typing import Tuple
from scipy.optimize import linear_sum_assignment
from functools import partial

from src.metrics import iou
from scipy.optimize import linear_sum_assignment
from functools import partial

class Ablation(Enum):
    NONE = auto()
    RANDOM = auto()
    SEQUENTIAL = auto()
    STOP_GRADIENTS = auto()
    NO_HUNGARIAN_ALGORITHM = auto()

    @classmethod
    def from_str(cls, label):
        if label == "no_ha":
            return cls.NO_HUNGARIAN_ALGORITHM
        elif label == "sg":
            return cls.STOP_GRADIENTS
        else:
            label = label.upper()
            return cls[label]


@dataclass
class SamMultimaskOutput(SamImageSegmentationOutput):
    loss: torch.FloatTensor = None
    input_points: torch.Tensor = None
    iou_targets: torch.Tensor = None
    iou_pred: torch.Tensor = None
    input_labels: torch.Tensor = None
    initial_pred_masks: torch.Tensor = None
    union: torch.Tensor = None
    intersection: torch.Tensor = None



class Model(ABC, SamPreTrainedModel):

    def __init__(self, config) -> None:
def compute_loss(
    pred_masks: torch.Tensor, 
    labels: torch.Tensor, 
    loss_fn: Callable,
    return_dict: bool = False
):
    """
    pred_masks: (bsz, 1, num_multimask_outputs, H, W)
    labels: (bsz, H, W)
    """
    bsz, _, num_preds, H, W = pred_masks.size()

    loss = loss_fn(
        pred_masks.reshape(-1, H, W), 
        labels.repeat_interleave(num_preds, dim=0)
    ).reshape(bsz, num_preds, -1)

    loss = loss.mean(dim=2)

    if not return_dict:
        return loss.min(dim=1)[0].mean()

    _min = loss.min(dim=1)

    return {
        "loss": _min[0].mean(),
        "indices": _min[1],
    }


class SamBaseline(SamPreTrainedModel):

    def __init__(self, config, multimask_output: bool = True):
        super().__init__(config)
        self.sam = SamModel(config)
        self.seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="none")
        self.multimask_output = multimask_output

    @property
    def mask_decoder(self):
        return self.sam.mask_decoder

    def forward(
        self, 
        pixel_values: Optional[torch.Tensor] = None,
        image_embeddings: Optional[torch.Tensor] = None,
        input_boxes: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        original_sizes: Optional[torch.Tensor] = None,
        reshaped_input_sizes: Optional[torch.Tensor] = None,
        label_mask: Optional[torch.Tensor] = None,
        **kwargs,

        label_mask: Optional[torch.Tensor] = None

    ):

        new_labels = []
        for i in range(len(labels)):
            while True:
                rand_idx = random.randint(0, labels.shape[1] - 1) 
                if label_mask[i, rand_idx] > 0:
                    break
            new_labels.append(labels[i, rand_idx])
        labels = torch.stack(new_labels, dim=0).to(self.device)
        
        if image_embeddings is None:
            image_embeddings = self.sam.get_image_embeddings(pixel_values)

        outputs = self.sam(
            image_embeddings=image_embeddings, 
            input_boxes=input_boxes,
            multimask_output=self.multimask_output,
        )

        loss = compute_loss(
            pred_masks=outputs.pred_masks, 
            labels=labels, 
            loss_fn=self.seg_loss, 
            return_dict=True
        )        

        loss = loss["loss"]

        return SamMultimaskOutput(
            loss=loss,
            iou_scores=outputs.iou_scores,
            pred_masks=outputs.pred_masks,
        )


class SamAR(Model):

    def __init__(self, config, processor, num_samples: int = 4, ablation="none"):
        super().__init__(config)
        self.sam = SamModel(config)
        self.processor = processor
        self.seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="none")
        self.multimask_output = True
        self.num_samples = num_samples
        self.ablation = ablation
        self.cell = CRnnCell()

    @property
    def mask_decoder(self):
        return self.sam.mask_decoder

    def forward(
        self, 
        pixel_values: Optional[torch.Tensor] = None,
        image_embeddings: Optional[torch.Tensor] = None,
        input_boxes: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        original_sizes: Optional[torch.Tensor] = None,
        reshaped_input_sizes: Optional[torch.Tensor] = None,
        label_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        
        self.cell.reset()
        
        if image_embeddings is None:
            image_embeddings = self.sam.get_image_embeddings(pixel_values)

        input_masks = None
        total_loss = 0.0
        pred_masks = []

        if self.ablation == "one":
            labels = labels[:, :1]
            label_mask = label_mask[:, :1]

        # Ensures we have enough samples for the labels
        num_samples = max([labels.shape[1], self.num_samples])
        
        for i in range(num_samples):
            self.sam.mask_decoder.forward = partial(forward, self=self.sam.mask_decoder, cell=self.cell)
            
            outputs = self.sam(
                image_embeddings=image_embeddings, 
                input_boxes=input_boxes,
                input_masks=input_masks,
                multimask_output=False,
            )
            input_masks = outputs.pred_masks.squeeze(2)

            if self.ablation == "sg":
                input_masks = input_masks.detach()
                        
            pred_masks.append(outputs.pred_masks)

        if self.training:
            if self.ablation == "random":
                rand_idxs = random.sample(range(num_samples), k=labels.shape[1])
                pred_masks = [pred_masks[i] for i in rand_idxs]

            elif self.ablation == "sequential":
                pred_masks = pred_masks[:labels.shape[1]]

            else:
                num_labels = labels.shape[1]

                def ceildiv(a, b):
                    return -(a // -b)
                
                chunk_size = ceildiv(num_samples, num_labels)
                new_pred_masks = []
                for i in range(0, num_samples, chunk_size):
                    masks = pred_masks[i:i+chunk_size]
                    new_pred_masks.append(random.choice(masks))
                pred_masks = new_pred_masks
            
        total_loss = 0.0
        pred_masks = torch.cat(pred_masks, dim=2)
        for i in range(labels.shape[0]):
            losses = torch.zeros(pred_masks.shape[2], labels.shape[1]).to(labels.device)
            for j in range(pred_masks.shape[2]):
                for k in range(labels.shape[1]):
                    loss = self.seg_loss(pred_masks[i, 0, j], labels[i, k]).mean()
                    losses[j, k] = loss
            
            
            if self.ablation != "no_ha":
                row_ind, col_ind = linear_sum_assignment(losses.detach().cpu().numpy())
            else:
                row_ind = torch.arange(losses.shape[0])
                col_ind = torch.arange(losses.shape[1])
            loss = losses[row_ind, col_ind]
            if self.training:
                loss *= label_mask[i].float()
            total_loss += loss.sum()
        
        total_loss /= labels.shape[1]
                
        return SamMultimaskOutput(
            loss=total_loss,
            iou_scores=outputs.iou_scores,
            pred_masks=pred_masks[:, :, :self.num_samples],
        )


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        truncated_normal_(m.bias, mean=0, std=0.001)


class CRnnCell(nn.Module):
    def __init__(self):
        # A cnn which goes from 512 x 64 x 64 -> 256 x 64 x 64
        super(CRnnCell, self).__init__()
        self.hidden_state = None
        self.conv_o = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv_h = nn.Conv2d(512, 256, kernel_size=3, padding=1)

    def forward(self, x):
        if self.hidden_state is None:
            self.hidden_state = torch.zeros(x.size(0), 256, x.size(2), x.size(3), device=x.device)
        combined = torch.cat([x, self.hidden_state], dim=1)
        o = self.conv_o(combined)
        h = self.conv_h(combined)
        self.hidden_state = h
        return o
    
    def reset(self):
        self.hidden_state = None


def forward(
        self,
        image_embeddings: torch.Tensor,
        image_positional_embeddings: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        output_attentions: Optional[bool] = None,
        attention_similarity: torch.Tensor = None,
        target_embedding: torch.Tensor = None,
        cell=None

    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Args:
            image_embeddings (`torch.Tensor`):
                the embeddings from the image encoder
            image_positional_embedding (`torch.Tensor`):
                positional encoding with the shape of image_embeddings
            sparse_prompt_embeddings (`torch.Tensor`):
                The embeddings of the points and boxes
            dense_prompt_embeddings (`torch.Tensor`):
                the embeddings of the mask inputs
            multimask_output (bool):
                Whether to return multiple masks or a single mask.
            output_attentions (bool, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
        """
        batch_size, num_channels, height, width = image_embeddings.shape
        point_batch_size = sparse_prompt_embeddings.shape[1]
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        
        output_tokens = output_tokens.repeat(batch_size, point_batch_size, 1, 1)

        if sparse_prompt_embeddings.sum().item() != 0:
            tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=2)
        else:
            tokens = output_tokens
        point_embeddings = tokens.to(self.iou_token.weight.dtype)

        # Expand per-image data in batch direction to be per-point
        dense_prompt_embeddings = cell(dense_prompt_embeddings)
        
        image_embeddings = image_embeddings + dense_prompt_embeddings
        image_embeddings = image_embeddings.repeat_interleave(point_batch_size, 0)
        image_positional_embeddings = image_positional_embeddings.repeat_interleave(point_batch_size, 0)

        # Run the transformer, image_positional_embedding are consumed
        point_embedding, image_embeddings, attentions = self.transformer(
            point_embeddings=point_embeddings,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            attention_similarity=attention_similarity,
            target_embedding=target_embedding,
            output_attentions=output_attentions,
        )
        iou_token_out = point_embedding[:, :, 0, :]
        mask_tokens_out = point_embedding[:, :, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        image_embeddings = image_embeddings.transpose(2, 3).reshape(
            batch_size * point_batch_size, num_channels, height, width
        )

        upscaled_embedding = self.upscale_conv1(image_embeddings)
        upscaled_embedding = self.activation(self.upscale_layer_norm(upscaled_embedding))
        upscaled_embedding = self.activation(self.upscale_conv2(upscaled_embedding))

        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            current_mlp = self.output_hypernetworks_mlps[i]
            hyper_in_list += [current_mlp(mask_tokens_out[:, :, i, :])]
        hyper_in = torch.stack(hyper_in_list, dim=2)

        _, num_channels, height, width = upscaled_embedding.shape
        upscaled_embedding = upscaled_embedding.reshape(batch_size, point_batch_size, num_channels, height * width)
        masks = (hyper_in @ upscaled_embedding).reshape(batch_size, point_batch_size, -1, height, width)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, :, mask_slice, :, :]
        iou_pred = iou_pred[:, :, mask_slice]

        outputs = (masks, iou_pred)

        if output_attentions:
            outputs = outputs + (attentions,)
        else:
            outputs = outputs + (None,)

        return outputs


class Encoder(nn.Module):
    """
    A convolutional neural network, consisting of len(num_filters) times a block of no_convs_per_block convolutional layers,
    after each block a pooling operation is performed. And after each convolutional layer a non-linear (ReLU) activation function is applied.
    """
    def __init__(self, input_channels, num_filters, no_convs_per_block, initializers, padding=True, posterior=False):
        super(Encoder, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.num_filters = num_filters

        if posterior:
            #To accomodate for the mask that is concatenated at the channel axis, we increase the input_channels.
            self.input_channels += 1

        layers = []
        for i in range(len(self.num_filters)):
            """
            Determine input_dim and output_dim of conv layers in this block. The first layer is input x output,
            All the subsequent layers are output x output.
            """
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = num_filters[i]

class CRnnCell(nn.Module):
    def __init__(self):
        # A cnn which goes from 512 x 64 x 64 -> 256 x 64 x 64
        super(CRnnCell, self).__init__()
        self.hidden_state = None
        self.conv_o = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv_h = nn.Conv2d(512, 256, kernel_size=3, padding=1)

    def forward(self, x):
        if self.hidden_state is None:
            self.hidden_state = torch.zeros(x.size(0), 256, x.size(2), x.size(3), device=x.device)
        combined = torch.cat([x, self.hidden_state], dim=1)
        o = self.conv_o(combined)
        h = self.conv_h(combined)
        self.hidden_state = h
        
        return o
    
    def reset(self):
        self.hidden_state = None


def wrap_forward(
    old_forward: callable,
    cell: CRnnCell,
    dense_prompt_embeddings: torch.Tensor,
    **kwargs
):

    dense_prompt_embeddings = cell(dense_prompt_embeddings)

    return old_forward(
        dense_prompt_embeddings=dense_prompt_embeddings,
        **kwargs
    )


class SeqSam(SamPreTrainedModel):

    def __init__(self, config, num_samples: int = 4, ablation: str = "none"):
        super().__init__(config)
        self.sam = SamModel(config)
        self.seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="none")
        self.multimask_output = True
        self.num_samples = num_samples
        self.ablation = Ablation.from_str(ablation)

        self.cell = CRnnCell()
        self.sam.mask_decoder.forward = partial(
            wrap_forward, 
            cell=self.cell, 
            old_forward=self.sam.mask_decoder.forward
        )

    @property
    def mask_decoder(self):
        return self.sam.mask_decoder
    
    def forward(
        self, 
        pixel_values: Optional[torch.Tensor] = None,
        image_embeddings: Optional[torch.Tensor] = None,
        input_boxes: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        label_mask: Optional[torch.Tensor] = None
    ):
        
        self.cell.reset()
        
        if image_embeddings is None:
            image_embeddings = self.sam.get_image_embeddings(pixel_values)

        input_masks = None
        total_loss = 0.0
        pred_masks = []

        # Ensures we have enough samples for the labels
        num_samples = max([labels.shape[1], self.num_samples])
        
        for i in range(num_samples):
            
            outputs = self.sam(
                image_embeddings=image_embeddings, 
                input_boxes=input_boxes,
                input_masks=input_masks,
                multimask_output=False,
            )
            input_masks = outputs.pred_masks.squeeze(2)

            if self.ablation == Ablation.STOP_GRADIENTS:
                input_masks = input_masks.detach()
                        
            pred_masks.append(outputs.pred_masks)

        if self.training:
            if self.ablation == Ablation.RANDOM:
                rand_idxs = random.sample(range(num_samples), k=labels.shape[1])
                pred_masks = [pred_masks[i] for i in rand_idxs]

            elif self.ablation == Ablation.SEQUENTIAL:
                pred_masks = pred_masks[:labels.shape[1]]

            else:
                num_labels = labels.shape[1]

                def ceildiv(a, b):
                    return -(a // -b)
                
                chunk_size = ceildiv(num_samples, num_labels)
                new_pred_masks = []
                for i in range(0, num_samples, chunk_size):
                    masks = pred_masks[i:i+chunk_size]
                    new_pred_masks.append(random.choice(masks))
                pred_masks = new_pred_masks
            

            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=int(padding)))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_per_block-1):
                layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=int(padding)))
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

        self.layers.apply(init_weights)

    def forward(self, input):
        output = self.layers(input)
        return output
    
        total_loss = 0.0
        pred_masks = torch.cat(pred_masks, dim=2)
        for i in range(labels.shape[0]):
            losses = torch.zeros(pred_masks.shape[2], labels.shape[1]).to(labels.device)
            for j in range(pred_masks.shape[2]):
                for k in range(labels.shape[1]):
                    loss = self.seg_loss(pred_masks[i, 0, j], labels[i, k]).mean()
                    losses[j, k] = loss 
            
            if self.ablation != Ablation.NO_HUNGARIAN_ALGORITHM:
                row_ind, col_ind = linear_sum_assignment(losses.detach().cpu().numpy())
            else:
                row_ind = torch.arange(losses.shape[0])
                col_ind = torch.arange(losses.shape[1])
            loss = losses[row_ind, col_ind]
            if self.training:
                loss *= label_mask[i].float()
            total_loss += loss.sum()
        
        total_loss /= labels.shape[1]
                
        return SamMultimaskOutput(
            loss=total_loss,
            iou_scores=outputs.iou_scores,
            pred_masks=pred_masks[:, :, :self.num_samples],
        )

