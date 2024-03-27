# ---------------------------------------------------------------------------------------------------
# CLIP-DINOiser
# authors: Monika Wysoczanska, Warsaw University of Technology

# Copyright (c) OpenMMLab. All rights reserved.
# Modified version of the original MaskCLIP code: https://github.com/chongzhou96/MaskCLIP/tree/master
# ---------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.ops import resize
from typing import List, Tuple
from torch import Tensor
from open_clip import get_tokenizer,  create_model_from_pretrained
from models.builder import MODELS
import torchvision.transforms as T
from .utils.prompt_templates import imagenet_templates

OPENAI_NORMALIZE = T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))


@MODELS.register_module()
class MaskClip(nn.Module):
    def __init__(
            self,
            backbone,
            decode_head,
            clip_model,
            class_names
        ):
        super(MaskClip, self).__init__()

        self.decode_head = eval(decode_head.get('type'))(clip_model, class_names, **decode_head)
        self.patch_size = backbone.get('patch_size')
        self.img_size = tuple([backbone.get('img_size', 224)]*2)
        pretrained = decode_head.get("pretrained")
        model, _ = create_model_from_pretrained(clip_model, pretrained=pretrained)
        model.eval()
        self.clip_T = OPENAI_NORMALIZE
        self.hook_features = {}
        self.backbone = model
        def hook_fn_forward(module, input, output):
            self.hook_features["v"] = output
        self.backbone.visual.transformer.resblocks[-2].register_forward_hook(hook_fn_forward)
        self._positional_embd = nn.Parameter(self.backbone.visual.positional_embedding.data.clone())

    @torch.no_grad()
    def extract_feat(self, inputs: Tensor) -> Tuple[Tensor]:
        """Extract features from images."""
        pos_embed = self.backbone.visual.positional_embedding

        B, C, H, W = inputs.shape
        hw_shape = (H // self.patch_size, W // self.patch_size)
        x_len, pos_len = hw_shape[0]*hw_shape[1], pos_embed.shape[0]

        if x_len != pos_len:
            if pos_len == (self.img_size[0] // self.patch_size) * (self.img_size[1] // self.patch_size) + 1:
                pos_h = self.img_size[0] // self.patch_size
                pos_w = self.img_size[1] // self.patch_size
            else:
                raise ValueError(
                    '{}, {}'.format(x_len, pos_len))

            self.backbone.visual.positional_embedding.data = self.resize_pos_embed(
                self._positional_embd[None], hw_shape,  (pos_h, pos_w), 'bicubic')[0]

        _ = self.backbone(inputs)
        v = self.hook_features["v"]
        v = self.extract_v(v, self.backbone.visual.transformer.resblocks[-1]).permute(1, 0, 2)
        v = self.backbone.visual.ln_post(v)
        v = v[:, 1:]
        v = v.reshape(B, hw_shape[0], hw_shape[1], -1).permute(0, 3, 1, 2).contiguous()

        self.backbone.visual.positional_embedding.data = self._positional_embd
        return v

    def extract_v(self, x, block):
        y = block.ln_1(x)
        y = torch.nn.functional.linear(y, block.attn.in_proj_weight, block.attn.in_proj_bias)
        B, N, C = y.shape
        y = y.view(B, N, 3, C // 3).permute(2, 0, 1, 3).reshape(3 * B, N, C // 3)
        y = F.linear(y, block.attn.out_proj.weight, block.attn.out_proj.bias)
        q, k, v = y.tensor_split(3, dim=0)
        v += x
        v += block.mlp(block.ln_2(v))
        return v


    @staticmethod
    def resize_pos_embed(pos_embed, input_shpae, pos_shape, mode):
        """Resize pos_embed weights.

        Resize pos_embed using bicubic interpolate method.
        Args:
            pos_embed (torch.Tensor): Position embedding weights.
            input_shpae (tuple): Tuple for (downsampled input image height,
                downsampled input image width).
            pos_shape (tuple): The resolution of downsampled origin training
                image.
            mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'nearest'``
        Return:
            torch.Tensor: The resized pos_embed of shape [B, L_new, C]
        """
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        pos_h, pos_w = pos_shape
        cls_token_weight = pos_embed[:, 0]
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
        pos_embed_weight = resize(
            pos_embed_weight, size=input_shpae, align_corners=False, mode=mode)
        cls_token_weight = cls_token_weight.unsqueeze(1)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
        return pos_embed

    def forward(self, inputs: Tensor, return_feat=False) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        inputs = self.clip_T(inputs)
        x = self.extract_feat(inputs)
        if return_feat:
            seg_logits, feats = self.decode_head(x, return_feat)
            return seg_logits, feats
        else:
            seg_logits = self.decode_head(x)
        return seg_logits

class MaskClipHead(nn.Module):
    def __init__(self, clip_model, class_names, in_channels=3, text_channels=512, use_templates=False, pretrained=None,
                 **kwargs):
        super(MaskClipHead, self).__init__()

        self.text_channels = text_channels
        self.clip_model = clip_model
        self.pretrained = pretrained
        self.class_names = class_names
        self.in_channels = in_channels
        self.use_templates = use_templates
        self.tokenizer = get_tokenizer(clip_model)
        model, _ = create_model_from_pretrained(clip_model, pretrained=pretrained)
        model.eval()
        self.register_buffer("class_embeddings", self._get_class_embeddings(model, class_names))
        self.proj = nn.Conv2d(self.in_channels, text_channels, 1, bias=False)
        self.proj.weight = nn.Parameter(model.visual.proj.t()[:, :, None, None])

    @torch.no_grad()
    def update_vocab(self, class_names):
        model, _ = create_model_from_pretrained(self.clip_model, pretrained=self.pretrained )
        model.eval()
        self.class_embeddings = self._get_class_embeddings(model, class_names)

    @torch.no_grad()
    def _embed_label(self, text_model: torch.nn.Module, label: str) -> torch.Tensor:
        """
        Encode label name into a single vector
        """
        if self.use_templates:
            templates = imagenet_templates
        elif "laion" in self.pretrained:
            templates = ['a photo of a {}', 'a photo of an {}']
        else:
            templates = ['a {}']
        all_prompts = [self.tokenizer(template.format(label)) for template in templates]
        out = text_model.encode_text(torch.cat(all_prompts))
        out /= out.norm(dim=-1, keepdim=True)
        out = out.mean(dim=0)
        return out

    def _get_class_embeddings(self, text_model: torch.nn.Module, class_names: List[str]):
        aug_embeddings = torch.stack([self._embed_label(text_model, label) for label in class_names])
        # normalize vector
        aug_embeddings = aug_embeddings / aug_embeddings.norm(dim=-1, keepdim=True)
        return aug_embeddings.squeeze(1)

    def forward(self, inputs, return_feat=False):
        v = inputs
        feat = self.proj(v)
        output = self.cls_seg(feat)
        if return_feat:
            return output, feat
        return output

    def cls_seg(self, feat):
        feat = feat / feat.norm(dim=1, keepdim=True)
        output = F.conv2d(feat, self.class_embeddings[:, :, None, None])
        output = F.softmax(output * 100, dim=1)
        return output
