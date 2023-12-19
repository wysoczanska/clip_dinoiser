# ------------------------------------------------------------------------------
# CLIP-DINOiser
# author: Monika Wysoczanska, Warsaw University of Technology
# ------------------------------------------------------------------------------
# Modified from OpenMMLab https://github.com/chongzhou96/MaskCLIP
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.ops import resize
from typing import Any, List
from torch import Tensor
from mmcv.utils import print_log
from mmseg.utils import get_root_logger
from open_clip import get_tokenizer,  create_model_from_pretrained
from models.builder import MODELS
from .vit import VisionTransformer
import torchvision.transforms as T
from .utils.embed import AdaptivePadding
from .utils.prompt_templates import imagenet_templates

OPENAI_NORMALIZE = T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))


def make_vision_transformer(backbone_cfg):
    model = VisionTransformer(**backbone_cfg)
    model.init_weights()
    return model


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
        self.backbone = make_vision_transformer(backbone)
        self.clip_T = OPENAI_NORMALIZE

        self.to_PIL = T.ToPILImage()
        self.patch_size = backbone.get('patch_size')
        self.padding = AdaptivePadding(self.patch_size, self.patch_size)

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        x = self.backbone(inputs)
        return x

    def forward(self, inputs: Tensor, return_feat=False) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        inputs = self.clip_T(inputs)
        x = self.extract_feat(inputs)

        seg_logits, feats, k = self.decode_head(x, return_feat)

        if return_feat:
            return seg_logits, feats, k
        return seg_logits

class MaskClipHead(nn.Module):
    def __init__(self, clip_model, class_names, visual_projs_path=None, in_index=-1, in_channels=3, norm_cfg=None, channels=0,
                 text_channels=512, attn_pooling=False, align_corners=False, model_prefix='hf-hub:laion', use_templates=False, **kwargs):
        super(MaskClipHead, self).__init__()

        self.text_channels = text_channels
        self.visual_projs_path = visual_projs_path
        self.clip_model = clip_model
        self.class_names = class_names
        self.in_channels = in_channels
        self.in_index = in_index # from base decode head default
        self._init_inputs(in_channels, in_index, None)
        self.channels = channels
        self.norm_cfg = norm_cfg
        self.align_corners = align_corners
        self.use_templates = use_templates

        self.proj = nn.Conv2d(self.in_channels, text_channels, 1, bias=False)
        self.load_visual_projs()

        self.attn_pooling = attn_pooling
        self.tokenizer = get_tokenizer(f'{model_prefix}/{clip_model}')
        self.hf_modelname = f'{model_prefix}/{clip_model}'
        model, _ = create_model_from_pretrained(f'{model_prefix}/{clip_model}')
        model.eval()
        self.register_buffer("class_embeddings", self._get_class_embeddings(model, class_names))

    @torch.no_grad()
    def update_vocab(self, class_names):
        model, _ = create_model_from_pretrained(self.hf_modelname)
        model.eval()
        self.class_embeddings = self._get_class_embeddings(model, class_names)

    @torch.no_grad()
    def _embed_label(self, text_model: torch.nn.Module, label: str) -> torch.Tensor:
        """
        Encode label name into a single vector
        """
        if self.use_templates:
            templates = imagenet_templates
        else:
            templates = ['a photo of an {}' if label.startswith('aeiou') else 'a photo of a {}']

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

    def load_visual_projs(self):
        loaded = torch.load(self.visual_projs_path, map_location='cuda')
        attrs = ['proj']
        for attr in attrs:
            current_attr = getattr(self, attr)
            state_dict = loaded[attr]
            for key in state_dict:
                if 'weight' in key:
                    state_dict[key] = state_dict[key][:, :, None, None]
            current_attr.load_state_dict(state_dict)
        print_log(f'Loaded proj weights from {self.visual_projs_path}', logger=get_root_logger())

    def forward(self, inputs, return_feat=False):
        x = self._transform_inputs(inputs)
        q, k, v, cls_token = None, None, None, None
        if isinstance(x, list) and len(x) == 4:
            x, q, k, v = x
        if isinstance(x, list) and len(x) == 2:
            x, cls_token = x
        if v is not None:
            feat = self.proj(v)
        else:
            feat = self.proj(x)
        output = self.cls_seg(feat)
        if return_feat:
            return output, feat, k

        return output

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def cls_seg(self, feat):
        feat = feat / feat.norm(dim=1, keepdim=True)
        output = F.conv2d(feat, self.class_embeddings[:, :, None, None])
        output = F.softmax(output * 100, dim=1) # softmax of similarities with temp scaling

        return output

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """
        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs
