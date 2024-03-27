# ------------------------------------------------------------------------------
import logging

import torch

log = logging.getLogger(__name__)
from mmseg.ops import resize
from mmseg.models import EncoderDecoder


class DinoCLIP_Infrencer(EncoderDecoder):
    def __init__(
            self,
            model,
            num_classes,
            test_cfg=dict(),
            **kwargs,
    ):
        super(EncoderDecoder, self).__init__()
        self.mode = test_cfg['mode']
        self.num_classes = num_classes
        self.model = model
        self.test_cfg = test_cfg
        self.align_corners = False

    @torch.no_grad()
    def encode_decode(self, img, meta_data):
        """
        """
        masks = self.model(img)
        masks = resize(
            input=masks,
            size=img.shape[-2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return masks
