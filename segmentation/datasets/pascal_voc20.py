# ------------------------------------------------------------------------------
import os.path as osp

from mmseg.datasets import CustomDataset
from mmseg.datasets import DATASETS


@DATASETS.register_module()
class PascalVOCDataset20(CustomDataset):
    """Pascal VOC dataset (the background class is ignored).

    Args:
        split (str): Split txt file for Pascal VOC.
    """

    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dog',
               'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa',
               'train', 'tvmonitor')

    PALETTE = [[128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
               [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
               [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
               [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
               [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]

    def __init__(self, split, **kwargs):
        super(PascalVOCDataset20, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            split=split,
            reduce_zero_label=True,
            **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
