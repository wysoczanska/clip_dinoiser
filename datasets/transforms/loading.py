from mmseg.datasets.builder import PIPELINES
from typing import Dict, Optional, Union
import cv2


@PIPELINES.register_module()
class ToRGB:
    def __call__(self, results):
        return self.transform(results)

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to go from BGR to RGB.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        results['img'] = cv2.cvtColor(results['img'], cv2.COLOR_BGR2RGB)
        return results