from ...imports import *
from ..templates import BaseProcessor

class TorchVTransform(BaseProcessor):
    def __init__(self, transform):
        self.transform = transform
    def _vprocess(self, video: Tensor) -> Tensor:
        return self.transform(video.transpose(0,1)).transpose(0,1) # transpose to T * C * H * W