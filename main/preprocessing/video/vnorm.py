from ...imports import *
from ..templates import BaseProcessor

class VNorm(BaseProcessor):
    def __init__(self,shift=0,scale=255):
        self.shift = shift
        self.scale = scale

    def _vprocess(self, video: Tensor) -> Tensor:
        return (video - self.shift) / self.scale