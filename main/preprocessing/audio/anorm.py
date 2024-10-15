from ..templates import BaseProcessor
from ...imports import *

class ANorm(BaseProcessor):
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std
    def _aprocess(self,audio):
        mean ,std = self.mean, self.std
        if not mean:
            mean = audio.mean(-1, keepdim=True)
        if not std:
            std = audio.std(-1, keepdim=True)
        return (audio - mean) / std
