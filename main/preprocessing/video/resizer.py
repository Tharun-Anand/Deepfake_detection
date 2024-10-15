from torchvision.transforms.functional import resize
from ..templates import BaseProcessor
from ...imports import *

class Resizer(BaseProcessor):
    def __init__(self,resize_shape,normalised_input):
        self.resize_shape = resize_shape
        self.normalised_input = normalised_input
    
    def _vprocess(self,video):
        """\
        Processes a single video. Input: C * T * H * W
        """
        device = video.device
        if self.normalised_input:
            video = (video * 255).astype(np.uint8)
        video = resize(video, self.resize_shape)
        if self.normalised_input:
            video = video / 255
        video = video.to(device)
        return video
    