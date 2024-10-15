from ..templates import BaseProcessor
from ...imports import *


class ChannelChanger(BaseProcessor):
    def __init__(self,target_channels):
        self.target_channels = target_channels
    def _aprocess(self,audio):
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)
        if audio.shape[0] == self.target_channels:
            return audio
        if self.target_channels == 1:
            return audio.mean(0,keepdim=True)
        else:
            return audio.mean(0,keepdim=True).expand(self.target_channels,-1)