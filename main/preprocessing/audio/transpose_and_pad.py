from ..templates import BaseProcessor
from ...imports import *


class TransposeAndPad(BaseProcessor):
    def __init__(self, max_length=1024):
        super().__init__()
        self.max_length = max_length

    def _aprocess(self, audio: Tensor) -> Tensor:
        """\
        Processes a single audio spectrogram. Input: C * W * H
        """
        audio = audio.transpose(1,2)
        audio = audio[:, :self.max_length]
        return F.pad(audio, (0, 0, 0, max(0, self.max_length - audio.shape[1])))