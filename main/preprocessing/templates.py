from abc import ABC, abstractmethod
from torch import Tensor
from typing import Tuple, Dict

class BaseProcessor(ABC):
    def process(self, data: Dict) -> Tuple[Tensor, Tensor]:
        """
        Process the given video and audio tensors by cropping the face from the video.

        Parameters:
            video (Tensor): The input video list. The shape should be list of (T, C, H, W).
            audio (Tensor): The input audio list. Ignored

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the cropped face video tensor and the original audio tensors.
        """
        video = data["video"]
        audio = data["audio"]

        video = [self._vprocess(v) for v in video]
        audio = [self._aprocess(a) for a in audio]

        data["video"] = video
        data["audio"] = audio

        return data

    def _vprocess(self, video: Tensor) -> Tensor:
        """\
        Processes a single video. Input: C * T * H * W
        """
        return video
    
    def _aprocess(self, audio: Tensor) -> Tensor:
        """\
        Processes a single audio. Input: K * T
        """
        return audio
    