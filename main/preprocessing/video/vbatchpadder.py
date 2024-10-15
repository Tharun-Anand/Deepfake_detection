from typing import Dict, Tuple
from ...imports import *
from ..templates import BaseProcessor


class VBatchPadder(BaseProcessor):
    def __init__(self,num_frames=None,pad_value=0,padding_strategy="constant",truncate=False):
        self.num_frames = num_frames
        self.pad_value = pad_value
        self.padding_strategy = padding_strategy
        self.truncate = truncate
        assert self.padding_strategy in ["max_length", "constant"]
        if self.padding_strategy == "constant":
            assert self.num_frames is not None, "num_frames must be specified if padding_strategy is 'constant'"

    def process(self, data: Dict) -> Dict:
        video = data["video"]
        num_frames = [v.shape[1] for v in video]
        max_num_frames = max(num_frames)
        if self.padding_strategy == "max_length":
            self.num_frames = max_num_frames

        processed = [self._vprocess(v) for v in video]
        video = [p[0] for p in processed]
        mask = [p[1] for p in processed]
        data["video"] = video
        data["batch_pad_mask"] = mask
        return data

    def _vprocess(self, video): # video: C * T * H * W
        device = video.device
        dtype = video.dtype
        cur_frames = video.shape[1]

        if cur_frames >= self.num_frames:
            if self.truncate:
                video = video[:, :self.num_frames, :, :]
                mask = torch.zeros(video.shape[1], dtype=torch.bool).to(device)
                # print("Truncated",video.shape)
            return video, mask
        
        pad_image = torch.ones(video.shape[0],1,*video.shape[-2:], dtype=dtype, device=device) * self.pad_value

        num_pads = self.num_frames - cur_frames
        pad_image = pad_image.repeat(1, num_pads, 1, 1)

        video = torch.cat([video, pad_image], dim=1)
        mask = torch.zeros(video.shape[1], dtype=torch.bool).to(device)
        mask[cur_frames:] = True
        video = video.to(dtype=dtype, device=device) # Ensure dtype and device are restored
        return video, mask