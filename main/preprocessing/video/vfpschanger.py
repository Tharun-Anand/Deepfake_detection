from ...imports import *
from ..templates import BaseProcessor



class VFPSChanger(BaseProcessor):
    def __init__(self, fps, method='drop-duplicate'):
        self.fps = fps
        self.method = method
        assert self.method in ['drop-duplicate', 'interpolate']

    def process(self, data):
        video = data['video']  # Expecting a list of video tensors
        info = data['info']    # Expecting a list of info dictionaries
        new_videos = []

        for v, i in zip(video, info):
            dtype = v.dtype
            old_fps = i['video_fps']
            i['video_fps'] = self.fps  # Update the FPS info

            num_frames = v.shape[1]
            new_num_frames = int(num_frames * self.fps / old_fps)

            if self.method == 'drop-duplicate':
                new_v = self._drop_duplicate(v, new_num_frames, num_frames, old_fps)
            elif self.method == 'interpolate':
                new_v = self._interpolate(v, new_num_frames)

            new_videos.append(new_v)

        data['video'] = new_videos

        return data
    
    def _interpolate(self, v, new_num_frames):
        new_v = F.interpolate(v.unsqueeze(0).to(dtype=torch.float), size=(new_num_frames, v.shape[2], v.shape[3]),
                               mode='trilinear', align_corners=False).squeeze(0)
        new_v = new_v.to(dtype=v.dtype)
        return new_v
    
    def _drop_duplicate(self, v, new_num_frames, num_frames, old_fps):
        if self.fps > old_fps:
            # Frame duplication: Use linspace and round to get closest frames
            indices = np.linspace(0, num_frames - 1, new_num_frames).round().astype(int)
        else:
            # Frame dropping: Use linspace and floor to get closest frames
            indices = np.linspace(0, num_frames - 1, new_num_frames).astype(int)

        # Select frames based on the calculated indices
        new_v = v[:, indices, :, :]

        return new_v
