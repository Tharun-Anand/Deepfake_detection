from ..templates import BaseProcessor
from ...imports import *


class CenterCropper(BaseProcessor):
    def __init__(self,crop_shape,normalised_input):
        self.crop_shape = crop_shape
        self.normalised_input = normalised_input

    def _vprocess(self,video):
        """\
        Processes a single video. Input: C * T * H * W
        """
        device = video.device
        video = rearrange(video, "c t h w -> t c h w").detach().cpu().numpy()
        if self.normalised_input:
            video = (video * 255).astype(np.uint8)
        frames = []

        for frame in video:
            # Crop face result: (H, W, C)
            center_cropped_img = self._center_crop(frame,*self.crop_shape)
            frames.append(torch.from_numpy(center_cropped_img))

        faces = torch.stack(frames)  # (T, H, W, C)
        faces = rearrange(faces, "t c h w -> c t h w").to(device)

        if self.normalised_input: # Convert back to [0,1]
            faces = faces / 255

        return faces
    
    def _center_crop(self, img, new_height=None, new_width=None):
        width = img.shape[2]
        height = img.shape[1]

        if new_width is None:
            new_width = min(width, height)

        if new_height is None:
            new_height = min(width, height)

        left = int(np.ceil((width - new_width) / 2))
        right = width - int(np.floor((width - new_width) / 2))

        top = int(np.ceil((height - new_height) / 2))
        bottom = height - int(np.floor((height - new_height) / 2))
        if len(img.shape) == 3:
            center_cropped_img = img[:, top:bottom, left:right]
        else:
            center_cropped_img = img[:, top:bottom, left:right, ...]
        
        return center_cropped_img


