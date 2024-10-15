from ..templates import BaseProcessor
from ...imports import *


class FramePadder(BaseProcessor):
    def __init__(self,pad_value,output_shape):
        self.pad_value = pad_value
        self.output_shape = output_shape

    def process(self, data: Dict) -> Dict:
        video = data["video"]

        processed = [self._vprocess(v) for v in video]
        video = [p[0] for p in processed]
        mask = [p[1] for p in processed]
        data["video"] = video
        data["frame_pad_mask"] = mask
        return data
    
    def _vprocess(self,video):
        """\
        Processes a single video. Input: C * T * H * W
        """
        return self.pad_to_output_shape(video, self.output_shape, self.pad_value)
        
    @staticmethod
    def pad_to_output_shape(image: torch.Tensor, output_shape: tuple, value: float = 0) -> torch.Tensor:
        """
        Pad a PyTorch image tensor symmetrically to match the required output shape.

        Args:
        - image (torch.Tensor): Input image tensor of shape (C, H, W) or (B, C, H, W).
        - output_shape (tuple): Desired output shape (C, H', W').

        Returns:
        - padded_image (torch.Tensor): Padded image tensor of shape (C, H', W').
        """

        # Get the current shape of the input image
        input_shape = image.shape[-2:]  # (H, W)

        # Calculate the amount of padding required
        pad_height = output_shape[0] - input_shape[0]
        pad_width = output_shape[1] - input_shape[1]

        # Compute the pad amounts for each side of the image
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        # Pad the image tensor symmetrically
        if len(image.shape) == 3:  # Single image (C, H, W)
            padded_image = F.pad(image.unsqueeze(0), (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=value)
            padded_image = padded_image.squeeze(0)
        elif len(image.shape) == 4:  # Batch of images (B, C, H, W)
            padded_image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=value)
        else:
            raise ValueError("Input tensor must be 3D (C, H, W) or 4D (B, C, H, W).")
        
        padding_mask = torch.zeros_like(padded_image)
        padding_mask[:, pad_top:pad_top+input_shape[0], pad_left:pad_left+input_shape[1]] = 1

        return padded_image, padding_mask

