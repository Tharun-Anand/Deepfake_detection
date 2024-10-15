from ..templates import BaseProcessor
from ...imports import *
from ...utils.face_crop import FaceXZooFaceDetector

class FaceCropper(BaseProcessor):
    def __init__(self,device,crop_shape,normalised_input):
        self.device = device
        self.crop_shape = crop_shape
        self.normalised_input = normalised_input
        if not FaceXZooFaceDetector.inited:
            FaceXZooFaceDetector.init(device=self.device)


    def _vprocess(self, video):
        """\
        Processes a single video. Input: C * T * H * W
        """
        video = rearrange(video, "c t h w -> t h w c").detach().cpu().numpy()
        if self.normalised_input:
            video = (video * 255).astype(np.uint8)

        face_frames = []
        for frame in video:
            # Crop face result: (H, W, C)
            face_crop = FaceXZooFaceDetector.crop_face(frame,imgsize=self.crop_shape)[0]
            face_frames.append(torch.from_numpy(face_crop))

        faces = torch.stack(face_frames)  # (T, H, W, C)
        faces = rearrange(faces, "t h w c -> c t h w").to(self.device)

        if self.normalised_input: # Convert back to [0,1]
            faces = faces / 255
        return faces

