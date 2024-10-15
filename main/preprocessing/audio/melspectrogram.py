from ...imports import *
from ..templates import BaseProcessor

class MelSpectrogram(BaseProcessor):
    def __init__(self, sr=16000, n_mels=128, n_fft=1024, hop_length=256, max_frames=1300, auto_adjust_hop_length=True, normalize=True):
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_frames = max_frames
        self.auto_adjust_hop_length = auto_adjust_hop_length
        self.normalize = normalize

    def _aprocess(self, audio):
        # Convert the audio tensor to a numpy array
        dtype, device = audio.dtype, audio.device
        audio_np = audio.detach().cpu().numpy()
        hop_length = self.hop_length
        if self.auto_adjust_hop_length:
            # Calculate the number of frames in the Mel-spectrogram with the current hop length
            num_samples = audio_np.shape[-1]
            num_frames = (num_samples - self.n_fft) // self.hop_length + 1

            # Adjust the hop length if the number of frames exceeds the max_frames limit
            if num_frames > self.max_frames:
                hop_length = (num_samples - self.n_fft) // (self.max_frames - 1)
                num_frames = self.max_frames

        # Compute the Mel-spectrogram
        S = librosa.feature.melspectrogram(y=audio_np, sr=self.sr, n_mels=self.n_mels,
                                           n_fft=self.n_fft, hop_length=hop_length)
        S = librosa.power_to_db(S, ref=np.max)

        # Normalize the Mel-spectrogram to the range 0 to 1 if the normalize flag is True
        if self.normalize:
            S_min, S_max = S.min(), S.max()
            S = (S - S_min) / (S_max - S_min)

        # Convert the Mel-spectrogram to a tensor
        S = torch.from_numpy(S).to(dtype=dtype, device=device)
        return S
