from ..templates import BaseProcessor
from ...imports import *


class SimpleSampler(BaseProcessor):
    def __init__(self, strategy, num_frames, stride=1):
        self.strategy = strategy
        self.num_frames = num_frames
        self.stride = stride
        assert self.stride > 0
        assert self.strategy in ["uniform", "random", "first_n", "last_n", "random_clip"]

    def _vprocess(self, video):
        """\
        Processes a single video. Input: C * T * H * W
        """
        total_frames = video.shape[1]
        indices = self.sample_frames(total_frames, self.strategy, self.num_frames, self.stride)
        return video[:,indices]
    

    @staticmethod
    def sample_indexes(total_frames: int, n_frames: int, temporal_sample_rate: int) -> Tensor:
        try:
            start_ind = torch.randint(0, total_frames - (n_frames * temporal_sample_rate) + 1, ())
        except RuntimeError as e:
            print(f"total_frames: {total_frames}, n_frames: {n_frames}, temporal_sample_rate: {temporal_sample_rate}")
            raise e
        return torch.arange(n_frames) * temporal_sample_rate + start_ind
    
    @staticmethod
    def sample_frames(total_frames, strategy, num_frames, stride=1):
        if strategy == "random":
            indices = np.random.choice(list(range(total_frames)), num_frames, replace=False)
        elif strategy == "uniform":
            indices = np.linspace(0, total_frames - 1, num_frames, endpoint=True).astype(int)
            #indices=sample_indexes(total_frames, num_frames, stride)
        elif strategy == "first_n":
            indices = np.arange(0, num_frames * stride, stride)
        elif strategy == "last_n":
            indices = np.arange(max(0, total_frames - num_frames * stride), total_frames, stride)
        elif strategy == "random_clip":
            start = np.random.randint(0, max(1, total_frames - num_frames * stride + 1))
            indices = np.arange(start, start + num_frames * stride, stride)
        else:
            raise ValueError("Unknown strategy")
        
        indices = indices[indices < total_frames]
        return list(indices)
