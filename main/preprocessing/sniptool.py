import math
from ..imports import *
from .templates import BaseProcessor

class SnipTool(BaseProcessor):
    def __init__(self, mode, start, end):
        self.mode = mode
        self.start = start
        self.end = end
        assert mode in ["video-frames", "audio-frames", "frac"]

    def process(self, data):
        video = data['video']
        audio = data['audio']
        info = data['info']
        for i in range(len(video)):
            video[i], audio[i] = self.snip(self.start, self.end, video[i], audio[i], info[i])
        data['video'] = video
        data['audio'] = audio
        return data

    def snip(self, start, end, video, audio, info):
        if self.mode == "video-frames":
            vstart, vend = start, end
            fstart, fend = start / info['video_fps'], end / info['video_fps']
            astart, aend = fstart * info['audio_fps'], fend * info['audio_fps']

        elif self.mode == "audio-frames":
            astart, aend = start, end
            fstart, fend = start / info['audio_fps'], end / info['audio_fps']
            vstart, vend = fstart * info['video_fps'], fend * info['video_fps']

        elif self.mode == "frac":
            fstart, fend = start, end
            vstart, vend = fstart * info['video_fps'], fend * info['video_fps']
            astart, aend = fstart * info['audio_fps'], fend * info['audio_fps']

        vstart, vend = int(math.floor(vstart)), int(math.ceil(vend))
        astart, aend = int(math.floor(astart)), int(math.ceil(aend))

        # Ensure indices are within bounds
        vstart, vend = max(vstart, 0), min(vend, video.shape[1])
        astart, aend = max(astart, 0), min(aend, audio.shape[1])

        return video[:, vstart:vend], audio[:, astart:aend]