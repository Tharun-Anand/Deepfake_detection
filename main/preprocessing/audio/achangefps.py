from ..templates import BaseProcessor
from ...imports import *

class AChangeFPS(BaseProcessor):
    def __init__(self,fps):
        self.fps = fps

    def process(self,data):
        audios = data['audio']
        infos = data['info']
        new_audios = []
        for audio, info in zip(audios, infos):
            new_audios.append(AF.resample(audio, info['audio_fps'], self.fps))
            info['audio_fps'] = self.fps
        data['audio'] = new_audios
        return data
