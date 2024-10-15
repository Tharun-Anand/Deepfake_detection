from .templates import BaseProcessor
from ..utils import read_video

class Loader(BaseProcessor):
    def __init__(self,device,debug=False):
        self.device = device
        self.debug = debug
    def process(self, data):
        """\
        data: {'video_links':[...]}
        """
        links = data['video_links']
        video = [read_video(link) for link in links]
        data['video'] = [v[0].to(self.device) for v in video]
        data['audio'] = [a[1].to(self.device) for a in video]
        data['info'] = [a[2] for a in video]
        if self.debug:
            data['video'] = [v[:,::10] for v in data['video']] # subsample for debugging
        return data