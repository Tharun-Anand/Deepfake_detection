from .templates import BaseProcessor
from ..imports import *

class Stacker(BaseProcessor):  
    def __init__(self):
        pass
    def process(self,data):
        data['video'] = torch.stack(data['video'])
        data['audio'] = torch.stack(data['audio'])
        for key in data:
            if 'mask' in key:
                data[key] = torch.stack(data[key])
        return data