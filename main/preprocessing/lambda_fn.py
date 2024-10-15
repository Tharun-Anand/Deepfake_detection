from .templates import BaseProcessor
from ..imports import *


class LambdaFn(BaseProcessor):
    def __init__(self,fn,on = None):
        self.fn = fn
        self.on = on
        assert on in ['audio','video']
        if on == 'audio':
            self._aprocess = fn
        elif on == 'video':
            self._vprocess = fn