from demucs.pretrained import get_model
from demucs.apply import apply_model
from ...imports import *
from ..templates import BaseProcessor
from . import ChannelChanger, AChangeFPS

class AudioSeparator(BaseProcessor):
    def __init__(self, model_name='htdemucs'): # mdx 
        self.model = get_model(model_name)
        self.model.eval()
        self.channel_shifter = ChannelChanger(2)
        self.resampler = AChangeFPS(self.model.samplerate)
    
    def process(self, data):
        speechs, backgrounds = [], []
        for audio in data['audio']:
            audio = self.resampler._aprocess(audio)
            audio = self.channel_shifter._aprocess(audio)
            speech, background = self._aprocess(audio)
            speechs.append(speech)
            backgrounds.append(background)
        data['speech'] = speechs
        data['background'] = backgrounds
        return data
    
    def _aprocess(self, audio_tensor):
        # Demucs expects a tensor with shape [batch_size, channels, length]
        audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension if necessary
        
        # Normalize the input audio tensor
        audio_tensor = audio_tensor.float() / audio_tensor.abs().max()

        with torch.no_grad():
            sources = apply_model(self.model, audio_tensor, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # 'sources' will have the shape [batch_size, num_sources, channels, length]
        # Assuming the first source is 'vocals' (speech) and the rest are 'background'
        speech = sources[0, -1]  # First source is typically vocals in Demucs
        background = sources[0, :-1].sum(dim=1)  # Sum all other sources as background

        return speech, background