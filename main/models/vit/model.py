from .audio import ViTAudio
from .video import ViTVideo
from .fusion import ViTFusion
from .decoder import ViTDecoder
from .masking_agent import MaskingAgent
from ...imports import *

class Model1(nn.Module):
    def __init__(self,masking_ratio=0): # 0.75 of input will be masked out!
        super().__init__()
        self.video_encoder = ViTVideo()
        self.audio_encoder = ViTAudio()
        self.fuser = ViTFusion()
        self.proj = nn.Linear(768,384)
        self.decoder = ViTDecoder(self.audio_encoder.num_patches, self.video_encoder.num_patches, 
                     audio_target_embed_dim=256, video_target_embed_dim=768*2)
        
        self.video_encoder.load_weights('/home/sushanth/deepfake_detection/tharun/initialization_weights/video.pt')

        self.masking_agent = MaskingAgent()
        self.video_mask_token = nn.Parameter(torch.randn(1, 1, self.video_encoder.embed_dim))
        self.audio_mask_token = nn.Parameter(torch.randn(1, 1, self.audio_encoder.embed_dim))
        self.masking_ratio = masking_ratio
    
    def forward(self, v):
        # Run through patching and positional encoding alone
        #encoded_audio = self.audio_encoder.encode_input(a)
        encoded_video = self.video_encoder.encode_input(v)
        

        # Select some timesteps alone to be run through the backbones
        N_video = int(encoded_video.shape[1] * (1-self.masking_ratio))
        #N_audio = int(encoded_audio.shape[1] * (1-self.masking_ratio))

        selected_video_encodings, video_shuffle_ids = self.masking_agent.shuffle_and_select(encoded_video, N_video)
        #selected_audio_encodings, audio_shuffle_ids = self.masking_agent.shuffle_and_select(encoded_audio, N_audio)

        # Run through the backbone
        video_embeddings = self.video_encoder.forward_features(selected_video_encodings)
        #audio_embeddings = self.audio_encoder.forward_features(selected_audio_encodings)

        # Fuse the embeddings together
        #audio_embeddings, video_embeddings = self.fuser(audio_embeddings, video_embeddings)
    
        # Reconstruct full shape with mask tokens
        # full_audio_embeddings, is_audio_pad_token_mask = self.masking_agent.pad_mask_token(audio_embeddings,
        #                                                            self.audio_mask_token, audio_shuffle_ids.shape[1])
        full_video_embeddings, is_video_pad_token_mask = self.masking_agent.pad_mask_token(video_embeddings,
                                                                   self.video_mask_token, video_shuffle_ids.shape[1])
        
        # full_audio_embeddings, is_audio_pad_token_mask = self.masking_agent.unshuffle(full_audio_embeddings,is_audio_pad_token_mask, audio_shuffle_ids)
        full_video_embeddings, is_video_pad_token_mask = self.masking_agent.unshuffle(full_video_embeddings,is_video_pad_token_mask, video_shuffle_ids)

        ## Run through the decoder
        # decoded_audio_embeddings, decoded_video_embeddings = self.decoder(full_audio_embeddings, full_video_embeddings)
        decoded_video_embeddings = self.decoder(self.proj(full_video_embeddings))
        
        # return decoded_video_embeddings,is_video_pad_token_mask
        return decoded_video_embeddings
