from ...imports import *
from .base import ViTBase

class ViTFusion(ViTBase):
    def __init__(self, embed_dim=768, depth=12,
        num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
        norm_layer="LayerNorm", init_values=0., tubelet_size=2):
        super().__init__(
            img_size=None, patch_size=None, n_frames=None,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            norm_layer=norm_layer, init_values=init_values, tubelet_size=tubelet_size
        )

    def init_patch_embedding(self, img_size,n_frames,patch_size,tubelet_size,embed_dim):
        self.num_patches = None
        pass # We dont need patch embedding in fusion

    def init_positional_embedding(self, num_patches, embed_dim, dropout_rate=0):
        pass
    
    def forward(self, audio_embedding, video_embedding):
        x = torch.cat([audio_embedding, video_embedding], dim=1)
        x = self.backbone(x)

        audio_embedding = x[:, :audio_embedding.shape[1]]
        video_embedding = x[:, -video_embedding.shape[1]:]
        return audio_embedding, video_embedding