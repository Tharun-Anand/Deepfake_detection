from ...imports import *
from .base import ViTBase
from .embeddings import AudioPatchEmbed


class ViTAudio(ViTBase):
    def __init__(self,img_size=(1024,128), patch_size=16, embed_dim=768, depth=12,
        num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
        norm_layer="LayerNorm", init_values=0., tubelet_size=None,n_frames=None):
        super().__init__(
            img_size=img_size, patch_size=patch_size, n_frames=n_frames,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            norm_layer=norm_layer, init_values=init_values, tubelet_size=tubelet_size
        )

    def init_patch_embedding(self,img_size,n_frames,patch_size,tubelet_size,embed_dim):
        self.patch_embedding = AudioPatchEmbed(
                img_size=img_size,
                patch_size=[16,16],
                in_chans=1,
                embed_dim=embed_dim,
            )

        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)