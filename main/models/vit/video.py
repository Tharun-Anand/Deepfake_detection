from ...imports import *
from .base1 import ViTBase
from .embeddings import PatchEmbedding3d


class ViTVideo(ViTBase):
    def __init__(self,img_size=224, patch_size=16, n_frames=16, embed_dim=768, depth=12,
        num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
        norm_layer="LayerNorm", init_values=0., tubelet_size=2):
        super().__init__(
            img_size=img_size, patch_size=patch_size, n_frames=n_frames,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            norm_layer=norm_layer, init_values=init_values, tubelet_size=tubelet_size
        )

    def init_patch_embedding(self, img_size,n_frames,patch_size,tubelet_size,embed_dim):
        self.patch_embedding1 = PatchEmbedding3d(
                    input_size=(3, n_frames, img_size, img_size),
                    patch_size=(tubelet_size, patch_size, patch_size),
                    embedding=embed_dim
                )
        #self.patch_embedding1.load_state_dict(torch.load('/home/sushanth/deepfake_detection/tharun/initialization_weights/patch_weights/video_encoder.patch_embedding1.pt'))

        
        self.patch_embedding2 = PatchEmbedding3d(
                    input_size=(3, n_frames, img_size, img_size),
                    patch_size=(tubelet_size, patch_size, patch_size),
                    embedding=embed_dim
                )
        #self.patch_embedding2.load_state_dict(torch.load('/home/sushanth/deepfake_detection/tharun/initialization_weights/patch_weights/video_encoder.patch_embedding2.pt'))
        
        # for param in self.patch_embedding2.parameters():
        #     param.requires_grad = False

        self.num_patches = (img_size // patch_size) * (img_size // patch_size) * (n_frames // tubelet_size)