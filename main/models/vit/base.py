from ...imports import *
from .block import Block
from .embeddings import SinCosPositionalEmbedding
from .masking_agent import MaskingAgent
import logging



class VitBackBone(nn.Module):
    def __init__(self,embed_dim=768, depth=12,
        num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
        norm_layer="LayerNorm", init_values=0.):
        super().__init__()
        
        if norm_layer == "LayerNorm":
            self.norm_layer = nn.LayerNorm
            self.norm = self.norm_layer(embed_dim)
        else:
            raise NotImplementedError("Only LayerNorm is supported")
        


        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, norm_layer=self.norm_layer, init_values=init_values
            )
            for _ in range(depth)
        ])



    def forward(self, x):
        for i, block in enumerate(self.blocks):
            #print(f"Passing through Block {i}")
            x = block(x)
        x = self.norm(x)
        return x


class ViTBase(nn.Module):
    def __init__(self, 
                img_size=224, n_frames=16, # Video Dimensions
                patch_size=16,tubelet_size=2, # Patch Dimensions
                embed_dim=768, # Embedding dimension
                depth=12, # Number of attention layers
                num_heads=12, # Number of attention heads
                mlp_ratio=4., # MLP Expansion or Contraction ratio. Neurons: dim, dim * MLP ratio, dim
                qkv_bias=False, 
                qk_scale=None, 
                drop_rate=0., 
                attn_drop_rate=0.,
                norm_layer="LayerNorm",
                  init_values=0.
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_size=patch_size
        
        self.init_patch_embedding(img_size,n_frames,patch_size,tubelet_size,embed_dim)
        self.init_positional_embedding(self.num_patches, embed_dim, drop_rate)
        self.init_backbone(embed_dim, depth, num_heads, 
                           mlp_ratio, qkv_bias, qk_scale,
                             drop_rate, attn_drop_rate, norm_layer,
                               init_values)

    
    
    def init_patch_embedding(self,img_size,n_frames,patch_size,tubelet_size,embed_dim):
        """
        initialize the patch embedding self.patch_embedding and return the number of patches
        """
        self.num_patches = None
        self.patch_embedding1 = None
        self.patch_embedding2 = None
        raise NotImplementedError("Not implemented yet")

    def init_positional_embedding(self,num_patches, embed_dim, dropout_rate=0.):
        self.pos_embedding1 = SinCosPositionalEmbedding((num_patches, embed_dim), dropout_rate=dropout_rate)        
        self.pos_embedding2 = SinCosPositionalEmbedding((num_patches, embed_dim), dropout_rate=dropout_rate)



    def init_backbone(self, embed_dim=768, depth=12, num_heads=12,
                       mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0.,
                         attn_drop_rate=0., norm_layer="LayerNorm", init_values=0.):
        
        self.backbone1 = VitBackBone(
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            norm_layer=norm_layer, init_values=init_values
        )
        self.backbone2 = VitBackBone(
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            norm_layer=norm_layer, init_values=init_values
        )
     
    def forward_features(self,x1,x2):


        t = [block(x2) for block in self.backbone2.blocks]
  


         # Process x1 through backbone1, combining with t if index > 1
        for i, block1 in enumerate(self.backbone1.blocks):
            if i > 1:
            # Normalize and combine x1 with corresponding t
                #x2 =   t[i - 1] / t[i - 1].max()  # Normalize t[i-1] for stability
                x1 = block1(x1,t[i-1])
            else:
                x1 = block1(x1)

        return x1
    

    def encode_input(self, x: Tensor) -> Tensor:
        """
        Returns without running backbone!, patch + pos embedding
        """
        emb1 = self.patch_embedding1(x)
        emb1 = self.pos_embedding1(emb1)

        emb2 = self.patch_embedding2(x)
        emb2 = self.pos_embedding2(emb2)

        return emb1,emb2
    
    def forward(self, x: Tensor) -> Tensor:
        emb1,emb2 = self.encode_input(x)
        emb = self.forward_features(emb1,emb2)
        return emb

    def extract_features(self, x: Tensor, seq_mean_pool: bool) -> Tensor:
        x = self.patch_embedding(x)
        x = self.pos_embedding(x)
        x = self.backbone.blocks(x)
        if seq_mean_pool:
            x = x.mean(dim=1)
        x = self.backbone.norm(x)
        return x
    
    def freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = True

    def get_device(self):
        return next(self.parameters()).device

    def load_weights(self, path):
        self.load_state_dict(torch.load(path, map_location=self.get_device()))
