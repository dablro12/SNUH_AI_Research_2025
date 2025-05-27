# import torch.nn as nn 
# import torch 
# import torch.nn.functional as F
# class PatchEmbedding(nn.Module) :
#     def __init__(self, in_channels = 1, embed_dim = 96, patch_size = 4) :
#         super().__init__()
#         self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size = patch_size, stride = patch_size)

#     def forward(self, x):
#         return self.proj(x)

# class PatchExpand(nn.Module) :
#     def __init__(self, in_channels, out_channels) :
#         super().__init__()
#         self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 2, stride = 2)

#     def forward(self, x) :
#         return self.up(x)

# class SwinBlock(nn.Module) :
#     def __init__(self, dim) :
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = nn.MultiheadAttention(dim, num_heads = 4, batch_first = True)
#         self.norm2 = nn.LayerNorm(dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, dim * 4),
#             nn.GELU(),
#             nn.Linear(dim * 4, dim),
#         )

#     def forward(self, x) :
#         B, C, H, W = x.shape
#         x = x.flatten(2).transpose(1, 2)
#         x = self.norm1(x)
#         attn_output, _ = self.attn(x, x, x)
#         x = x + attn_output
#         x = self.norm2(x)
#         x = x + self.mlp(x)
#         x = x.transpose(1, 2).reshape(B, C, H, W)
        
#         return x

# class SwinUNet(nn.Module) :
#     def __init__(self, in_channels = 3, out_channels = 3, base_dim = 96):
#         super().__init__()
#         self.patch_embed = PatchEmbedding(in_channels, embed_dim = base_dim)

#         self.encoder1 = SwinBlock(base_dim)
#         self.down1 = nn.Conv2d(base_dim, base_dim * 2, kernel_size = 2, stride = 2)
#         self.encoder2 = SwinBlock(base_dim * 2)
#         self.down2 = nn.Conv2d(base_dim * 2, base_dim * 4, kernel_size = 2, stride = 2)
#         self.bottleneck = SwinBlock(base_dim * 4)

#         self.up2 = PatchExpand(base_dim * 4, base_dim * 2)
#         self.decoder2 = SwinBlock(base_dim * 2)
#         self.up1 = PatchExpand(base_dim * 2, base_dim)
#         self.decoder1 = SwinBlock(base_dim)

#         self.final = nn.Conv2d(base_dim, out_channels, kernel_size = 1)

#     def forward(self, x) :
#         x = self.patch_embed(x)
#         e1 = self.encoder1(x)
#         e2 = self.encoder2(self.down1(e1))
#         b = self.bottleneck(self.down2(e2))

#         d2 = self.decoder2(self.up2(b) + e2)
#         d1 = self.decoder1(self.up1(d2) + e1)
#         out = self.final(d1)
        
#         return out
    
# def build_model(ckpt_path:str = None, device:str='cpu'):
#     model = SwinUNet(in_channels=1, out_channels=1, base_dim=64).to(device)
#     if ckpt_path is not None:
#         checkpoint = torch.load(ckpt_path, map_location = 'cuda')
#         model.load_state_dict(checkpoint)
#     return model