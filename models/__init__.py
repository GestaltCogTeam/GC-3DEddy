from typing import Tuple
import torch.nn as nn

from .quant import VectorQuantizer2
from .tat import TAT
from .vqvae import VQVAE


def build_vae_tat(
    # Shared args
    device, patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
    # VQVAE args
    V=4096, Cvae=32, ch=160, share_quant_resi=4,
    # TAT args
    depth=16, shared_aln=False, attn_l2_norm=True,
    flash_if_available=True, fused_if_available=True,
    init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1,    # init_std < 0: automated
    vae_if_train=False,
    time_patch_num=1,
    patch_size=2,
    data_load_channel=7,
    data_load_reso=36,
    vae_in_channels=25,
) -> Tuple[VQVAE, TAT]:
    heads = depth
    width = depth * 64
    dpr = 0.1 * depth/24
    
    # disable built-in initialization for speed
    if not vae_if_train:
        for clz in (nn.Linear, nn.LayerNorm, nn.BatchNorm2d, nn.SyncBatchNorm, nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d):
            setattr(clz, 'reset_parameters', lambda self: None)
    
    # build models
    vae_local = VQVAE(vocab_size=V, z_channels=Cvae, ch=ch, test_mode=(not vae_if_train), share_quant_resi=share_quant_resi, v_patch_nums=patch_nums, patch_size=patch_size, vae_in_channels=vae_in_channels).to(device)
    if vae_if_train:
        return vae_local, None
    tat_wo_ddp = TAT(
        vae_local=vae_local,
        depth=depth, embed_dim=width, num_heads=heads, drop_rate=0., attn_drop_rate=0., drop_path_rate=dpr,
        norm_eps=1e-6, shared_aln=shared_aln, cond_drop_rate=0.1,
        attn_l2_norm=attn_l2_norm,
        patch_nums=patch_nums,
        flash_if_available=flash_if_available, fused_if_available=fused_if_available,
        data_load_channel=data_load_channel, data_load_reso=data_load_reso, time_patch_num=time_patch_num,
    ).to(device)
    tat_wo_ddp.init_weights(init_adaln=init_adaln, init_adaln_gamma=init_adaln_gamma, init_head=init_head, init_std=init_std)
    
    return vae_local, tat_wo_ddp
