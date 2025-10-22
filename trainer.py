import time
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import torch.nn.functional as F

import dist
from models import TAT, VQVAE, VectorQuantizer2
from utils.amp_sc import AmpOptimizer
from utils.misc import MetricLogger, TensorboardLogger
from utils.metrics import masked_mse_loss

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor


class TATTrainer(object):
    def __init__(
        self, device, patch_nums: Tuple[int, ...], resos: Tuple[int, ...],
        vae_local: VQVAE, tat_wo_ddp: TAT, tat: DDP,
        tat_opt: AmpOptimizer, label_smooth: float,
        time_patch_num: int = 1,
    ):
        super(TATTrainer, self).__init__()
        
        self.tat, self.vae_local, self.quantize_local = tat, vae_local, vae_local.quantize
        self.quantize_local: VectorQuantizer2
        self.tat_wo_ddp: TAT = tat_wo_ddp  # after torch.compile
        self.tat_opt = tat_opt
        
        del self.tat_wo_ddp.rng
        self.tat_wo_ddp.rng = torch.Generator(device=device)
        
        self.label_smooth = label_smooth
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean')
        self.L = sum(pn * pn for pn in patch_nums)
        self.last_l = patch_nums[-1] * patch_nums[-1]
        self.loss_weight = torch.ones(1, self.L*time_patch_num, device=device) / (self.L*time_patch_num)
        
        self.patch_nums, self.resos = patch_nums, resos
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(patch_nums):
            self.begin_ends.append((cur, cur + pn * pn))
            cur += pn*pn
        
        self.prog_it = 0
        self.last_prog_si = -1
        self.first_prog = True
    
    @torch.no_grad()
    def eval_ep(self, ld_val: DataLoader):
        tot = 0
        L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
        stt = time.time()
        training = self.tat_wo_ddp.training
        self.tat_wo_ddp.eval()
        for inp_BCHW, surface_BCHW in ld_val:
            B, P, C, H, W = inp_BCHW.shape
            V = self.vae_local.vocab_size
            inp_BCHW = inp_BCHW.to(dist.get_device(), non_blocking=True)
            surface_BCHW = surface_BCHW.to(dist.get_device(), non_blocking=True)
            inp_BCHW_wo_nan = torch.nan_to_num(inp_BCHW)
            surface_BCHW_wo_nan = torch.nan_to_num(surface_BCHW)
            
            gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_BCHW_wo_nan.view(-1, C, H, W))
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            gt_BL = gt_BL.view(B, -1)
            x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_tat_input(gt_idx_Bl)
            _, L_wo_first_l, Cv = x_BLCv_wo_first_l.shape
            x_BLCv_wo_first_l = x_BLCv_wo_first_l.view(B, P, L_wo_first_l, Cv)
            
            self.tat_wo_ddp.forward
            logits_BLV = self.tat_wo_ddp(surface_BCHW_wo_nan, x_BLCv_wo_first_l)
            L_mean += self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)) * B
            #L_tail += self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V), gt_BL[:, -self.last_l:].reshape(-1)) * B
            cur_L_tail = 0
            cur_acc_tail = 0
            for p in range(P):
                p_start = p * self.L + self.L - self.last_l
                p_end = p_start + self.last_l
                cur_L_tail += self.val_loss(logits_BLV.data[:, p_start:p_end].reshape(-1, V), gt_BL[:, p_start:p_end].reshape(-1))
                cur_acc_tail += (logits_BLV.data[:, p_start:p_end].argmax(dim=-1) == gt_BL[:, p_start:p_end]).sum() * (100 / self.last_l)
            L_tail += cur_L_tail / P * B
            acc_mean += (logits_BLV.data.argmax(dim=-1) == gt_BL).sum() * (100/gt_BL.shape[1])
            #acc_tail += (logits_BLV.data[:, -self.last_l:].argmax(dim=-1) == gt_BL[:, -self.last_l:]).sum() * (100 / self.last_l)
            acc_tail += cur_acc_tail / P
            tot += B
        self.tat_wo_ddp.train(training)
        
        stats = L_mean.new_tensor([L_mean.item(), L_tail.item(), acc_mean.item(), acc_tail.item(), tot])
        dist.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= tot
        L_mean, L_tail, acc_mean, acc_tail, _ = stats.tolist()
        return L_mean, L_tail, acc_mean, acc_tail, tot, time.time()-stt
    
    def train_step(
        self, it: int, g_it: int, stepping: bool, metric_lg: MetricLogger, tb_lg: TensorboardLogger,
        inp_BCHW: FTen, surface_BCHW: Union[ITen, FTen], prog_si: int, prog_wp_it: float,
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        # if progressive training
        self.tat_wo_ddp.prog_si = self.vae_local.quantize.prog_si = prog_si
        if self.last_prog_si != prog_si:
            if self.last_prog_si != -1: self.first_prog = False
            self.last_prog_si = prog_si
            self.prog_it = 0
        self.prog_it += 1
        prog_wp = max(min(self.prog_it / prog_wp_it, 1), 0.01)
        if self.first_prog: prog_wp = 1    # no prog warmup at first prog stage, as it's already solved in wp
        if prog_si == len(self.patch_nums) - 1: prog_si = -1    # max prog, as if no prog
        
        # forward
        B, P, C, H, W = inp_BCHW.shape
        V = self.vae_local.vocab_size
        self.tat.require_backward_grad_sync = stepping
        
        inp_BCHW_wo_nan = torch.nan_to_num(inp_BCHW)
        surface_BCHW_wo_nan = torch.nan_to_num(surface_BCHW)
        gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_BCHW_wo_nan.view(-1, C, H, W))
        gt_BL = torch.cat(gt_idx_Bl, dim=1)
        gt_BL = gt_BL.view(B, -1)
        x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_tat_input(gt_idx_Bl)
        _, L_wo_first_l, Cv = x_BLCv_wo_first_l.shape
        x_BLCv_wo_first_l = x_BLCv_wo_first_l.view(B, P, L_wo_first_l, Cv)
        
        with self.tat_opt.amp_ctx:
            self.tat_wo_ddp.forward
            logits_BLV = self.tat(surface_BCHW_wo_nan, x_BLCv_wo_first_l)
            loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1)
            if prog_si >= 0:    # in progressive training
                bg, ed = self.begin_ends[prog_si]
                assert logits_BLV.shape[1] == gt_BL.shape[1] == ed
                lw = self.loss_weight[:, :ed].clone()
                lw[:, bg:ed] *= min(max(prog_wp, 0), 1)
            else:               # not in progressive training
                lw = self.loss_weight
            loss = loss.mul(lw).sum(dim=-1).mean()
        
        # backward
        grad_norm, scale_log2 = self.tat_opt.backward_clip_step(loss=loss, stepping=stepping)
        
        # log
        pred_BL = logits_BLV.data.argmax(dim=-1)
        if it == 0 or it in metric_lg.log_iters:
            Lmean = self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)).item()
            acc_mean = (pred_BL == gt_BL).float().mean().item() * 100
            if prog_si >= 0:    # in progressive training
                Ltail = acc_tail = -1
            else:               # not in progressive training
                #Ltail = self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V), gt_BL[:, -self.last_l:].reshape(-1)).item()
                #acc_tail = (pred_BL[:, -self.last_l:] == gt_BL[:, -self.last_l:]).float().mean().item() * 100
                cur_Ltail = 0
                cur_acc_tail = 0
                for p in range(P):
                    p_start = p * self.L + self.L - self.last_l
                    p_end = p_start + self.last_l
                    cur_Ltail += self.val_loss(logits_BLV.data[:, p_start:p_end].reshape(-1, V), gt_BL[:, p_start:p_end].reshape(-1)).item()
                    cur_acc_tail += (pred_BL[:, p_start:p_end] == gt_BL[:, p_start:p_end]).float().mean().item() * 100
                Ltail = cur_Ltail / P
                acc_tail = cur_acc_tail / P
            grad_norm = grad_norm.item()
            metric_lg.update(Lm=Lmean, Lt=Ltail, Accm=acc_mean, Acct=acc_tail, tnm=grad_norm)
        
        # log to tensorboard
        if g_it == 0 or (g_it + 1) % 500 == 0:
            prob_per_class_is_chosen = pred_BL.view(-1).bincount(minlength=V).float()
            dist.allreduce(prob_per_class_is_chosen)
            prob_per_class_is_chosen /= prob_per_class_is_chosen.sum()
            cluster_usage = (prob_per_class_is_chosen > 0.001 / V).float().mean().item() * 100
            if dist.is_master():
                if g_it == 0:
                    tb_lg.update(head='AR_iter_loss', z_voc_usage=cluster_usage, step=-10000)
                    tb_lg.update(head='AR_iter_loss', z_voc_usage=cluster_usage, step=-1000)
                kw = dict(z_voc_usage=cluster_usage)
                for si, (bg, ed) in enumerate(self.begin_ends):
                    if 0 <= prog_si < si: break
                    pred, tar = logits_BLV.data[:, bg:ed].reshape(-1, V), gt_BL[:, bg:ed].reshape(-1)
                    acc = (pred.argmax(dim=-1) == tar).float().mean().item() * 100
                    ce = self.val_loss(pred, tar).item()
                    kw[f'acc_{self.resos[si]}'] = acc
                    kw[f'L_{self.resos[si]}'] = ce
                tb_lg.update(head='AR_iter_loss', **kw, step=g_it)
                tb_lg.update(head='AR_iter_schedule', prog_a_reso=self.resos[prog_si], prog_si=prog_si, prog_wp=prog_wp, step=g_it)
        
        self.tat_wo_ddp.prog_si = self.vae_local.quantize.prog_si = -1
        return grad_norm, scale_log2
    
    def get_config(self):
        return {
            'patch_nums':   self.patch_nums, 'resos': self.resos,
            'label_smooth': self.label_smooth,
            'prog_it':      self.prog_it, 'last_prog_si': self.last_prog_si, 'first_prog': self.first_prog,
        }
    
    def state_dict(self):
        state = {'config': self.get_config()}
        for k in ('tat_wo_ddp', 'vae_local', 'tat_opt'):
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                state[k] = m.state_dict()
        return state
    
    def load_state_dict(self, state, strict=True, skip_vae=False):
        for k in ('tat_wo_ddp', 'vae_local', 'tat_opt'):
            if skip_vae and 'vae' in k: continue
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                ret = m.load_state_dict(state[k], strict=strict)
                if ret is not None:
                    missing, unexpected = ret
                    print(f'[TATTrainer.load_state_dict] {k} missing:  {missing}')
                    print(f'[TATTrainer.load_state_dict] {k} unexpected:  {unexpected}')
        
        config: dict = state.pop('config', None)
        self.prog_it = config.get('prog_it', 0)
        self.last_prog_si = config.get('last_prog_si', -1)
        self.first_prog = config.get('first_prog', True)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[TAT.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict: raise AttributeError(err)
                    else: print(err)

class VAETrainer(object):
    def __init__(
        self, device, patch_nums: Tuple[int, ...], resos: Tuple[int, ...],
        vae_local: VQVAE, vae: DDP,
        vae_opt: AmpOptimizer,
    ):
        super(VAETrainer, self).__init__()

        self.vae, self.vae_local, self.quantize_local = vae, vae_local, vae_local.quantize
        self.quantize_local: VectorQuantizer2
        self.vae_opt = vae_opt

        self.L = sum(pn * pn for pn in patch_nums)
        self.last_l = patch_nums[-1] * patch_nums[-1]
        self.loss_weight = torch.ones(1, self.L, device=device) / self.L

    @torch.no_grad()
    def eval_ep(self, ld_val: DataLoader):
        tot = 0
        stt = time.time()
        R_loss, VQ_loss = 0, 0
        training = self.vae_local.training
        self.vae_local.eval()
        for inp_BCHW, _ in ld_val:
            B, C = inp_BCHW.shape[0], inp_BCHW.shape[2]
            inp_BCHW = inp_BCHW.squeeze(1)
            inp_BCHW = inp_BCHW.to(dist.get_device(), non_blocking=True)
            inp_BCHW_wo_nan = torch.nan_to_num(inp_BCHW)

            self.vae_local.forward
            reconstructed_B3HW, _, vq_loss = self.vae_local(inp_BCHW_wo_nan)
            VQ_loss += vq_loss * B
            #R_loss += F.mse_loss(reconstructed_B3HW, inp_BCHW, reduction='sum')
            R_loss += masked_mse_loss(reconstructed_B3HW, inp_BCHW)
            tot += B

        self.vae_local.train(training)
        stats = R_loss.new_tensor([R_loss.item(), VQ_loss.item(), tot])
        dist.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= tot

        R_loss, VQ_loss, _ = stats.tolist()
        return R_loss, VQ_loss, tot, time.time()-stt

    def train_step(
        self, it: int, g_it: int, stepping: bool, metric_lg: MetricLogger, tb_lg: TensorboardLogger,
        inp_BCHW: FTen,
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:

        # forward
        B, C = inp_BCHW.shape[0], inp_BCHW.shape[2]
        self.vae.require_backward_grad_sync = stepping
        inp_BCHW = inp_BCHW.squeeze(1)
        inp_BCHW_wo_nan = torch.nan_to_num(inp_BCHW)

        with self.vae_opt.amp_ctx:
            self.vae_local.forward
            reconstructed_B3HW, _, VQ_loss = self.vae(inp_BCHW_wo_nan)
            #R_loss = F.mse_loss(reconstructed_B3HW, inp_BCHW, reduction='sum') * (1. / B)
            R_loss = masked_mse_loss(reconstructed_B3HW, inp_BCHW) * (1. / B)
            loss = R_loss + VQ_loss * 0.2

        # backward
        grad_norm, scale_log2 = self.vae_opt.backward_clip_step(loss=loss, stepping=stepping)

        # log
        if it == 0 or it in metric_lg.log_iters:
            grad_norm = grad_norm.item()
            metric_lg.update(R_loss=R_loss, VQ_loss=VQ_loss, tnm=grad_norm)

        # log to tensorboard
        if g_it == 0 or (g_it + 1) % 500 == 0:
           if dist.is_master():
                kw = {'R_loss': R_loss.item(), 'VQ_loss': VQ_loss.item()}
                tb_lg.update(head='VAE_iter_loss', **kw, step=g_it)

        return grad_norm, scale_log2

    def get_config(self):
        return {
        }

    def state_dict(self):
        state = {'config': self.get_config()}
        for k in ('vae_local', 'vae_opt'):
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                state[k] = m.state_dict()
        return state

    def load_state_dict(self, state, strict=True, skip_vae=False):
        for k in ('vae_local', 'vae_opt'):
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                ret = m.load_state_dict(state[k], strict=strict)
                if ret is not None:
                    missing, unexpected = ret
                    print(f'[VAETrainer.load_state_dict] {k} missing:  {missing}')
                    print(f'[VAETrainer.load_state_dict] {k} unexpected:  {unexpected}')
