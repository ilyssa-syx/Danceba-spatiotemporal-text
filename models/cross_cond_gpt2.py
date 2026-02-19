
"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

from curses import window
import math
import logging
from pydoc import text

import torch
import torch.nn as nn
from torch.nn import functional as F

from .mlp import GatedMLP, Mlp
from .mamba_simple import Mamba
from .LPE import LPE_1

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class CrossCondGPT2(nn.Module):
    """  Danceba Pipeline  """
    def __init__(self, config):
        super().__init__()
        self.gpt_base = CrossCondGPTBase(config.base)
        self.gpt_head = CrossCondGPTHead(config.head)
        self.block_size = config.block_size

    def get_block_size(self):
        return self.block_size

    def sample(self, xs, cond, text_upper, text_lower, text_torso, text_whole, text_meta, shift=None):
        print("do sample!!!")
        block_size = self.get_block_size() - 1
        if shift is not None:
            block_shift = min(shift, block_size)
        else:
            block_shift = block_size
        x_up, x_down = xs
        for k in range(cond.size(1)):
            current_len = x_up.size(1)
            if current_len <= block_size:
                x_cond_up = x_up
                x_cond_down = x_down
                window_abs_start = 0
            else:
                offset = (
                    block_shift
                    + (k - block_size - 1) % (block_size - block_shift + 1)
                )
                window_start = -offset
                x_cond_up = x_up[:, window_start:]
                x_cond_down = x_down[:, window_start:]
                window_abs_start = current_len - offset
            # keep music the same
            cond_input = cond[:, :k+1] if k < block_size else cond[:, k-(block_shift+(k-block_size-1)%(block_size-block_shift+1))+1:k+1]
            logits, _ = self.forward(idxs=(x_cond_up, x_cond_down), music=cond_input, text_upper=text_upper, text_lower=text_lower, text_torso=text_torso, text_whole=text_whole, text_meta=text_meta, window_abs_start=window_abs_start, targets=None)
            logit_up, logit_down = logits
            logit_up = logit_up[:, -1, :]
            logit_down = logit_down[:, -1, :]

            probs_up = F.softmax(logit_up, dim=-1)
            probs_down = F.softmax(logit_down, dim=-1)

            _, ix_up = torch.topk(probs_up, k=1, dim=-1)
            _, ix_down = torch.topk(probs_down, k=1, dim=-1)

            # append to the sequence and continue
            x_up = torch.cat((x_up, ix_up), dim=1)
            x_down = torch.cat((x_down, ix_down), dim=1)

        return ([x_up], [x_down])

    def forward(self, idxs, music, text_upper, text_lower, text_torso, text_whole, text_meta, targets=None, window_abs_start=0): # cond: music condition ("music_seq[:, config.ds_rate//music_relative_rate:]")
        idx_up, idx_down = idxs
        
        targets_up, targets_down = None, None
        if targets is not None:
            targets_up, targets_down = targets
        b, N, text_dim = text_upper.size()
        T_motion = idx_up.size(1)
        # print("L98:",cond.shape)
        feat = self.gpt_base(idx_up, idx_down, music, text_upper, text_lower, text_torso, text_whole, text_meta, window_abs_start=window_abs_start)
        logits_up, logits_down, loss_up, loss_down = self.gpt_head(feat, text_meta=text_meta, T_motion=T_motion, N=N, targets=targets, window_abs_start=window_abs_start)
        # logits_down, loss_down = self.down_half_gpt(feat, targets_down)
        
        if loss_up is not None and loss_down is not None:
            loss = loss_up + loss_down
        else:
            loss = None

        return (logits_up, logits_down), loss


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CausalCrossConditionalSelfAttention(nn.Module):
    """
    优化版本的多头掩码自注意力层
    - 使用向量化操作替代Python循环，大幅提升GPU性能
    - 修复fallback逻辑，防止在包含文本特征时出现错误
    - 修复滑动窗口推理时的文本时间对齐问题 ⭐ NEW
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, text_meta=None, T_motion=None, N=None, window_abs_start=0):
        """
        Args:
            x: Input tensor
            text_meta: Text temporal metadata
            T_motion: Motion sequence length
            N: Number of text segments
            window_abs_start: 当前窗口在完整序列中的绝对起始帧位置 ⭐ KEY FIX
            return_attention: 是否返回attention weights
        """
        B, Total_Len, C = x.size()  # Total_Len = 3*T_motion + 4*N*1 (music, up, down, 4 text features)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, Total_Len, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, Total_Len, hs)
        q = self.query(x).view(B, Total_Len, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, Total_Len, hs)
        v = self.value(x).view(B, Total_Len, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, Total_Len, hs)
        
        # causal self-attention; Self-attend: (B, nh, Total_Len, hs) x (B, nh, hs, Total_Len) -> (B, nh, Total_Len, Total_Len)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Construct spatio-temporal mask
        if text_meta is not None and T_motion is not None and N is not None:
            T_motion = int(T_motion)
            N = int(N)
            
            # Create st_mask: shape (B, 1, Total_Len, Total_Len)
            # True = Forbid (will be filled with -inf), False = Allow
            st_mask = torch.zeros(B, 1, Total_Len, Total_Len, device=x.device, dtype=torch.bool)
            
            # Logic 1: Motion-to-Motion - Vectorized Causal Mask for 3-stream (music, up, down)
            motion_len = 3 * T_motion
            
            # 序列结构：[music_0...music_{T-1}, up_0...up_{T-1}, down_0...down_{T-1}]
            # 索引范围：music[0:T], up[T:2T], down[2T:3T]
            
            # 初始化：默认全部禁止
            st_mask[:, :, :motion_len, :motion_len] = True
            
            # 🚀 向量化实现：使用torch.tril一次性构建所有causal masks
            # 创建因果mask模板 (下三角矩阵，包括对角线)
            causal_template = torch.tril(torch.ones(T_motion, T_motion, device=x.device, dtype=torch.bool))
            
            # === Music stream 的因果关系 ===
            # music[t] 能看到 music[0], ..., music[t] (包括自己)
            st_mask[:, :, 0:T_motion, 0:T_motion] = ~causal_template
            
            # === Up stream 的因果关系 ===
            # up[t] 能看到：
            # - music[0], ..., music[t] (当前时刻的music)
            # - up[0], ..., up[t-1] (之前时刻的up，不包括自己)
            # - down[0], ..., down[t-1] (之前时刻的down)
            
            # Up -> Music: 使用完整的causal mask
            st_mask[:, :, T_motion:2*T_motion, 0:T_motion] = ~causal_template
            
            # Up -> Up: 使用严格下三角（不包括对角线）
            strictly_lower = torch.tril(torch.ones(T_motion, T_motion, device=x.device, dtype=torch.bool), diagonal=-1)
            st_mask[:, :, T_motion:2*T_motion, T_motion:2*T_motion] = ~strictly_lower
            
            # Up -> Down: 使用严格下三角
            st_mask[:, :, T_motion:2*T_motion, 2*T_motion:3*T_motion] = ~strictly_lower
            
            # === Down stream 的因果关系 ===
            # down[t] 能看到：
            # - music[0], ..., music[t] (当前时刻的music)
            # - up[0], ..., up[t] (当前时刻的up，包括up[t])
            # - down[0], ..., down[t-1] (之前时刻的down，不包括自己)
            
            # Down -> Music: 使用完整的causal mask
            st_mask[:, :, 2*T_motion:3*T_motion, 0:T_motion] = ~causal_template
            
            # Down -> Up: 使用完整的causal mask (包括对角线)
            st_mask[:, :, 2*T_motion:3*T_motion, T_motion:2*T_motion] = ~causal_template
            
            # Down -> Down: 使用严格下三角
            st_mask[:, :, 2*T_motion:3*T_motion, 2*T_motion:3*T_motion] = ~strictly_lower
            
            # Logic 2: Motion-to-Text - Default ALL FORBIDDEN
            st_mask[:, :, :motion_len, motion_len:] = True
            
            # Define text feature regions
            text_start = motion_len
            text_upper_start = text_start
            text_lower_start = text_start + N
            text_torso_start = text_start + 2 * N
            text_whole_start = text_start + 3 * N
            
            # 🚀 优化：使用向量化操作替代Python循环
            # Unlock specific regions based on text_meta
            for b in range(B):
                if text_meta[b] is None or len(text_meta[b]) == 0:
                    continue
                
                for i, meta in enumerate(text_meta[b]):
                    if i >= N:  # Safety check
                        break
                    
                    # ⭐⭐⭐ 核心修复：将绝对时间映射到当前窗口的相对时间 ⭐⭐⭐
                    # Meta里存储的是绝对帧号（相对于完整序列起始）
                    # 例如：meta['start_frame'] = 100, 表示从完整序列的第100帧开始
                    # 当前窗口从第80帧开始 (window_abs_start = 80)
                    # 那么在当前窗口里，这个文本段的相对位置是: 100 - 80 = 20
                    
                    # 🔥 重要：text_meta中的帧号需要除以8进行下采样对齐
                    # Motion数据已经下采样，但text_meta中的start_frame/end_frame是原始尺度
                    abs_start = meta['start_frame'] // 8
                    abs_end = meta['end_frame'] // 8
                    rel_start = abs_start - window_abs_start
                    rel_end = abs_end - window_abs_start
                    # print(f"Text segment {i}: abs_start={abs_start}, abs_end={abs_end}, rel_start={rel_start}, rel_end={rel_end}")
                    # Calculate token range for segment i in each text feature
                    seg_start = i
                    seg_end = i + 1
                    
                    # 裁剪到当前窗口范围 [0, T_motion)
                    # 如果文本段完全在窗口外，frame_start >= frame_end，不会设置mask
                    frame_start = max(0, rel_start)
                    frame_end = min(T_motion, rel_end)
                    
                    if frame_end > frame_start:
                        # Rule for Up Body (global indices T_motion to 2*T_motion)
                        # 使用切片一次性设置整个帧范围
                        up_slice = slice(T_motion + frame_start, T_motion + frame_end)
                        text_upper_slice = slice(text_upper_start + seg_start, text_upper_start + seg_end)
                        text_torso_slice = slice(text_torso_start + seg_start, text_torso_start + seg_end)
                        text_whole_slice = slice(text_whole_start + seg_start, text_whole_start + seg_end)
                        
                        # ALLOW: Text_Upper[i], Text_Torso[i], Text_Whole[i]
                        st_mask[b, 0, up_slice, text_upper_slice] = False
                        st_mask[b, 0, up_slice, text_torso_slice] = False
                        st_mask[b, 0, up_slice, text_whole_slice] = False
                        
                        # Rule for Down Body (global indices 2*T_motion to 3*T_motion)
                        down_slice = slice(2 * T_motion + frame_start, 2 * T_motion + frame_end)
                        text_lower_slice = slice(text_lower_start + seg_start, text_lower_start + seg_end)
                        
                        # ALLOW: Text_Lower[i], Text_Torso[i], Text_Whole[i]
                        st_mask[b, 0, down_slice, text_lower_slice] = False
                        st_mask[b, 0, down_slice, text_torso_slice] = False
                        st_mask[b, 0, down_slice, text_whole_slice] = False
            
            # Apply mask to attention scores
            att = att.masked_fill(st_mask, float('-inf'))
        else:
            # 🛡️ 改进的Fallback逻辑：只对motion部分应用causal mask
            # 假设没有text_meta时，要么是纯motion数据，要么应该mask掉所有文本
            
            # 检测是否可能包含文本特征
            if T_motion is not None:
                motion_len = 3 * int(T_motion)
                
                if Total_Len == motion_len:
                    # 纯motion数据：应用正确的3-stream causal mask (向量化版本)
                    st_mask = torch.zeros(1, 1, Total_Len, Total_Len, device=x.device, dtype=torch.bool)
                    st_mask[:, :, :, :] = True  # 默认全部禁止
                    
                    T = int(T_motion)
                    causal_template = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
                    strictly_lower = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=-1)
                    
                    # Music -> Music: causal
                    st_mask[:, :, 0:T, 0:T] = ~causal_template
                    # Up -> Music/Up/Down
                    st_mask[:, :, T:2*T, 0:T] = ~causal_template
                    st_mask[:, :, T:2*T, T:2*T] = ~strictly_lower
                    st_mask[:, :, T:2*T, 2*T:3*T] = ~strictly_lower
                    # Down -> Music/Up/Down
                    st_mask[:, :, 2*T:3*T, 0:T] = ~causal_template
                    st_mask[:, :, 2*T:3*T, T:2*T] = ~causal_template
                    st_mask[:, :, 2*T:3*T, 2*T:3*T] = ~strictly_lower
                    
                    att = att.masked_fill(st_mask, float('-inf'))
                elif Total_Len > motion_len:
                    # 包含文本特征：对motion应用causal mask，motion不能attend到文本 (向量化版本)
                    st_mask = torch.zeros(1, 1, Total_Len, Total_Len, device=x.device, dtype=torch.bool)
                    st_mask[:, :, :, :] = True  # 默认全部禁止
                    
                    # Motion部分的正确causal mask (向量化)
                    T = int(T_motion)
                    causal_template = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
                    strictly_lower = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=-1)
                    
                    st_mask[:, :, 0:T, 0:T] = ~causal_template
                    st_mask[:, :, T:2*T, 0:T] = ~causal_template
                    st_mask[:, :, T:2*T, T:2*T] = ~strictly_lower
                    st_mask[:, :, T:2*T, 2*T:3*T] = ~strictly_lower
                    st_mask[:, :, 2*T:3*T, 0:T] = ~causal_template
                    st_mask[:, :, 2*T:3*T, T:2*T] = ~causal_template
                    st_mask[:, :, 2*T:3*T, 2*T:3*T] = ~strictly_lower
                    
                    # 文本部分全部mask（没有text_meta就不知道如何处理）
                    # st_mask[:, :, motion_len:, :] = True  # 已经默认是True
                    
                    att = att.masked_fill(st_mask, float('-inf'))
                else:
                    raise ValueError(f"Total_Len ({Total_Len}) < expected motion_len ({motion_len})")
            else:
                # 完全不知道结构：假设是等分的3段motion (向量化版本)
                t = Total_Len // 3
                if Total_Len == t * 3:
                    st_mask = torch.zeros(1, 1, Total_Len, Total_Len, device=x.device, dtype=torch.bool)
                    st_mask[:, :, :, :] = True
                    
                    causal_template = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool))
                    strictly_lower = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=-1)
                    
                    st_mask[:, :, 0:t, 0:t] = ~causal_template
                    st_mask[:, :, t:2*t, 0:t] = ~causal_template
                    st_mask[:, :, t:2*t, t:2*t] = ~strictly_lower
                    st_mask[:, :, t:2*t, 2*t:3*t] = ~strictly_lower
                    st_mask[:, :, 2*t:3*t, 0:t] = ~causal_template
                    st_mask[:, :, 2*t:3*t, t:2*t] = ~causal_template
                    st_mask[:, :, 2*t:3*t, 2*t:3*t] = ~strictly_lower
                    
                    att = att.masked_fill(st_mask, float('-inf'))
                else:
                    raise ValueError(
                        f"Cannot infer structure: Total_Len={Total_Len}, T_motion={T_motion}, N={N}. "
                        "Please provide text_meta, T_motion, and N for sequences with text features."
                    )
        
        att = F.softmax(att, dim=-1)
        
        
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, Total_Len, Total_Len) x (B, nh, Total_Len, hs) -> (B, nh, Total_Len, hs)
        y = y.transpose(1, 2).contiguous().view(B, Total_Len, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        
        return y


class Block_Base(nn.Module):
    """ an Temporal-Gated Causal Attention (TGCA) block """

    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd)
        self.norm2 = RMSNorm(config.n_embd)
        self.in_proj = nn.Linear(config.n_embd, config.n_embd)
        self.act_proj = nn.Linear(config.n_embd, config.n_embd)
        self.act = nn.SiLU()
        self.sigmoid =  nn.Sigmoid()
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn = CausalCrossConditionalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )


    def forward(self, x, text_meta=None, T=None, N=None, window_abs_start=0):
        shortcut = x
        x = self.norm1(x)
        act_res = self.act(self.act_proj(x)) 
        # SiLU # {'fid_k': (12.918347226610166-6.601375950773499e-07j), 'fid_g': (14.377040083933046-2.347936684761542e-08j), 'div_k': 8.241817106803259, 'div_g': 7.440401058930617} 0.2896913823938913
        # act_res = self.sigmoid(self.act_proj(x)) 
        # Sigmoid # {'fid_k': (22.10337116047633-4.1560030123329423e-07j), 'fid_g': 12.051192377432386, 'div_k': 7.663574515092067, 'div_g': 6.547586472982015} 0.2630817338399422
        x = self.in_proj(x)
        x = self.act(x)
        x = self.attn(x, text_meta=text_meta, T_motion=T, N=N, window_abs_start=window_abs_start)
        x = self.out_proj(x * act_res)
        x = shortcut + x

        x = x + self.mlp(self.norm2(x))
        return x


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class SMR(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, linear = False):
        super(SMR, self).__init__()
        self.conv = nn.Conv1d(in_features, out_features, kernel_size, stride=1)
        self.use_linear = linear
        if linear:
            self.linear = nn.Linear(in_features, out_features)
        self.pad = (kernel_size- 1, 0)
    def forward(self, x):
        # Input shape: (B, H, L)
        # Output shape: (B, H, L)
        if self.use_linear:
            factor = self.linear(self.conv(F.pad(x, self.pad, mode='constant',value=0.0)).transpose(1, 2)).transpose(1, 2)
        else:
            factor = self.conv(F.pad(x, self.pad, mode='constant', value=0.0))
        return torch.sigmoid(factor) * x

class Block_Head(nn.Module):
    """ an Parallel Mamba block """

    def __init__(self, config):
        super().__init__()
        self.norm_music_1 = RMSNorm(config.n_embd)
        self.norm_music_2 = RMSNorm(config.n_embd)
        self.norm_up_1 = RMSNorm(config.n_embd)
        self.norm_up_2 = RMSNorm(config.n_embd)
        self.norm_down_1 = RMSNorm(config.n_embd)
        self.norm_down_2 = RMSNorm(config.n_embd)
        self.norm_text_upper = RMSNorm(config.n_embd)
        self.norm_text_lower = RMSNorm(config.n_embd)
        self.norm_text_torso = RMSNorm(config.n_embd)
        self.norm_text_whole = RMSNorm(config.n_embd)
        self.mamba_music = Mamba(
            d_model=config.n_embd,
            d_state=128,
            d_conv=4,
            expand=4,
        )
        self.mamba_up = Mamba(
            d_model=config.n_embd,
            d_state=128,
            d_conv=4,
            expand=4,
        )
        self.mamba_down = Mamba(
            d_model=config.n_embd,
            d_state=128,
            d_conv=4,
            expand=4,
        )
            
        self.gate_mlp_1 = GatedMLP(
            in_features=config.n_embd,
            hidden_features=768,
            out_features=config.n_embd,
        )
        self.gate_mlp_2 = GatedMLP(
            in_features=config.n_embd,
            hidden_features=768,
            out_features=config.n_embd,
        )
        self.gate_mlp_3 = GatedMLP(
            in_features=config.n_embd,
            hidden_features=768,
            out_features=config.n_embd,
        )
        self.text_mlp = GatedMLP(
            in_features=config.n_embd,
            hidden_features=config.n_embd*4,
            out_features=config.n_embd,
        )

    def forward(self, x, T_motion, text_meta):
        t = T_motion
        B, Total_Len, D = x.shape
        
        # Calculate N and sequence boundaries
        text_total_len = Total_Len - 3 * t
        N = text_total_len // 4
        text_segment_len = N
        motion_len = 3 * t
        
        music = x[:, :t, :]
        up = x[:, t:2*t, :]
        down = x[:, 2*t:, :]
        text_upper = x[:, motion_len:motion_len+text_segment_len, :]
        text_lower = x[:, motion_len+text_segment_len:motion_len+2*text_segment_len, :]
        text_torso = x[:, motion_len+2*text_segment_len:motion_len+3*text_segment_len, :]
        text_whole = x[:, motion_len+3*text_segment_len:motion_len+4*text_segment_len, :]

        music = music + self.mamba_music(self.norm_music_1(music))
        music = music + self.gate_mlp_1(self.norm_music_2(music))

        up = up + self.mamba_up(self.norm_up_1(up))
        up = up + self.gate_mlp_2(self.norm_up_2(up))

        down = down + self.mamba_down(self.norm_down_1(down))
        down = down + self.gate_mlp_3(self.norm_down_2(down))

        text_upper = text_upper + self.text_mlp(self.norm_text_upper(text_upper))
        text_lower = text_lower + self.text_mlp(self.norm_text_lower(text_lower))
        text_torso = text_torso + self.text_mlp(self.norm_text_torso(text_torso))
        text_whole = text_whole + self.text_mlp(self.norm_text_whole(text_whole))

        x = torch.cat([music, up, down, text_upper, text_lower, text_torso, text_whole], dim=1)

        return x

class CrossCondGPTBase(nn.Module):
    """  the Global Beat Attention via Temporal Gating in Sec 3.3 """

    def __init__(self, config):
        super().__init__()

        self.tok_emb_up = nn.Embedding(config.vocab_size_up, config.n_embd  )
        self.tok_emb_down = nn.Embedding(config.vocab_size_down, config.n_embd)
        """  Phase-Based Rhythm Feature Extraction in Sec 3.2  """
        self.pos_emb = LPE_1(config)
        self.position_scale = nn.Parameter(torch.tensor(1e-6))
        self.music_cond_emb = nn.Linear(config.n_music, config.n_embd)
        self.text_cond_emb = nn.Linear(config.n_text, config.n_embd)
        self.text_modality_emb = nn.Parameter(torch.randn(1, 1, config.n_embd) * 0.02)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.Sequential(*[Block_Base(config) for _ in range(config.n_layer)])
        self.block_size = config.block_size
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # module.weight.data.uniform_(math.sqrt(6.0/sum(module.weight.size())))
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, RMSNorm)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                if 'text_modality_emb' in pn or 'position_scale' in pn or 'learnable_padding_token' in pn:
                    no_decay.add(fpn)
                    continue
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        missing_keys = param_dict.keys() - union_params
        assert len(missing_keys) == 0, f"Critical Error: parameters {str(missing_keys)} were not separated into either decay/no_decay set!"                                            % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx_up, idx_down, music, text_upper, text_lower, text_torso, text_whole, text_meta, window_abs_start=0):
        b, t = idx_up.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        b, t = idx_down.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        token_embeddings_up = self.tok_emb_up(idx_up)  # each index maps to a (learnable) vector
        token_embeddings_down = self.tok_emb_down(idx_down)  # each index maps to a (learnable) vector
        
        # ✅ 修正Bug：使用 self.music_cond_emb 而不是 self.cond_emb
        token_embeddings = torch.cat([self.music_cond_emb(music), token_embeddings_up, token_embeddings_down], dim=1)
        
        position_embeddings = self.pos_emb(music)
        # 注意这里是3个并列的lpe预测然后concatenate。
        pos_size = token_embeddings.shape[1]
        position_embeddings = position_embeddings[:, :pos_size, :]
        position_embeddings = self.position_scale * position_embeddings
        token_embeddings = token_embeddings + position_embeddings
        
        # deal with text modality
        # reshape text conditions from (32, N, 1, 512) to (32, N*1, 512)
        N = text_upper.size(1)
        text_upper = text_upper.view(b, -1, text_upper.size(-1))  # (b, len, dim)
        text_lower = text_lower.view(b, -1, text_lower.size(-1))  # (b, len, dim)
        text_torso = text_torso.view(b, -1, text_torso.size(-1))  # (b, len, dim)
        text_whole = text_whole.view(b, -1, text_whole.size(-1))  # (b, len, dim)
        text_cond = torch.cat([text_upper, text_lower, text_torso, text_whole], dim=1)  # (b, total_len, dim)
        text_cond_emb = self.text_cond_emb(text_cond)  # (b, total_len, n_embd)
        text_modality_emb = self.text_modality_emb.repeat(1, text_cond_emb.size(1), 1)  # (1, total_len, n_embd)
        text_cond_emb = text_cond_emb + text_modality_emb

        final_embedding = torch.cat([token_embeddings, text_cond_emb], dim=1)
        x = self.drop(final_embedding)

        T_motion = t  # Length of each motion segment (music/up/down)
        
        
        for i, block in enumerate(self.blocks):
            block_output = block(x, text_meta=text_meta, T=T_motion, N=N, window_abs_start=window_abs_start)
            
            x = block_output

        return x

        
class CrossCondGPTHead(nn.Module):
    """  the Mamba-Based Parallel Motion Modeling in Sec 3.4 """

    def __init__(self, config):
        super().__init__()

        self.blocks = nn.Sequential(*[Block_Head(config) for _ in range(config.n_layer)])
        self.block_base = Block_Base(config)
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.RMS_f = RMSNorm(config.n_embd)
        self.block_size = config.block_size
        self.head_up = nn.Linear(config.n_embd, config.vocab_size_up, bias=False)
        self.head_down = nn.Linear(config.n_embd, config.vocab_size_down, bias=False)
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        
        # 【修复】加入 Conv1d 支持 Mamba
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d)
        
        # 【修复】显式加入自定义 RMSNorm 类
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, RMSNorm)
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                # 【修复】优先捕获独立的 Parameter (text_modality_emb, position_scale, learnable_padding_token)
                # 只要名字里包含这些特定的词，直接进入 no_decay (不减肥)
                if 'text_modality_emb' in pn or 'position_scale' in pn or 'learnable_padding_token' in pn:
                    no_decay.add(fpn)
                    continue  # 处理完直接跳过，防止后面逻辑干扰

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        
        # 如果还是报错，打印出来看看到底是谁漏了
        missing_keys = param_dict.keys() - union_params
        assert len(missing_keys) == 0, f"Critical Error: parameters {str(missing_keys)} were not separated into either decay/no_decay set!"

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, x, text_meta=None, T_motion=None, N=None, targets=None, window_abs_start=0):
        """
        Args:
            window_abs_start: 窗口绝对起始帧位置（传递给最后的block_base）
            return_attention: 是否返回attention weights
        """
        for block in self.blocks:
            x = block(x, T_motion=T_motion, text_meta=text_meta)
        
        # [修改] 传递 window_abs_start 和 return_attention 给 block_base
        block_base_output = self.block_base(x, text_meta=text_meta, T=T_motion, N=N, window_abs_start=window_abs_start)
        
        x = block_base_output
        
        x = self.RMS_f(x)
        logits_up = self.head_up(x[:, T_motion:T_motion*2, :])
        logits_down = self.head_down(x[:, T_motion*2:T_motion*3, :])  # down half

        loss_up, loss_down = None, None

        if targets is not None:
            targets_up, targets_down = targets

            loss_up = F.cross_entropy(logits_up.view(-1, logits_up.size(-1)), targets_up.view(-1))
            loss_down = F.cross_entropy(logits_down.view(-1, logits_down.size(-1)), targets_down.view(-1))

        return logits_up, logits_down, loss_up, loss_down