"""
SmartOTFS 性能分析脚本 —— 合并版

对最新版七组注意力/位置编码配置进行全面性能分析：
  1. 时域互相关 + PAPR CCDF + DD 域误差
  2. 时域 MSE / EVM / PSD error 指标
  3. DD 域探针能量泄漏分析 (3D/2D)
  4. AWGN / EVA / WATER 信道 BER 仿真
  5. EVA / WATER 信道矩阵缓存复用

分析配置与 SmartOTFS_20260826.py 保持一致：
  Config_A: Pos=Standard  + Attn=Standard
  Config_B: Pos=Standard  + Attn=GQA
  Config_C: Pos=Standard  + Attn=Differential
  Config_D: Pos=Standard  + Attn=MLA
  Config_E: Pos=Standard  + Attn=Dual_Axis
  Config_F: Pos=Standard  + Attn=BoltAttention
  Config_G: Pos=PhaseLoom + Attn=BoltAttention

运行方式：python analyze_performance_20260827.py
"""

import os
import sys
import math
import time
import pickle
import warnings
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.interpolate import RectBivariateSpline

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

# ======================= 全局配置 =======================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

DATA_VAL_PATH = os.path.join("DataSet", "val_data_4_QAM.pkl")


def find_existing_path(*candidates):
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


QAM_GEN_PATH = find_existing_path(
    os.path.join("training_models", "shared", "best_qam_generator.pth"),
    os.path.join("training_models", "best_qam_generator.pth"),
)


def config_weight(config_name, pos_name, attn_name):
    artifact_stem = f"{config_name}_Pos_{pos_name}_Attn_{attn_name}"
    config_dir = os.path.join("training_models", artifact_stem)
    return find_existing_path(
        os.path.join(config_dir, f"best_end2end_finetuned_{artifact_stem}.pth"),
    )

CONFIG_MAP = {
    "A": {
        "weight": config_weight("Config_A", "Standard", "Standard"),
        "pos_encoding": "Standard", "attn_type": "Standard",
        "label": "Config_A (Pos=Standard, Attn=Standard)",
        "short_label": "Config_A",
    },
    "B": {
        "weight": config_weight("Config_B", "Standard", "GQA"),
        "pos_encoding": "Standard", "attn_type": "GQA",
        "label": "Config_B (Pos=Standard, Attn=GQA)",
        "short_label": "Config_B",
    },
    "C": {
        "weight": config_weight("Config_C", "Standard", "Differential"),
        "pos_encoding": "Standard", "attn_type": "Differential",
        "label": "Config_C (Pos=Standard, Attn=Differential)",
        "short_label": "Config_C",
    },
    "D": {
        "weight": config_weight("Config_D", "Standard", "MLA"),
        "pos_encoding": "Standard", "attn_type": "MLA",
        "label": "Config_D (Pos=Standard, Attn=MLA)",
        "short_label": "Config_D",
    },
    "E": {
        "weight": config_weight("Config_E", "Standard", "Dual_Axis"),
        "pos_encoding": "Standard", "attn_type": "Dual_Axis",
        "label": "Config_E (Pos=Standard, Attn=Dual_Axis)",
        "short_label": "Config_E",
    },
    "F": {
        "weight": config_weight("Config_F", "Standard", "BoltAttention"),
        "pos_encoding": "Standard", "attn_type": "BoltAttention",
        "label": "Config_F (Pos=Standard, Attn=BoltAttention)",
        "short_label": "Config_F",
    },
    "G": {
        "weight": config_weight("Config_G", "PhaseLoom", "BoltAttention"),
        "pos_encoding": "PhaseLoom", "attn_type": "BoltAttention",
        "label": "Config_G (Pos=PhaseLoom, Attn=BoltAttention)",
        "short_label": "Config_G",
    },
}

ANALYSIS_CONFIGS = ["A", "B", "C", "D", "E", "F", "G"]
PRIMARY_CONFIG = "G"
CONFIG_COLORS = {
    "A": "#1f77b4", "B": "#9467bd", "C": "#8c564b", "D": "#17becf",
    "E": "#ff7f0e", "F": "#2ca02c", "G": "#d62728",
}
CONFIG_MARKERS = {
    "A": "o-", "B": "v-", "C": "P-", "D": "X-",
    "E": "D-", "F": "s-", "G": "^--",
}
SAVE_DD_ERROR_HEATMAP_FIG = False
SAVE_EQUIV_BASIS_3D_FIG = False
SAVE_EQUIV_BASIS_2D_FIG = False


def active_config_labels(results=None):
    if results is None:
        return ANALYSIS_CONFIGS
    return [lbl for lbl in ANALYSIS_CONFIGS if lbl in results]

M, N = 16, 16
BITS_PER_SYMBOL = 2
QAM_D_MODEL, QAM_NHEAD, QAM_NUM_LAYERS, QAM_DIM_FF, QAM_DROPOUT = 256, 4, 4, 512, 0.1
DD_D_MODEL, DD_NHEAD, DD_NUM_LAYERS, DD_DIM_FF, DD_DROPOUT = 256, 8, 6, 2048, 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAMMA_MIN, GAMMA_MAX, GAMMA_STEP = 0.0, 10.0, 0.1

# BER 仿真参数
SNR_dB_RANGE = np.arange(0, 22, 2)
MODULATION_ORDER = 4

# 水声信道模块导入：读取根目录 water_channel_data 中的本地 WATER MAT。
_BELLHOP_DIRS = [
    ".",
]
for _bellhop_dir in _BELLHOP_DIRS:
    if os.path.isdir(_bellhop_dir) and _bellhop_dir not in sys.path:
        sys.path.append(_bellhop_dir)
_BELLHOP_IMPORT_ERROR = None
try:
    from bellhop_water_channel import (water_channel_gen, load_channel_metadata,
                                        extract_multipath_taps, estimate_doppler)
    _HAS_BELLHOP = True
except Exception as exc:
    _HAS_BELLHOP = False
    _BELLHOP_IMPORT_ERROR = exc
    print(f"  [INFO] bellhop_water_channel 模块不可用，水声信道 BER 将无法生成: {exc}")

WATER_CHANNEL_REL = os.path.join("water_channel_data", "water_channel_otfs.mat")
WATER_CHANNEL_ENV = "SMARTOTFS_WATER_CHANNEL_MAT"
WATER_DELTA_F = 250.0
WATER_FC = 3000.0
REQUIRE_WATER_BER = True


# ======================= 模型定义 =======================
class DDPMPE2D(nn.Module):
    def __init__(self, M, N, d_model):
        super().__init__()
        self.M, self.N, self.d_model = M, N, d_model
        self.R = d_model // 2
        self.phi_tau = nn.Parameter(torch.randn(self.R) * 0.1)
        self.phi_nu = nn.Parameter(torch.randn(self.R) * 0.1)

    def forward(self):
        device = self.phi_tau.device
        m = torch.arange(self.M, dtype=torch.float32, device=device).view(-1, 1, 1)
        n = torch.arange(self.N, dtype=torch.float32, device=device).view(1, -1, 1)
        phase = 2 * math.pi * ((m / self.M) * self.phi_tau + (n / self.N) * self.phi_nu)
        pe = torch.stack([torch.cos(phase), torch.sin(phase)], dim=-1).reshape(self.M, self.N, -1)
        return pe


class HAPEPE2D(nn.Module):
    def __init__(self, M, N, d_model, B=4, lambda_scale=2.0, eps_nu=None):
        super().__init__()
        self.M, self.N, self.d_model = M, N, d_model
        self.B, self.lambda_scale = B, lambda_scale
        self.R = d_model // 2
        if self.R % B != 0:
            raise ValueError(f"d_model//2={self.R} must be divisible by B={B}")
        self.R_b = self.R // B
        self.eps_nu = eps_nu if eps_nu is not None else 0.05 / N
        r_idx = torch.arange(self.R, dtype=torch.float32)
        self.register_buffer('f_nu_base', (r_idx % N) / N)
        self.w_nu = nn.Parameter(torch.zeros(self.R) * 0.01)
        self.register_buffer('band_scales', torch.tensor(
            [lambda_scale ** b for b in range(B)], dtype=torch.float32).view(1, B, 1))
        self.alpha_tau = nn.Parameter(torch.randn(B, self.R_b) * 0.1)
        self.beta_raw = nn.Parameter(torch.full((B, self.R_b), 0.5413))
        self.g_tau = nn.Parameter(torch.randn(self.R) * 0.01)
        self.g_nu = nn.Parameter(torch.randn(self.R) * 0.01)
        self.g_bias = nn.Parameter(torch.full((self.R,), 2.0))
        self.gamma_raw = nn.Parameter(torch.tensor(0.5413))
        self.eta_raw = nn.Parameter(torch.tensor(-2.1972))

    def forward(self):
        device = self.w_nu.device
        m_hat = torch.arange(self.M, dtype=torch.float32, device=device).view(self.M, 1, 1) / self.M
        n = torch.arange(self.N, dtype=torch.float32, device=device).view(1, self.N, 1)
        n_hat = n / self.N
        f_nu = self.f_nu_base.view(1, 1, self.R) + self.eps_nu * torch.tanh(self.w_nu.view(1, 1, self.R))
        theta_nu = 2 * math.pi * f_nu * n
        alpha = self.alpha_tau.view(1, self.B, self.R_b)
        beta = F.softplus(self.beta_raw.view(1, self.B, self.R_b))
        theta_tau_band = 2 * math.pi * self.band_scales * beta * m_hat.view(self.M, 1, 1) + alpha
        theta_tau = theta_tau_band.reshape(self.M, 1, self.R)
        gamma = F.softplus(self.gamma_raw)
        gate_in = gamma * (m_hat * self.g_tau.view(1, 1, self.R) +
                           n_hat * self.g_nu.view(1, 1, self.R) +
                           self.g_bias.view(1, 1, self.R))
        G = torch.sigmoid(gate_in)
        theta_gate = math.pi * (1.0 - G)
        psi = theta_nu + theta_tau + theta_gate
        G_mean = G.mean(dim=(0, 1))
        G_mean_band = G_mean.view(self.B, self.R_b).mean(dim=1)
        eta = F.softplus(self.eta_raw)
        A_band = torch.sqrt(1.0 / (1.0 + eta * (1.0 - G_mean_band)))
        A = A_band.view(self.B, 1).expand(self.B, self.R_b).reshape(self.R)
        pe = torch.stack([A.view(1, 1, self.R) * torch.cos(psi),
                          A.view(1, 1, self.R) * torch.sin(psi)], dim=-1)
        return pe.reshape(self.M, self.N, self.d_model)


class StandardPositionalEncoding2D(nn.Module):
    def __init__(self, M, N, d_model):
        super().__init__()
        self.M, self.N = M, N
        half_d = d_model // 2
        pe_m = torch.zeros(M, half_d)
        pm = torch.arange(0, M, dtype=torch.float32).unsqueeze(1)
        div_m = torch.exp(torch.arange(0, half_d, 2, dtype=torch.float32) * (-math.log(10000.0) / half_d))
        pe_m[:, 0::2] = torch.sin(pm * div_m)
        pe_m[:, 1::2] = torch.cos(pm * div_m)
        pe_n = torch.zeros(N, half_d)
        pn = torch.arange(0, N, dtype=torch.float32).unsqueeze(1)
        div_n = torch.exp(torch.arange(0, half_d, 2, dtype=torch.float32) * (-math.log(10000.0) / half_d))
        pe_n[:, 0::2] = torch.sin(pn * div_n)
        pe_n[:, 1::2] = torch.cos(pn * div_n)
        self.register_buffer('pe_m', pe_m)
        self.register_buffer('pe_n', pe_n)

    def forward(self):
        return torch.cat([self.pe_m.unsqueeze(1).expand(-1, self.N, -1),
                          self.pe_n.unsqueeze(0).expand(self.M, -1, -1)], dim=-1)


class QAMGenerator(nn.Module):
    def __init__(self, M=16, N=16, bits_per_symbol=2, d_model=256, nhead=4,
                 num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.M, self.N = M, N
        self.bits_per_symbol = bits_per_symbol
        self.d_model = d_model
        self.bit_embed = nn.Linear(bits_per_symbol, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, M, N, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.out_proj = nn.Linear(d_model, 2)

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, self.M, self.N, self.bits_per_symbol)
        x = self.bit_embed(x) + self.pos_embed
        x = x.view(B, self.M * self.N, self.d_model)
        return self.out_proj(self.transformer(x)).view(B, self.M, self.N, 2)


class DualAxisSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model, self.nhead = d_model, nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5
        self.qkv_tau = nn.Linear(d_model, 3 * d_model)
        self.qkv_nu = nn.Linear(d_model, 3 * d_model)
        self.proj_tau = nn.Linear(d_model, d_model)
        self.proj_nu = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _reshape_to_attention(self, x, axis):
        B, Ml, Nl, D = x.shape
        if axis == 0:
            return x.permute(0, 2, 1, 3).reshape(B * Nl, Ml, D), Nl, Ml
        return x.reshape(B * Ml, Nl, D), Ml, Nl

    def _attention(self, q, k, v):
        B2, L, D = q.shape
        q = q.view(B2, L, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B2, L, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B2, L, self.nhead, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v,
                                             dropout_p=self.dropout.p if self.training else 0.0,
                                             scale=self.scale)
        return out.transpose(1, 2).reshape(B2, L, D)

    def forward(self, x):
        xt, on_, Ml = self._reshape_to_attention(x, axis=0)
        qt, kt, vt = self.qkv_tau(xt).chunk(3, dim=-1)
        ot = self.proj_tau(self._attention(qt, kt, vt))
        ot = ot.reshape(-1, on_, Ml, self.d_model).permute(0, 2, 1, 3)
        xn, om_, Nl = self._reshape_to_attention(x, axis=1)
        qn, kn, vn = self.qkv_nu(xn).chunk(3, dim=-1)
        on_ = self.proj_nu(self._attention(qn, kn, vn))
        on_ = on_.reshape(-1, om_, Nl, self.d_model)
        return ot + on_ + x


class DualAxisTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = DualAxisSelfAttention(d_model, nhead, dropout)
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(dim_feedforward, d_model), nn.Dropout(dropout))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.self_attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class DualAxisTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([DualAxisTransformerBlock(d_model, nhead, dim_feedforward, dropout)
                                     for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class StandardTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        B, Ml, Nl, D = x.shape
        return self.encoder(x.reshape(B, Ml * Nl, D)).reshape(B, Ml, Nl, D)


class SACIASelfAttention(nn.Module):
    def __init__(self, d_model, nhead, M, N, dropout=0.1, eta_init=1.0):
        super().__init__()
        self.d_model, self.nhead, self.M, self.N = d_model, nhead, M, N
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5
        self.linear_gate_tau = nn.Linear(d_model, d_model)
        self.linear_gate_nu = nn.Linear(d_model, d_model)
        self.qkv_tau = nn.Linear(d_model, 3 * d_model)
        self.qkv_nu = nn.Linear(d_model, 3 * d_model)
        self.proj_tau = nn.Linear(d_model, d_model)
        self.proj_nu = nn.Linear(d_model, d_model)
        self.lambda_tau = nn.Parameter(torch.tensor(0.01))
        self.lambda_nu = nn.Parameter(torch.tensor(0.01))
        self.gamma_tau = nn.Parameter(torch.tensor(1.0))
        self.gamma_nu = nn.Parameter(torch.tensor(1.0))
        self.eta = nn.Parameter(torch.tensor(eta_init))
        self.alpha_tau = nn.Parameter(torch.ones(d_model))
        self.alpha_nu = nn.Parameter(torch.ones(d_model))
        self.dropout = nn.Dropout(dropout)
        self._build_base_matrices(M, N)

    def _build_base_matrices(self, Mv, Nv):
        it = torch.arange(Mv, dtype=torch.float32)
        self.register_buffer('base_dist_tau', torch.abs(it.unsqueeze(1) - it.unsqueeze(0)))
        in_ = torch.arange(Nv, dtype=torch.float32)
        ad = torch.abs(in_.unsqueeze(1) - in_.unsqueeze(0))
        self.register_buffer('base_circ_dist_nu', torch.min(ad, Nv - ad))
        self.register_buffer('base_sin2_nu',
                             torch.sin(torch.pi * (in_.unsqueeze(1) - in_.unsqueeze(0)) / Nv) ** 2)

    def _compute_Phi(self):
        Pt = self.base_dist_tau ** self.gamma_tau.abs()
        Pn = (self.base_circ_dist_nu ** self.gamma_nu.abs()) * (1.0 + self.eta.abs() * self.base_sin2_nu)
        return Pt, Pn

    def _sparse_attention(self, q, k, v, Phi, lam, L):
        Bp, Lv, D = q.shape
        q = q.view(Bp, Lv, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(Bp, Lv, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(Bp, Lv, self.nhead, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores - lam.abs() * Phi.unsqueeze(0).unsqueeze(0)
        attn = self.dropout(F.softmax(scores, dim=-1))
        return torch.matmul(attn, v).transpose(1, 2).reshape(Bp, Lv, D)

    def forward(self, x):
        B, Mv, Nv, D = x.shape
        c_tau = x.mean(dim=2); c_nu = x.mean(dim=1)
        G_nt = torch.sigmoid(self.linear_gate_tau(c_nu)).unsqueeze(1)
        G_tn = torch.sigmoid(self.linear_gate_nu(c_tau)).unsqueeze(2)
        Xt = x * G_nt + x; Xn = x * G_tn + x
        Pt, Pn = self._compute_Phi()
        xt = Xt.permute(0, 2, 1, 3).reshape(B * Nv, Mv, D)
        qt, kt, vt = self.qkv_tau(xt).chunk(3, dim=-1)
        Ht = self.proj_tau(self._sparse_attention(qt, kt, vt, Pt, self.lambda_tau, Mv))
        Ht = Ht.reshape(B, Nv, Mv, D).permute(0, 2, 1, 3)
        xn = Xn.reshape(B * Mv, Nv, D)
        qn, kn, vn = self.qkv_nu(xn).chunk(3, dim=-1)
        Hn = self.proj_nu(self._sparse_attention(qn, kn, vn, Pn, self.lambda_nu, Nv))
        Hn = Hn.reshape(B, Mv, Nv, D)
        return self.alpha_tau * Ht + self.alpha_nu * Hn + x


class SACIACoderLayer(nn.Module):
    def __init__(self, d_model, nhead, M, N, dim_feedforward=1024, dropout=0.1, eta_init=1.0):
        super().__init__()
        self.self_attn = SACIASelfAttention(d_model, nhead, M, N, dropout, eta_init)
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(dim_feedforward, d_model), nn.Dropout(dropout))
        self.norm1 = nn.LayerNorm(d_model)

    def forward(self, x):
        H_attn = self.self_attn(x)
        H_norm = self.norm1(H_attn)
        return self.ffn(H_norm) + H_norm


class SACIATransformer(nn.Module):
    def __init__(self, d_model, nhead, M, N, num_layers, dim_feedforward=1024, dropout=0.1, eta_init=1.0):
        super().__init__()
        self.layers = nn.ModuleList([SACIACoderLayer(d_model, nhead, M, N, dim_feedforward, dropout, eta_init)
                                     for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class GQASelfAttention(nn.Module):
    def __init__(self, d_model, nhead, kv_heads=None, dropout=0.1):
        super().__init__()
        self.d_model, self.nhead = d_model, nhead
        self.head_dim = d_model // nhead
        self.kv_heads = kv_heads or max(1, nhead // 4)
        self.group_size = nhead // self.kv_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, self.kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, self.kv_heads * self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout

    def forward(self, x):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.kv_heads, self.head_dim).transpose(1, 2)
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p if self.training else 0.0)
        return self.out_proj(out.transpose(1, 2).reshape(B, L, D))


class DifferentialSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model, self.nhead = d_model, nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5
        self.q1_proj = nn.Linear(d_model, d_model)
        self.k1_proj = nn.Linear(d_model, d_model)
        self.q2_proj = nn.Linear(d_model, d_model)
        self.k2_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.lambda_param = nn.Parameter(torch.tensor(math.log(math.exp(0.1) - 1.0)))
        self.dropout = nn.Dropout(dropout)

    def _shape(self, x):
        B, L, _ = x.shape
        return x.view(B, L, self.nhead, self.head_dim).transpose(1, 2)

    def forward(self, x):
        q1, k1 = self._shape(self.q1_proj(x)), self._shape(self.k1_proj(x))
        q2, k2 = self._shape(self.q2_proj(x)), self._shape(self.k2_proj(x))
        v = self._shape(self.v_proj(x))
        a1 = F.softmax(torch.matmul(q1, k1.transpose(-2, -1)) * self.scale, dim=-1)
        a2 = F.softmax(torch.matmul(q2, k2.transpose(-2, -1)) * self.scale, dim=-1)
        out = torch.matmul(self.dropout(a1 - F.softplus(self.lambda_param) * a2), v)
        B, _, L, _ = out.shape
        return self.out_proj(out.transpose(1, 2).reshape(B, L, self.d_model))


class MLASelfAttention(nn.Module):
    def __init__(self, d_model, nhead, latent_dim=None, dropout=0.1):
        super().__init__()
        self.d_model, self.nhead = d_model, nhead
        self.head_dim = d_model // nhead
        self.latent_dim = latent_dim or max(d_model // 4, self.head_dim)
        self.q_down = nn.Linear(d_model, self.latent_dim)
        self.q_up = nn.Linear(self.latent_dim, d_model)
        self.kv_down = nn.Linear(d_model, self.latent_dim)
        self.k_up = nn.Linear(self.latent_dim, d_model)
        self.v_up = nn.Linear(self.latent_dim, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout

    def _shape(self, x):
        B, L, _ = x.shape
        return x.view(B, L, self.nhead, self.head_dim).transpose(1, 2)

    def forward(self, x):
        B, L, D = x.shape
        q = self._shape(self.q_up(self.q_down(x)))
        latent = self.kv_down(x)
        k = self._shape(self.k_up(latent))
        v = self._shape(self.v_up(latent))
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p if self.training else 0.0)
        return self.out_proj(out.transpose(1, 2).reshape(B, L, D))


class FlatAttentionTransformerBlock(nn.Module):
    def __init__(self, attn_module, d_model, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = attn_module
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(dim_feedforward, d_model), nn.Dropout(dropout))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.self_attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class FlatAttentionTransformer(nn.Module):
    def __init__(self, attn_type, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        builders = {
            "GQA": lambda: GQASelfAttention(d_model, nhead, dropout=dropout),
            "Differential": lambda: DifferentialSelfAttention(d_model, nhead, dropout),
            "MLA": lambda: MLASelfAttention(d_model, nhead, dropout=dropout),
        }
        self.layers = nn.ModuleList([
            FlatAttentionTransformerBlock(builders[attn_type](), d_model, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        B, Ml, Nl, D = x.shape
        x = x.reshape(B, Ml * Nl, D)
        for layer in self.layers:
            x = layer(x)
        return x.reshape(B, Ml, Nl, D)


class DDGeneratorV3(nn.Module):
    def __init__(self, M=16, N=16, d_model=256, nhead=8, num_layers=6,
                 dim_feedforward=2048, dropout=0.1, pos_encoding_type='Standard', attn_type='Standard'):
        super().__init__()
        self.M, self.N = M, N
        self.d_model = d_model
        self.qam_embed = nn.Linear(2, d_model)
        if pos_encoding_type == 'DDPMPE':
            self.pos_encoder = DDPMPE2D(M, N, d_model)
        elif pos_encoding_type == 'PhaseLoom':
            self.pos_encoder = HAPEPE2D(M, N, d_model)
        elif pos_encoding_type == 'Standard':
            self.pos_encoder = StandardPositionalEncoding2D(M, N, d_model)
        else:
            raise ValueError(f"Unknown pos_encoding_type: {pos_encoding_type}")
        if attn_type == 'Dual_Axis':
            self.transformer = DualAxisTransformer(d_model=d_model, nhead=nhead, num_layers=num_layers,
                                                   dim_feedforward=dim_feedforward, dropout=dropout)
        elif attn_type == 'Standard':
            self.transformer = StandardTransformer(d_model=d_model, nhead=nhead, num_layers=num_layers,
                                                   dim_feedforward=dim_feedforward, dropout=dropout)
        elif attn_type == 'BoltAttention':
            self.transformer = SACIATransformer(d_model=d_model, nhead=nhead, M=M, N=N, num_layers=num_layers,
                                                dim_feedforward=dim_feedforward, dropout=dropout)
        elif attn_type in ('GQA', 'Differential', 'MLA'):
            self.transformer = FlatAttentionTransformer(attn_type=attn_type, d_model=d_model, nhead=nhead,
                                                        num_layers=num_layers, dim_feedforward=dim_feedforward,
                                                        dropout=dropout)
        else:
            raise ValueError(f"Unknown attn_type: {attn_type}")
        self.out_proj = nn.Linear(d_model, 2)

    def forward(self, x):
        B = x.shape[0]
        x = self.qam_embed(x) + self.pos_encoder().unsqueeze(0)
        return self.out_proj(self.transformer(x))


class EndToEndOTFS(nn.Module):
    def __init__(self, qam_gen, dd_gen):
        super().__init__()
        self.qam_gen = qam_gen
        self.dd_gen = dd_gen
        self.M = qam_gen.M
        self.N = qam_gen.N

    def forward(self, bits):
        return self.dd_gen(self.qam_gen(bits))


class LegacyOTFSReceiver(nn.Module):
    def __init__(self, M=16, N=16):
        super().__init__()
        self.M, self.N = M, N

    def forward(self, x):
        xc = torch.complex(x[..., 0], x[..., 1]) if x.shape[-1] == 2 else x
        return torch.fft.fft(xc, dim=-1) / torch.sqrt(torch.tensor(self.M * self.N, dtype=xc.dtype, device=xc.device))


# ======================= 数据集加载 =======================
class OTFSDataset(Dataset):
    def __init__(self, data_file):
        super().__init__()
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        self.bits_matrices = data['bits_matrices']
        self.idft_matrices = data['idft_matrices']
        self.qam_matrices = data.get('qam_matrices', None)
        self.M = data['M']
        self.N = data['N']
        self.num_frames = data['num_frames']
        print(f"  数据集加载成功: {data_file}")
        print(f"  M={self.M}, N={self.N}, frames={self.num_frames}")

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        bits = torch.FloatTensor(self.bits_matrices[idx])
        target = torch.FloatTensor(self.idft_matrices[idx])
        qam = (torch.FloatTensor(self.qam_matrices[idx]) if self.qam_matrices is not None else torch.zeros(1))
        return bits, target, qam


# ======================= 工具函数 =======================
def strip_module_prefix(state_dict):
    return {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}


def build_model(config_label):
    cfg = CONFIG_MAP[config_label]
    qam_gen = QAMGenerator(M=M, N=N, bits_per_symbol=BITS_PER_SYMBOL, d_model=QAM_D_MODEL,
                           nhead=QAM_NHEAD, num_layers=QAM_NUM_LAYERS,
                           dim_feedforward=QAM_DIM_FF, dropout=QAM_DROPOUT)
    qam_state = strip_module_prefix(torch.load(QAM_GEN_PATH, map_location=DEVICE))
    qam_gen.load_state_dict(qam_state)
    qam_gen.to(DEVICE).eval()
    dd_gen = DDGeneratorV3(M=M, N=N, d_model=DD_D_MODEL, nhead=DD_NHEAD,
                           num_layers=DD_NUM_LAYERS, dim_feedforward=DD_DIM_FF,
                           dropout=DD_DROPOUT, pos_encoding_type=cfg["pos_encoding"],
                           attn_type=cfg["attn_type"])
    model = EndToEndOTFS(qam_gen, dd_gen).to(DEVICE).eval()
    return model


def load_end2end_weights(model, weight_path):
    model.load_state_dict(strip_module_prefix(torch.load(weight_path, map_location=DEVICE)))
    print(f"  已加载权重: {os.path.basename(weight_path)}")


def compute_ccdf(papr_dB, gamma):
    return np.mean(papr_dB[:, np.newaxis] > gamma[np.newaxis, :], axis=0)


# ======================= Part 1: 互相关 / PAPR / DD误差 (原分析) =======================
def analyze_config(config_label, val_dataset, receiver):
    cfg = CONFIG_MAP[config_label]
    print(f"\n{'='*60}")
    print(f"  {cfg['label']}")
    print(f"  pos_encoding='{cfg['pos_encoding']}', attn_type='{cfg['attn_type']}'")
    print(f"{'='*60}")
    print("  [1/3] 构建模型并加载权重...")
    model = build_model(config_label)
    load_end2end_weights(model, cfg["weight"])
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    L = M * N
    corr_sum = np.zeros(2 * L - 1, dtype=np.float64)
    papr_dB_list, target_papr_dB_list = [], []
    err_sum = np.zeros((M, N), dtype=np.float64)
    gen_signals, target_signals, qam_originals = [], [], []
    peak_heights, peak_shifts = [], []
    mse_list, evm_list = [], []

    print(f"  [2/3] 逐帧推理 ({val_dataset.num_frames} frames)...")
    progress_step = max(1, val_dataset.num_frames // 10)

    with torch.no_grad():
        for idx, (bits, target_dd, qam_orig) in enumerate(val_loader):
            bits, target_dd, qam_orig = bits.to(DEVICE), target_dd.to(DEVICE), qam_orig.to(DEVICE)
            gen_dd = model(bits)
            recovered = receiver(gen_dd)
            qam_cplx_t = torch.complex(qam_orig[..., 0], qam_orig[..., 1])
            rec_power = torch.mean(torch.abs(recovered) ** 2)
            qam_power = torch.mean(torch.abs(qam_cplx_t) ** 2)
            scale_per_frame = torch.sqrt(qam_power / (rec_power + 1e-12))
            recovered_scaled = recovered * scale_per_frame
            err_t = torch.abs(recovered_scaled.squeeze(0) - qam_cplx_t.squeeze(0)) ** 2
            err_sum += err_t.cpu().numpy()

            gen_dd_np = gen_dd.squeeze(0).cpu().numpy()
            target_np = target_dd.squeeze(0).cpu().numpy()
            qam_np = qam_orig.squeeze(0).cpu().numpy()
            gen_cplx = gen_dd_np[..., 0] + 1j * gen_dd_np[..., 1]
            target_cplx = target_np[..., 0] + 1j * target_np[..., 1]
            qam_cplx = qam_np[..., 0] + 1j * qam_np[..., 1]

            recovered_np = recovered_scaled.squeeze(0).cpu().numpy()
            evm_val = np.sqrt(np.mean(np.abs(recovered_np - qam_cplx) ** 2) /
                              (np.mean(np.abs(qam_cplx) ** 2) + 1e-12)) * 100
            evm_list.append(evm_val)
            mse_val = np.mean(np.abs(recovered_np - qam_cplx) ** 2) * 100
            mse_list.append(mse_val)

            gen_flat, target_flat = gen_cplx.flatten(), target_cplx.flatten()
            E_gen = np.sum(np.abs(gen_flat) ** 2) + 1e-12
            E_target = np.sum(np.abs(target_flat) ** 2) + 1e-12
            corr = np.correlate(gen_flat, target_flat, mode='full')
            corr_norm = np.abs(corr) / np.sqrt(E_gen * E_target)
            corr_sum += corr_norm
            peak_idx = np.argmax(corr_norm)
            peak_heights.append(corr_norm[peak_idx])
            peak_shifts.append(peak_idx - (L - 1))

            def compute_papr_np(signal):
                power = np.abs(signal.flatten()) ** 2
                return 10.0 * np.log10(np.max(power) / (np.mean(power) + 1e-12))

            papr_dB_list.append(compute_papr_np(gen_cplx))
            target_papr_dB_list.append(compute_papr_np(target_cplx))
            gen_signals.append(gen_cplx)
            target_signals.append(target_cplx)
            qam_originals.append(qam_cplx)

            if (idx + 1) % progress_step == 0:
                print(f"    进度: {idx + 1}/{val_dataset.num_frames} frames")

    avg_corr = corr_sum / len(gen_signals)
    papr_dB = np.array(papr_dB_list, dtype=np.float64)
    target_papr_dB = np.array(target_papr_dB_list, dtype=np.float64)
    mean_err = err_sum / len(gen_signals)
    peak_heights_arr = np.array(peak_heights, dtype=np.float64)
    peak_shifts_arr = np.array(peak_shifts, dtype=np.float64)
    mse_arr = np.array(mse_list, dtype=np.float64)
    evm_arr = np.array(evm_list, dtype=np.float64)

    print(f"  [3/3] 分析完成")
    print(f"    互相关峰值: {np.mean(peak_heights_arr):.4f} +/- {np.std(peak_heights_arr):.4f}")
    print(f"    PAPR (生成): {np.mean(papr_dB):.2f} +/- {np.std(papr_dB):.2f} dB")
    print(f"    MSE: {np.mean(mse_arr):.4f}% | EVM: {np.mean(evm_arr):.4f}%")
    print(f"    整体平均 MSE: {10.0 * np.log10(np.mean(mean_err) + 1e-12):.2f} dB")

    return {"avg_corr": avg_corr, "papr_dB": papr_dB, "target_papr_dB": target_papr_dB,
            "mean_err": mean_err, "gen_signals": gen_signals, "target_signals": target_signals,
            "qam_originals": qam_originals, "peak_heights": peak_heights_arr,
            "peak_shifts": peak_shifts_arr, "mse_arr": mse_arr, "evm_arr": evm_arr}


def compute_target_autocorr(val_dataset):
    L = M * N
    autocorr_sum = np.zeros(2 * L - 1, dtype=np.float64)
    for idx in range(len(val_dataset)):
        _, target_dd, _ = val_dataset[idx]
        target_np = target_dd.numpy()
        target_cplx = target_np[..., 0] + 1j * target_np[..., 1]
        target_flat = target_cplx.flatten()
        E = np.sum(np.abs(target_flat) ** 2) + 1e-12
        auto = np.correlate(target_flat, target_flat, mode='full')
        autocorr_sum += np.abs(auto) / E
    return autocorr_sum / len(val_dataset)


def plot_cross_correlation(results, target_auto, fig_dir):
    fig, ax = plt.subplots(figsize=(12, 7))
    L = M * N
    x = np.arange(-(L - 1), L)
    ax.plot(x, target_auto, 'k-', linewidth=2.0, label='Target Autocorr. (Ref.)', alpha=0.8)
    for lbl in active_config_labels(results):
        ax.plot(x, results[lbl]["avg_corr"], color=CONFIG_COLORS[lbl], linewidth=1.5, label=CONFIG_MAP[lbl]["label"], alpha=0.85)
    ax.set_xlabel("Delay offset (sample)", fontsize=13)
    ax.set_ylabel("Normalized correlation coefficient", fontsize=13)
    ax.set_title("Time-domain Waveform Cross-correlation Comparison", fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right'); ax.grid(True, alpha=0.3)
    ax.set_xlim(-L + 1, L - 1)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "cross_correlation_Config_A_to_G.png"), dpi=150, bbox_inches='tight')
    plt.close(fig); print(f"  已保存: cross_correlation_Config_A_to_G.png")


def plot_papr_ccdf(results, target_papr_dB, fig_dir):
    fig, ax = plt.subplots(figsize=(12, 7))
    gamma = np.arange(GAMMA_MIN, GAMMA_MAX + GAMMA_STEP, GAMMA_STEP)
    ax.semilogy(gamma, compute_ccdf(target_papr_dB, gamma), 'k-', linewidth=2.0, label='Target', alpha=0.8)
    for lbl in active_config_labels(results):
        ax.semilogy(gamma, compute_ccdf(results[lbl]["papr_dB"], gamma), color=CONFIG_COLORS[lbl],
                    linewidth=1.5, label=CONFIG_MAP[lbl]["label"], alpha=0.85)
    ax.set_xlabel("PAPR threshold (dB)", fontsize=13)
    ax.set_ylabel("CCDF P(PAPR > threshold)", fontsize=13)
    ax.set_title("PAPR Complementary Cumulative Distribution Function", fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right'); ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(GAMMA_MIN, GAMMA_MAX)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "papr_ccdf_Config_A_to_G.png"), dpi=150, bbox_inches='tight')
    plt.close(fig); print(f"  已保存: papr_ccdf_Config_A_to_G.png")


def plot_dd_error_heatmap(results, fig_dir):
    labels = active_config_labels(results)
    total = len(labels) + 1
    ncols = 4 if total > 6 else 3
    nrows = int(np.ceil(total / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 5.0 * nrows))
    axes = np.atleast_1d(axes).ravel()
    eps = 1e-12
    configs = [("Target (Zero Reference)", None, axes[0])]
    configs += [(CONFIG_MAP[lbl]["label"], results[lbl]["mean_err"], axes[i + 1])
                for i, lbl in enumerate(labels)]
    for title, err_mat, ax in configs:
        err_db = 10.0 * np.log10(err_mat + eps) if err_mat is not None else np.zeros((M, N))
        vmax = max(0, np.max(err_db)) if err_mat is not None else 0
        vmin = min(-40, np.min(err_db)) if err_mat is not None else -40
        im = ax.imshow(err_db, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_xlabel("Doppler index (n)"); ax.set_ylabel("Delay index (m)")
        ax.set_title(title, fontsize=12, fontweight='bold'); plt.colorbar(im, ax=ax, label="Mean MSE (dB)")
    for ax in axes[len(configs):]:
        ax.set_visible(False)
    fig.suptitle("DD-domain Error Spatial Distribution Comparison", fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "dd_error_heatmap_Config_A_to_G.png"), dpi=150, bbox_inches='tight')
    plt.close(fig); print(f"  已保存: dd_error_heatmap_Config_A_to_G.png")


def plot_dd_error_profiles(results, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    eps = 1e-12
    for ax_idx, axis in enumerate([1, 0]):
        ax = axes[ax_idx]
        for lbl in active_config_labels(results):
            profile = np.mean(results[lbl]["mean_err"], axis=axis)
            ax.plot(range(M if axis == 1 else N), 10.0 * np.log10(profile + eps),
                    CONFIG_MARKERS[lbl], color=CONFIG_COLORS[lbl],
                    label=CONFIG_MAP[lbl]["short_label"], markersize=6)
        ax.set_xlabel("Delay index (m)" if axis == 1 else "Doppler index (n)", fontsize=12)
        ax.set_ylabel("Mean MSE (dB)", fontsize=12)
        ax.set_title(f"Error profile along {'delay' if axis == 1 else 'Doppler'} axis", fontsize=13, fontweight='bold')
        ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "dd_error_profiles_Config_A_to_G.png"), dpi=150, bbox_inches='tight')
    plt.close(fig); print(f"  已保存: dd_error_profiles_Config_A_to_G.png")


# ======================= Part 2: 时域 MSE / EVM / PSD Error 指标 =======================
def compute_psd_error(gen_signal, target_signal):
    """计算 PSD error: 生成信号与目标信号的功率谱差异百分比"""
    gen_psd = np.abs(np.fft.fft(gen_signal.flatten())) ** 2
    target_psd = np.abs(np.fft.fft(target_signal.flatten())) ** 2
    gen_psd = gen_psd / (np.sum(gen_psd) + 1e-12)
    target_psd = target_psd / (np.sum(target_psd) + 1e-12)
    return np.sum(np.abs(gen_psd - target_psd)) * 50  # 百分比


def compute_metrics_for_config(config_label, val_dataset, receiver):
    """对单组配置计算 MSE/EVM/PSD error 指标"""
    cfg = CONFIG_MAP[config_label]
    print(f"\n  Computing time-domain metrics for {cfg['label']}...")
    model = build_model(config_label)
    load_end2end_weights(model, cfg["weight"])

    mse_list, evm_list, psd_list, papr_list = [], [], [], []
    with torch.no_grad():
        for idx in range(len(val_dataset)):
            bits, target_dd, qam_orig = val_dataset[idx]
            bits = bits.unsqueeze(0).to(DEVICE)
            qam_orig_t = qam_orig.unsqueeze(0).to(DEVICE)
            gen_dd = model(bits)
            recovered = receiver(gen_dd)
            qam_cplx_t = torch.complex(qam_orig_t[..., 0], qam_orig_t[..., 1])
            rec_power = torch.mean(torch.abs(recovered) ** 2)
            qam_power = torch.mean(torch.abs(qam_cplx_t) ** 2)
            recovered_scaled = recovered * torch.sqrt(qam_power / (rec_power + 1e-12))

            recovered_np = recovered_scaled.squeeze(0).cpu().numpy()
            qam_cplx = qam_orig.squeeze(0).numpy()
            qam_cplx = qam_cplx[..., 0] + 1j * qam_cplx[..., 1]

            mse_val = np.mean(np.abs(recovered_np - qam_cplx) ** 2)
            evm_val = np.sqrt(np.mean(np.abs(recovered_np - qam_cplx) ** 2) /
                              (np.mean(np.abs(qam_cplx) ** 2) + 1e-12))
            mse_list.append(mse_val * 100)
            evm_list.append(evm_val * 100)

            gen_np = gen_dd.squeeze(0).cpu().numpy()
            gen_cplx = gen_np[..., 0] + 1j * gen_np[..., 1]
            target_np = target_dd.numpy()
            target_cplx = target_np[..., 0] + 1j * target_np[..., 1]
            psd_list.append(compute_psd_error(gen_cplx, target_cplx))
            power = np.abs(gen_cplx.flatten()) ** 2
            papr_list.append(10.0 * np.log10(np.max(power) / (np.mean(power) + 1e-12)))

    mse_arr = np.array(mse_list, dtype=np.float64)
    evm_arr = np.array(evm_list, dtype=np.float64)
    psd_arr = np.array(psd_list, dtype=np.float64)
    papr_arr = np.array(papr_list, dtype=np.float64)

    metrics = {"mse_mean": float(np.mean(mse_arr)), "mse_std": float(np.std(mse_arr)),
               "evm_mean": float(np.mean(evm_arr)), "evm_std": float(np.std(evm_arr)),
               "psd_error_mean": float(np.mean(psd_arr)), "psd_error_std": float(np.std(psd_arr)),
               "papr_mean": float(np.mean(papr_arr)), "papr_std": float(np.std(papr_arr)),
               "mse_all": mse_arr, "evm_all": evm_arr, "psd_error_all": psd_arr}
    print(f"    MSE={metrics['mse_mean']:.4f}%, EVM={metrics['evm_mean']:.4f}%, PSDerror={metrics['psd_error_mean']:.4f}%")
    return metrics


# ======================= Part 3: 单点探针响应 (Single-input Probe Response) =======================
def _recover_qam_np(time_signal, qam_ref=None):
    recovered = np.fft.fft(time_signal, axis=-1) / np.sqrt(M * N)
    if qam_ref is not None:
        rec_power = np.mean(np.abs(recovered) ** 2)
        ref_power = np.mean(np.abs(qam_ref) ** 2)
        recovered = recovered * np.sqrt(ref_power / (rec_power + 1e-12))
    return recovered


def _normalized_psd_db(signal):
    spec = np.fft.fft(signal.flatten())
    psd = np.abs(spec) ** 2
    psd = psd / (np.max(psd) + 1e-12)
    return 10.0 * np.log10(psd + 1e-12)


def build_visual_example_data(results, constellation_frames=64):
    """Collect Fig1/Fig2/Fig7 data: Regulated OTFS vs SmartOTFS-I."""
    labels = active_config_labels(results)
    primary = PRIMARY_CONFIG if PRIMARY_CONFIG in results else labels[0]
    target_first = results[primary]["target_signals"][0]
    smart_first = results[primary]["gen_signals"][0]
    qam_ref_first = results[primary]["qam_originals"][0]
    smart_qam_first = _recover_qam_np(smart_first, qam_ref_first)

    L = M * N
    const_frames = min(constellation_frames, len(results[primary]["gen_signals"]))
    const_data = {"constellation_frames": const_frames, "primary_config": primary}
    for lbl in labels:
        pts = []
        for frame_idx in range(const_frames):
            pts.append(_recover_qam_np(results[lbl]["gen_signals"][frame_idx],
                                       results[lbl]["qam_originals"][frame_idx]).flatten())
        const_data[f"qam_{lbl}"] = np.concatenate(pts)

    return {
        "M": M, "N": N,
        "flat_index": np.arange(L),
        "regulated_first": target_first.flatten(),
        "smart_first": smart_first.flatten(),
        "regulated_first_real": np.real(target_first.flatten()),
        "regulated_first_imag": np.imag(target_first.flatten()),
        "smart_first_real": np.real(smart_first.flatten()),
        "smart_first_imag": np.imag(smart_first.flatten()),
        "psd_freq": np.linspace(0.0, N / 2.0, L),
        "regulated_psd_dB": _normalized_psd_db(target_first),
        "smart_psd_dB": _normalized_psd_db(smart_first),
        "regulated_qam_first": qam_ref_first.flatten(),
        "smart_qam_first": smart_qam_first.flatten(),
        **const_data,
    }


def plot_visual_examples(visual_data, fig_dir):
    x = visual_data["flat_index"]
    regulated = visual_data["regulated_first"]
    smart = visual_data["smart_first"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    axes[0].plot(x, np.real(regulated), 'b-', linewidth=0.9, label='Regulated OTFS')
    axes[0].plot(x, np.real(smart), 'r--', linewidth=0.9, label='SmartOTFS')
    axes[0].set_xlabel('Flattened Index (MxN)')
    axes[0].set_ylabel('Real Part Amplitude')
    axes[0].legend(fontsize=8, loc='upper right')
    axes[0].grid(True, alpha=0.35)
    axes[0].set_xlim([0, len(x) - 1])

    axes[1].plot(x, np.imag(regulated), 'b-', linewidth=0.9, label='Regulated OTFS')
    axes[1].plot(x, np.imag(smart), 'r--', linewidth=0.9, label='SmartOTFS')
    axes[1].set_xlabel('Flattened Index (MxN)')
    axes[1].set_ylabel('Imaginary Part Amplitude')
    axes[1].legend(fontsize=8, loc='upper right')
    axes[1].grid(True, alpha=0.35)
    axes[1].set_xlim([0, len(x) - 1])
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig1_time_domain_iq_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.plot(visual_data["psd_freq"], visual_data["regulated_psd_dB"], 'b-', linewidth=0.9, label='Regulated OTFS')
    ax.plot(visual_data["psd_freq"], visual_data["smart_psd_dB"], 'r--', linewidth=0.9, label='SmartOTFS')
    ax.set_xlabel('Normalized Frequency (cycles/sample)')
    ax.set_ylabel('PSD (dB/Hz)')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.35)
    ax.set_xlim([visual_data["psd_freq"][0], visual_data["psd_freq"][-1]])
    y_min = min(np.min(visual_data["regulated_psd_dB"]), np.min(visual_data["smart_psd_dB"]))
    y_max = max(np.max(visual_data["regulated_psd_dB"]), np.max(visual_data["smart_psd_dB"]))
    ax.set_ylim([max(-60, np.floor(y_min / 10) * 10), min(10, np.ceil(y_max / 10) * 10)])
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig2_psd_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    labels = [lbl for lbl in ANALYSIS_CONFIGS if f"qam_{lbl}" in visual_data]
    ncols = min(4, max(1, len(labels)))
    nrows = int(np.ceil(len(labels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, lbl in zip(axes, labels):
        pts = visual_data[f"qam_{lbl}"]
        ax.scatter(np.real(pts), np.imag(pts), s=10, c='b', alpha=0.75, edgecolors='none')
        ax.set_title(CONFIG_MAP[lbl]["short_label"], fontsize=10)
        ax.set_xlabel('In-phase (I)')
        ax.set_ylabel('Quadrature (Q)')
        ax.grid(True, alpha=0.35)
        ax.set_aspect('equal', adjustable='box')
        lim = max(1.0, np.max(np.abs(np.concatenate([np.real(pts), np.imag(pts)]))) * 1.15)
        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
    for ax in axes[len(labels):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig7_qam_constellation_Config_A_to_G.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  saved: fig1_time_domain_iq_comparison.png, fig2_psd_comparison.png, fig7_qam_constellation_Config_A_to_G.png")


def qam4_probe_symbols():
    amp = 1.0 / np.sqrt(2.0)
    return [amp + 1j * amp, amp - 1j * amp, -amp + 1j * amp, -amp - 1j * amp]


def compute_probe_energy(config_label, m0, n0, model, receiver):
    """在 (m0,n0) 放置 one-hot 探针，通过模型后计算 DD 域能量分布"""
    dd_gen = model.dd_gen
    energy_acc = np.zeros((M, N), dtype=np.float64)
    with torch.no_grad():
        for sym in qam4_probe_symbols():
            qam_real = np.zeros((1, M, N), dtype=np.float32)
            qam_imag = np.zeros((1, M, N), dtype=np.float32)
            qam_real[0, m0, n0] = np.real(sym)
            qam_imag[0, m0, n0] = np.imag(sym)
            qam_in = torch.stack([torch.from_numpy(qam_real), torch.from_numpy(qam_imag)], dim=-1).to(DEVICE)
            rec = receiver(dd_gen(qam_in))
            energy_acc += np.abs(rec[0].cpu().numpy()) ** 2
    energy = energy_acc / len(qam4_probe_symbols())
    peak_pos = np.unravel_index(np.argmax(energy), energy.shape)
    concentration = energy[peak_pos] / (np.sum(energy) + 1e-15)
    return energy, concentration, peak_pos


def plot_probe_energy(results, fig_dir, probe_results):
    """绘制探针能量 3D 和 2D 图"""
    m0, n0 = 6, 4
    configs = ["Target"] + active_config_labels(results)
    fine_res = 200
    x_fine = np.linspace(0, N - 1, fine_res)
    y_fine = np.linspace(0, M - 1, fine_res)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine)

    # 3D
    ncols = 4 if len(configs) > 6 else 3
    nrows = int(np.ceil(len(configs) / ncols))
    fig = plt.figure(figsize=(6.0 * ncols, 5.0 * nrows))
    for idx, cfg_label in enumerate(configs):
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')
        energy_16x16 = probe_results[cfg_label]["energy"]
        spline = RectBivariateSpline(np.arange(M, dtype=float), np.arange(N, dtype=float), energy_16x16, kx=3, ky=3)
        energy_smooth = np.maximum(0, spline(y_fine, x_fine, grid=True))
        ax.plot_surface(X_fine, Y_fine, energy_smooth, cmap='hot', linewidth=0, antialiased=True, alpha=0.85, rstride=2, cstride=2)
        ax.scatter(n0, m0, energy_smooth[fine_res * m0 // M, fine_res * n0 // N], color='lime', s=100, marker='*')
        conc = probe_results[cfg_label]["concentration"]
        ax.set_xlabel('Doppler (n)'); ax.set_ylabel('Delay (m)'); ax.set_zlabel('Energy')
        ax.set_title(f'{cfg_label}: Conc={conc:.4f}', fontsize=10, fontweight='bold')
        ax.view_init(elev=28, azim=-55)
    fig.suptitle(f'3D Probe Energy Distribution (probe m={m0}, n={n0})', fontsize=14, fontweight='bold')
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "probe_energy_3d_Config_A_to_G.png"), dpi=150, bbox_inches='tight')
    plt.close(fig); print(f"  已保存: probe_energy_3d_Config_A_to_G.png")

    # 2D
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 5.0 * nrows))
    axes2 = np.atleast_1d(axes2).ravel()
    for idx, cfg_label in enumerate(configs):
        ax = axes2[idx]
        energy_16x16 = probe_results[cfg_label]["energy"]
        spline = RectBivariateSpline(np.arange(M, dtype=float), np.arange(N, dtype=float), energy_16x16, kx=3, ky=3)
        energy_smooth = np.maximum(0, spline(y_fine, x_fine, grid=True))
        energy_dB = 10.0 * np.log10(energy_smooth + 1e-15)
        vmax = np.max(energy_dB); vmin = max(-50, vmax - 50)
        im = ax.pcolormesh(x_fine, y_fine, energy_dB, shading='auto', cmap='hot', vmin=vmin, vmax=vmax)
        ax.plot(n0, m0, marker='*', color='lime', markersize=18, markeredgewidth=2, markeredgecolor='white')
        conc = probe_results[cfg_label]["concentration"]
        ax.set_xlabel('Doppler index (n)'); ax.set_ylabel('Delay index (m)')
        ax.set_title(f'{cfg_label}: Concentr.={conc:.4f}', fontsize=10, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Energy (dB)')
    for ax in axes2[len(configs):]:
        ax.set_visible(False)
    fig2.suptitle(f'2D DD Grid Energy | Probe at (m={m0}, n={n0})', fontsize=13, fontweight='bold')
    fig2.tight_layout(); fig2.savefig(os.path.join(fig_dir, "probe_energy_2d_Config_A_to_G.png"), dpi=150, bbox_inches='tight')
    plt.close(fig2); print(f"  已保存: probe_energy_2d_Config_A_to_G.png")


# ======================= Part 3b: Equivalent Basis Extraction Analysis =======================
#  从真实验证样本中用岭回归估计"全局等效基矩阵" B，
#  验证生成OTFS是否保持标准OTFS的DD域基函数聚焦特性。

def complex_flatten(mat):
    """将 [M,N,2] 或 [M,N] 转为 complex 后 C-order 展平为 [MN]"""
    if mat.ndim == 3 and mat.shape[-1] == 2:
        return (mat[..., 0] + 1j * mat[..., 1]).flatten()
    elif np.iscomplexobj(mat):
        return mat.flatten()
    else:
        return mat.flatten()


def collect_basis_matrices(val_dataset, models, max_frames=None):
    """收集验证集 QAM 输入 X 和各路径输出 S。

    Returns:
        X:      [R, MN] complex — 原始 QAM/DD 符号（展平）
        S_std:  [R, MN] complex — 标准 IDFT 输出
        S_A, S_G, S_I: [R, MN] complex — 生成 OTFS 输出
    """
    num_frames = len(val_dataset)
    if max_frames is not None:
        num_frames = min(num_frames, max_frames)
    MN = M * N
    X = np.zeros((num_frames, MN), dtype=complex)
    S_std = np.zeros((num_frames, MN), dtype=complex)
    S_A = np.zeros((num_frames, MN), dtype=complex)
    S_D = np.zeros((num_frames, MN), dtype=complex)
    S_G = np.zeros((num_frames, MN), dtype=complex)
    S_I = np.zeros((num_frames, MN), dtype=complex)

    receiver = LegacyOTFSReceiver(M=M, N=N).to(DEVICE).eval()

    print(f"  Collecting basis matrices ({num_frames} frames)...")
    for i in range(num_frames):
        bits, target_dd, qam_orig = val_dataset[i]
        qam_np = qam_orig.numpy()
        qam_cplx = qam_np[..., 0] + 1j * qam_np[..., 1]
        X[i, :] = complex_flatten(qam_cplx)

        target_np = target_dd.numpy()
        target_cplx = target_np[..., 0] + 1j * target_np[..., 1]
        S_std[i, :] = complex_flatten(target_cplx)

        qam_t = qam_orig.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            for lbl, S_mat in [("A", S_A), ("D", S_D), ("G", S_G), ("I", S_I)]:
                if lbl in models:
                    out = models[lbl].dd_gen(qam_t)
                    out_np = out.squeeze(0).cpu().numpy()
                    out_cplx = out_np[..., 0] + 1j * out_np[..., 1]
                    S_mat[i, :] = complex_flatten(out_cplx)

        if (i + 1) % 100 == 0:
            print(f"    Frame: {i + 1}/{num_frames}")

    return X, S_std, S_A, S_D, S_G, S_I


def estimate_equivalent_basis(X, S, reg_ratio=1e-6):
    """岭回归估计等效基矩阵 B = (X^H X + λI)^(-1) X^H S

    B 的第 i 行 = 第 i 个 DD 输入符号对应的等效基信号 ph_i ∈ C^{MN}
    展平顺序: C-order, i = m0 * N + n0
    """
    MN = X.shape[1]
    XHX = X.conj().T @ X  # [MN, MN] Hermitian
    tr_val = np.real(np.trace(XHX))
    lam = reg_ratio * tr_val / MN if tr_val > 1e-15 else reg_ratio
    reg_XHX = XHX + lam * np.eye(MN, dtype=complex)
    XHS = X.conj().T @ S  # [MN, MN]
    B = np.linalg.solve(reg_XHX, XHS)  # [MN, MN]
    return B


def dd_response_from_basis(phi, M, N):
    """对等效基信号 phi (time-domain, flattened) 使用 FFT 得 DD 域能量分布

    归一化与 LegacyOTFSReceiver 一致: FFT / sqrt(M*N)
    """
    phi_2d = phi.reshape(M, N)  # C-order reshape
    response = np.fft.fft(phi_2d, axis=-1) / np.sqrt(M * N)
    energy = np.abs(response) ** 2
    return energy


def compute_basis_metrics(phi_std, phi_gen, energy_gen, m0, n0, Mval, Nval):
    """计算等效基信号指标"""
    MN = Mval * Nval
    E_2d = energy_gen  # [M, N]
    E_sum = np.sum(E_2d) + 1e-15

    peak_idx = np.argmax(E_2d)
    peak_m, peak_n = np.unravel_index(peak_idx, (Mval, Nval))

    C_target = E_2d[m0, n0] / E_sum
    C_max = np.max(E_2d) / E_sum

    # 主瓣区域: |m-m0|<=1, circular |n-n0|<=1
    main_mask = np.zeros((Mval, Nval), dtype=bool)
    for m in range(Mval):
        for n in range(Nval):
            if abs(m - m0) <= 1:
                d_circ = min(abs(n - n0), Nval - abs(n - n0))
                if d_circ <= 1:
                    main_mask[m, n] = True
    MainLobeRatio = np.sum(E_2d[main_mask]) / E_sum

    SidelobeLeakage = 1.0 - MainLobeRatio

    # NMSE to standard basis
    phi_std_vec = phi_std.ravel()
    phi_gen_vec = phi_gen.ravel()
    alpha = np.vdot(phi_std_vec, phi_gen_vec) / (np.vdot(phi_std_vec, phi_std_vec) + 1e-15)
    nmse = np.sum(np.abs(phi_gen_vec - alpha * phi_std_vec) ** 2) / (np.sum(np.abs(phi_std_vec) ** 2) + 1e-15)

    # Correlation
    corr = np.abs(np.dot(phi_gen_vec.conj(), phi_std_vec)) / (
        np.sqrt(np.sum(np.abs(phi_gen_vec) ** 2) * np.sum(np.abs(phi_std_vec) ** 2)) + 1e-15)

    peak_error = np.sqrt((peak_m - m0) ** 2 + min(abs(peak_n - n0), Nval - abs(peak_n - n0)) ** 2)

    return {
        "peak_pos": (peak_m, peak_n),
        "peak_error": float(peak_error),
        "max_concentration": float(C_max),
        "target_ratio": float(C_target),
        "main_lobe_ratio": float(MainLobeRatio),
        "sidelobe_leakage": float(SidelobeLeakage),
        "nmse_to_std": float(nmse),
        "corr_to_std": float(corr),
    }


def plot_equivalent_basis_3d(energies, m0, n0, fig_dir):
    """绘制等效基响应 3D 表面图"""
    from scipy.interpolate import RectBivariateSpline
    fine_res = 120
    x_fine = np.linspace(0, N - 1, fine_res)
    y_fine = np.linspace(0, M - 1, fine_res)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine)

    configs = [("std", "Standard OTFS Equivalent Basis")]
    configs += [(lbl, CONFIG_MAP[lbl]["short_label"]) for lbl in ANALYSIS_CONFIGS if lbl in energies]

    n_cols = 4
    n_rows = int(np.ceil(len(configs) / n_cols))
    fig = plt.figure(figsize=(5.2 * n_cols, 4.8 * n_rows))
    for idx, (lbl, title) in enumerate(configs):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')
        E16 = energies[lbl]
        spline = RectBivariateSpline(np.arange(M, dtype=float), np.arange(N, dtype=float), E16, kx=3, ky=3)
        E_smooth = np.maximum(0, spline(y_fine, x_fine, grid=True))
        ax.plot_surface(X_fine, Y_fine, E_smooth, cmap='hot', linewidth=0, antialiased=True, alpha=0.85, rstride=2, cstride=2)
        ax.scatter(n0, m0, E_smooth[fine_res * m0 // M, fine_res * n0 // N], color='lime', s=100, marker='*')
        c_target = E16[m0, n0] / (np.sum(E16) + 1e-15)
        c_max = np.max(E16) / (np.sum(E16) + 1e-15)
        ax.set_xlabel('Doppler (n)'); ax.set_ylabel('Delay (m)'); ax.set_zlabel('Energy')
        ax.set_title(f'{title}\nMaxC={c_max:.3f}, TargetC={c_target:.3f}', fontsize=10, fontweight='bold')
        ax.view_init(elev=28, azim=-55)
    fig.suptitle(f'Data-driven Equivalent Basis 3D  (probe DD pos m={m0}, n={n0})', fontsize=14, fontweight='bold')
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "equivalent_basis_3d_Config_A_to_G.png"), dpi=150, bbox_inches='tight')
    plt.close(fig); print(f"  已保存: equivalent_basis_3d_Config_A_to_G.png")


def plot_equivalent_basis_2d(energies, m0, n0, fig_dir):
    """绘制等效基响应 2D 热力图"""
    fine_res = 120
    x_fine = np.linspace(0, N - 1, fine_res)
    y_fine = np.linspace(0, M - 1, fine_res)

    configs = [("std", "Standard OTFS Equivalent Basis")]
    configs += [(lbl, CONFIG_MAP[lbl]["short_label"]) for lbl in ANALYSIS_CONFIGS if lbl in energies]

    n_cols = 4
    n_rows = int(np.ceil(len(configs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.5 * n_rows))
    axes = np.atleast_2d(axes)
    for idx, (lbl, title) in enumerate(configs):
        ax = axes[idx // n_cols][idx % n_cols]
        E16 = energies[lbl]
        E_dB = 10.0 * np.log10(E16 + 1e-15)
        vmax = np.max(E_dB); vmin = max(-50, vmax - 50)
        im = ax.imshow(E_dB, aspect='auto', origin='lower', cmap='hot', vmin=vmin, vmax=vmax,
                       extent=[0, N - 1, 0, M - 1])
        ax.plot(n0, m0, marker='*', color='lime', markersize=16, markeredgewidth=2, markeredgecolor='white')
        c_main = energies[lbl][max(0, m0 - 1):min(M, m0 + 2), :].sum() / (np.sum(energies[lbl]) + 1e-15)
        ax.set_xlabel('Doppler index (n)'); ax.set_ylabel('Delay index (m)')
        ax.set_title(f'{title}\nMain={c_main:.3f}', fontsize=10, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Energy (dB)')
    for idx in range(len(configs), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis('off')
    fig.suptitle(f'Data-driven Equivalent Basis 2D  (probe DD pos m={m0}, n={n0})', fontsize=13, fontweight='bold')
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "equivalent_basis_2d_Config_A_to_G.png"), dpi=150, bbox_inches='tight')
    plt.close(fig); print(f"  已保存: equivalent_basis_2d_Config_A_to_G.png")


def plot_equivalent_basis_metrics(metrics_all, fig_dir):
    """绘制等效基指标柱状图"""
    eps = 1e-15
    metric_labels = ["Target Leakage (dB)", "Sidelobe Leakage (dB)", "Basis NMSE (dB)"]
    n_metrics = len(metric_labels)
    x = np.arange(n_metrics)
    labels = [lbl for lbl in ANALYSIS_CONFIGS if lbl in metrics_all]
    width = min(0.12, 0.75 / max(len(labels), 1))

    fig, ax = plt.subplots(figsize=(12, 6))
    offsets = {lbl: (i - (len(labels) - 1) / 2.0) * width for i, lbl in enumerate(labels)}

    for lbl in labels:
        m = metrics_all[lbl]
        vals = [
            10.0 * np.log10(max(1.0 - m["target_ratio"], eps)),
            10.0 * np.log10(max(m["sidelobe_leakage"], eps)),
            10.0 * np.log10(max(m["nmse_to_std"], eps)),
        ]
        ax.bar(x + offsets[lbl], vals, width, label=CONFIG_MAP[lbl]["short_label"],
               color=CONFIG_COLORS[lbl], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylabel("Error / leakage level (dB, lower is better)", fontsize=12)
    ax.set_title("Equivalent Basis Error Metrics Comparison", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "equivalent_basis_metrics_Config_A_to_G.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: equivalent_basis_metrics_Config_A_to_G.png")


def run_equivalent_basis_analysis(results, val_dataset, fig_dir, mat_dir):
    """主入口: 等效基提取与分析"""
    print("\n" + "=" * 60)
    print("  Part 3b: Data-driven Equivalent Basis Analysis")
    print("=" * 60)

    m0, n0 = 6, 4  # DD probe position, aligned with single-input probe analysis
    probe_idx = m0 * N + n0
    max_frames = min(len(val_dataset), 500)

    labels = active_config_labels(results)
    print("\n  Building models for dynamic equivalent basis extraction...")
    models = {}
    for lbl in labels:
        models[lbl] = build_model(lbl)
        load_end2end_weights(models[lbl], CONFIG_MAP[lbl]["weight"])

    num_frames = max_frames
    MN = M * N
    X = np.zeros((num_frames, MN), dtype=complex)
    S_std = np.zeros((num_frames, MN), dtype=complex)
    S_by_label = {lbl: np.zeros((num_frames, MN), dtype=complex) for lbl in labels}

    print(f"  Collecting dynamic basis matrices ({num_frames} frames)...")
    for i in range(num_frames):
        _, target_dd, qam_orig = val_dataset[i]
        qam_np = qam_orig.numpy()
        X[i, :] = complex_flatten(qam_np[..., 0] + 1j * qam_np[..., 1])
        target_np = target_dd.numpy()
        S_std[i, :] = complex_flatten(target_np[..., 0] + 1j * target_np[..., 1])
        qam_t = qam_orig.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            for lbl in labels:
                out = models[lbl].dd_gen(qam_t).squeeze(0).cpu().numpy()
                S_by_label[lbl][i, :] = complex_flatten(out[..., 0] + 1j * out[..., 1])
        if (i + 1) % 100 == 0:
            print(f"    Frame: {i + 1}/{num_frames}")

    print("\n  Estimating dynamic equivalent basis matrices...")
    B_std = estimate_equivalent_basis(X, S_std)
    phi_std = B_std[probe_idx, :]
    energy_std = dd_response_from_basis(phi_std, M, N)
    metrics_all = {}
    basis_mat = {
        "M": M, "N": N, "m0": m0, "n0": n0,
        "flatten_order": "C_order_index_m_times_N_plus_n",
        "energy_basis_std": np.ascontiguousarray(energy_std),
        "phi_basis_std_real": np.ascontiguousarray(np.real(phi_std).reshape(M, N)),
        "phi_basis_std_imag": np.ascontiguousarray(np.imag(phi_std).reshape(M, N)),
        "config_labels": np.array(labels, dtype=object),
        "metric_names": np.array(["max_concentration", "target_ratio", "main_lobe_ratio",
                                  "sidelobe_leakage", "nmse_to_std", "corr_to_std"], dtype=object),
    }
    for lbl in labels:
        B_lbl = estimate_equivalent_basis(X, S_by_label[lbl])
        phi_lbl = B_lbl[probe_idx, :]
        energy_lbl = dd_response_from_basis(phi_lbl, M, N)
        metrics_lbl = compute_basis_metrics(phi_std, phi_lbl, energy_lbl, m0, n0, M, N)
        metrics_all[lbl] = metrics_lbl
        basis_mat[f"energy_basis_{lbl}"] = np.ascontiguousarray(energy_lbl)
        basis_mat[f"phi_basis_{lbl}_real"] = np.ascontiguousarray(np.real(phi_lbl).reshape(M, N))
        basis_mat[f"phi_basis_{lbl}_imag"] = np.ascontiguousarray(np.imag(phi_lbl).reshape(M, N))
        basis_mat[f"metrics_{lbl}"] = np.array([
            metrics_lbl[k] for k in ["max_concentration", "target_ratio",
                                     "main_lobe_ratio", "sidelobe_leakage",
                                     "nmse_to_std", "corr_to_std"]
        ])
        print(f"    {lbl}: target_ratio={metrics_lbl['target_ratio']:.4f}, "
              f"main_lobe={metrics_lbl['main_lobe_ratio']:.4f}, "
              f"nmse={10*np.log10(metrics_lbl['nmse_to_std']+1e-15):.2f} dB")

    energies = {"std": energy_std}
    for lbl in labels:
        energies[lbl] = basis_mat[f"energy_basis_{lbl}"]
    if SAVE_EQUIV_BASIS_3D_FIG:
        plot_equivalent_basis_3d(energies, m0, n0, fig_dir)
    else:
        print("  跳过保存: equivalent_basis_3d_Config_A_to_G.png")
    if SAVE_EQUIV_BASIS_2D_FIG:
        plot_equivalent_basis_2d(energies, m0, n0, fig_dir)
    else:
        print("  跳过保存: equivalent_basis_2d_Config_A_to_G.png")
    plot_equivalent_basis_metrics(metrics_all, fig_dir)

    savemat(os.path.join(mat_dir, "equivalent_basis_Config_A_to_G.mat"), basis_mat, do_compression=True)
    print(f"  已保存: equivalent_basis_Config_A_to_G.mat")
    return metrics_all


# ======================= Part 4: BER 仿真 (AWGN / EVA / WATER) =======================
def generate_qam_constellation(modulation_order=4):
    symbols_per_dim = int(math.sqrt(modulation_order))
    max_val = symbols_per_dim - 1
    values = np.linspace(-max_val, max_val, symbols_per_dim)
    constellation = np.zeros((modulation_order, 2))
    idx = 0
    for i in range(symbols_per_dim):
        for j in range(symbols_per_dim):
            constellation[idx, 0] = values[j]; constellation[idx, 1] = values[i]
            idx += 1
    energy = np.mean(constellation[:, 0] ** 2 + constellation[:, 1] ** 2)
    return constellation / np.sqrt(energy)


def qam_demap_to_bits(symbols_complex, constellation):
    mod_order = constellation.shape[0]
    bits_per_symbol = int(np.log2(mod_order))
    syms = symbols_complex.flatten()
    const_cplx = constellation[:, 0] + 1j * constellation[:, 1]
    dist = np.abs(syms[:, np.newaxis] - const_cplx[np.newaxis, :]) ** 2
    idx = np.argmin(dist, axis=1)
    bits = np.zeros(len(idx) * bits_per_symbol, dtype=np.float32)
    for b in range(bits_per_symbol):
        bits[b::bits_per_symbol] = (idx >> (bits_per_symbol - 1 - b)) & 1
    return bits


def eva_channel_gen(M, N):
    """生成 EVA 信道参数"""
    car_fre = 4e9; delta_f = 15e3; T = 1.0 / delta_f
    delays_ns = np.array([0, 30, 150, 310, 370, 710, 1090, 1730, 2510])
    delays = delays_ns * 1e-9
    pdp_dB = np.array([0, -1.5, -1.4, -3.6, -0.6, -9.1, -7.0, -12.0, -16.9])
    one_delay_tap = 1.0 / (M * delta_f)
    delay_taps = np.round(delays / one_delay_tap).astype(int)
    taps = len(delays)
    max_speed = 120; max_UE_speed = max_speed * (1000.0 / 3600.0)
    Doppler_vel = (max_UE_speed * car_fre) / 299792458.0
    one_doppler_tap = 1.0 / (N * T)
    max_Doppler_tap = round(Doppler_vel / one_doppler_tap)
    Doppler_taps = np.round(max_Doppler_tap * np.cos(2 * np.pi * np.random.rand(taps))).astype(int)
    pow_prof = 10.0 ** (pdp_dB / 10.0); pow_prof = pow_prof / np.sum(pow_prof)
    chan_coef = np.sqrt(pow_prof) * (np.sqrt(0.5) * (np.random.randn(taps) + 1j * np.random.randn(taps)))
    return taps, delay_taps, Doppler_taps, chan_coef


def gen_time_domain_channel(M, N, taps, delay_taps, Doppler_taps, chan_coef):
    """构建时域信道矩阵 G"""
    z = np.exp(1j * 2 * np.pi / (N * M))
    l_max = int(np.max(delay_taps))
    gs = np.zeros((l_max + 1, N * M), dtype=complex)
    G = np.zeros((N * M, N * M), dtype=complex)
    for q in range(N * M):
        for i in range(taps):
            if q >= delay_taps[i]:
                gs[delay_taps[i], q] += chan_coef[i] * (z ** (Doppler_taps[i] * (q - delay_taps[i])))
    for q in range(N * M):
        for l in range(l_max + 1):
            if (q % M) >= l:
                G[q, q - l] = gs[l, q]
    return G


def gen_dd_channel_matrix(M, N, G):
    """构建 DD 域等效信道矩阵 H_DD"""
    L = M * N
    I_L = np.eye(L, dtype=complex)
    I_3d = I_L.reshape(M, N, L, order='F')
    X_mod = np.fft.ifft(I_3d, axis=1, norm='ortho').reshape(L, L, order='F')
    Y_time = G @ X_mod
    Y_3d = Y_time.reshape(M, N, L, order='F')
    H_DD = np.fft.fft(Y_3d, axis=1, norm='ortho').reshape(L, L, order='F')
    return H_DD


CHANNEL_CACHE_DIR = "channel_cache"


def channel_cache_path(channel_type, num_items):
    os.makedirs(CHANNEL_CACHE_DIR, exist_ok=True)
    if channel_type == "EVA":
        name = f"eva_M{M}_N{N}_frames{num_items}_v120_fc4GHz_df15k.npz"
    elif channel_type == "WATER":
        name = f"water_M{M}_N{N}_snap{num_items}_df{int(WATER_DELTA_F)}_fc{int(WATER_FC)}.npz"
    else:
        raise ValueError(f"Unsupported channel cache type: {channel_type}")
    return os.path.join(CHANNEL_CACHE_DIR, name)


def save_channel_cache(path, cache):
    np.savez_compressed(
        path,
        G=np.ascontiguousarray(cache["G"]),
        H_H=np.ascontiguousarray(cache["H_H"]),
        eigvals=np.ascontiguousarray(cache["eigvals"]),
        eigvecs=np.ascontiguousarray(cache["eigvecs"]),
    )
    print(f"    Saved channel cache: {path}")


def load_channel_cache(path):
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=False)
    cache = {key: data[key] for key in ["G", "H_H", "eigvals", "eigvecs"]}
    print(f"    Loaded channel cache: {path}")
    return cache


def build_eva_channel_cache(num_frames):
    path = channel_cache_path("EVA", num_frames)
    cached = load_channel_cache(path)
    if cached is not None:
        return cached

    print(f"\n  Pre-computing EVA channel matrices...")
    t0 = time.perf_counter()
    L = M * N
    cache = {
        "G": np.zeros((num_frames, L, L), dtype=np.complex128),
        "H_H": np.zeros((num_frames, L, L), dtype=np.complex128),
        "eigvals": np.zeros((num_frames, L), dtype=np.float64),
        "eigvecs": np.zeros((num_frames, L, L), dtype=np.complex128),
    }
    for frame_idx in range(num_frames):
        taps, delay_taps, Doppler_taps, chan_coef = eva_channel_gen(M, N)
        G = gen_time_domain_channel(M, N, taps, delay_taps, Doppler_taps, chan_coef)
        H_DD = np.sqrt(N) * gen_dd_channel_matrix(M, N, G)
        H_H = H_DD.conj().T
        eigvals, eigvecs = np.linalg.eigh(H_H @ H_DD)
        cache["G"][frame_idx] = G
        cache["H_H"][frame_idx] = H_H
        cache["eigvals"][frame_idx] = eigvals
        cache["eigvecs"][frame_idx] = eigvecs
        if (frame_idx + 1) % 100 == 0:
            print(f"    EVA channel: {frame_idx + 1}/{num_frames}")
    save_channel_cache(path, cache)
    print(f"    Done ({time.perf_counter() - t0:.1f}s), {num_frames} EVA frames cached")
    return cache


def build_water_channel_cache(water_lt_tot, water_hmat, water_dt, water_df, water_all_paths, water_all_dopplers):
    path = channel_cache_path("WATER", water_lt_tot)
    cached = load_channel_cache(path)
    if cached is not None:
        return cached

    print(f"\n  Pre-computing WATER channel matrices...")
    t0 = time.perf_counter()
    L = M * N
    cache = {
        "G": np.zeros((water_lt_tot, L, L), dtype=np.complex128),
        "H_H": np.zeros((water_lt_tot, L, L), dtype=np.complex128),
        "eigvals": np.zeros((water_lt_tot, L), dtype=np.float64),
        "eigvecs": np.zeros((water_lt_tot, L, L), dtype=np.complex128),
    }
    for snap_idx in range(water_lt_tot):
        taps, delay_taps, Doppler_taps, chan_coef = water_channel_gen(
            M, N, delta_f=WATER_DELTA_F, fc=WATER_FC, frame_idx=snap_idx,
            hmat_preloaded=water_hmat, dt_preloaded=water_dt, df_preloaded=water_df,
            all_paths_preloaded=water_all_paths, all_dopplers_preloaded=water_all_dopplers)
        G = gen_time_domain_channel(M, N, taps, delay_taps, Doppler_taps, chan_coef)
        H_DD = np.sqrt(N) * gen_dd_channel_matrix(M, N, G)
        H_H = H_DD.conj().T
        eigvals, eigvecs = np.linalg.eigh(H_H @ H_DD)
        cache["G"][snap_idx] = G
        cache["H_H"][snap_idx] = H_H
        cache["eigvals"][snap_idx] = eigvals
        cache["eigvecs"][snap_idx] = eigvecs
    save_channel_cache(path, cache)
    print(f"    Done ({time.perf_counter() - t0:.1f}s), {water_lt_tot} WATER snapshots cached")
    return cache


def resolve_water_channel_mat():
    """Find the WATER channel MAT file from env, common relative paths, or project search."""
    tried = []
    env_path = os.environ.get(WATER_CHANNEL_ENV, "").strip()
    candidates = []

    if env_path:
        candidates.append(env_path)
    candidates.extend([
        os.path.join(PROJECT_ROOT, WATER_CHANNEL_REL),
        os.path.join(PROJECT_ROOT, "water_channel_otfs.mat"),
        os.path.join(PROJECT_ROOT, "generate_bellhop_water_channel", "water_channel_otfs.mat"),
        os.path.join(PROJECT_ROOT, "channel_simulator", "water_channel_otfs.mat"),
    ])

    for path in candidates:
        abs_path = os.path.abspath(path)
        tried.append(abs_path)
        if os.path.exists(abs_path):
            return abs_path, tried

    for dirpath, _, filenames in os.walk(PROJECT_ROOT):
        if "performance_analysis_result" in dirpath:
            continue
        for filename in filenames:
            if filename == "water_channel_otfs.mat":
                abs_path = os.path.join(dirpath, filename)
                tried.append(abs_path)
                return abs_path, tried

    return None, tried


def select_ber_channel_types(require_water=True):
    """Select BER channels and fail loudly if requested WATER data is unavailable."""
    channel_types = ["AWGN", "EVA"]
    water_reasons = []
    water_mat, tried_paths = resolve_water_channel_mat()

    if not _HAS_BELLHOP:
        water_reasons.append(f"bellhop_water_channel import failed: {_BELLHOP_IMPORT_ERROR}")
    if water_mat is None:
        water_reasons.append(
            "missing WATER channel file. Tried: " + ", ".join(tried_paths)
        )

    if water_reasons:
        message = "WATER BER cannot be generated because " + "; ".join(water_reasons)
        if require_water:
            raise RuntimeError(message)
        print(f"  [WARNING] {message}")
        return channel_types, None

    channel_types.append("WATER")
    print(f"  WATER channel file: {os.path.relpath(water_mat, PROJECT_ROOT)}")
    print(f"  WATER parameters: delta_f={WATER_DELTA_F:.1f} Hz, fc={WATER_FC:.1f} Hz")
    return channel_types, water_mat


def run_ber_simulation(val_dataset, fig_dir, mat_dir):
    """对 Config_A_to_G 四组配置进行 AWGN / EVA / WATER 信道 BER 仿真"""
    print("\n" + "=" * 60)
    print("  BER Simulation — AWGN / EVA / WATER Channels")
    print("=" * 60)

    L = M * N
    constellation = generate_qam_constellation(MODULATION_ORDER)
    model_names = [lbl for lbl in ANALYSIS_CONFIGS if os.path.exists(CONFIG_MAP[lbl]["weight"])]
    signal_names = ["Target"] + model_names

    # 信道类型：AWGN、EVA、水声 WATER。WATER 默认必需，避免静默漏图。
    channel_selection = select_ber_channel_types(require_water=REQUIRE_WATER_BER)
    if isinstance(channel_selection, tuple):
        channel_types, water_channel_mat = channel_selection
    else:
        channel_types = channel_selection
        water_channel_mat = None

    # 构建模型（Config_A_to_G）
    models = {}
    for lbl in model_names:
        print(f"  构建模型 {lbl}...")
        models[lbl] = build_model(lbl)
        load_end2end_weights(models[lbl], CONFIG_MAP[lbl]["weight"])

    # 预生成所有信号（使用全部验证帧以保证 BER 统计量级）
    num_frames = min(len(val_dataset), 500)
    print(f"\n  预生成所有配置的信号 ({num_frames} 帧)...")
    all_bits = []
    for i in range(num_frames):
        bits, _, _ = val_dataset[i]
        all_bits.append(bits.numpy().astype(np.float32).flatten())

    all_signals = {}
    for lbl in model_names:
        sigs = []
        with torch.no_grad():
            for i in range(num_frames):
                bits_t = torch.FloatTensor(val_dataset[i][0]).unsqueeze(0).to(DEVICE)
                dd_out = models[lbl](bits_t)
                dd_np = dd_out.squeeze(0).cpu().numpy()
                sigs.append(dd_np[..., 0] + 1j * dd_np[..., 1])
        all_signals[lbl] = sigs
        print(f"    {lbl}: {num_frames} frames")

    # 加载 Target (IDFT 基线) 信号，与 Config_A_to_G 使用完全相同的帧
    print(f"    Loading Target Regulated OTFS signals...")
    target_signals = []
    for i in range(num_frames):
        _, target_dd, _ = val_dataset[i]
        target_np = target_dd.numpy()
        target_signals.append(target_np[..., 0] + 1j * target_np[..., 1])
    all_signals["Target"] = target_signals
    print(f"    Target: {num_frames} frames")

    # WATER 信道预计算（若启用）
    eva_cache = build_eva_channel_cache(num_frames) if "EVA" in channel_types else None
    water_cache = None
    water_lt_tot = 0
    if "WATER" in channel_types:
        from scipy.io import loadmat as _loadmat
        try:
            water_data = _loadmat(water_channel_mat)
            water_hmat = water_data['hmat']
            water_dt = float(water_data['dt'].ravel()[0])
            water_df = float(water_data['df'].ravel()[0])
            water_meta = load_channel_metadata(water_channel_mat)
            water_lt_tot = int(water_meta['Lt_tot'])
            if water_lt_tot <= 0:
                raise ValueError("WATER channel has no time snapshots")
            print(f"    Water channel: {water_meta['Lf']} freq x {water_lt_tot} time steps")

            water_all_paths = extract_multipath_taps(water_hmat, water_dt, water_df, peak_threshold_db=-20)
            water_all_dopplers = estimate_doppler(water_all_paths, water_dt, WATER_FC)
            print(f"    Extracted multipath for {water_lt_tot} snapshots")
            water_cache = build_water_channel_cache(
                water_lt_tot, water_hmat, water_dt, water_df, water_all_paths, water_all_dopplers)
        except Exception as exc:
            water_rel = os.path.relpath(water_channel_mat, PROJECT_ROOT) if water_channel_mat else WATER_CHANNEL_REL
            raise RuntimeError(f"Failed to prepare WATER BER channel from {water_rel}: {exc}") from exc

    ber_results = {}
    for ch_type in channel_types:
        print(f"\n  --- {ch_type} Channel ---")
        ber_results[ch_type] = {s: np.zeros(len(SNR_dB_RANGE)) for s in signal_names}
        ber_errors = {s: np.zeros(len(SNR_dB_RANGE), dtype=np.float64) for s in signal_names}
        ber_totals = {s: np.zeros(len(SNR_dB_RANGE), dtype=np.float64) for s in signal_names}

        for frame_idx in range(num_frames):
            bits_flat = all_bits[frame_idx]

            if ch_type == "EVA":
                G = eva_cache["G"][frame_idx]
                H_H = eva_cache["H_H"][frame_idx]
                eigvals = eva_cache["eigvals"][frame_idx]
                eigvecs = eva_cache["eigvecs"][frame_idx]
            elif ch_type == "WATER":
                water_frame_idx = frame_idx % water_lt_tot
                G = water_cache["G"][water_frame_idx]
                H_H = water_cache["H_H"][water_frame_idx]
                eigvals = water_cache["eigvals"][water_frame_idx]
                eigvecs = water_cache["eigvecs"][water_frame_idx]

            for snr_idx, snr_db in enumerate(SNR_dB_RANGE):
                snr_lin = 10.0 ** (snr_db / 10.0)
                for sig_name in signal_names:
                    sig_cplx = all_signals[sig_name][frame_idx]
                    x = sig_cplx.flatten('F')
                    Es = np.mean(np.abs(x) ** 2)
                    sigma2 = Es / snr_lin
                    noise = np.sqrt(sigma2 / 2.0) * (np.random.randn(L) + 1j * np.random.randn(L))

                    if ch_type == "AWGN":
                        y = x + noise
                        Y_DDgrid = np.fft.fft(y.reshape(M, N, order='F'), axis=1, norm='ortho')
                        est_bits = qam_demap_to_bits(Y_DDgrid, constellation)
                    else:
                        y = G @ x + noise
                        Y_DDgrid = np.fft.fft(y.reshape(M, N, order='F'), axis=1, norm='ortho')
                        y_dd_vec = Y_DDgrid.flatten('F')
                        b = H_H @ y_dd_vec
                        b_rot = eigvecs.conj().T @ b
                        b_rot = b_rot / (eigvals + sigma2)
                        x_hat = eigvecs @ b_rot
                        est_bits = qam_demap_to_bits(x_hat.reshape(M, N, order='F'), constellation)

                    num_b = len(bits_flat)
                    ber_errors[sig_name][snr_idx] += np.sum(est_bits != bits_flat)
                    ber_totals[sig_name][snr_idx] += num_b

            if (frame_idx + 1) % 100 == 0:
                print(f"    Frame: {frame_idx + 1}/{num_frames}")

        for snr_idx in range(len(SNR_dB_RANGE)):
            for sig_name in signal_names:
                ber_results[ch_type][sig_name][snr_idx] = (
                    ber_errors[sig_name][snr_idx] / max(ber_totals[sig_name][snr_idx], 1))

        # 输出 BER 表
        header = f"{'SNR':>5s}" + "".join(f"  {s:>12s}" for s in signal_names)
        print(f"\n  {header}")
        print("  " + "-" * (5 + 14 * len(signal_names)))
        for snr_idx, snr_db in enumerate(SNR_dB_RANGE):
            print(f"  {snr_db:5d}" + "".join(f"  {ber_results[ch_type][s][snr_idx]:12.2e}" for s in signal_names))

    # ---- 绘制 BER 曲线（参照原版 plot_ber_ADG.m 风格）----
    print("\n=== Plotting BER curves ===")
    colors_ber = {"Target": [0.00, 0.00, 0.00],
                  **{lbl: CONFIG_COLORS[lbl] for lbl in model_names}}
    markers_ber = {"Target": "o-", **{lbl: CONFIG_MARKERS[lbl] for lbl in model_names}}
    labels_ber = {"Target": "Target (Regulated OTFS Baseline)",
                  **{lbl: CONFIG_MAP[lbl]["label"] for lbl in model_names}}
    ms_ber, lw_ber = 8, 2.0

    for ch_type in channel_types:
        fig, ax = plt.subplots(figsize=(9, 6.5))
        for sig_name in signal_names:
            ber_vals = ber_results[ch_type][sig_name]
            # 将 0 值替换为很小的值避免 log(0)
            ber_vals_plot = np.maximum(ber_vals, 1e-8)
            ax.semilogy(SNR_dB_RANGE, ber_vals_plot, markers_ber[sig_name],
                        color=colors_ber[sig_name], linewidth=lw_ber, markersize=ms_ber,
                        markerfacecolor='none',
                        label=labels_ber[sig_name])
        ax.set_xlabel("SNR (dB)", fontsize=14)
        ax.set_ylabel("BER", fontsize=14)
        ax.set_title(f"BER Performance -- {ch_type} Channel  (M={M}, N={N}, 4-QAM, {num_frames} frames)",
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='lower left')
        ax.grid(True, alpha=0.3, which='both')
        # 自适应 y 轴范围（参照原版 adjust_ber_axis）
        all_ber_vals = np.concatenate([np.maximum(ber_results[ch_type][s], 1e-8) for s in signal_names])
        all_ber_vals = all_ber_vals[all_ber_vals > 0]
        if len(all_ber_vals) > 0:
            y_lo = 10 ** (max(-6, np.floor(np.log10(np.min(all_ber_vals)))))
            y_hi = 10 ** (min(0, np.ceil(np.log10(np.max(all_ber_vals)))))
            if y_hi <= y_lo:
                y_hi = 1
            ax.set_ylim([max(y_lo, 1e-6), min(y_hi, 1)])
        else:
            ax.set_ylim([1e-4, 1])
        fig.tight_layout()
        fpath = os.path.join(fig_dir, f"ber_{ch_type.lower()}.png")
        fig.savefig(fpath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  saved: ber_{ch_type.lower()}.png")

    # ---- 保存 BER MAT 数据 ----
    print("\n=== Saving BER MAT data ===")
    if REQUIRE_WATER_BER and "WATER" not in ber_results:
        raise RuntimeError("WATER BER was required but no WATER results were generated.")

    ber_mat = {"SNR_dB": np.ascontiguousarray(SNR_dB_RANGE), "M": M, "N": N,
               "modulation_order": MODULATION_ORDER, "num_frames": num_frames,
               "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               "EVA_delays_ns": np.array([0, 30, 150, 310, 370, 710, 1090, 1730, 2510]),
               "EVA_pdp_dB": np.array([0, -1.5, -1.4, -3.6, -0.6, -9.1, -7.0, -12.0, -16.9]),
               "EVA_carrier_freq_GHz": 4.0, "EVA_max_speed_kmh": 120.0,
               "EVA_subcarrier_spacing_kHz": 15.0}
    if "WATER" in channel_types:
        ber_mat["WATER_depth_m"] = 100.0; ber_mat["WATER_range_m"] = 500.0
        ber_mat["WATER_TX_depth_m"] = 50.0; ber_mat["WATER_RX_depth_m"] = 50.0
        ber_mat["WATER_BW_Hz"] = 4000.0; ber_mat["WATER_fc_Hz"] = WATER_FC
        ber_mat["WATER_delta_f_Hz"] = WATER_DELTA_F; ber_mat["WATER_sound_speed_mps"] = 1500.0
    for ch_type in channel_types:
        for sig_name in signal_names:
            ber_mat[f"ber_{sig_name}_{ch_type}"] = np.ascontiguousarray(ber_results[ch_type][sig_name])

    if REQUIRE_WATER_BER:
        missing_water = [f"ber_{sig_name}_WATER" for sig_name in signal_names
                         if f"ber_{sig_name}_WATER" not in ber_mat]
        if missing_water:
            raise RuntimeError(f"WATER BER fields missing from MAT output: {missing_water}")

    savemat(os.path.join(mat_dir, "ber_results.mat"), ber_mat, do_compression=True)
    print(f"  saved: ber_results.mat")
    if "WATER" in channel_types:
        print(f"  saved: ber_water.png and ber_*_WATER fields")

    return ber_results


# ======================= 主流程 =======================
def main():
    print("=" * 60)
    print("  SmartOTFS Performance Analysis — Config_A to Config_G")
    print(f"  Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = os.path.join("performance_analysis_result", f"analysis_{timestamp}")
    fig_dir = os.path.join(root_dir, "performance_figures")
    mat_dir = os.path.join(root_dir, "performance_mat_data")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(mat_dir, exist_ok=True)
    print(f"\n[OUTPUT] {root_dir}")

    print("\n=== Loading validation dataset ===")
    val_dataset = OTFSDataset(DATA_VAL_PATH)
    receiver = LegacyOTFSReceiver(M=M, N=N).to(DEVICE).eval()

    # ---- Part 1: 互相关 / PAPR / DD误差 ----
    results = {}
    all_target_papr = None
    for config_label in ANALYSIS_CONFIGS:
        cfg = CONFIG_MAP[config_label]
        if not os.path.exists(cfg["weight"]):
            print(f"  [WARNING] 权重文件不存在: {cfg['weight']}，跳过配置 {config_label}")
            continue
        result = analyze_config(config_label, val_dataset, receiver)
        results[config_label] = result
        if all_target_papr is None:
            all_target_papr = result["target_papr_dB"]

    if len(results) < 3:
        print("\n[WARNING] 部分配置未能加载")
        if len(results) == 0:
            print("[ERROR] 无有效配置"); sys.exit(1)

    target_auto = compute_target_autocorr(val_dataset)

    print("\n=== Plotting Part 1: 互相关 / PAPR / DD误差 ===")
    plot_cross_correlation(results, target_auto, fig_dir)
    plot_papr_ccdf(results, all_target_papr, fig_dir)
    if SAVE_DD_ERROR_HEATMAP_FIG:
        plot_dd_error_heatmap(results, fig_dir)
    else:
        print("  跳过保存: dd_error_heatmap_Config_A_to_G.png")
    plot_dd_error_profiles(results, fig_dir)

    # ---- 保存 Part 1 MAT ----
    print("\n=== Saving Part 1 MAT data ===")
    gamma = np.arange(GAMMA_MIN, GAMMA_MAX + GAMMA_STEP, GAMMA_STEP)
    corr_mat = {}
    for lbl in active_config_labels(results):
        for key, val in [("corr", "avg_corr"), ("peak_heights", "peak_heights"), ("peak_shifts", "peak_shifts")]:
            corr_mat[f"{key}_{lbl}"] = np.ascontiguousarray(results[lbl][val])
        for stat in ["peak_height", "peak_shift"]:
            arr = results[lbl][f"{stat}s"]
            corr_mat[f"{stat}_mean_{lbl}"] = float(np.mean(arr))
            corr_mat[f"{stat}_std_{lbl}"] = float(np.std(arr))
    corr_mat["corr_target_auto"] = np.ascontiguousarray(target_auto)
    savemat(os.path.join(mat_dir, "cross_correlation_Config_A_to_G.mat"), corr_mat, do_compression=True)

    ccdf_mat = {"gamma": np.ascontiguousarray(gamma),
                "ccdf_target": np.ascontiguousarray(compute_ccdf(all_target_papr, gamma))}
    for lbl in active_config_labels(results):
        ccdf_mat[f"ccdf_{lbl}"] = np.ascontiguousarray(compute_ccdf(results[lbl]["papr_dB"], gamma))
        ccdf_mat[f"papr_{lbl}_mean"] = float(np.mean(results[lbl]["papr_dB"]))
        ccdf_mat[f"papr_{lbl}_std"] = float(np.std(results[lbl]["papr_dB"]))
    ccdf_mat["papr_target_mean"] = float(np.mean(all_target_papr))
    ccdf_mat["papr_target_std"] = float(np.std(all_target_papr))
    savemat(os.path.join(mat_dir, "papr_ccdf_Config_A_to_G.mat"), ccdf_mat, do_compression=True)

    eps = 1e-12
    dd_err_mat = {}
    for lbl in active_config_labels(results):
        dd_err_mat[f"err_{lbl}"] = np.ascontiguousarray(results[lbl]["mean_err"])
        dd_err_mat[f"profile_tau_{lbl}"] = np.ascontiguousarray(np.mean(results[lbl]["mean_err"], axis=1))
        dd_err_mat[f"profile_nu_{lbl}"] = np.ascontiguousarray(np.mean(results[lbl]["mean_err"], axis=0))
        dd_err_mat[f"overall_mse_dB_{lbl}"] = float(10.0 * np.log10(np.mean(results[lbl]["mean_err"]) + eps))
    dd_err_mat["err_target"] = np.ascontiguousarray(np.zeros((M, N), dtype=np.float64))
    savemat(os.path.join(mat_dir, "dd_error_Config_A_to_G.mat"), dd_err_mat, do_compression=True)
    print("\n=== Saving requested Fig1/Fig2/Fig7 visual example data ===")
    visual_examples = build_visual_example_data(results, constellation_frames=64)
    plot_visual_examples(visual_examples, fig_dir)
    savemat(os.path.join(mat_dir, "smartotfs_visual_examples_Config_A_to_G.mat"),
            {k: np.ascontiguousarray(v) if isinstance(v, np.ndarray) else v
             for k, v in visual_examples.items()},
            do_compression=True)
    print(f"  saved: smartotfs_visual_examples_Config_A_to_G.mat")
    print(f"  已保存: cross_correlation_Config_A_to_G.mat, papr_ccdf_Config_A_to_G.mat, dd_error_Config_A_to_G.mat")

    # ---- Part 2: 时域指标 (MSE/EVM/PSD error) ----
    print("\n=== Part 2: Computing MSE / EVM / PSD Error metrics ===")
    metrics_all = {}
    for config_label in ANALYSIS_CONFIGS:
        cfg = CONFIG_MAP[config_label]
        if config_label not in results:
            continue
        m = compute_metrics_for_config(config_label, val_dataset, receiver)
        metrics_all[config_label] = m
    metrics_mat = {}
    for lbl in active_config_labels(results):
        if lbl in metrics_all:
            for k, v in metrics_all[lbl].items():
                if k.endswith("_all"):
                    metrics_mat[f"{k[:-4]}_{lbl}"] = np.ascontiguousarray(v)
                else:
                    metrics_mat[f"{k}_{lbl}"] = v
    savemat(os.path.join(mat_dir, "time_domain_metrics_Config_A_to_G.mat"), metrics_mat, do_compression=True)
    print(f"  已保存: time_domain_metrics_Config_A_to_G.mat")

    # ---- Part 3: 单点探针响应 (Single-input Probe Response) ----
    print("\n=== Part 3: Single-input Probe Response ===")
    m0, n0 = 6, 4
    probe_results = {}
    # Target (IDFT 基线): 完美 delta
    energy_target = np.zeros((M, N), dtype=np.float64)
    for sym in qam4_probe_symbols():
        qam_probe = np.zeros((M, N), dtype=complex)
        qam_probe[m0, n0] = sym
        x_time = np.fft.ifft(qam_probe, axis=-1, norm='ortho') * np.sqrt(N)
        recovered_target = np.fft.fft(x_time, axis=-1, norm='ortho') / np.sqrt(N)
        energy_target += np.abs(recovered_target) ** 2
    energy_target /= len(qam4_probe_symbols())
    probe_results["Target"] = {"energy": energy_target, "concentration": 1.0,
                                "peak_pos": (m0, n0)}
    print(f"  Target: concentration=1.0000 (delta function)")

    for config_label in ANALYSIS_CONFIGS:
        if config_label not in results:
            continue
        print(f"  Computing probe energy for {CONFIG_MAP[config_label]['label']}...")
        model = build_model(config_label)
        load_end2end_weights(model, CONFIG_MAP[config_label]["weight"])
        energy, conc, peak_pos = compute_probe_energy(config_label, m0, n0, model, receiver)
        probe_results[config_label] = {"energy": energy, "concentration": conc, "peak_pos": peak_pos}
        print(f"    Concentration={conc:.4f}, Peak=({peak_pos[0]},{peak_pos[1]})")

    plot_probe_energy(results, fig_dir, probe_results)

    # 保存探针 MAT
    probe_symbols = np.array(qam4_probe_symbols(), dtype=complex)
    probe_mat = {"M": M, "N": N, "probe_m": m0, "probe_n": n0,
                 "probe_symbols": probe_symbols}
    for lbl in ["Target"] + active_config_labels(results):
        if lbl in probe_results:
            probe_mat[f"energy_{lbl}"] = probe_results[lbl]["energy"]
            probe_mat[f"concentration_{lbl}"] = probe_results[lbl]["concentration"]
    savemat(os.path.join(mat_dir, "probe_energy_Config_A_to_G.mat"), probe_mat, do_compression=True)
    print(f"  已保存: probe_energy_Config_A_to_G.mat")

    # ---- Part 3b: 等效基提取分析 ----
    eb_metrics = run_equivalent_basis_analysis(results, val_dataset, fig_dir, mat_dir)

    # ---- Part 4: BER 仿真 ----
    print("\n=== Part 4: BER Simulation ===")
    ber_results = run_ber_simulation(val_dataset, fig_dir, mat_dir)

    # ---- Dynamic final summary for all available configurations ----
    summary_labels = active_config_labels(results)
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY - SmartOTFS Performance Analysis")
    print("=" * 70)
    header = f"{'Metric':<30}" + "".join(f"{lbl:>12}" for lbl in summary_labels) + f"{'Target':>12}"
    print(header)
    print("-" * max(70, 30 + 12 * (len(summary_labels) + 1)))
    print(f"{'Cross-corr peak mean':<30}" +
          "".join(f"{np.mean(results[lbl]['peak_heights']):>12.4f}" for lbl in summary_labels) +
          f"{'1.0000':>12}")
    print(f"{'PAPR mean (dB)':<30}" +
          "".join(f"{np.mean(results[lbl]['papr_dB']):>12.2f}" for lbl in summary_labels) +
          f"{np.mean(all_target_papr):>12.2f}")
    print(f"{'Overall MSE (dB)':<30}" +
          "".join(f"{10.0*np.log10(np.mean(results[lbl]['mean_err'])+eps):>12.2f}" for lbl in summary_labels) +
          f"{'--':>12}")

    print(f"\n{'Time-domain Metrics':<30}" + "".join(f"{lbl:>12}" for lbl in summary_labels))
    print("-" * max(70, 30 + 12 * len(summary_labels)))
    for metric_name in ["mse_mean", "evm_mean", "psd_error_mean"]:
        print(f"{metric_name:<30}" +
              "".join(f"{metrics_all.get(lbl, {}).get(metric_name, 0):>11.4f}%" for lbl in summary_labels))

    print(f"\n{'Probe Concentration':<30}")
    print("-" * 70)
    for lbl in summary_labels:
        if lbl in probe_results:
            print(f"  {CONFIG_MAP[lbl]['short_label']:<28} {probe_results[lbl]['concentration']:>12.4f}")

    for ch_type in ["AWGN", "EVA", "WATER"]:
        if ch_type not in ber_results:
            continue
        sigs = ["Target"] + [lbl for lbl in summary_labels if lbl in ber_results[ch_type]]
        print(f"\n{'BER @SNR=10dB (' + ch_type + ')':<35}" + "".join(f"{sig:>12}" for sig in sigs))
        print("-" * max(70, 35 + 12 * len(sigs)))
        snr10_idx = np.where(SNR_dB_RANGE == 10)[0]
        if len(snr10_idx) > 0:
            snr10_idx = snr10_idx[0]
            print(f"  {'BER':<33}" +
                  "".join(f"{ber_results[ch_type][sig][snr10_idx]:>12.2e}" for sig in sigs))

    print(f"\n{'='*70}")
    print(f"  ALL DONE! Output: {root_dir}")
    print(f"{'='*70}")
    return


if __name__ == "__main__":
    main()
