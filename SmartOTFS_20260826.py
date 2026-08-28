"""
渐进式 OTFS 生成系统（模块二创新：双轴解耦注意力 + 对照实验）
- 阶段1: 训练比特→QAM (QAMGenerator) - 增强容量
- 阶段2: 训练 QAM→DD域 (DDGenerator with Dual-Axis Attention / Standard Attention) - 增强容量
- 阶段3: 联合微调（启用）
评估与可视化保持与原 V3 一致

对照实验说明：
  当前运行七种配置组合：
    Config_A: Pos=Standard + Attn=Standard
    Config_B: Pos=Standard + Attn=GQA
    Config_C: Pos=Standard + Attn=Differential
    Config_D: Pos=Standard + Attn=MLA
    Config_E: Pos=Standard + Attn=Dual_Axis
    Config_F: Pos=Standard + Attn=BoltAttention
    Config_G: Pos=PhaseLoom + Attn=BoltAttention
  每种配置独立训练、验证，保存关键指标到 .mat 文件，所有实验在同一个 Python 文件中运行。

运行方式：
  - 默认运行全部七种对照实验（COMPARISON_MODE = True）
  - 如需运行单一配置，将 COMPARISON_MODE 设为 False，并修改 SINGLE_CONFIG 字典
"""

import os
import pickle
import warnings
import math
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from scipy import signal
from scipy.io import savemat, loadmat
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('training.log', mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# ==================== 全局运行模式设置 ====================
COMPARISON_MODE = True          # True: 运行四种配置对照实验; False: 运行单一配置
SINGLE_CONFIG = {               # 仅在 COMPARISON_MODE=False 时生效
    "pos_encoding_type": "Standard",    # 'Standard', 'PhaseLoom' 或 'DDPMPE'
    "attn_type": "Standard",            # 'Standard', 'Dual_Axis', 'BoltAttention', 'GQA', 'Differential' 或 'MLA'
}

# ==================== 训练跳过控制 ====================
FORCE_RETRAIN = False           # True: 强制重新训练，忽略已有权重; False: 自动跳过
VERBOSE_SKIP = True             # True: 跳过训练时打印详细验证信息
DO_WEIGHT_VERIFY = True         # True: 跳过时验证权重文件完整性（尝试加载并检查参数匹配）
FORCE_SAVE_MAT = True           # True: 强制保存 .mat 文件（即使训练被跳过）
SAVE_MAT_ON_ERROR = True        # True: 即使单个 .mat 保存失败也继续执行后续代码
USE_AMP = True                  # True: 使用混合精度训练 (torch.cuda.amp)，可提速 1.5-2x
USE_MULTI_GPU = True            # True: 使用所有可用 GPU 通过 DataParallel 并行训练

# ==================== 多 GPU 工具函数 ====================
def _unwrap_model(model: nn.Module) -> nn.Module:
    """剥离 DataParallel 包装，返回底层模型"""
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def _get_model_attr(model: nn.Module, attr: str):
    """安全获取模型属性（兼容 DataParallel 包装）"""
    return getattr(_unwrap_model(model), attr)


def _wrap_model(model: nn.Module, device: torch.device) -> nn.Module:
    """如果启用了多 GPU 且条件满足，则用 DataParallel 包装模型"""
    if (USE_MULTI_GPU and device.type == 'cuda'
            and torch.cuda.device_count() > 1
            and not isinstance(model, nn.DataParallel)):
        gpu_ids = list(range(torch.cuda.device_count()))
        logger.info(f"DataParallel: 使用 {len(gpu_ids)} 张 GPU (IDs: {gpu_ids})")
        return nn.DataParallel(model, device_ids=gpu_ids)
    return model

# ==================== 随机种子设置 ====================
BASE_SEED = 42

def set_seed(seed: int):
    """设置随机种子以确保实验可重复"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(BASE_SEED)

# 创建输出文件夹
TRAINING_PLOT_DIR = "training_plots"
TRAINING_MODEL_DIR = "training_models"
METRICS_MAT_DIR = "metrics_MAT_data"

os.makedirs(TRAINING_PLOT_DIR, exist_ok=True)
os.makedirs(TRAINING_MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_MAT_DIR, exist_ok=True)

CONFIG_METADATA = {
    ("Standard", "Standard"): {
        "name": "Config_A", "pos": "Standard", "attn": "Standard"
    },
    ("Standard", "GQA"): {
        "name": "Config_B", "pos": "Standard", "attn": "GQA"
    },
    ("Standard", "Differential"): {
        "name": "Config_C", "pos": "Standard", "attn": "Differential"
    },
    ("Standard", "MLA"): {
        "name": "Config_D", "pos": "Standard", "attn": "MLA"
    },
    ("Standard", "Dual_Axis"): {
        "name": "Config_E", "pos": "Standard", "attn": "Dual_Axis"
    },
    ("Standard", "BoltAttention"): {
        "name": "Config_F", "pos": "Standard", "attn": "BoltAttention"
    },
    ("PhaseLoom", "BoltAttention"): {
        "name": "Config_G", "pos": "PhaseLoom", "attn": "BoltAttention"
    },
}


def get_config_metadata(pos_type: str, attn_type: str) -> dict:
    return CONFIG_METADATA.get(
        (pos_type, attn_type),
        {
            "name": "Config_Custom",
            "pos": pos_type[:1].upper() + pos_type[1:],
            "attn": attn_type[:1].upper() + attn_type[1:],
        }
    )


def get_config_name(pos_type: str, attn_type: str) -> str:
    return get_config_metadata(pos_type, attn_type)["name"]


def get_config_display(pos_type: str, attn_type: str) -> str:
    meta = get_config_metadata(pos_type, attn_type)
    return f"{meta['name']} (Pos={meta['pos']}, Attn={meta['attn']})"


def get_artifact_stem(pos_type: str, attn_type: str) -> str:
    meta = get_config_metadata(pos_type, attn_type)
    return f"{meta['name']}_Pos_{meta['pos']}_Attn_{meta['attn']}"


def get_experiment_dir_name(pos_type: str, attn_type: str) -> str:
    return get_artifact_stem(pos_type, attn_type)


def get_model_dir(pos_type: str, attn_type: str) -> str:
    return os.path.join(TRAINING_MODEL_DIR, get_experiment_dir_name(pos_type, attn_type))


def get_metric_dir(pos_type: str, attn_type: str) -> str:
    return os.path.join(METRICS_MAT_DIR, get_experiment_dir_name(pos_type, attn_type))


def get_training_plot_dir(pos_type: str, attn_type: str) -> str:
    return os.path.join(TRAINING_PLOT_DIR, get_experiment_dir_name(pos_type, attn_type))


def ensure_experiment_dirs(pos_type: str, attn_type: str):
    model_dir = get_model_dir(pos_type, attn_type)
    metric_dir = get_metric_dir(pos_type, attn_type)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)
    return model_dir, metric_dir


# ==================== 工具函数：安全 mat 保存 & 权重验证 ====================
def safe_savemat(file_path: str, data_dict: dict, label: str = "") -> bool:
    """
    安全保存 .mat 文件，包含完整的错误处理和验证。

    Args:
        file_path: 保存路径
        data_dict: 要保存的数据字典
        label: 日志标签

    Returns:
        bool: 保存成功返回 True，失败返回 False
    """
    prefix = f"[{label}] " if label else ""
    target_path = os.path.normpath(file_path)

    # 1. 验证输入
    if not isinstance(data_dict, dict) or len(data_dict) == 0:
        print(f"{prefix}❌ 错误: 数据字典为空，跳过保存 {target_path}")
        return False

    # 2. 检查并转换数据
    clean_dict = {}
    for key, value in data_dict.items():
        try:
            if isinstance(value, np.ndarray):
                # 确保数组是连续的（非连续数组可能导致 savemat 静默失败）
                if not value.flags['C_CONTIGUOUS']:
                    clean_dict[key] = np.ascontiguousarray(value)
                else:
                    clean_dict[key] = value
            elif isinstance(value, str):
                # 字符串在 MATLAB v5 格式中需要特殊处理
                clean_dict[key] = value
            elif isinstance(value, (int, float, bool, complex)):
                clean_dict[key] = value
            elif isinstance(value, list):
                clean_dict[key] = np.array(value)
            elif value is None or (isinstance(value, float) and np.isnan(value)):
                clean_dict[key] = float('nan')
            else:
                print(f"{prefix}⚠️ 警告: 键 '{key}' 的类型 {type(value)} 不常见，尝试转换为 numpy 数组")
                clean_dict[key] = np.array(value)
        except Exception as e:
            print(f"{prefix}⚠️ 警告: 键 '{key}' 数据转换失败: {e}，将跳过该字段")
            continue

    if len(clean_dict) == 0:
        print(f"{prefix}❌ 错误: 所有数据字段转换失败，跳过保存 {abs_path}")
        return False

    # 3. 确保目标目录存在
    os.makedirs(os.path.dirname(target_path) or '.', exist_ok=True)

    # 4. 执行保存
    try:
        savemat(target_path, clean_dict, format='5', do_compression=False)
    except TypeError as e:
        print(f"{prefix}❌ savemat TypeError: {e}")
        print(f"{prefix}   尝试使用 v7.3 格式重新保存...")
        try:
            savemat(target_path, clean_dict, format='7.3', do_compression=False)
        except Exception as e2:
            print(f"{prefix}❌ v7.3 格式也失败: {e2}")
            return False
    except Exception as e:
        print(f"{prefix}❌ savemat 失败 ({type(e).__name__}): {e}")
        return False

    # 5. 验证文件已生成
    if not os.path.exists(target_path):
        print(f"{prefix}❌ 错误: 文件保存后未找到 {target_path}")
        return False

    file_size = os.path.getsize(target_path)
    if file_size == 0:
        print(f"{prefix}❌ 错误: 文件大小为 0: {target_path}")
        return False

    print(f"{prefix}✅ .mat 文件已保存: {target_path} ({file_size:,} bytes, {len(clean_dict)} 个变量)")
    return True


def verify_model_weights(model: nn.Module, weight_path: str, label: str = "") -> bool:
    """
    验证模型权重文件是否可以安全加载。

    Args:
        model: 模型实例（用于比对参数结构）
        weight_path: 权重文件路径
        label: 日志标签

    Returns:
        bool: 验证通过返回 True
    """
    prefix = f"[{label}] " if label else ""
    target_path = os.path.normpath(weight_path)

    # 1. 检查文件存在
    if not os.path.exists(target_path):
        if VERBOSE_SKIP:
            print(f"{prefix}🔍 权重文件不存在: {target_path}")
        return False

    # 2. 检查文件大小
    file_size = os.path.getsize(target_path)
    if file_size < 1024:  # 小于 1KB 可能是损坏文件
        print(f"{prefix}⚠️ 权重文件异常小 ({file_size} bytes)，可能损坏: {target_path}")
        return False

    # 3. 尝试加载
    try:
        state_dict = torch.load(target_path, map_location='cpu')
    except Exception as e:
        print(f"{prefix}❌ 权重文件无法加载: {e}")
        return False

    # 4. 如果提供了模型，验证参数匹配
    if model is not None and DO_WEIGHT_VERIFY:
        model_keys = set(model.state_dict().keys())
        loaded_keys = set(state_dict.keys())
        missing = model_keys - loaded_keys
        unexpected = loaded_keys - model_keys

        if missing:
            print(f"{prefix}⚠️ 权重文件缺少参数 ({len(missing)} 个): {list(missing)[:5]}...")
            return False
        if unexpected:
            print(f"{prefix}⚠️ 权重文件包含意外参数 ({len(unexpected)} 个): {list(unexpected)[:5]}...")
            return False

        if VERBOSE_SKIP:
            print(f"{prefix}✅ 权重验证通过: {target_path} ({file_size:,} bytes, {len(loaded_keys)} 个参数匹配)")

    return True


# ==================== 数据集类（不变） ====================
class OTFSDataset(Dataset):
    def __init__(self, data_file: str):
        super().__init__()
        self.data_file = data_file
        self.bits_matrices = []
        self.idft_matrices = []
        self.qam_matrices = None
        self.M = 0
        self.N = 0
        self.num_frames = 0
        self._load_data()

    def _load_data(self):
        try:
            with open(self.data_file, 'rb') as f:
                data = pickle.load(f)
            self.bits_matrices = data['bits_matrices']
            self.idft_matrices = data['idft_matrices']
            self.qam_matrices = data.get('qam_matrices', None)
            self.M = data['M']
            self.N = data['N']
            self.num_frames = data['num_frames']
            print(f"成功加载数据集: {self.data_file}")
            print(f"M={self.M}, N={self.N}, 帧数={self.num_frames}")
        except Exception as e:
            raise ValueError(f"加载数据集失败: {e}")

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx: int):
        bits = torch.FloatTensor(self.bits_matrices[idx])          # [M, N*bits]
        target = torch.FloatTensor(self.idft_matrices[idx])        # [M, N, 2]
        qam = torch.FloatTensor(self.qam_matrices[idx]) if self.qam_matrices is not None else torch.zeros(1)
        return bits, target, qam


# ==================== 二维 DD-PMPE 位置编码（原 V3） ====================
class DDPMPE2D(nn.Module):
    """DD-PMPE: 可学习的二维位置编码，基于时延-多普勒相位"""
    def __init__(self, M: int, N: int, d_model: int):
        super().__init__()
        self.M = M
        self.N = N
        self.d_model = d_model
        self.R = d_model // 2
        self.phi_tau = nn.Parameter(torch.randn(self.R) * 0.1)
        self.phi_nu = nn.Parameter(torch.randn(self.R) * 0.1)

    def forward(self):
        device = self.phi_tau.device
        m = torch.arange(self.M, dtype=torch.float32, device=device).view(-1, 1, 1)
        n = torch.arange(self.N, dtype=torch.float32, device=device).view(1, -1, 1)
        phase = 2 * math.pi * ((m / self.M) * self.phi_tau + (n / self.N) * self.phi_nu)
        cos_vals = torch.cos(phase)
        sin_vals = torch.sin(phase)
        pe = torch.stack([cos_vals, sin_vals], dim=-1).reshape(self.M, self.N, -1)
        return pe


# ==================== HAPE 位置编码（层级自适应相位编码） ====================
class HAPEPE2D(nn.Module):
    """DD-MCPMPE: 多尺度交叉耦合可学习二维位置编码

    相比 DD-PMPE 的改进：
    1. 引入时延-多普勒交叉耦合项 (m̂·n̂ 乘积项)
    2. 多尺度频率分解 (K 个对数线性尺度 s_k = 2^k)
    3. 可学习相位偏置 (解除原点相位恒为 0 的约束)

    数学定义 —— 对于位置 (m,n) 和全局频率索引 r' (映射至尺度 k, 尺度内索引 r):
        m̂ = m/M,  n̂ = n/N
        ψ_{k,r}(m,n) = 2π·2^k·[m̂·φ^τ_{k,r} + n̂·φ^ν_{k,r} + m̂·n̂·φ^c_{k,r}] + β^τ_{k,r} + β^ν_{k,r}
        PE(m,n, 2r') = cos(ψ),  PE(m,n, 2r'+1) = sin(ψ)

    参数量: 5R = 2.5·d_model (原 DD-PMPE 为 2R = d_model)

    Args:
        M, N: DD 网格尺寸
        d_model: 编码维度
        K: 频率尺度数 (默认 4，需满足 d_model//2 可被 K 整除)
    """
    def __init__(self, M: int, N: int, d_model: int, B: int = 4,
                 lambda_scale: float = 2.0, eps_nu: float = None):
        super().__init__()
        self.M = M
        self.N = N
        self.d_model = d_model
        self.B = B
        self.lambda_scale = lambda_scale
        self.R = d_model // 2
        if self.R % B != 0:
            raise ValueError(f"d_model//2={self.R} must be divisible by B={B}")
        self.R_b = self.R // B
        self.eps_nu = eps_nu if eps_nu is not None else 0.05 / N

        # ========== Innovation 1: Fourier-Aligned Doppler Basis ==========
        r_idx = torch.arange(self.R, dtype=torch.float32)
        f_nu_base = (r_idx % N) / N
        self.register_buffer('f_nu_base', f_nu_base)
        self.w_nu = nn.Parameter(torch.zeros(self.R) * 0.01)

        # ========== Innovation 2: Hierarchical Multi-Band Delay ==========
        s_vals = torch.tensor([lambda_scale ** b for b in range(B)], dtype=torch.float32)
        self.register_buffer('band_scales', s_vals.view(1, B, 1))
        self.alpha_tau = nn.Parameter(torch.randn(B, self.R_b) * 0.1)
        self.beta_raw = nn.Parameter(torch.full((B, self.R_b), 0.5413))

        # ========== Innovation 3: Frequency-Selective Sparse Gating ==========
        self.g_tau = nn.Parameter(torch.randn(self.R) * 0.01)
        self.g_nu = nn.Parameter(torch.randn(self.R) * 0.01)
        self.g_bias = nn.Parameter(torch.full((self.R,), 2.0))
        self.gamma_raw = nn.Parameter(torch.tensor(0.5413))

        # ========== Innovation 4: Band-level Amplitude Modulation ==========
        self.eta_raw = nn.Parameter(torch.tensor(-2.1972))

    def forward(self):
        device = self.w_nu.device

        m_hat = torch.arange(self.M, dtype=torch.float32,
                             device=device).view(self.M, 1, 1) / self.M
        n = torch.arange(self.N, dtype=torch.float32,
                         device=device).view(1, self.N, 1)
        n_hat = n / self.N

        # Innovation 1: Fourier-Aligned Doppler
        f_nu = self.f_nu_base.view(1, 1, self.R) + \
               self.eps_nu * torch.tanh(self.w_nu.view(1, 1, self.R))
        theta_nu = 2 * math.pi * f_nu * n

        # Innovation 2: Hierarchical Multi-Band Delay
        alpha = self.alpha_tau.view(1, self.B, self.R_b)
        beta = F.softplus(self.beta_raw.view(1, self.B, self.R_b))
        theta_tau_band = 2 * math.pi * self.band_scales * beta * \
                         m_hat.view(self.M, 1, 1) + alpha
        theta_tau = theta_tau_band.reshape(self.M, 1, self.R)

        # Innovation 3: Frequency-Selective Sparse Gating
        gamma = F.softplus(self.gamma_raw)
        gate_in = gamma * (
            m_hat * self.g_tau.view(1, 1, self.R) +
            n_hat * self.g_nu.view(1, 1, self.R) +
            self.g_bias.view(1, 1, self.R)
        )
        G = torch.sigmoid(gate_in)
        theta_gate = math.pi * (1.0 - G)

        # Total phase
        psi = theta_nu + theta_tau + theta_gate

        # Innovation 4: Band-level Amplitude Modulation + cos/sin
        G_mean = G.mean(dim=(0, 1))
        G_mean_band = G_mean.view(self.B, self.R_b).mean(dim=1)
        eta = F.softplus(self.eta_raw)
        A_band = torch.sqrt(1.0 / (1.0 + eta * (1.0 - G_mean_band)))
        A = A_band.view(self.B, 1).expand(self.B, self.R_b).reshape(self.R)

        cos_vals = A.view(1, 1, self.R) * torch.cos(psi)
        sin_vals = A.view(1, 1, self.R) * torch.sin(psi)
        pe = torch.stack([cos_vals, sin_vals], dim=-1).reshape(
            self.M, self.N, self.d_model)
        return pe


# ==================== 标准 2D 正弦余弦位置编码 ====================
class StandardPositionalEncoding2D(nn.Module):
    """
    标准的 2D 正弦余弦位置编码（不可学习，固定编码）
    将 d_model 对半分为 M 维编码和 N 维编码，分别使用不同频率的 sin/cos，
    最后沿最后一个维度拼接，得到 [M, N, d_model] 的位置编码矩阵。
    """
    def __init__(self, M: int, N: int, d_model: int):
        super().__init__()
        self.M = M
        self.N = N
        self.d_model = d_model
        half_d = d_model // 2

        # --- M 维度（时延轴）的位置编码: [M, half_d] ---
        pe_m = torch.zeros(M, half_d)
        position_m = torch.arange(0, M, dtype=torch.float32).unsqueeze(1)  # [M, 1]
        div_term_m = torch.exp(
            torch.arange(0, half_d, 2, dtype=torch.float32) * (-math.log(10000.0) / half_d)
        )  # [half_d/2]
        pe_m[:, 0::2] = torch.sin(position_m * div_term_m)
        pe_m[:, 1::2] = torch.cos(position_m * div_term_m)

        # --- N 维度（多普勒轴）的位置编码: [N, half_d] ---
        pe_n = torch.zeros(N, half_d)
        position_n = torch.arange(0, N, dtype=torch.float32).unsqueeze(1)  # [N, 1]
        div_term_n = torch.exp(
            torch.arange(0, half_d, 2, dtype=torch.float32) * (-math.log(10000.0) / half_d)
        )  # [half_d/2]
        pe_n[:, 0::2] = torch.sin(position_n * div_term_n)
        pe_n[:, 1::2] = torch.cos(position_n * div_term_n)

        # 注册为 buffer（不参与梯度更新，随模型移动到正确的 device）
        self.register_buffer('pe_m', pe_m)  # [M, half_d]
        self.register_buffer('pe_n', pe_n)  # [N, half_d]

    def forward(self):
        """
        返回: [M, N, d_model]
        对于位置 (i, j)，编码 = [pe_m[i], pe_n[j]]，即 M 维编码和 N 维编码的拼接
        """
        # pe_m: [M, 1, half_d], pe_n: [1, N, half_d]
        pe = torch.cat([
            self.pe_m.unsqueeze(1).expand(-1, self.N, -1),  # [M, N, half_d]
            self.pe_n.unsqueeze(0).expand(self.M, -1, -1)   # [M, N, half_d]
        ], dim=-1)  # [M, N, d_model]
        return pe


# ==================== 模块一：比特 → QAM（增强容量，不变） ====================
class QAMGenerator(nn.Module):
    def __init__(self, M: int = 16, N: int = 16, bits_per_symbol: int = 2,
                 d_model: int = 256, nhead: int = 4, num_layers: int = 4,
                 dim_feedforward: int = 512, dropout: float = 0.1):
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
        x = self.bit_embed(x)
        x = x + self.pos_embed
        x = x.view(B, self.M * self.N, self.d_model)
        x = self.transformer(x)
        out = self.out_proj(x)
        return out.view(B, self.M, self.N, 2)


# ==================== 创新注意力模块：双轴解耦注意力（原 V3，不变） ====================
class DualAxisSelfAttention(nn.Module):
    """
    双轴自注意力：分别在时延轴（M维）和多普勒轴（N维）上执行独立注意力
    输入 shape: (batch, M, N, d_model)
    输出 shape: (batch, M, N, d_model)
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        assert d_model % nhead == 0
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5

        self.qkv_tau = nn.Linear(d_model, 3 * d_model)
        self.qkv_nu = nn.Linear(d_model, 3 * d_model)

        self.proj_tau = nn.Linear(d_model, d_model)
        self.proj_nu = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _reshape_to_attention(self, x: torch.Tensor, axis: int):
        B, M, N, D = x.shape
        if axis == 0:  # 沿 M 轴（时延），将 N 合并到 batch
            x_reshaped = x.permute(0, 2, 1, 3).reshape(B * N, M, D)
            other_dim = N
            axis_len = M
        else:  # 沿 N 轴（多普勒），将 M 合并到 batch
            x_reshaped = x.reshape(B * M, N, D)
            other_dim = M
            axis_len = N
        return x_reshaped, other_dim, axis_len

    def _attention(self, q, k, v):
        """使用 PyTorch scaled_dot_product_attention (Flash Attention 后端)"""
        B, L, D = q.shape
        q = q.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.nhead, self.head_dim).transpose(1, 2)

        # 使用 PyTorch 2.0+ 的融合注意力内核（自动选择 Flash Attention / Memory-Efficient）
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            scale=self.scale
        )
        out = out.transpose(1, 2).reshape(B, L, D)
        return out

    def forward(self, x: torch.Tensor):
        # 1. 沿时延轴（M）的自注意力
        x_tau, other_n, M_len = self._reshape_to_attention(x, axis=0)
        qkv_tau = self.qkv_tau(x_tau).chunk(3, dim=-1)
        q, k, v = qkv_tau
        out_tau = self._attention(q, k, v)
        out_tau = self.proj_tau(out_tau)
        out_tau = out_tau.reshape(-1, other_n, M_len, self.d_model).permute(0, 2, 1, 3)

        # 2. 沿多普勒轴（N）的自注意力
        x_nu, other_m, N_len = self._reshape_to_attention(x, axis=1)
        qkv_nu = self.qkv_nu(x_nu).chunk(3, dim=-1)
        q, k, v = qkv_nu
        out_nu = self._attention(q, k, v)
        out_nu = self.proj_nu(out_nu)
        out_nu = out_nu.reshape(-1, other_m, N_len, self.d_model)

        # 3. 融合两个轴的输出（简单相加 + 残差连接）
        out = out_tau + out_nu + x
        return out


class DualAxisTransformerBlock(nn.Module):
    """双轴注意力 + 前馈网络的标准 Transformer Block"""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.self_attn = DualAxisSelfAttention(d_model, nhead, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.self_attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class DualAxisTransformer(nn.Module):
    """堆叠多个双轴 Transformer Block"""
    def __init__(self, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DualAxisTransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ==================== SACIA: 稀疏感知跨轴交互注意力机制（新增） ====================
class SACIASelfAttention(nn.Module):
    """
    SACIA 核心注意力模块 —— 实现三项创新:
      (1) 双向跨轴门控交互   (Section 3, Eq. 2-7)
      (2) 距离感知稀疏约束   (Section 4, Eq. 8-11)
      (3) 自适应融合权重     (Section 2, Eq. 1)

    输入:  X ∈ R^{B×M×N×D}     (batch, 时延维度, 多普勒维度, 特征维度)
    输出:  H_attn ∈ R^{B×M×N×D} (含自适应融合与残差连接的注意力输出)
    """

    def __init__(self, d_model: int, nhead: int, M: int, N: int,
                 dropout: float = 0.1, eta_init: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.M = M
        self.N = N
        assert d_model % nhead == 0, \
            f"d_model ({d_model}) 必须能被 nhead ({nhead}) 整除"
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5

        # ---- (1) 跨轴门控交互模块 (Section 3, Eq. 4-5) ----
        # Linear_gate_tau: c_nu → G_{ν→τ}，多普勒上下文调制时延轴
        self.linear_gate_tau = nn.Linear(d_model, d_model)
        # Linear_gate_nu:  c_tau → G_{τ→ν}，时延上下文调制多普勒轴
        self.linear_gate_nu = nn.Linear(d_model, d_model)

        # ---- (2) 双轴独立 QKV 投影 ----
        self.qkv_tau = nn.Linear(d_model, 3 * d_model)   # 时延轴 QKV
        self.qkv_nu  = nn.Linear(d_model, 3 * d_model)   # 多普勒轴 QKV
        self.proj_tau = nn.Linear(d_model, d_model)       # 时延轴输出投影
        self.proj_nu  = nn.Linear(d_model, d_model)       # 多普勒轴输出投影

        # ---- (3) 距离感知稀疏参数 (Section 4, Eq. 8-11) ----
        self.lambda_tau = nn.Parameter(torch.tensor(0.01))   # λ_τ: 时延轴稀疏强度
        self.lambda_nu  = nn.Parameter(torch.tensor(0.01))   # λ_ν: 多普勒轴稀疏强度
        self.gamma_tau  = nn.Parameter(torch.tensor(1.0))    # γ_τ: 时延衰减指数
        self.gamma_nu   = nn.Parameter(torch.tensor(1.0))    # γ_ν: 多普勒衰减指数
        self.eta = nn.Parameter(torch.tensor(eta_init))      # η:   周期调制强度

        # ---- (4) 自适应融合权重 (Section 2, Eq. 1) ----
        self.alpha_tau = nn.Parameter(torch.ones(d_model))   # α_τ (初始全1)
        self.alpha_nu  = nn.Parameter(torch.ones(d_model))   # α_ν (初始全1)

        self.dropout = nn.Dropout(dropout)

        # ---- (5) 预构建距离矩阵基底 (不参与梯度更新，仅查表使用) ----
        self._build_base_matrices(M, N)

    # ------------------------------------------------------------------
    # 预构建距离基底
    # ------------------------------------------------------------------
    def _build_base_matrices(self, M: int, N: int):
        """构建 |i-j| 与圆周距离的基底矩阵，注册为 buffer 自动随模型迁移设备"""
        # --- 时延轴: |i-j|  (Eq. 9 的基底) ---
        i_tau = torch.arange(M, dtype=torch.float32).unsqueeze(1)   # [M, 1]
        j_tau = torch.arange(M, dtype=torch.float32).unsqueeze(0)   # [1, M]
        base_dist_tau = torch.abs(i_tau - j_tau)                     # [M, M]
        self.register_buffer('base_dist_tau', base_dist_tau)

        # --- 多普勒轴: 圆周距离 |i-j|_circ 与 sin² 周期性项基底 (Eq. 10) ---
        i_nu = torch.arange(N, dtype=torch.float32).unsqueeze(1)    # [N, 1]
        j_nu = torch.arange(N, dtype=torch.float32).unsqueeze(0)    # [1, N]
        abs_diff = torch.abs(i_nu - j_nu)
        base_circ_dist_nu = torch.min(abs_diff, N - abs_diff)        # [N, N]
        base_sin2_nu = torch.sin(torch.pi * (i_nu - j_nu) / N) ** 2 # [N, N]
        self.register_buffer('base_circ_dist_nu', base_circ_dist_nu)
        self.register_buffer('base_sin2_nu', base_sin2_nu)

    # ------------------------------------------------------------------
    # 前向传播中按当前 γ/η 计算惩罚矩阵 Φ (Eq. 9-10)
    # ------------------------------------------------------------------
    def _compute_Phi(self):
        """
        根据当前可学习参数构造惩罚矩阵。
        Φ 本身不注册为 nn.Parameter，仅通过 γ, η 间接参与梯度更新。

        Eq. 9:  Φ_τ[i,j] = |i - j|^{γ_τ}
        Eq. 10: Φ_ν[i,j] = |i - j|_{circ}^{γ_ν} · (1 + η · sin²(π(i-j)/N))
        """
        Phi_tau = self.base_dist_tau ** self.gamma_tau.abs()               # [M, M]

        Phi_nu = (self.base_circ_dist_nu ** self.gamma_nu.abs()) * \
                 (1.0 + self.eta.abs() * self.base_sin2_nu)                # [N, N]

        return Phi_tau, Phi_nu

    # ------------------------------------------------------------------
    # 距离感知稀疏注意力计算 (Section 4)
    # ------------------------------------------------------------------
    def _sparse_attention(self, q: torch.Tensor, k: torch.Tensor,
                          v: torch.Tensor, Phi: torch.Tensor,
                          lambda_val: torch.nn.Parameter,
                          seq_len: int) -> torch.Tensor:
        """
        带距离惩罚的缩放点积注意力 (Eq. 8 / Eq. 11)。

        Args:
            q, k, v:     [B', L, D]  合并批次后的序列
            Phi:         [L, L]      惩罚矩阵
            lambda_val:  可学习标量 (λ_τ 或 λ_ν)
            seq_len:     序列长度 (M 或 N)

        Returns:
            out: [B', L, D]
        """
        B_prime, L, D = q.shape

        # 多头重塑: [B', L, D] → [B', h, L, d_k]
        q = q.view(B_prime, L, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B_prime, L, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B_prime, L, self.nhead, self.head_dim).transpose(1, 2)

        # 缩放点积得分: QKᵀ / √d_k
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale     # [B', h, L, L]

        # 距离感知稀疏惩罚: scores − λ·Φ
        penalty = lambda_val.abs() * Phi.unsqueeze(0).unsqueeze(0)     # [1, 1, L, L]
        scores = scores - penalty

        # Softmax + Dropout
        attn_weights = F.softmax(scores, dim=-1)                        # [B', h, L, L]
        attn_weights = self.dropout(attn_weights)

        # Weighted sum: softmax(...)·V
        out = torch.matmul(attn_weights, v)                             # [B', h, L, d_k]
        out = out.transpose(1, 2).reshape(B_prime, L, D)               # [B', L, D]
        return out

    # ------------------------------------------------------------------
    # SACIA 前向传播 (Section 2-4)
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:  [B, M, N, D]

        Returns:
            H_attn: [B, M, N, D]  →  α_τ⊙H_τ + α_ν⊙H_ν + X
        """
        B, M, N, D = x.shape

        # ============ Step 1: 跨轴门控交互 (Section 3, Eq. 2-7) ============

        # Eq. 2: c_τ = (1/N)·Σ_n X  →  [B, M, D]
        c_tau = x.mean(dim=2)
        # Eq. 3: c_ν = (1/M)·Σ_m X  →  [B, N, D]
        c_nu = x.mean(dim=1)

        # Eq. 4: G_{ν→τ} = σ(Linear_gate_tau(c_ν))  →  [B, 1, N, D]
        G_nu_to_tau = torch.sigmoid(self.linear_gate_tau(c_nu)).unsqueeze(1)

        # Eq. 5: G_{τ→ν} = σ(Linear_gate_nu(c_τ))  →  [B, M, 1, D]
        G_tau_to_nu = torch.sigmoid(self.linear_gate_nu(c_tau)).unsqueeze(2)

        # Eq. 6: X̃_τ = X ⊙ G_{ν→τ} + X   (broadcast [B, 1, N, D] → [B, M, N, D])
        X_tilde_tau = x * G_nu_to_tau + x
        # Eq. 7: X̃_ν = X ⊙ G_{τ→ν} + X   (broadcast [B, M, 1, D] → [B, M, N, D])
        X_tilde_nu = x * G_tau_to_nu + x

        # 按当前 γ/η 计算惩罚矩阵
        Phi_tau, Phi_nu = self._compute_Phi()

        # ============ Step 2: 时延轴稀疏多头注意力 (Section 4.1, Eq. 8-9) ============
        # 重塑: [B, M, N, D] → [B×N, M, D]
        x_tau = X_tilde_tau.permute(0, 2, 1, 3).reshape(B * N, M, D)
        qkv_tau = self.qkv_tau(x_tau).chunk(3, dim=-1)
        H_tau = self._sparse_attention(
            qkv_tau[0], qkv_tau[1], qkv_tau[2], Phi_tau, self.lambda_tau, M)
        H_tau = self.proj_tau(H_tau)                                    # [B×N, M, D]
        H_tau = H_tau.reshape(B, N, M, D).permute(0, 2, 1, 3)          # [B, M, N, D]

        # ============ Step 3: 多普勒轴稀疏多头注意力 (Section 4.2, Eq. 10-11) ============
        # 重塑: [B, M, N, D] → [B×M, N, D]
        x_nu = X_tilde_nu.reshape(B * M, N, D)
        qkv_nu = self.qkv_nu(x_nu).chunk(3, dim=-1)
        H_nu = self._sparse_attention(
            qkv_nu[0], qkv_nu[1], qkv_nu[2], Phi_nu, self.lambda_nu, N)
        H_nu = self.proj_nu(H_nu)                                       # [B×M, N, D]
        H_nu = H_nu.reshape(B, M, N, D)                                 # [B, M, N, D]

        # ============ Step 4: 自适应融合 + 残差连接 (Section 2, Eq. 1) ============
        # H_attn = α_τ ⊙ H_τ + α_ν ⊙ H_ν + X
        H_attn = self.alpha_tau * H_tau + self.alpha_nu * H_nu + x

        return H_attn


class SACIACoderLayer(nn.Module):
    """
    SACIA 完整编码器层 (Section 6, Eq. 14)

    执行流程:
      跨轴门控 → 双路稀疏多头注意力 → 自适应融合残差 → LayerNorm → FFN → 残差
    """

    def __init__(self, d_model: int, nhead: int, M: int, N: int,
                 dim_feedforward: int = 1024, dropout: float = 0.1,
                 eta_init: float = 1.0):
        super().__init__()

        # SACIA 核心注意力（内含跨轴门控 + 稀疏约束 + 自适应融合）
        self.self_attn = SACIASelfAttention(d_model, nhead, M, N, dropout, eta_init)

        # 前馈网络 (Eq. 14: FFN = Linear → GELU → Dropout → Linear → Dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        # 层归一化 (Section 6: post-attention norm)
        self.norm1 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, M, N, D]
        Returns:
            H_out: [B, M, N, D]
        """
        # H_attn = α_τ⊙H_τ + α_ν⊙H_ν + X   (Section 2, Eq. 1)
        H_attn = self.self_attn(x)             # [B, M, N, D]

        # H_norm = LayerNorm(H_attn)           (Section 6, Eq. 14)
        H_norm = self.norm1(H_attn)            # [B, M, N, D]

        # H_out = FFN(H_norm) + H_norm         (Section 6, Eq. 14)
        H_out = self.ffn(H_norm) + H_norm      # [B, M, N, D]

        return H_out


class SACIATransformer(nn.Module):
    """
    堆叠多个 SACIA 编码器层。

    输入:  [B, M, N, D]
    输出:  [B, M, N, D]
    """

    def __init__(self, d_model: int, nhead: int, M: int, N: int,
                 num_layers: int, dim_feedforward: int = 1024,
                 dropout: float = 0.1, eta_init: float = 1.0):
        super().__init__()
        self.layers = nn.ModuleList([
            SACIACoderLayer(d_model, nhead, M, N,
                            dim_feedforward, dropout, eta_init)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ==================== 标准 Transformer 自注意力（新增） ====================
class StandardTransformer(nn.Module):
    """
    标准全局自注意力：将 2D 网格展平为 1D 序列，使用 PyTorch 原生 TransformerEncoder
    输入 shape: (batch, M, N, d_model)
    输出 shape: (batch, M, N, d_model)
    """
    def __init__(self, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, M, N, D = x.shape
        x = x.reshape(B, M * N, D)               # [B, M*N, d_model]
        x = self.encoder(x)                       # [B, M*N, d_model]
        x = x.reshape(B, M, N, D)                 # [B, M, N, d_model]
        return x


# ==================== 模块二：QAM → DD域（修改：支持可配置位置编码和注意力类型） ====================
class GQASelfAttention(nn.Module):
    """Grouped-query attention over the flattened DD grid."""
    def __init__(self, d_model: int, nhead: int, kv_heads: int = None,
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.kv_heads = kv_heads or max(1, nhead // 4)
        assert nhead % self.kv_heads == 0
        self.group_size = nhead // self.kv_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, self.kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, self.kv_heads * self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.kv_heads, self.head_dim).transpose(1, 2)
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)


class DifferentialSelfAttention(nn.Module):
    """Differential attention: softmax(Q1K1^T) - lambda * softmax(Q2K2^T)."""
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
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

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        return x.view(B, L, self.nhead, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q1 = self._shape(self.q1_proj(x))
        k1 = self._shape(self.k1_proj(x))
        q2 = self._shape(self.q2_proj(x))
        k2 = self._shape(self.k2_proj(x))
        v = self._shape(self.v_proj(x))

        a1 = F.softmax(torch.matmul(q1, k1.transpose(-2, -1)) * self.scale, dim=-1)
        a2 = F.softmax(torch.matmul(q2, k2.transpose(-2, -1)) * self.scale, dim=-1)
        lam = F.softplus(self.lambda_param)
        attn = self.dropout(a1 - lam * a2)
        out = torch.matmul(attn, v)
        B, _, L, _ = out.shape
        out = out.transpose(1, 2).reshape(B, L, self.d_model)
        return self.out_proj(out)


class GatedDeltaSelfAttention(nn.Module):
    """Linear recurrent attention with gated DeltaNet-style state updates."""
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.alpha_proj = nn.Linear(d_model, nhead)
        self.beta_proj = nn.Linear(d_model, nhead)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        return x.view(B, L, self.nhead, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        q = F.elu(self._shape(self.q_proj(x))) + 1.0
        k = F.elu(self._shape(self.k_proj(x))) + 1.0
        v = self._shape(self.v_proj(x))
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

        alpha = torch.sigmoid(self.alpha_proj(x)).transpose(1, 2).unsqueeze(-1)
        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2).unsqueeze(-1)
        state = x.new_zeros(B, self.nhead, self.head_dim, self.head_dim)
        outs = []
        for t in range(L):
            kt = k[:, :, t, :]
            vt = v[:, :, t, :]
            qt = q[:, :, t, :]
            old_v = torch.matmul(state.transpose(-2, -1), kt.unsqueeze(-1)).squeeze(-1)
            delta_v = vt - old_v
            write = torch.matmul(kt.unsqueeze(-1), delta_v.unsqueeze(-2))
            state = alpha[:, :, t, :].unsqueeze(-1) * state + beta[:, :, t, :].unsqueeze(-1) * write
            ot = torch.matmul(state.transpose(-2, -1), qt.unsqueeze(-1)).squeeze(-1)
            outs.append(ot)

        out = torch.stack(outs, dim=2).transpose(1, 2).reshape(B, L, self.d_model)
        return self.out_proj(self.dropout(out))


class MLASelfAttention(nn.Module):
    """Multi-head latent attention with low-rank query and KV compression."""
    def __init__(self, d_model: int, nhead: int, latent_dim: int = None,
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.latent_dim = latent_dim or max(d_model // 4, self.head_dim)

        self.q_down = nn.Linear(d_model, self.latent_dim)
        self.q_up = nn.Linear(self.latent_dim, d_model)
        self.kv_down = nn.Linear(d_model, self.latent_dim)
        self.k_up = nn.Linear(self.latent_dim, d_model)
        self.v_up = nn.Linear(self.latent_dim, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        return x.view(B, L, self.nhead, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        q_latent = self.q_down(x)
        kv_latent = self.kv_down(x)

        q = self._shape(self.q_up(q_latent))
        k = self._shape(self.k_up(kv_latent))
        v = self._shape(self.v_up(kv_latent))

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)


class FlatAttentionTransformerBlock(nn.Module):
    """Shared flattened-grid Transformer block for new attention variants."""
    def __init__(self, attn_module: nn.Module, d_model: int,
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = attn_module
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class FlatAttentionTransformer(nn.Module):
    """Stack flattened-grid blocks and restore the DD grid shape."""
    def __init__(self, attn_type: str, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        builders = {
            'GQA': lambda: GQASelfAttention(d_model, nhead, dropout=dropout),
            'Differential': lambda: DifferentialSelfAttention(d_model, nhead, dropout),
            'Gated_Delta': lambda: GatedDeltaSelfAttention(d_model, nhead, dropout),
            'MLA': lambda: MLASelfAttention(d_model, nhead, dropout=dropout),
        }
        if attn_type not in builders:
            raise ValueError(f"Unknown flat attention type: {attn_type}")
        self.layers = nn.ModuleList([
            FlatAttentionTransformerBlock(
                builders[attn_type](), d_model, dim_feedforward, dropout
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, M, N, D = x.shape
        x = x.reshape(B, M * N, D)
        for layer in self.layers:
            x = layer(x)
        return x.reshape(B, M, N, D)


class DDGeneratorV3(nn.Module):
    """
    输入 QAM: [batch, M, N, 2]
    输出 DD 域符号: [batch, M, N, 2]

    Args:
        pos_encoding_type: 'DDPMPE' (可学习DD-PMPE), 'PhaseLoom' (层级自适应HAPE),
                           或 'Standard' (固定正弦余弦)
        attn_type: 'Dual_Axis' (双轴解耦) 或 'Standard' (标准全局自注意力) 或 'BoltAttention'
    """
    def __init__(self, M: int = 16, N: int = 16,
                 d_model: int = 256, nhead: int = 8, num_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 pos_encoding_type: str = 'DDPMPE',
                 attn_type: str = 'Dual_Axis'):
        super().__init__()
        self.M = M
        self.N = N
        self.d_model = d_model
        self.pos_encoding_type = pos_encoding_type
        self.attn_type = attn_type

        # 输入嵌入：2 通道 (I/Q) -> d_model
        self.qam_embed = nn.Linear(2, d_model)

        # ---- 位置编码 ----
        if pos_encoding_type == 'DDPMPE':
            self.pos_encoder = DDPMPE2D(M, N, d_model)
        elif pos_encoding_type == 'PhaseLoom':
            self.pos_encoder = HAPEPE2D(M, N, d_model)
        elif pos_encoding_type == 'Standard':
            self.pos_encoder = StandardPositionalEncoding2D(M, N, d_model)
        else:
            raise ValueError(f"Unknown pos_encoding_type: {pos_encoding_type}")

        # ---- 注意力机制 ----
        if attn_type == 'Dual_Axis':
            self.transformer = DualAxisTransformer(
                d_model=d_model, nhead=nhead, num_layers=num_layers,
                dim_feedforward=dim_feedforward, dropout=dropout
            )
        elif attn_type == 'Standard':
            self.transformer = StandardTransformer(
                d_model=d_model, nhead=nhead, num_layers=num_layers,
                dim_feedforward=dim_feedforward, dropout=dropout
            )
        elif attn_type == 'BoltAttention':
            self.transformer = SACIATransformer(
                d_model=d_model, nhead=nhead, M=M, N=N, num_layers=num_layers,
                dim_feedforward=dim_feedforward, dropout=dropout
            )
        elif attn_type in ('GQA', 'Differential', 'Gated_Delta', 'MLA'):
            self.transformer = FlatAttentionTransformer(
                attn_type=attn_type, d_model=d_model, nhead=nhead,
                num_layers=num_layers, dim_feedforward=dim_feedforward,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown attn_type: {attn_type}")

        # 输出投影：d_model -> 2
        self.out_proj = nn.Linear(d_model, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.qam_embed(x)                           # [B, M, N, d_model]
        x = x + self.pos_encoder().unsqueeze(0)         # [B, M, N, d_model]
        x = self.transformer(x)                         # [B, M, N, d_model]
        out = self.out_proj(x)                          # [B, M, N, 2]
        return out


# ==================== 传统接收机（不变） ====================
class LegacyOTFSReceiver(nn.Module):
    def __init__(self, M: int = 16, N: int = 16):
        super().__init__()
        self.M = M
        self.N = N

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == 2:
            x_complex = torch.complex(x[..., 0], x[..., 1])
        else:
            x_complex = x
        X = torch.fft.fft(x_complex, dim=-1)
        X = X / torch.sqrt(torch.tensor(self.M * self.N, dtype=x_complex.dtype))
        return X


# ==================== PSD 损失、EVM 计算（不变） ====================
def psd_mse_loss(signal_complex: torch.Tensor, target_complex: torch.Tensor) -> torch.Tensor:
    batch_size = signal_complex.size(0)
    sig_flat = signal_complex.view(batch_size, -1)
    tgt_flat = target_complex.view(batch_size, -1)
    sig_fft = torch.fft.fft(sig_flat, dim=-1)
    tgt_fft = torch.fft.fft(tgt_flat, dim=-1)
    sig_psd = torch.abs(sig_fft) ** 2
    tgt_psd = torch.abs(tgt_fft) ** 2
    sig_psd = sig_psd / (sig_psd.sum(dim=-1, keepdim=True) + 1e-8)
    tgt_psd = tgt_psd / (tgt_psd.sum(dim=-1, keepdim=True) + 1e-8)
    log_sig = torch.log(sig_psd + 1e-8)
    log_tgt = torch.log(tgt_psd + 1e-8)
    per_sample_mse = torch.mean((log_sig - log_tgt) ** 2, dim=-1)
    return torch.mean(per_sample_mse)


def compute_evm(original_qam: torch.Tensor, recovered_qam: torch.Tensor) -> torch.Tensor:
    if not torch.is_complex(recovered_qam):
        raise ValueError("recovered_qam must be complex")
    if original_qam.dim() == 4 and original_qam.shape[-1] == 2:
        original_qam = torch.complex(original_qam[..., 0], original_qam[..., 1])
    batch_size = original_qam.size(0)
    orig_flat = original_qam.reshape(batch_size, -1)
    rec_flat = recovered_qam.reshape(batch_size, -1)
    error_power = torch.abs(rec_flat - orig_flat) ** 2
    signal_power = torch.abs(orig_flat) ** 2
    mse = torch.mean(error_power, dim=1)
    avg_signal_power = torch.mean(signal_power, dim=1)
    evm = torch.sqrt(mse / (avg_signal_power + 1e-12))
    return evm


# ===== PAPR 新增 =====
def compute_papr(signal_complex: torch.Tensor) -> torch.Tensor:
    """
    计算时域复数信号的 PAPR (Peak-to-Average Power Ratio，线性值，非 dB)。

    PAPR = max(|s_i|^2) / mean(|s_i|^2)

    Args:
        signal_complex: 复数张量，形状 [B, M, N] 或 [B, L]

    Returns:
        papr: 形状 [B]，每个样本的 PAPR 值（线性值）
    """
    batch_size = signal_complex.size(0)
    s_flat = signal_complex.reshape(batch_size, -1)
    power = torch.abs(s_flat) ** 2
    papr = power.max(dim=-1).values / (power.mean(dim=-1) + 1e-12)
    return papr
# ===== PAPR 新增结束 =====


# ==================== 统一组合损失计算（消除 5 处重复代码） ====================
def compute_combined_loss(out_dd: torch.Tensor, target_dd: torch.Tensor,
                          qam_batch: torch.Tensor, receiver: nn.Module,
                          loss_weights: dict, device: torch.device):
    """
    统一计算 MSE + PSD + EVM 组合损失，所有训练/验证/评估均调用此函数。

    Args:
        out_dd: 模型输出 DD 域符号 [B, M, N, 2]
        target_dd: 目标 DD 域符号 [B, M, N, 2]
        qam_batch: QAM 批次 [B, M, N, 2]
        receiver: LegacyOTFSReceiver 实例
        loss_weights: {'mse': w1, 'psd': w2, 'evm': w3}
        device: 计算设备

    Returns:
        total_loss, mse_loss, psd_loss, evm_loss (均为标量 Tensor)
    """
    mse_loss = F.mse_loss(out_dd, target_dd)

    # 强制转为 FP32：AMP autocast 会产出 FP16 复数，但 FFT/sqrt 不支持 ComplexHalf
    target_complex = torch.complex(target_dd[..., 0].float(), target_dd[..., 1].float())
    out_complex = torch.complex(out_dd[..., 0].float(), out_dd[..., 1].float())

    psd_loss = psd_mse_loss(out_complex, target_complex) if loss_weights.get('psd', 0) > 0 \
        else torch.tensor(0.0, device=device)

    evm_loss = torch.tensor(0.0, device=device)
    if loss_weights.get('evm', 0) > 0:
        # receiver 内部也需要 FP32，传入 float() 版本
        recovered_qam = receiver(out_dd.float())
        evm_val = compute_evm(qam_batch.float(), recovered_qam)
        evm_loss = torch.mean(evm_val)

    total_loss = (loss_weights['mse'] * mse_loss +
                  loss_weights['psd'] * psd_loss +
                  loss_weights['evm'] * evm_loss)
    return total_loss, mse_loss, psd_loss, evm_loss


# ==================== 训练函数（阶段2），修改：返回全部指标 + 早停 ====================
def train_dd_generator_v3(qam_gen, dd_gen, train_loader, val_loader, device,
                          epochs=200, lr=5e-4, loss_weights=None,
                          best_model_path=os.path.join(TRAINING_MODEL_DIR, "shared", "best_dd_generator_v3.pth"),
                          early_stopping_patience: int = 50,
                          config_label: str = ""):
    """
    训练 DDGenerator（阶段2）

    Args:
        early_stopping_patience: 早停耐心值，若验证损失连续该轮数未改善则提前停止
        config_label: 配置标签（用于打印标识）

    Returns:
        train_losses, val_losses, train_mse_hist, val_mse_hist,
        train_psd_hist, val_psd_hist, train_evm_hist, val_evm_hist
    """
    if loss_weights is None:
        loss_weights = {'mse': 1.0, 'psd': 0.001, 'evm': 0.001}

    qam_gen.eval()
    for param in qam_gen.parameters():
        param.requires_grad = False

    # 多 GPU 包装
    dd_gen = _wrap_model(dd_gen, device)

    receiver = LegacyOTFSReceiver(M=_get_model_attr(dd_gen, 'M'),
                                   N=_get_model_attr(dd_gen, 'N')).to(device)
    for param in receiver.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(dd_gen.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.999))
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device.type == 'cuda'))

    warmup_epochs = 20
    scheduler1 = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    scheduler2 = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-7)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_epochs])

    best_val_loss = float('inf')
    epochs_no_improve = 0
    stopped_early = False

    train_losses, val_losses = [], []
    train_mse_hist, val_mse_hist = [], []
    train_psd_hist, val_psd_hist = [], []
    train_evm_hist, val_evm_hist = [], []

    label_str = f" [{config_label}]" if config_label else ""
    os.makedirs(os.path.dirname(best_model_path) or '.', exist_ok=True)

    for epoch in range(epochs):
        # ---- 训练 ----
        dd_gen.train()
        total_loss = 0.0
        total_mse = 0.0
        total_psd = 0.0
        total_evm = 0.0

        for bits, target_dd, qam_batch in train_loader:
            bits, target_dd, qam_batch = bits.to(device), target_dd.to(device), qam_batch.to(device)
            with torch.no_grad():
                qam = qam_gen(bits)

            with torch.cuda.amp.autocast(enabled=(USE_AMP and device.type == 'cuda')):
                out_dd = dd_gen(qam)
            # 损失计算在 autocast 外部（FP32），避免 FFT/sqrt 对 ComplexHalf 不兼容
            loss, mse_loss, psd_loss, evm_loss = compute_combined_loss(
                out_dd, target_dd, qam_batch, receiver, loss_weights, device)

            if torch.isnan(loss) or loss > 100:
                logger.warning(f"Skipping batch due to loss={loss.item():.3f}")
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(dd_gen.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()

            batch_size = bits.size(0)
            total_loss += loss.item() * batch_size
            total_mse += mse_loss.item() * batch_size
            total_psd += psd_loss.item() * batch_size
            total_evm += evm_loss.item() * batch_size

        # ---- 验证 ----
        dd_gen.eval()
        val_loss = 0.0
        val_mse_sum = 0.0
        val_psd_sum = 0.0
        val_evm_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for bits, target_dd, qam_batch in val_loader:
                bits, target_dd, qam_batch = bits.to(device), target_dd.to(device), qam_batch.to(device)
                qam = qam_gen(bits)
                out_dd = dd_gen(qam)
                loss, mse_loss, psd_loss, evm_loss = compute_combined_loss(
                    out_dd, target_dd, qam_batch, receiver, loss_weights, device)

                batch_sz = bits.size(0)
                val_loss += loss.item() * batch_sz
                val_mse_sum += mse_loss.item() * batch_sz
                val_psd_sum += psd_loss.item() * batch_sz
                val_evm_sum += evm_loss.item() * batch_sz
                val_count += batch_sz

        n_train = len(train_loader.dataset)
        n_val = max(val_count, 1)
        epoch_train_loss = total_loss / n_train
        epoch_val_loss = val_loss / n_val
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_mse_hist.append(total_mse / n_train)
        val_mse_hist.append(val_mse_sum / n_val)
        train_psd_hist.append(total_psd / n_train)
        val_psd_hist.append(val_psd_sum / n_val)
        train_evm_hist.append(total_evm / n_train)
        val_evm_hist.append(val_evm_sum / n_val)

        # 保存最佳模型 + 早停检查
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            torch.save(_unwrap_model(dd_gen).state_dict(), best_model_path)
            print(f"Epoch {epoch+1}{label_str}: val loss={epoch_val_loss:.6f} (best, saved)")
        else:
            epochs_no_improve += 1

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}{label_str}: train loss={epoch_train_loss:.6f} "
                  f"(MSE:{train_mse_hist[-1]:.6f} PSD:{train_psd_hist[-1]:.6f} EVM:{train_evm_hist[-1]:.6f}) | "
                  f"val loss={epoch_val_loss:.6f} "
                  f"(MSE:{val_mse_hist[-1]:.6f} PSD:{val_psd_hist[-1]:.6f} EVM:{val_evm_hist[-1]:.6f})")

        # 早停判断
        if epochs_no_improve >= early_stopping_patience:
            print(f"早停触发于 Epoch {epoch+1}{label_str}，{early_stopping_patience} 轮内验证损失未改善。")
            stopped_early = True
            break

    total_epochs = len(train_losses)
    print(f"阶段2完成{label_str}：共训练 {total_epochs} 轮"
          f"{'（早停）' if stopped_early else ''}，最佳验证损失: {best_val_loss:.6f}")

    return (train_losses, val_losses,
            train_mse_hist, val_mse_hist,
            train_psd_hist, val_psd_hist,
            train_evm_hist, val_evm_hist)


# ==================== 联合微调函数（阶段3，修改：返回最终验证损失） ====================
def joint_finetune(end2end_model, train_loader, val_loader, device,
                   epochs=80, lr=1e-5, loss_weights=None,
                   best_model_path=os.path.join(TRAINING_MODEL_DIR, "shared", "best_end2end_finetuned.pth"),
                   config_label: str = ""):
    if loss_weights is None:
        loss_weights = {'mse': 1.0, 'psd': 0.001, 'evm': 0.001}

    # 多 GPU 包装
    end2end_model = _wrap_model(end2end_model, device)

    receiver = LegacyOTFSReceiver(M=_get_model_attr(end2end_model, 'M'),
                                   N=_get_model_attr(end2end_model, 'N')).to(device)
    for param in receiver.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(end2end_model.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device.type == 'cuda'))
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    best_val_loss = float('inf')
    final_val_loss = float('inf')
    label_str = f" [{config_label}]" if config_label else ""
    os.makedirs(os.path.dirname(best_model_path) or '.', exist_ok=True)

    for epoch in range(epochs):
        end2end_model.train()
        total_loss = 0.0
        for bits, target_dd, qam_batch in train_loader:
            bits, target_dd, qam_batch = bits.to(device), target_dd.to(device), qam_batch.to(device)
            with torch.cuda.amp.autocast(enabled=(USE_AMP and device.type == 'cuda')):
                out_dd = end2end_model(bits)
            # 损失计算在 autocast 外部（FP32），避免 FFT/sqrt 对 ComplexHalf 不兼容
            loss, _, _, _ = compute_combined_loss(
                out_dd, target_dd, qam_batch, receiver, loss_weights, device)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * bits.size(0)

        # 验证
        end2end_model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for bits, target_dd, qam_batch in val_loader:
                bits, target_dd, qam_batch = bits.to(device), target_dd.to(device), qam_batch.to(device)
                out_dd = end2end_model(bits)
                loss, _, _, _ = compute_combined_loss(
                    out_dd, target_dd, qam_batch, receiver, loss_weights, device)
                val_loss += loss.item() * bits.size(0)
                val_count += bits.size(0)

        epoch_val_loss = val_loss / max(val_count, 1)
        final_val_loss = epoch_val_loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(_unwrap_model(end2end_model).state_dict(), best_model_path)

        scheduler.step()
        if (epoch + 1) % 20 == 0:
            print(f"Joint Epoch {epoch+1}{label_str}: val loss={epoch_val_loss:.6f}")

    print(f"阶段3完成{label_str}：最佳验证损失: {best_val_loss:.6f}，最终验证损失: {final_val_loss:.6f}")
    return best_val_loss, final_val_loss


# ==================== 辅助函数（原 V3，完全保留） ====================
def _build_qam_constellation(modulation_order: int) -> np.ndarray:
    """构建通用 QAM 星座图，支持 4/16/64-QAM，返回 [modulation_order] 的复数数组"""
    symbols_per_dim = int(math.sqrt(modulation_order))
    if symbols_per_dim ** 2 != modulation_order:
        raise ValueError(f"调制阶数 {modulation_order} 必须是完全平方数（如 4, 16, 64）")
    max_val = symbols_per_dim - 1
    values = np.linspace(-max_val, max_val, symbols_per_dim)
    constellation = np.zeros(modulation_order, dtype=complex)
    idx = 0
    for i in range(symbols_per_dim):
        for j in range(symbols_per_dim):
            constellation[idx] = values[j] + 1j * values[i]
            idx += 1
    energy = np.mean(np.abs(constellation) ** 2)
    constellation = constellation / np.sqrt(energy)
    return constellation


def bits_to_qam(bits_tensor, M, N, bits_per_symbol=2):
    """将比特张量转换为 QAM 符号（支持任意阶数 QAM）"""
    bits = bits_tensor.numpy().astype(int)
    bits_reshaped = bits.reshape(M, N, bits_per_symbol)
    modulation_order = 2 ** bits_per_symbol
    constellation = _build_qam_constellation(modulation_order)

    # 将每组的 bits_per_symbol 个比特转换为十进制索引
    powers = 2 ** np.arange(bits_per_symbol - 1, -1, -1)
    indices = np.sum(bits_reshaped * powers.reshape(1, 1, -1), axis=-1)  # [M, N]

    qam = constellation[indices]  # [M, N] complex
    return qam


def calculate_psd_scipy(signal_data, fs=16):
    signal_flat = signal_data.flatten()
    f, psd = signal.welch(signal_flat, fs=fs, nperseg=256, scaling='density')
    return f, psd


def plot_loss_curves(train_losses, val_losses, best_info, save_path=None):
    if save_path is None:
        save_path = os.path.join(TRAINING_PLOT_DIR, "loss_curves.png")
    plt.figure(figsize=(12, 6))
    epochs = np.arange(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Training Loss', linewidth=2.5, marker='o', markersize=4)
    plt.plot(epochs, val_losses, label='Validation Loss', linewidth=2.5, marker='s', markersize=4)
    best_epoch = best_info['best_epoch']
    best_val_loss = best_info['best_val_loss']
    plt.scatter(best_epoch, best_val_loss, color='red', s=100, zorder=5, label=f'Best Val Loss (Epoch {best_epoch})')
    plt.annotate(f'Loss: {best_val_loss:.6f}', xy=(best_epoch, best_val_loss), xytext=(10, 10),
                 textcoords='offset points', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                 arrowprops=dict(arrowstyle='->'))
    plt.title('Training and Validation Loss Curves', fontsize=16)
    plt.xlabel('Epoch'); plt.ylabel('Total Loss'); plt.grid(True, alpha=0.3); plt.legend(); plt.yscale('log')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"损失曲线已保存至: {save_path}")


def load_loss_history_from_mat(mat_path: str):
    """从结果 MAT 文件恢复训练/验证损失历史。"""
    if not os.path.exists(mat_path):
        return [], [], [], [], [], [], [], []
    try:
        data = loadmat(mat_path)
    except Exception:
        return [], [], [], [], [], [], [], []

    def _get_list(key):
        value = data.get(key, None)
        if value is None:
            return []
        arr = np.asarray(value).squeeze()
        if arr.size == 0:
            return []
        return arr.astype(np.float64).tolist()

    return (
        _get_list('train_loss'),
        _get_list('val_loss'),
        _get_list('train_mse'),
        _get_list('val_mse'),
        _get_list('train_psd'),
        _get_list('val_psd'),
        _get_list('train_evm'),
        _get_list('val_evm'),
    )


def evaluate_on_validation_set(model, val_loader, device):
    """在验证集上批量计算 MSE、EVM、PSD 误差、PAPR（已向量化优化）"""
    model.eval()
    receiver = LegacyOTFSReceiver(M=model.module.M if hasattr(model, 'module') else model.M,
                                   N=model.module.N if hasattr(model, 'module') else model.N).to(device)
    mse_list = []
    evm_list = []
    psd_error_list = []
    all_original_qam = []
    all_recovered_qam = []
    # ===== PAPR 新增 =====
    target_papr_list = []
    generated_papr_list = []
    # ===== PAPR 新增结束 =====
    with torch.no_grad():
        for bits_batch, target_batch, qam_batch in val_loader:
            bits_batch = bits_batch.to(device)
            target_batch = target_batch.to(device)
            qam_batch = qam_batch.to(device)
            outputs = model(bits_batch)
            batch_size = bits_batch.size(0)

            # --- MSE per sample (批量向量化) ---
            mse_per_sample = F.mse_loss(
                outputs.reshape(batch_size, -1),
                target_batch.reshape(batch_size, -1),
                reduction='none'
            ).mean(dim=-1)  # [B]
            mse_list.extend(mse_per_sample.cpu().tolist())

            # --- EVM per sample (批量计算) ---
            recovered_qam = receiver(outputs)
            evm_per_sample = compute_evm(qam_batch, recovered_qam)  # [B]
            evm_list.extend(evm_per_sample.cpu().tolist())

            # --- PSD error per sample (批量向量化) ---
            outputs_complex = torch.complex(outputs[..., 0], outputs[..., 1])
            target_complex = torch.complex(target_batch[..., 0], target_batch[..., 1])
            sig_flat = outputs_complex.reshape(batch_size, -1)
            tgt_flat = target_complex.reshape(batch_size, -1)
            sig_fft = torch.fft.fft(sig_flat, dim=-1)
            tgt_fft = torch.fft.fft(tgt_flat, dim=-1)
            sig_psd = torch.abs(sig_fft) ** 2
            tgt_psd = torch.abs(tgt_fft) ** 2
            sig_psd = sig_psd / (sig_psd.sum(dim=-1, keepdim=True) + 1e-8)
            tgt_psd = tgt_psd / (tgt_psd.sum(dim=-1, keepdim=True) + 1e-8)
            log_sig = torch.log(sig_psd + 1e-8)
            log_tgt = torch.log(tgt_psd + 1e-8)
            psd_per_sample = torch.mean((log_sig - log_tgt) ** 2, dim=-1)  # [B]
            psd_error_list.extend(psd_per_sample.cpu().tolist())

            # ===== PAPR 新增 =====
            # 目标时域信号 PAPR（target_batch 是时域信号 [B, M, N, 2]）
            target_papr_per_sample = compute_papr(target_complex)  # [B]
            target_papr_list.extend(target_papr_per_sample.cpu().tolist())
            # 模型生成时域信号 PAPR（outputs 是时域信号 [B, M, N, 2]）
            generated_papr_per_sample = compute_papr(outputs_complex)  # [B]
            generated_papr_list.extend(generated_papr_per_sample.cpu().tolist())
            # ===== PAPR 新增结束 =====

            # --- 保存原始和恢复的 QAM ---
            all_original_qam.append(qam_batch.cpu().numpy())
            all_recovered_qam.append(recovered_qam.cpu().numpy())

    original_qam_all = np.concatenate(all_original_qam, axis=0)
    recovered_qam_all = np.concatenate(all_recovered_qam, axis=0)
    # ===== PAPR 新增 =====
    target_papr_all = np.array(target_papr_list, dtype=np.float64)
    generated_papr_all = np.array(generated_papr_list, dtype=np.float64)
    # ===== PAPR 新增结束 =====
    metrics = {
        'MSE_mean': float(np.mean(mse_list)), 'MSE_std': float(np.std(mse_list)),
        'EVM_mean': float(np.mean(evm_list)), 'EVM_std': float(np.std(evm_list)),
        'PSD_error_mean': float(np.mean(psd_error_list)), 'PSD_error_std': float(np.std(psd_error_list)),
        # ===== PAPR 新增 =====
        'target_papr_mean': float(np.mean(target_papr_list)),
        'target_papr_std': float(np.std(target_papr_list)),
        'generated_papr_mean': float(np.mean(generated_papr_list)),
        'generated_papr_std': float(np.std(generated_papr_list)),
        # ===== PAPR 新增结束 =====
    }
    return metrics, original_qam_all, recovered_qam_all, target_papr_all, generated_papr_all


def save_qam_constellation_data(original_qam_all, recovered_qam_all,
                                save_path=None):
    if save_path is None:
        save_path = os.path.join(METRICS_MAT_DIR, "summary", "qam_constellation_data.mat")
    # 确保输入为复数数组，避免实值数组 .real/.imag 行为异常
    if not np.iscomplexobj(original_qam_all):
        original_qam_all = original_qam_all[..., 0] + 1j * original_qam_all[..., 1]
    if not np.iscomplexobj(recovered_qam_all):
        recovered_qam_all = recovered_qam_all[..., 0] + 1j * recovered_qam_all[..., 1]
    data_dict = {
        'original_qam_real': np.ascontiguousarray(original_qam_all.real),
        'original_qam_imag': np.ascontiguousarray(original_qam_all.imag),
        'recovered_qam_real': np.ascontiguousarray(recovered_qam_all.real),
        'recovered_qam_imag': np.ascontiguousarray(recovered_qam_all.imag)
    }
    safe_savemat(save_path, data_dict, label="save_qam_constellation")


# ==================== EndToEnd 模型（不变） ====================
class EndToEndOTFS(nn.Module):
    def __init__(self, qam_gen, dd_gen):
        super().__init__()
        self.qam_gen = qam_gen
        self.dd_gen = dd_gen
        self.M = qam_gen.M
        self.N = qam_gen.N

    def forward(self, bits):
        qam = self.qam_gen(bits)
        dd = self.dd_gen(qam)
        return dd


# ==================== 辅助工厂函数（消除重复的模型构建代码） ====================
def _build_qam_generator(config: dict, bits_per_symbol: int, device: torch.device) -> QAMGenerator:
    """统一构建 QAMGenerator 实例（消除 6 处重复的构造函数调用）"""
    return QAMGenerator(
        M=config['M'], N=config['N'], bits_per_symbol=bits_per_symbol,
        d_model=config['qam_d_model'], nhead=config['qam_nhead'],
        num_layers=config['qam_num_layers'],
        dim_feedforward=config['qam_dim_feedforward'],
        dropout=config['qam_dropout']
    ).to(device)


def _build_dd_generator(config: dict, pos_type: str, attn_type: str,
                        device: torch.device) -> DDGeneratorV3:
    """统一构建 DDGeneratorV3 实例"""
    return DDGeneratorV3(
        M=config['M'], N=config['N'],
        d_model=config['dd_d_model'], nhead=config['dd_nhead'],
        num_layers=config['dd_num_layers'],
        dim_feedforward=config['dd_dim_feedforward'],
        dropout=config['dd_dropout'],
        pos_encoding_type=pos_type,
        attn_type=attn_type
    ).to(device)


def _save_experiment_mat_files(config: dict, pos_type: str, attn_type: str,
                                config_label: str, results: dict,
                                final_metrics: dict, val_loader, device):
    """保存单个实验配置的所有 .mat 文件（QAM 星座图 + 时频谱 + 结果汇总）"""
    if not FORCE_SAVE_MAT:
        return
    _, metric_dir = ensure_experiment_dirs(pos_type, attn_type)
    artifact_stem = get_artifact_stem(pos_type, attn_type)

    # QAM 星座图数据
    end2end = results.get('_end2end')
    val_loader_full = DataLoader(val_loader.dataset, batch_size=config['batch_size'], shuffle=False)
    if end2end is None:
        return
    _, original_qam_all, recovered_qam_all, target_papr_all, generated_papr_all = evaluate_on_validation_set(
        end2end, val_loader_full, device)

    # 确保 original_qam_all 为复数数组，避免实值数组 .real/.imag 行为异常
    if not np.iscomplexobj(original_qam_all):
        original_qam_all = original_qam_all[..., 0] + 1j * original_qam_all[..., 1]
    if not np.iscomplexobj(recovered_qam_all):
        recovered_qam_all = recovered_qam_all[..., 0] + 1j * recovered_qam_all[..., 1]

    qam_const_path = os.path.join(metric_dir, f"qam_constellation_{artifact_stem}.mat")
    qam_dict = {
        'original_qam_real': np.ascontiguousarray(original_qam_all.real),
        'original_qam_imag': np.ascontiguousarray(original_qam_all.imag),
        'recovered_qam_real': np.ascontiguousarray(recovered_qam_all.real),
        'recovered_qam_imag': np.ascontiguousarray(recovered_qam_all.imag),
        'config_name': get_config_name(pos_type, attn_type),
        'config_label': get_config_display(pos_type, attn_type),
        'pos_encoding_type': pos_type, 'attn_type': attn_type,
    }
    success = safe_savemat(qam_const_path, qam_dict, label=config_label)
    if not success and not SAVE_MAT_ON_ERROR:
        raise RuntimeError(f"[{config_label}] QAM 数据保存失败。")

    # 时域波形和频谱数据（第一帧）
    val_dataset = val_loader.dataset
    bits_0, target_0, _ = val_dataset[0]
    bits_0 = bits_0.unsqueeze(0).to(device)
    target_0_np = target_0.cpu().numpy()
    with torch.no_grad():
        out_dd_0 = end2end(bits_0).squeeze(0).cpu().numpy()

    generated_complex = out_dd_0[:, :, 0] + 1j * out_dd_0[:, :, 1]
    target_complex = target_0_np[:, :, 0] + 1j * target_0_np[:, :, 1]
    f_gen, psd_gen = calculate_psd_scipy(np.abs(generated_complex))
    f_tgt, psd_tgt = calculate_psd_scipy(np.abs(target_complex))
    psd_error = np.abs(psd_gen - psd_tgt)

    time_spec_path = os.path.join(metric_dir, f"time_spectrum_{artifact_stem}.mat")
    time_spec_dict = {
        'generated_real': np.ascontiguousarray(out_dd_0[:, :, 0].flatten()),
        'generated_imag': np.ascontiguousarray(out_dd_0[:, :, 1].flatten()),
        'target_real': np.ascontiguousarray(target_0_np[:, :, 0].flatten()),
        'target_imag': np.ascontiguousarray(target_0_np[:, :, 1].flatten()),
        'freqs': np.ascontiguousarray(f_gen),
        'psd_generated': np.ascontiguousarray(psd_gen),
        'psd_target': np.ascontiguousarray(psd_tgt),
        'psd_error': np.ascontiguousarray(psd_error),
        'M': config['M'], 'N': config['N'],
        'config_name': get_config_name(pos_type, attn_type),
        'config_label': get_config_display(pos_type, attn_type),
        'pos_encoding_type': pos_type, 'attn_type': attn_type,
    }
    success = safe_savemat(time_spec_path, time_spec_dict, label=config_label)
    if not success and not SAVE_MAT_ON_ERROR:
        raise RuntimeError(f"[{config_label}] 时域/频谱数据保存失败。")

    # 结果汇总
    mat_save_path = os.path.join(metric_dir, f"results_{artifact_stem}.mat")
    mat_dict = {
        'train_loss': np.array(results.get('train_losses', [])),
        'val_loss': np.array(results.get('val_losses', [])),
        'train_mse': np.array(results.get('train_mse', [])),
        'val_mse': np.array(results.get('val_mse', [])),
        'train_psd': np.array(results.get('train_psd', [])),
        'val_psd': np.array(results.get('val_psd', [])),
        'train_evm': np.array(results.get('train_evm', [])),
        'val_evm': np.array(results.get('val_evm', [])),
        'MSE_mean': final_metrics['MSE_mean'], 'MSE_std': final_metrics['MSE_std'],
        'EVM_mean': final_metrics['EVM_mean'], 'EVM_std': final_metrics['EVM_std'],
        'PSD_error_mean': final_metrics['PSD_error_mean'], 'PSD_error_std': final_metrics['PSD_error_std'],
        'PSD_error_first_frame': float(np.mean(psd_error)),
        'best_finetune_loss': float(results.get('best_finetune_loss', float('nan'))),
        'final_finetune_loss': float(results.get('final_finetune_loss', float('nan'))),
        # ===== PAPR 新增 =====
        'target_papr_mean': final_metrics['target_papr_mean'],
        'target_papr_std': final_metrics['target_papr_std'],
        'target_papr_all': np.ascontiguousarray(target_papr_all),
        'generated_papr_mean': final_metrics['generated_papr_mean'],
        'generated_papr_std': final_metrics['generated_papr_std'],
        'generated_papr_all': np.ascontiguousarray(generated_papr_all),
        # ===== PAPR 新增结束 =====
        'pos_encoding_type': pos_type, 'attn_type': attn_type,
        'config_name': get_config_name(pos_type, attn_type),
        'config_label': get_config_display(pos_type, attn_type),
        'config': str(config),
    }
    success = safe_savemat(mat_save_path, mat_dict, label=config_label)
    if not success and not SAVE_MAT_ON_ERROR:
        raise RuntimeError(f"[{config_label}] 结果 .mat 文件保存失败。")


# ==================== 单次实验运行函数（新增：封装阶段1-3完整流程） ====================
def run_single_experiment(config, train_loader, val_loader, device,
                          qam_gen_pretrained=None):
    """
    运行一次完整的实验（阶段1可选 + 阶段2 + 阶段3 + 评估）

    Args:
        config: 实验超参数字典
        train_loader, val_loader: 数据加载器
        device: 计算设备
        qam_gen_pretrained: 预训练的 QAMGenerator（如果提供则跳过阶段1并冻结复用）

    Returns:
        results: 包含所有训练曲线和最终指标的字典
    """
    pos_type = config['pos_encoding_type']
    attn_type = config['attn_type']
    config_name = get_config_name(pos_type, attn_type)
    config_label = get_config_display(pos_type, attn_type)
    artifact_stem = get_artifact_stem(pos_type, attn_type)
    model_dir, metric_dir = ensure_experiment_dirs(pos_type, attn_type)
    plot_dir = get_training_plot_dir(pos_type, attn_type)
    os.makedirs(plot_dir, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"开始实验: {config_label}")
    print(f"{'='*70}")

    # ---- 阶段1：QAMGenerator（如果未提供预训练模型则从头训练） ----
    bits_per_symbol = int(math.log2(config['modulation_order']))

    if qam_gen_pretrained is not None:
        # 🔑 关键修复: 使用独立副本而非直接引用，防止各配置间的权重交叉污染
        print(f"[{config_label}] 复用预训练 QAMGenerator（创建独立副本），跳过阶段1。")
        qam_gen = _build_qam_generator(config, bits_per_symbol, device)
        qam_gen.load_state_dict(qam_gen_pretrained.state_dict())
        qam_gen.eval()
        for param in qam_gen.parameters():
            param.requires_grad = False
        print(f"[{config_label}] QAMGenerator 独立副本已创建并冻结。")
    else:
        best_qam_path = os.path.join(model_dir, f"best_qam_generator_{artifact_stem}.pth")

        # —— 增强的跳过逻辑 ——
        should_skip_qam = False
        if not FORCE_RETRAIN:
            if VERBOSE_SKIP:
                print(f"[{config_label}] 🔍 检查 QAMGenerator 权重: {os.path.normpath(best_qam_path)}")
            should_skip_qam = verify_model_weights(None, best_qam_path, label=config_label)
        else:
            print(f"[{config_label}] ⚠️  FORCE_RETRAIN=True，强制重新训练 QAMGenerator。")

        if should_skip_qam:
            print(f"[{config_label}] 发现已存在的 QAMGenerator 权重，跳过阶段1训练。")
            print(f"  加载: {best_qam_path}")
            qam_gen = _build_qam_generator(config, bits_per_symbol, device)
            qam_gen.load_state_dict(torch.load(best_qam_path, map_location=device))
            qam_gen.eval()
        else:
            print(f"[{config_label}] 阶段1：训练 QAMGenerator")
            qam_gen = _build_qam_generator(config, bits_per_symbol, device)
            optimizer_qam = optim.Adam(qam_gen.parameters(), lr=config['qam_lr'])
            best_qam_loss = float('inf')
            for epoch in range(config['qam_epochs']):
                qam_gen.train()
                total_loss = 0.0
                for bits, _, qam_target in train_loader:
                    bits, qam_target = bits.to(device), qam_target.to(device)
                    optimizer_qam.zero_grad()
                    qam_pred = qam_gen(bits)
                    loss = F.mse_loss(qam_pred, qam_target)
                    loss.backward()
                    optimizer_qam.step()
                    total_loss += loss.item() * bits.size(0)
                avg_loss = total_loss / len(train_loader.dataset)
                if avg_loss < best_qam_loss:
                    best_qam_loss = avg_loss
                    torch.save(qam_gen.state_dict(), best_qam_path)
                if (epoch + 1) % 5 == 0:
                    print(f"  QAM Epoch {epoch+1}: MSE = {avg_loss:.6f}")
            qam_gen.load_state_dict(torch.load(best_qam_path, map_location=device))
            qam_gen.eval()
            print(f"[{config_label}] 阶段1完成，最佳 QAM MSE: {best_qam_loss:.6f}")

    # ---- 阶段2：训练 DDGenerator ----
    best_dd_path = os.path.join(model_dir, f"best_dd_generator_{artifact_stem}.pth")
    result_mat_path = os.path.join(metric_dir, f"results_{artifact_stem}.mat")

    # 重置随机种子以确保不同配置的 DDGenerator 从相同的初始状态开始（公平对比）
    set_seed(BASE_SEED)

    dd_gen = _build_dd_generator(config, pos_type, attn_type, device)

    # —— 增强的跳过逻辑 ——
    should_skip_dd = False
    if not FORCE_RETRAIN:
        if VERBOSE_SKIP:
            print(f"[{config_label}] 🔍 检查阶段2权重: {os.path.normpath(best_dd_path)}")
        should_skip_dd = verify_model_weights(dd_gen, best_dd_path, label=config_label)
    else:
        print(f"[{config_label}] ⚠️  FORCE_RETRAIN=True，强制重新训练阶段2。")

    if should_skip_dd:
        print(f"[{config_label}] ✅ 发现已存在 DDGenerator 权重，跳过阶段2训练。")
        print(f"    加载: {os.path.normpath(best_dd_path)}")
        dd_gen.load_state_dict(torch.load(best_dd_path, map_location=device))
        dd_gen.eval()
        (train_losses, val_losses,
         train_mse, val_mse,
         train_psd, val_psd,
         train_evm, val_evm) = load_loss_history_from_mat(result_mat_path)
        if len(train_losses) > 0 and len(val_losses) > 0:
            print(f"[{config_label}] 已从 {result_mat_path} 加载损失历史，将用于绘制损失曲线。")
        else:
            print(f"[{config_label}] 未找到已记录的损失历史，跳过损失曲线绘制。")
    else:
        print(f"[{config_label}] 阶段2：训练 DDGenerator（共 {config['dd_epochs']} 轮）")
        (train_losses, val_losses,
         train_mse, val_mse,
         train_psd, val_psd,
         train_evm, val_evm) = train_dd_generator_v3(
            qam_gen, dd_gen, train_loader, val_loader, device,
            epochs=config['dd_epochs'], lr=config['dd_lr'],
            loss_weights=config['dd_loss_weights'],
            best_model_path=best_dd_path,
            early_stopping_patience=config.get('early_stopping_patience', 50),  # ===== 修改点 2：默认值 25 → 50 =====
            config_label=config_label
        )
        dd_gen.load_state_dict(torch.load(best_dd_path, map_location=device))
        dd_gen.eval()

    # ---- 构建端到端模型 ----
    end2end = EndToEndOTFS(qam_gen, dd_gen).to(device)

    # ---- 阶段3：联合微调 ----
    finetuned_path = os.path.join(model_dir, f"best_end2end_finetuned_{artifact_stem}.pth")

    # —— 增强的跳过逻辑 ——
    should_skip_ft = False
    if not FORCE_RETRAIN:
        if VERBOSE_SKIP:
            print(f"[{config_label}] 🔍 检查阶段3权重: {os.path.normpath(finetuned_path)}")
        should_skip_ft = verify_model_weights(end2end, finetuned_path, label=config_label)
    else:
        print(f"[{config_label}] ⚠️  FORCE_RETRAIN=True，强制重新训练阶段3。")

    if should_skip_ft:
        print(f"[{config_label}] ✅ 发现已存在的微调模型权重，跳过阶段3训练。")
        print(f"    加载: {os.path.normpath(finetuned_path)}")
        state_dict = torch.load(finetuned_path, map_location=device)
        end2end.load_state_dict(state_dict)
        end2end.eval()
        best_finetune_loss = float('nan')
        final_finetune_loss = float('nan')
    else:
        print(f"[{config_label}] 阶段3：联合微调（共 {config['finetune_epochs']} 轮）")
        best_finetune_loss, final_finetune_loss = joint_finetune(
            end2end, train_loader, val_loader, device,
            epochs=config['finetune_epochs'], lr=config['finetune_lr'],
            loss_weights=config['dd_loss_weights'],
            best_model_path=finetuned_path,
            config_label=config_label
        )
        end2end.load_state_dict(torch.load(finetuned_path, map_location=device))
        end2end.eval()

    # ---- 最终评估 ----
    print(f"[{config_label}] 最终评估：在验证集上计算指标")
    val_loader_full = DataLoader(val_loader.dataset, batch_size=config['batch_size'], shuffle=False)
    final_metrics, original_qam_all, recovered_qam_all, target_papr_all, generated_papr_all = evaluate_on_validation_set(
        end2end, val_loader_full, device
    )

    print(f"[{config_label}] 最终指标:")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    # ---- 保存 QAM 星座图数据 ----
    if FORCE_SAVE_MAT:
        # 确保 original_qam_all 为复数数组，避免实值数组 .real/.imag 行为异常
        if not np.iscomplexobj(original_qam_all):
            original_qam_all = original_qam_all[..., 0] + 1j * original_qam_all[..., 1]
        if not np.iscomplexobj(recovered_qam_all):
            recovered_qam_all = recovered_qam_all[..., 0] + 1j * recovered_qam_all[..., 1]

        qam_const_path = os.path.join(metric_dir, f"qam_constellation_{artifact_stem}.mat")
        qam_dict = {
            'original_qam_real': np.ascontiguousarray(original_qam_all.real),
            'original_qam_imag': np.ascontiguousarray(original_qam_all.imag),
            'recovered_qam_real': np.ascontiguousarray(recovered_qam_all.real),
            'recovered_qam_imag': np.ascontiguousarray(recovered_qam_all.imag),
            'config_name': config_name,
            'config_label': config_label,
            'pos_encoding_type': pos_type,
            'attn_type': attn_type,
        }
        success = safe_savemat(qam_const_path, qam_dict, label=config_label)
        if not success and not SAVE_MAT_ON_ERROR:
            raise RuntimeError(f"[{config_label}] QAM 星座图数据保存失败，程序终止。设置 SAVE_MAT_ON_ERROR=True 可跳过此错误。")

    # ---- 保存第一帧时域波形和频谱数据 ----
    if FORCE_SAVE_MAT:
        val_dataset = val_loader.dataset
        bits_0, target_0, _ = val_dataset[0]
        bits_0 = bits_0.unsqueeze(0).to(device)
        target_0_np = target_0.cpu().numpy()
        end2end.eval()
        with torch.no_grad():
            out_dd_0 = end2end(bits_0).squeeze(0).cpu().numpy()

        generated_real = out_dd_0[:, :, 0].flatten()
        generated_imag = out_dd_0[:, :, 1].flatten()
        target_real = target_0_np[:, :, 0].flatten()
        target_imag = target_0_np[:, :, 1].flatten()

        generated_complex = out_dd_0[:, :, 0] + 1j * out_dd_0[:, :, 1]
        target_complex = target_0_np[:, :, 0] + 1j * target_0_np[:, :, 1]
        f_gen, psd_gen = calculate_psd_scipy(np.abs(generated_complex))
        f_tgt, psd_tgt = calculate_psd_scipy(np.abs(target_complex))
        psd_error = np.abs(psd_gen - psd_tgt)

        time_spec_path = os.path.join(metric_dir, f"time_spectrum_{artifact_stem}.mat")
        time_spec_dict = {
            'generated_real': np.ascontiguousarray(generated_real),
            'generated_imag': np.ascontiguousarray(generated_imag),
            'target_real': np.ascontiguousarray(target_real),
            'target_imag': np.ascontiguousarray(target_imag),
            'freqs': np.ascontiguousarray(f_gen),
            'psd_generated': np.ascontiguousarray(psd_gen),
            'psd_target': np.ascontiguousarray(psd_tgt),
            'psd_error': np.ascontiguousarray(psd_error),
            'M': config['M'],
            'N': config['N'],
            'config_name': config_name,
            'config_label': config_label,
            'pos_encoding_type': pos_type,
            'attn_type': attn_type,
        }
        success = safe_savemat(time_spec_path, time_spec_dict, label=config_label)
        if not success and not SAVE_MAT_ON_ERROR:
            raise RuntimeError(f"[{config_label}] 时域/频谱数据保存失败，程序终止。设置 SAVE_MAT_ON_ERROR=True 可跳过此错误。")

    # ---- 保存本配置的 .mat 文件 ----
    if FORCE_SAVE_MAT:
        mat_save_path = result_mat_path
        mat_dict = {
            # 训练曲线（跳过训练时为空 np.array([])）
            'train_loss': np.array(train_losses),
            'val_loss': np.array(val_losses),
            'train_mse': np.array(train_mse),
            'val_mse': np.array(val_mse),
            'train_psd': np.array(train_psd),
            'val_psd': np.array(val_psd),
            'train_evm': np.array(train_evm),
            'val_evm': np.array(val_evm),
            # 最终指标
            'MSE_mean': final_metrics['MSE_mean'],
            'MSE_std': final_metrics['MSE_std'],
            'EVM_mean': final_metrics['EVM_mean'],
            'EVM_std': final_metrics['EVM_std'],
            'PSD_error_mean': final_metrics['PSD_error_mean'],
            'PSD_error_std': final_metrics['PSD_error_std'],
            'PSD_error_first_frame': float(np.mean(psd_error)),
            # 阶段3 结果
            'best_finetune_loss': float(best_finetune_loss),
            'final_finetune_loss': float(final_finetune_loss),
            # ===== PAPR 新增 =====
            'target_papr_mean': final_metrics['target_papr_mean'],
            'target_papr_std': final_metrics['target_papr_std'],
            'target_papr_all': np.ascontiguousarray(target_papr_all),
            'generated_papr_mean': final_metrics['generated_papr_mean'],
            'generated_papr_std': final_metrics['generated_papr_std'],
            'generated_papr_all': np.ascontiguousarray(generated_papr_all),
            # ===== PAPR 新增结束 =====
            # 配置信息
            'config_name': config_name,
            'config_label': config_label,
            'pos_encoding_type': pos_type,
            'attn_type': attn_type,
            'config': str(config),
        }
        success = safe_savemat(mat_save_path, mat_dict, label=config_label)
        if not success and not SAVE_MAT_ON_ERROR:
            raise RuntimeError(f"[{config_label}] 结果 .mat 文件保存失败，程序终止。设置 SAVE_MAT_ON_ERROR=True 可跳过此错误。")

    if len(train_losses) > 0 and len(val_losses) > 0:
        best_info = {
            'best_epoch': int(np.argmin(val_losses)) + 1,
            'best_val_loss': float(min(val_losses))
        }
        plot_loss_curves(
            train_losses, val_losses, best_info,
            save_path=os.path.join(plot_dir, f"loss_curves_{artifact_stem}.png")
        )

    # ---- 返回结果汇总 ----
    results = {
        'config_name': config_name,
        'config_label': config_label,
        'pos_encoding_type': pos_type,
        'attn_type': attn_type,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_mse': train_mse,
        'val_mse': val_mse,
        'train_psd': train_psd,
        'val_psd': val_psd,
        'train_evm': train_evm,
        'val_evm': val_evm,
        'final_metrics': final_metrics,
        'best_finetune_loss': best_finetune_loss,
        'final_finetune_loss': final_finetune_loss,
        'psd_error_first_frame': float(np.mean(psd_error)),
        # ===== PAPR 新增 =====
        'target_papr_mean': float(final_metrics['target_papr_mean']),
        'target_papr_std': float(final_metrics['target_papr_std']),
        'target_papr_all': target_papr_all,
        'generated_papr_mean': float(final_metrics['generated_papr_mean']),
        'generated_papr_std': float(final_metrics['generated_papr_std']),
        'generated_papr_all': generated_papr_all,
        # ===== PAPR 新增结束 =====
        'best_dd_path': best_dd_path,
        'finetuned_path': finetuned_path,
    }
    return results


def print_comparison_table(all_results):
    """打印各配置的对比汇总表（含 PAPR）"""
    label_width = max(len('配置'), *(len(r['config_label']) for r in all_results)) + 2
    metric_width = 12
    table_width = label_width + metric_width * 6 + 6

    print(f"\n{'='*table_width}")
    print("对照实验汇总对比表")
    print(f"{'='*table_width}")
    header = (f"{'配置':<{label_width}}"
              f"{'MSE_mean':>{metric_width}}"
              f"{'EVM_mean':>{metric_width}}"
              f"{'PSD_err':>{metric_width}}"
              f"{'TargetPAPR':>{metric_width}}"
              f"{'GenPAPR':>{metric_width}}"
              f"{'微调loss':>{metric_width}}")
    print(header)
    print("-" * table_width)
    for r in all_results:
        m = r['final_metrics']
        label = r['config_label']
        # ===== PAPR 新增 =====
        t_papr = m.get('target_papr_mean', float('nan'))
        g_papr = m.get('generated_papr_mean', float('nan'))
        print(f"{label:<{label_width}}"
              f"{m['MSE_mean']:>{metric_width}.6f}"
              f"{m['EVM_mean']:>{metric_width}.6f}"
              f"{m['PSD_error_mean']:>{metric_width}.6f}"
              f"{t_papr:>{metric_width}.4f}"
              f"{g_papr:>{metric_width}.4f}"
              f"{r['best_finetune_loss']:>{metric_width}.6f}")
        # ===== PAPR 新增结束 =====
    print(f"{'='*table_width}")

    # 找出各项最佳
    best_mse = min(all_results, key=lambda r: r['final_metrics']['MSE_mean'])
    best_evm = min(all_results, key=lambda r: r['final_metrics']['EVM_mean'])
    best_psd = min(all_results, key=lambda r: r['final_metrics']['PSD_error_mean'])
    print(f"\n最佳 MSE:  {best_mse['config_label']} ({best_mse['final_metrics']['MSE_mean']:.6f})")
    print(f"最佳 EVM:  {best_evm['config_label']} ({best_evm['final_metrics']['EVM_mean']:.6f})")
    print(f"最佳 PSD:  {best_psd['config_label']} ({best_psd['final_metrics']['PSD_error_mean']:.6f})")
    # ===== PAPR 新增 =====
    lowest_papr = min(all_results, key=lambda r: r['final_metrics'].get('generated_papr_mean', float('inf')))
    lp_g = lowest_papr['final_metrics'].get('generated_papr_mean', float('nan'))
    lp_t = lowest_papr['final_metrics'].get('target_papr_mean', float('nan'))
    print(f"最低生成 PAPR: {lowest_papr['config_label']} "
          f"(Generated={lp_g:.4f}, Target={lp_t:.4f})")
    # ===== PAPR 新增结束 =====


# ==================== 主程序 ====================
if __name__ == "__main__":
    # ======== 全局配置 ========
    config = {
        "train_file": "DataSet/train_data_4_QAM.pkl",
        "val_file": "DataSet/val_data_4_QAM.pkl",
        "batch_size": 32,
        "M": 16,
        "N": 16,
        "modulation_order": 4,
        # QAMGenerator 增强配置
        "qam_epochs": 20,
        "qam_lr": 1e-3,
        "qam_d_model": 256,
        "qam_nhead": 4,
        "qam_num_layers": 4,
        "qam_dim_feedforward": 512,
        "qam_dropout": 0.1,
        # DDGenerator 增强配置
        "dd_epochs": 200,
        "dd_lr": 5e-4,
        "dd_loss_weights": {"mse": 1.0, "psd": 0.001, "evm": 0.001},
        "dd_d_model": 256,
        "dd_nhead": 8,
        "dd_num_layers": 6,
        "dd_dim_feedforward": 2048,
        "dd_dropout": 0.1,
        # 联合微调配置
        "finetune_epochs": 80,
        "finetune_lr": 1e-5,
        # 早停配置
        "early_stopping_patience": 50,  # ===== 修改点 2：早停耐心值 25 → 50 =====
        # 文件路径
        "best_qam_path": os.path.join(TRAINING_MODEL_DIR, "shared", "best_qam_generator.pth"),
    }

    bits_per_symbol = int(math.log2(config['modulation_order']))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"调制阶数: {config['modulation_order']}-QAM, 每符号比特数: {bits_per_symbol}")
    print(f"使用设备: {device}")

    # 加载数据集
    print("\n=== 加载数据集 ===")
    train_dataset = OTFSDataset(config['train_file'])
    val_dataset = OTFSDataset(config['val_file'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # ======== 确定要运行的实验配置列表 ========
    if COMPARISON_MODE:
        # ===== 实验配置 A～G：按当前命名重新编号 =====
        experiment_configs = [
            {"pos_encoding_type": "Standard",  "attn_type": "Standard"},      # Config_A (Pos=Standard, Attn=Standard)
            {"pos_encoding_type": "Standard",  "attn_type": "GQA"},           # Config_B (Pos=Standard, Attn=GQA)
            {"pos_encoding_type": "Standard",  "attn_type": "Differential"},  # Config_C (Pos=Standard, Attn=Differential)
            {"pos_encoding_type": "Standard",  "attn_type": "MLA"},           # Config_D (Pos=Standard, Attn=MLA)
            {"pos_encoding_type": "Standard",  "attn_type": "Dual_Axis"},     # Config_E (Pos=Standard, Attn=Dual_Axis)
            {"pos_encoding_type": "Standard",  "attn_type": "BoltAttention"}, # Config_F (Pos=Standard, Attn=BoltAttention)
            {"pos_encoding_type": "PhaseLoom", "attn_type": "BoltAttention"}, # Config_G (Pos=PhaseLoom, Attn=BoltAttention)
            # {"pos_encoding_type": "PhaseLoom", "attn_type": "Standard"},   # 旧编号 Config_C 已注释，不参与实验
            # {"pos_encoding_type": "PhaseLoom", "attn_type": "Dual_Axis"},  # 旧编号 Config_F 已注释，不参与实验
        ]
        print("\n===== 对照实验模式：将运行 7 种配置组合（Config_A～Config_G） =====")
    else:
        experiment_configs = [SINGLE_CONFIG]
        print(f"\n===== 单一配置模式：{SINGLE_CONFIG} =====")

    # ======== 阶段1：训练 QAMGenerator（仅训练一次，后续复用） ========
    if COMPARISON_MODE:
        qam_shared_path = config['best_qam_path']
        os.makedirs(os.path.dirname(qam_shared_path), exist_ok=True)

        # —— 增强的跳过逻辑 ——
        should_skip_qam = False
        if not FORCE_RETRAIN:
            if VERBOSE_SKIP:
                print(f"\n🔍 检查 QAMGenerator 权重: {os.path.normpath(qam_shared_path)}")
            if os.path.exists(qam_shared_path):
                # 验证文件可加载，且参数 shape 匹配当前模型
                try:
                    test_state = torch.load(qam_shared_path, map_location='cpu')
                    # 构建临时模型验证 shape 兼容性
                    test_model = _build_qam_generator(config, bits_per_symbol, torch.device('cpu'))
                    model_keys = set(test_model.state_dict().keys())
                    loaded_keys = set(test_state.keys())
                    if model_keys != loaded_keys:
                        print(f"   ⚠️ 权重参数结构不匹配（可能调制阶数不同），将重新训练")
                        should_skip_qam = False
                    else:
                        # 逐参数校验 shape
                        for key in model_keys:
                            if test_state[key].shape != test_model.state_dict()[key].shape:
                                print(f"   ⚠️ 参数 '{key}' 维度不匹配: "
                                      f"文件{tuple(test_state[key].shape)} vs 模型{tuple(test_model.state_dict()[key].shape)}")
                                should_skip_qam = False
                                break
                        else:
                            file_size = os.path.getsize(qam_shared_path)
                            if VERBOSE_SKIP:
                                print(f"   ✅ 权重文件存在且参数匹配 ({file_size:,} bytes, {len(test_state)} 个参数)")
                            should_skip_qam = True
                except Exception as e:
                    print(f"   ⚠️ 权重文件验证失败: {e}")
                    should_skip_qam = False
        else:
            print(f"\n⚠️  FORCE_RETRAIN=True，强制重新训练 QAMGenerator。")

        if should_skip_qam:
            print(f"\n===== 阶段1：发现已存在的 QAMGenerator 权重，跳过训练 =====")
            print(f"  加载: {os.path.normpath(qam_shared_path)}")
            qam_gen_shared = _build_qam_generator(config, bits_per_symbol, device)
            qam_gen_shared.load_state_dict(torch.load(qam_shared_path, map_location=device))
            qam_gen_shared.eval()
            for param in qam_gen_shared.parameters():
                param.requires_grad = False
            print(f"阶段1：QAMGenerator 权重已加载并冻结，所有配置将通过深拷贝复用此模型。")
        else:
            print("\n===== 阶段1：训练 QAMGenerator（四种配置共用） =====")
            set_seed(BASE_SEED)
            qam_gen_shared = _build_qam_generator(config, bits_per_symbol, device)

            optimizer_qam = optim.Adam(qam_gen_shared.parameters(), lr=config['qam_lr'])
            best_qam_loss = float('inf')
            for epoch in range(config['qam_epochs']):
                qam_gen_shared.train()
                total_loss = 0.0
                for bits, _, qam_target in train_loader:
                    bits, qam_target = bits.to(device), qam_target.to(device)
                    optimizer_qam.zero_grad()
                    qam_pred = qam_gen_shared(bits)
                    loss = F.mse_loss(qam_pred, qam_target)
                    loss.backward()
                    optimizer_qam.step()
                    total_loss += loss.item() * bits.size(0)
                avg_loss = total_loss / len(train_loader.dataset)
                if avg_loss < best_qam_loss:
                    best_qam_loss = avg_loss
                    torch.save(qam_gen_shared.state_dict(), qam_shared_path)
                if (epoch + 1) % 5 == 0:
                    print(f"  QAM Epoch {epoch+1}: MSE = {avg_loss:.6f}")
            qam_gen_shared.load_state_dict(torch.load(qam_shared_path, map_location=device))
            qam_gen_shared.eval()
            print(f"阶段1完成，最佳 QAM MSE: {best_qam_loss:.6f}，权重已保存至 {qam_shared_path}，所有配置将复用此模型。")
    else:
        qam_gen_shared = None  # 单一模式在 run_single_experiment 中自行训练

    # ======== 运行所有实验配置 ========
    all_results = []
    for i, exp_cfg in enumerate(experiment_configs):
        pos_t = exp_cfg['pos_encoding_type']
        attn_t = exp_cfg['attn_type']
        config_name = get_config_name(pos_t, attn_t)
        config_label = get_config_display(pos_t, attn_t)
        artifact_stem = get_artifact_stem(pos_t, attn_t)
        model_dir = get_model_dir(pos_t, attn_t)
        print(f"\n{'#'*70}")
        print(f"# 实验 {i+1}/{len(experiment_configs)}: {config_label}")
        print(f"{'#'*70}")

        # 快速评估提示：检查是否所有权重均已存在（使用增强验证）
        dd_path = os.path.join(model_dir, f"best_dd_generator_{artifact_stem}.pth")
        ft_path = os.path.join(model_dir, f"best_end2end_finetuned_{artifact_stem}.pth")
        dd_exists = verify_model_weights(None, dd_path, label=config_name)
        ft_exists = verify_model_weights(None, ft_path, label=config_name)
        if dd_exists and ft_exists:
            print(f"{config_name} 所有权重已存在，将进入快速评估模式（跳过阶段2和阶段3训练）")
        elif dd_exists:
            print(f"{config_name} DDGenerator 权重已存在，但微调权重缺失，将仅训练阶段3")
        elif ft_exists:
            print(f"{config_name} 微调权重已存在，但 DDGenerator 权重缺失，将重新训练阶段2和阶段3")

        # 合并配置
        run_config = {**config, **exp_cfg}

        # 使用共享的 QAMGenerator（对比模式）或自行训练（单一模式）
        result = run_single_experiment(
            run_config, train_loader, val_loader, device,
            qam_gen_pretrained=qam_gen_shared
        )
        all_results.append(result)

    # ======== 汇总对比 ========
    print_comparison_table(all_results)

    # ======== 保存总对比文件 ========
    comparison_data = {}
    for r in all_results:
        prefix = r['config_name']
        m = r['final_metrics']
        comparison_data[f"{prefix}_MSE_mean"] = m['MSE_mean']
        comparison_data[f"{prefix}_MSE_std"] = m['MSE_std']
        comparison_data[f"{prefix}_EVM_mean"] = m['EVM_mean']
        comparison_data[f"{prefix}_EVM_std"] = m['EVM_std']
        comparison_data[f"{prefix}_PSD_error_mean"] = m['PSD_error_mean']
        comparison_data[f"{prefix}_PSD_error_std"] = m['PSD_error_std']
        comparison_data[f"{prefix}_best_finetune_loss"] = float(r['best_finetune_loss'])
        comparison_data[f"{prefix}_final_finetune_loss"] = float(r['final_finetune_loss'])
        comparison_data[f"{prefix}_PSD_error_first_frame"] = float(r['psd_error_first_frame'])
        comparison_data[f"{prefix}_pos_encoding_type"] = r['pos_encoding_type']
        comparison_data[f"{prefix}_attn_type"] = r['attn_type']
        comparison_data[f"{prefix}_train_loss"] = np.array(r['train_losses'])
        comparison_data[f"{prefix}_val_loss"] = np.array(r['val_losses'])
        # ===== PAPR 新增 =====
        comparison_data[f"{prefix}_target_papr_mean"] = float(r['target_papr_mean'])
        comparison_data[f"{prefix}_target_papr_std"] = float(r['target_papr_std'])
        comparison_data[f"{prefix}_target_papr_all"] = np.ascontiguousarray(r['target_papr_all'])
        comparison_data[f"{prefix}_generated_papr_mean"] = float(r['generated_papr_mean'])
        comparison_data[f"{prefix}_generated_papr_std"] = float(r['generated_papr_std'])
        comparison_data[f"{prefix}_generated_papr_all"] = np.ascontiguousarray(r['generated_papr_all'])
        # ===== PAPR 新增结束 =====

    comparison_data['num_configs'] = len(all_results)
    comparison_data['config_summary'] = [r['config_label'] for r in all_results]
    summary_dir = os.path.join(METRICS_MAT_DIR, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    safe_savemat(os.path.join(summary_dir, "comparison_all.mat"), comparison_data, label="comparison")

    print("\n===== 运行总结 =====")
    print(f"完成实验配置数: {len(all_results)}")
    for r in all_results:
        print(f"  {r['config_name']} ({r['config_label']}): "
              f"MSE={r['final_metrics']['MSE_mean']:.6f}, "
              f"EVM={r['final_metrics']['EVM_mean']:.6f}, "
              f"PSD_err={r['final_metrics']['PSD_error_mean']:.6f}")
    print(f"所有 MAT 结果文件已按实验配置保存至 {METRICS_MAT_DIR}/ 目录")
    print("==============================")
    print("仿真结束")
