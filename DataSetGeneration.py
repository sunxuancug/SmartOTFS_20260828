import torch
import math
import numpy as np
import os
import pickle
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List
import warnings
from scipy.io import savemat

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

warnings.filterwarnings('ignore')

# ==================== 通用QAM调制逻辑（支持4/16/64等） ====================
def generate_qam_constellation(modulation_order: int = 4) -> torch.Tensor:
    symbols_per_dim = int(math.sqrt(modulation_order))
    if symbols_per_dim ** 2 != modulation_order:
        raise ValueError(f"调制阶数 {modulation_order} 必须是完全平方数（如4,16,64）")

    max_val = symbols_per_dim - 1
    values = torch.linspace(-max_val, max_val, symbols_per_dim)

    constellation = torch.zeros((modulation_order, 2))
    idx = 0
    for i in range(symbols_per_dim):
        for j in range(symbols_per_dim):
            constellation[idx, 0] = values[j]
            constellation[idx, 1] = values[i]
            idx += 1

    energy = torch.mean(constellation[:, 0] ** 2 + constellation[:, 1] ** 2)
    constellation = constellation / torch.sqrt(energy)
    return constellation


def bits_to_symbol_indices(bits: torch.Tensor, bits_per_symbol: int) -> torch.Tensor:
    bits = bits.long()
    batch_size, seq_len = bits.shape

    if seq_len % bits_per_symbol != 0:
        padding_len = bits_per_symbol - (seq_len % bits_per_symbol)
        padding = torch.zeros((batch_size, padding_len), dtype=torch.long, device=bits.device)
        bits = torch.cat([bits, padding], dim=1)
        seq_len = bits.shape[1]

    num_symbols = seq_len // bits_per_symbol
    bits_reshaped = bits.reshape(batch_size, num_symbols, bits_per_symbol)

    symbol_indices = torch.zeros((batch_size, num_symbols), dtype=torch.long, device=bits.device)
    for k in range(bits_per_symbol):
        symbol_indices = symbol_indices * 2 + bits_reshaped[:, :, k]

    return symbol_indices


def traditional_qam_modulation(bits: torch.Tensor, modulation_order: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    bits_per_symbol = int(math.log2(modulation_order))
    constellation = generate_qam_constellation(modulation_order)
    symbol_indices = bits_to_symbol_indices(bits, bits_per_symbol)
    symbols = constellation[symbol_indices]
    return symbols, constellation


# ==================== 数据集类（无信道） ====================
class QAMIDFTDataset(Dataset):
    """
    生成QAM+IDFT数据集。
    输出：
        bits_mat: [M, N * bits_per_symbol]
        qam_mat:  [M, N, 2]
        idft_mat: [M, N, 2]
    """

    def __init__(self,
                 num_frames: int = 1000,
                 M: int = 16,
                 N: int = 16,
                 modulation_order: int = 4,
                 data_file: str = None,
                 is_train: bool = True,
                 val_split_ratio: float = 0.2):
        super().__init__()
        self.modulation_order = modulation_order
        self.bits_per_symbol = int(math.log2(modulation_order))
        if 2 ** self.bits_per_symbol != modulation_order:
            raise ValueError(f"调制阶数 {modulation_order} 必须是2的幂")

        self.num_frames = num_frames
        self.M = M
        self.N = N
        self.total_bits_per_row = self.N * self.bits_per_symbol
        self.data_file = data_file
        self.is_train = is_train
        self.val_split_ratio = val_split_ratio
        self.train_frames = int(num_frames * (1 - val_split_ratio))
        self.val_frames = num_frames - self.train_frames

        # 数据存储
        self.bits_matrices = []
        self.qam_matrices = []
        self.idft_matrices = []

        # 加载/生成数据
        if data_file and os.path.exists(data_file):
            print(f"正在从文件加载数据集: {data_file}")
            self._load_from_file()
        else:
            print(f"正在生成 {num_frames} 帧 {modulation_order}-QAM + Regulated OTFS 数据 (M={M}, N={N})...")
            self._generate_data()
            if data_file:
                self._save_split_data()

    def _generate_single_frame(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """生成单帧：比特矩阵 -> QAM -> IDFT（已向量化优化）"""
        # 1. 比特矩阵 [M, N*bits_per_symbol]
        bits_matrix = torch.randint(0, 2, (self.M, self.total_bits_per_row), dtype=torch.float32)

        # 2. QAM调制 [M, N, 2] — 批量处理所有 M 行
        qam_matrix, _ = traditional_qam_modulation(bits_matrix, self.modulation_order)

        # 3. IDFT [M, N, 2] — 批量 FFT 沿最后一维
        qam_complex = torch.complex(qam_matrix[..., 0], qam_matrix[..., 1])  # [M, N]
        idft_complex = torch.fft.ifft(qam_complex, dim=-1) * self.N           # [M, N]
        idft_matrix = torch.stack([idft_complex.real, idft_complex.imag], dim=-1)  # [M, N, 2]

        return bits_matrix, qam_matrix, idft_matrix

    def _generate_data(self):
        """批量生成所有帧数据"""
        self.bits_matrices.clear()
        self.qam_matrices.clear()
        self.idft_matrices.clear()

        for frame_idx in range(self.num_frames):
            bits_mat, qam_mat, idft_mat = self._generate_single_frame()
            self.bits_matrices.append(bits_mat.numpy())
            self.qam_matrices.append(qam_mat.numpy())
            self.idft_matrices.append(idft_mat.numpy())

            if (frame_idx + 1) % 100 == 0:
                print(f"已生成 {frame_idx + 1}/{self.num_frames} 帧数据")
        print("数据生成完成！")

    def _save_split_data(self):
        """将数据按训练/验证集拆分并分别保存为 PKL 和 MAT 文件"""
        output_dir = os.path.dirname(self.data_file) or '.'
        train_stem = f"train_data_{self.modulation_order}_QAM"
        val_stem = f"val_data_{self.modulation_order}_QAM"
        train_file = os.path.join(output_dir, f"{train_stem}.pkl")
        val_file = os.path.join(output_dir, f"{val_stem}.pkl")

        train_mat_file = os.path.join(output_dir, f"{train_stem}.mat")
        val_mat_file = os.path.join(output_dir, f"{val_stem}.mat")

        os.makedirs(output_dir, exist_ok=True)

        indices = np.arange(self.num_frames)
        np.random.shuffle(indices)
        train_indices = indices[:self.train_frames]
        val_indices = indices[self.train_frames:]

        # ========== 保存训练集 PKL ==========
        train_data = {
            'num_frames': self.train_frames,
            'M': self.M,
            'N': self.N,
            'modulation_order': self.modulation_order,
            'bits_per_symbol': self.bits_per_symbol,
            'bits_matrices': [self.bits_matrices[i] for i in train_indices],
            'qam_matrices': [self.qam_matrices[i] for i in train_indices],
            'idft_matrices': [self.idft_matrices[i] for i in train_indices],
        }
        with open(train_file, 'wb') as f:
            pickle.dump(train_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"训练集 PKL 已保存到: {train_file} (共{self.train_frames}帧, {self.modulation_order}-QAM)")

        # ========== 保存验证集 PKL ==========
        val_data = {
            'num_frames': self.val_frames,
            'M': self.M,
            'N': self.N,
            'modulation_order': self.modulation_order,
            'bits_per_symbol': self.bits_per_symbol,
            'bits_matrices': [self.bits_matrices[i] for i in val_indices],
            'qam_matrices': [self.qam_matrices[i] for i in val_indices],
            'idft_matrices': [self.idft_matrices[i] for i in val_indices],
        }
        with open(val_file, 'wb') as f:
            pickle.dump(val_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"验证集 PKL 已保存到: {val_file} (共{self.val_frames}帧, {self.modulation_order}-QAM)")

        # ========== 保存训练集 MAT ==========
        self._save_mat_file(
            train_mat_file,
            [self.bits_matrices[i] for i in train_indices],
            [self.qam_matrices[i] for i in train_indices],
            [self.idft_matrices[i] for i in train_indices],
            self.train_frames
        )
        print(f"训练集 MAT 已保存到: {train_mat_file}")

        # ========== 保存验证集 MAT ==========
        self._save_mat_file(
            val_mat_file,
            [self.bits_matrices[i] for i in val_indices],
            [self.qam_matrices[i] for i in val_indices],
            [self.idft_matrices[i] for i in val_indices],
            self.val_frames
        )
        print(f"验证集 MAT 已保存到: {val_mat_file}")

    def _save_mat_file(self, mat_file, bits_list, qam_list, idft_list, num_frames):
        """保存数据为 MAT 文件格式"""
        bits_mat = np.stack(bits_list, axis=0)  # [frames, M, N*bits_per_symbol]
        qam_mat = np.stack(qam_list, axis=0)   # [frames, M, N, 2]
        idft_mat = np.stack(idft_list, axis=0) # [frames, M, N, 2]

        qam_real = qam_mat[..., 0]
        qam_imag = qam_mat[..., 1]
        idft_real = idft_mat[..., 0]
        idft_imag = idft_mat[..., 1]

        mat_dict = {
            'num_frames': num_frames,
            'M': self.M,
            'N': self.N,
            'modulation_order': self.modulation_order,
            'bits_per_symbol': self.bits_per_symbol,
            'bits_matrices': bits_mat.astype(np.float32),
            'qam_real': qam_real.astype(np.float32),
            'qam_imag': qam_imag.astype(np.float32),
            'idft_real': idft_real.astype(np.float32),
            'idft_imag': idft_imag.astype(np.float32),
        }
        savemat(mat_file, mat_dict, do_compression=True)

    def _load_from_file(self):
        """加载训练/验证集文件"""
        try:
            with open(self.data_file, 'rb') as f:
                data = pickle.load(f)

            # 检查调制阶数兼容性
            if 'modulation_order' not in data:
                warnings.warn("加载的数据没有 modulation_order 字段，假设为 4-QAM")
                file_modulation = 4
            else:
                file_modulation = data['modulation_order']

            if file_modulation != self.modulation_order:
                print(f"警告：调制阶数不匹配！文件={file_modulation}, 请求={self.modulation_order}")
                self.modulation_order = file_modulation
                self.bits_per_symbol = int(math.log2(self.modulation_order))
                self.total_bits_per_row = self.N * self.bits_per_symbol

            # 校验维度
            if data['M'] != self.M or data['N'] != self.N:
                print(f"警告：矩阵维度不匹配！文件={data['M']}x{data['N']}, 请求={self.M}x{self.N}")
                self.M = data['M']
                self.N = data['N']
                self.total_bits_per_row = self.N * self.bits_per_symbol

            self.bits_matrices = data['bits_matrices']
            self.qam_matrices = data['qam_matrices']
            self.idft_matrices = data['idft_matrices']

            print(f"数据集加载完成！{len(self.bits_matrices)}帧, {self.M}x{self.N}, {self.modulation_order}-QAM")

        except Exception as e:
            print(f"加载失败，重新生成数据：{e}")
            self._generate_data()
            self._save_split_data()

    def __len__(self):
        return self.train_frames if self.is_train else self.val_frames

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bits_mat = torch.FloatTensor(self.bits_matrices[idx])
        qam_mat = torch.FloatTensor(self.qam_matrices[idx])
        idft_mat = torch.FloatTensor(self.idft_matrices[idx])
        return bits_mat, qam_mat, idft_mat


# ==================== 数据集生成接口 ====================
def generate_qam_idft_split_datasets(
        num_frames: int = 1000,
        M: int = 16,
        N: int = 16,
        modulation_order: int = 4,
        base_save_path: str = "DataSet/qam_idft_dataset",
        val_split_ratio: float = 0.2,
        validate: bool = True
) -> Tuple[QAMIDFTDataset, QAMIDFTDataset]:
    """
    生成并保存QAM+IDFT训练集/验证集，自动保存 PKL 和 MAT 两种格式。
    文件命名示例：train_data_4_QAM.pkl, val_data_4_QAM.pkl
    """
    # 生成全量数据（会触发拆分保存）
    full_dataset = QAMIDFTDataset(
        num_frames=num_frames,
        M=M,
        N=N,
        modulation_order=modulation_order,
        data_file=f"{base_save_path}.pkl",
        val_split_ratio=val_split_ratio
    )

    output_dir = os.path.dirname(base_save_path) or '.'
    train_file = os.path.join(output_dir, f"train_data_{modulation_order}_QAM.pkl")
    val_file = os.path.join(output_dir, f"val_data_{modulation_order}_QAM.pkl")

    train_dataset = QAMIDFTDataset(
        num_frames=num_frames,
        M=M,
        N=N,
        modulation_order=modulation_order,
        data_file=train_file,
        is_train=True,
        val_split_ratio=val_split_ratio
    )

    val_dataset = QAMIDFTDataset(
        num_frames=num_frames,
        M=M,
        N=N,
        modulation_order=modulation_order,
        data_file=val_file,
        is_train=False,
        val_split_ratio=val_split_ratio
    )

    if validate:
        print("\n=== 训练/验证集数据验证 ===")
        train_bits, train_qam, train_idft = train_dataset[0]
        print(f"训练集单帧维度：")
        print(f"  比特矩阵: {train_bits.shape} | QAM: {train_qam.shape} | Regulated OTFS: {train_idft.shape}")

        val_bits, val_qam, val_idft = val_dataset[0]
        print(f"验证集单帧维度：")
        print(f"  比特矩阵: {val_bits.shape} | QAM: {val_qam.shape} | Regulated OTFS: {val_idft.shape}")

        print(f"\n数据集总数验证：")
        print(f"  总帧数: {num_frames} | 训练集帧数: {len(train_dataset)} | 验证集帧数: {len(val_dataset)}")
        print(f"  训练+验证: {len(train_dataset) + len(val_dataset)} (应等于总帧数)")

    return train_dataset, val_dataset


# ==================== 运行示例 ====================
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    config = {
        "num_frames": 6000,          # 总帧数
        "M": 16,                     # 子载波数
        "N": 16,                     # 符号数（多普勒维）
        "modulation_order": 4,       # 4-QAM
        "base_save_path": "DataSet/qam_idft_dataset",
        "val_split_ratio": 0.2,
        "validate": True
    }

    train_dataset, val_dataset = generate_qam_idft_split_datasets(**config)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print("\n=== 训练集DataLoader迭代示例 ===")
    for batch_idx, (bits_batch, qam_batch, idft_batch) in enumerate(train_dataloader):
        print(f"Train Batch {batch_idx + 1}:")
        print(f"  比特矩阵: {bits_batch.shape}")   # [32, 16, 32] for 4-QAM
        print(f"  QAM矩阵: {qam_batch.shape}")     # [32, 16, 16, 2]
        print(f"  Regulated OTFS 矩阵: {idft_batch.shape}")   # [32, 16, 16, 2]
        break

    print("\n=== 验证集DataLoader迭代示例 ===")
    for batch_idx, (bits_batch, qam_batch, idft_batch) in enumerate(val_dataloader):
        print(f"Val Batch {batch_idx + 1}:")
        print(f"  比特矩阵: {bits_batch.shape}")
        print(f"  QAM矩阵: {qam_batch.shape}")
        print(f"  Regulated OTFS 矩阵: {idft_batch.shape}")
        break
