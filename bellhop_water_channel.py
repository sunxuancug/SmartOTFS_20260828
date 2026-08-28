"""
水声信道模块 —— 从 Bellhop/channel_simulator 生成的 .mat 文件中
提取信道参数，转换为 OTFS 系统所需的 (taps, delay_taps, Doppler_taps, chan_coef) 格式。

用法:
    from bellhop_water_channel import water_channel_gen, load_channel_metadata

    taps, delay_taps, Doppler_taps, chan_coef = water_channel_gen(
        M=16, N=16, mat_file_path="water_channel_otfs.mat"
    )

依赖: scipy, numpy
"""

import numpy as np
from scipy.io import loadmat
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)


def load_channel_metadata(mat_file_path):
    """
    加载水声信道 .mat 文件的元数据，不提取完整 hmat。

    返回:
        meta: dict, 包含:
            - 'dt': 时间分辨率 [s]
            - 'df': 频率分辨率 [Hz]
            - 'Lf': 频率采样点数
            - 'Lt_tot': 时间采样点数
            - 'freq_range': (f_min_approx, f_max_approx) 从 .prm 参数推算
            - 'total_duration': 信道总时长 [s]
            - 'delay_resolution': 时延分辨率 [s]
            - 'max_delay': 最大无模糊时延 [s]
    """
    mat = loadmat(mat_file_path)
    hmat = mat['hmat']         # [Lf, Lt_tot] complex
    dt = float(mat['dt'].ravel()[0])
    df = float(mat['df'].ravel()[0])

    Lf, Lt_tot = hmat.shape
    total_duration = Lt_tot * dt
    delay_resolution = 1.0 / (Lf * df)
    max_delay = 1.0 / df

    meta = {
        'dt': dt,
        'df': df,
        'Lf': Lf,
        'Lt_tot': Lt_tot,
        'total_duration': total_duration,
        'delay_resolution': delay_resolution,
        'max_delay': max_delay,
    }
    return meta


def extract_multipath_taps(hmat, dt, df, peak_threshold_db=-20, min_peak_spacing=1):
    """
    从时变频率响应 hmat 中提取多径抽头参数。

    对 hmat 的每一列（每个时刻）做 IFFT 得到时域 CIR，
    使用峰值检测识别多径分量。

    参数:
        hmat: [Lf, Lt_tot] 复数矩阵，频域信道响应
        dt: 时间分辨率 [s]
        df: 频率分辨率 [Hz]
        peak_threshold_db: 峰值检测阈值 [dB]，低于最大峰值此值以下的忽略
        min_peak_spacing: 峰值最小间隔 [采样点]

    返回:
        all_paths: list of dict, 每个元素对应一个时刻的多径信息:
            {
                'delays': [float],      # 时延 [s] (相对于第一个到达路径)
                'gains': [complex],     # 复增益
                'num_paths': int,       # 路径数
            }
    """
    Lf, Lt_tot = hmat.shape
    all_paths = []

    # 阈值转换为线性
    threshold_linear = 10.0 ** (peak_threshold_db / 20.0)

    for t in range(Lt_tot):
        # hmat[:, t] 已经是时域 CIR（channel_simulator 内部做了 IFFT）
        cir = hmat[:, t]  # [Lf], complex
        cir_mag = np.abs(cir)

        # 峰值检测
        max_mag = np.max(cir_mag)
        if max_mag < 1e-12:
            all_paths.append({'delays': [0.0], 'gains': [1.0+0j], 'num_paths': 1})
            continue

        # 找到所有超过阈值的局部最大值
        peaks = []
        for i in range(len(cir_mag)):
            if cir_mag[i] < max_mag * threshold_linear:
                continue
            # 检查是否为局部最大值
            left = cir_mag[i - 1] if i > 0 else 0
            right = cir_mag[i + 1] if i < Lf - 1 else 0
            if cir_mag[i] >= left and cir_mag[i] > right:
                peaks.append(i)

        # 合并间距过近的峰值（保留幅度最大的）
        if min_peak_spacing > 1 and len(peaks) > 1:
            merged = []
            i = 0
            while i < len(peaks):
                group = [peaks[i]]
                j = i + 1
                while j < len(peaks) and peaks[j] - peaks[j-1] < min_peak_spacing:
                    group.append(peaks[j])
                    j += 1
                # 此组中取幅度最大者
                best = max(group, key=lambda x: cir_mag[x])
                merged.append(best)
                i = j
            peaks = merged

        # 提取延迟和增益
        delays_abs = np.array([p / (Lf * df) for p in peaks])  # 绝对时延 [s]
        gains = cir[peaks]

        # 转换为相对于首个到达路径的时延
        delays_rel = delays_abs - delays_abs[0]

        all_paths.append({
            'delays': delays_rel.tolist(),
            'gains': gains.tolist(),
            'num_paths': len(peaks),
        })

    return all_paths


def estimate_doppler(all_paths, dt, fc):
    """
    从时变路径相位中估计多普勒频移。

    参数:
        all_paths: extract_multipath_taps 的输出
        dt: 时间分辨率 [s]
        fc: 载波频率 [Hz]

    返回:
        doppler_per_path: list of list, 每个时刻每条路径的多普勒频移 [Hz]
    """
    Lt_tot = len(all_paths)
    if Lt_tot < 2:
        return [[0.0] * ap['num_paths'] for ap in all_paths]

    doppler_per_path = []
    for t in range(Lt_tot):
        num_paths = all_paths[t]['num_paths']
        dopplers = []
        for p in range(num_paths):
            if t == 0:
                g0 = all_paths[0]['gains'][p] if p < len(all_paths[0]['gains']) else 0
                g1 = all_paths[1]['gains'][p] if p < len(all_paths[1]['gains']) else 0
            elif t == Lt_tot - 1:
                g0 = all_paths[t-1]['gains'][p] if p < len(all_paths[t-1]['gains']) else 0
                g1 = all_paths[t]['gains'][p] if p < len(all_paths[t]['gains']) else 0
            else:
                g0 = all_paths[t-1]['gains'][p] if p < len(all_paths[t-1]['gains']) else 0
                g1 = all_paths[t+1]['gains'][p] if p < len(all_paths[t+1]['gains']) else 0

            if abs(g0) < 1e-12 or abs(g1) < 1e-12:
                dopplers.append(0.0)
                continue

            dphi = np.angle(g1 * np.conj(g0))
            fd = dphi / (2.0 * np.pi * 2.0 * dt)
            dopplers.append(float(fd))

        doppler_per_path.append(dopplers)

    return doppler_per_path


def water_channel_gen(M, N, mat_file_path=None, delta_f=None, fc=None,
                      frame_idx=0, peak_threshold_db=-20,
                      hmat_preloaded=None, dt_preloaded=None, df_preloaded=None,
                      all_paths_preloaded=None, all_dopplers_preloaded=None):
    """
    从水声信道 .mat 文件生成 OTFS 信道参数。

    参数:
        M, N: OTFS 网格维度
        mat_file_path: .mat 文件路径（与 hmat_preloaded 二选一）
        delta_f: OTFS 子载波间隔 [Hz]。默认 None = 自动计算 B/M。
        fc: 载波频率 [Hz]。默认 None = 使用水声信道中心频率。
        frame_idx: 使用 hmat 中第几个时间快照
        peak_threshold_db: 多径峰值检测阈值 [dB]
        hmat_preloaded, dt_preloaded, df_preloaded: 预加载数据（避免重复 loadmat）
        all_paths_preloaded: 预提取的多径参数（避免重复 extract_multipath_taps）
        all_dopplers_preloaded: 预估计的多普勒（避免重复 estimate_doppler）

    返回:
        taps, delay_taps, Doppler_taps, chan_coef
    """
    # ===== 加载数据 =====
    if hmat_preloaded is not None:
        hmat = hmat_preloaded
        dt_val = dt_preloaded
        df_val = df_preloaded
    else:
        if mat_file_path is None:
            raise ValueError("必须提供 mat_file_path 或 hmat_preloaded")
        mat = loadmat(mat_file_path)
        hmat = mat['hmat']
        dt_val = float(mat['dt'].ravel()[0])
        df_val = float(mat['df'].ravel()[0])

    Lf, Lt_tot = hmat.shape
    B = Lf * df_val

    if fc is None:
        fc = B / 2.0
    if delta_f is None:
        delta_f = B / M

    T = 1.0 / delta_f
    one_delay_tap = 1.0 / (M * delta_f)
    one_doppler_tap = 1.0 / (N * T)

    # ===== 提取多径（使用预提取数据或重新计算）=====
    if all_paths_preloaded is not None:
        all_paths = all_paths_preloaded
    else:
        all_paths = extract_multipath_taps(
            hmat, dt_val, df_val, peak_threshold_db=peak_threshold_db)

    t_idx = min(frame_idx, Lt_tot - 1)
    paths = all_paths[t_idx]

    # ===== 估计多普勒（使用预估计或重新计算）=====
    if all_dopplers_preloaded is not None:
        all_dopplers = all_dopplers_preloaded
    else:
        all_dopplers = estimate_doppler(all_paths, dt_val, fc)
    dopplers = all_dopplers[t_idx]

    # ===== 转换为 OTFS 参数 =====
    delays_sec = np.array(paths['delays'])
    gains = np.array(paths['gains'])
    dopplers_hz = np.array(dopplers)

    delay_taps = np.round(delays_sec / one_delay_tap).astype(int)
    delay_taps = np.maximum.accumulate(delay_taps)

    max_doppler_tap = N // 2
    Doppler_taps = np.round(dopplers_hz / one_doppler_tap).astype(int)
    Doppler_taps = np.clip(Doppler_taps, -max_doppler_tap, max_doppler_tap).astype(int)

    # ===== 功率归一化 =====
    pow_prof = np.abs(gains) ** 2
    pow_sum = np.sum(pow_prof)
    if pow_sum > 1e-12:
        pow_prof = pow_prof / pow_sum
    chan_coef = np.sqrt(pow_prof) * (gains / (np.abs(gains) + 1e-12))

    taps = len(gains)

    # ===== ZP 截断警告（仅首次）=====
    max_d = int(np.max(delay_taps))
    if max_d >= M and frame_idx == 0:
        n_truncated = np.sum(delay_taps >= M)
        import warnings as _w
        _w.warn(
            f"水声信道有 {n_truncated}/{taps} 条路径的延迟 (max={max_d}) "
            f"超过 OTFS ZP 保护间隔 M={M}。这些路径会被部分截断。"
            f"建议增大 M 或提高 delta_f (当前={delta_f:.1f} Hz)。"
        )

    return taps, delay_taps, Doppler_taps, chan_coef


def print_channel_info(taps, delay_taps, Doppler_taps, chan_coef, delta_f, N):
    """打印信道参数摘要"""
    T = 1.0 / delta_f
    one_delay_tap = 1.0 / (len(delay_taps) * delta_f) if taps > 0 else 0
    one_doppler_tap = 1.0 / (N * T)

    print(f"\n{'='*60}")
    print(f"  Water Acoustic Channel Summary")
    print(f"{'='*60}")
    print(f"  Number of paths (taps): {taps}")
    print(f"  {'Path':<6s} {'Delay [s]':<14s} {'Delay [tap]':<12s} "
          f"{'Doppler [Hz]':<14s} {'Doppler [tap]':<14s} {'Gain [dB]':<10s}")
    print(f"  {'-'*70}")

    for i in range(taps):
        delay_s = delay_taps[i] * one_delay_tap if one_delay_tap > 0 else 0
        dopp_hz = Doppler_taps[i] * one_doppler_tap
        gain_db = 20.0 * np.log10(abs(chan_coef[i]) + 1e-12)
        print(f"  {i:<6d} {delay_s:<14.6f} {int(delay_taps[i]):<12d} "
              f"{dopp_hz:<14.4f} {int(Doppler_taps[i]):<14d} {gain_db:<10.2f}")

    # 统计
    max_delay_s = np.max(delay_taps) * one_delay_tap if taps > 0 else 0
    max_delay_taps = int(np.max(delay_taps)) if taps > 0 else 0
    print(f"  {'-'*70}")
    print(f"  Max delay: {max_delay_s*1000:.3f} ms = {max_delay_taps} OTFS taps")
    print(f"  OTFS frame duration: {N*T*1000:.3f} ms")
    print(f"  Delay resolution: {one_delay_tap*1e6:.2f} µs")
    print(f"  Doppler resolution: {one_doppler_tap:.4f} Hz")
    print(f"{'='*60}\n")


# ======================= 测试入口 =======================
if __name__ == "__main__":
    import sys

    # 查找 .mat 文件
    MAT_DIR = 'water_channel_data'
    MAT_FILE = os.path.join(MAT_DIR, 'water_channel_otfs.mat')

    if not os.path.exists(MAT_FILE):
        print(f"❌ .mat 文件不存在: {MAT_FILE}")
        print(f"   请先在 MATLAB 中运行 matlab program/bellhop_channel/run_water_channel.m")
        print(f"   生成水声信道数据后再运行此脚本。")
        sys.exit(1)

    print(f"📁 加载信道数据: {MAT_FILE}")

    # 查看元数据
    meta = load_channel_metadata(MAT_FILE)
    print(f"\n信道元数据:")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    # 生成 OTFS 信道参数
    M, N = 16, 16
    delta_f = meta['df'] * meta['Lf'] / M  # B / M
    print(f"\nOTFS 参数: M={M}, N={N}, delta_f={delta_f:.1f} Hz")

    taps, delay_taps, Doppler_taps, chan_coef = water_channel_gen(
        M, N, MAT_FILE, delta_f=delta_f
    )

    print_channel_info(taps, delay_taps, Doppler_taps, chan_coef, delta_f, N)
    print("✅ 水声信道参数提取成功！")
