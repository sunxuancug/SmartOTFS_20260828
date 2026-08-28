# SmartOTFS

SmartOTFS 是一个面向 OTFS 调制波形学习与性能分析的实验工程。项目主要完成三类工作：

- 生成 4-QAM OTFS 训练/验证数据集；
- 训练并比较多种位置编码与注意力机制组合的神经网络 OTFS 波形生成模型；
- 对训练后的模型进行性能分析，并输出可用于论文绘图的 MATLAB/Python 图像与 `.mat` 数据。

当前实验配置主要包含 Config A-G：

| 配置 | 位置编码 | 注意力机制 | 图中显示名 |
|---|---|---|---|
| Config_A | Standard | Standard | Std-TF OTFS |
| Config_B | Standard | GQA | GQA-TF OTFS |
| Config_C | Standard | Differential | Differential-TF OTFS |
| Config_D | Standard | MLA | MLA-TF OTFS |
| Config_E | Standard | Dual_Axis | Dual-Axis-TF OTFS |
| Config_F | Standard | BoltAttention | BoltAttention-TF OTFS |
| Config_G | PhaseLoom | BoltAttention | Smart OTFS |

## 目录结构

```text
SmartOTFS/
├── DataSet/
├── training_models/
├── training_plots/
├── metrics_MAT_data/
├── performance_analysis_result/
├── Plot_Results_Code/
├── water_channel_data/
├── generate_bellhop_water_channel/
├── channel_cache/
├── DataSetGeneration.py
├── SmartOTFS_20260826.py
├── analyze_performance_20260827.py
├── bellhop_water_channel.py
└── training.log
```

## 主要文件夹说明

### `DataSet/`

保存 OTFS 训练集和验证集。

典型文件：

```text
train_data_4_QAM.pkl
val_data_4_QAM.pkl
train_data_4_QAM.mat
val_data_4_QAM.mat
```

`.pkl` 文件供 Python 训练和性能分析脚本读取；`.mat` 文件便于 MATLAB 或其他工具检查数据。

数据通常由 `DataSetGeneration.py` 生成。

### `training_models/`

保存各配置训练得到的模型权重。

典型子目录：

```text
Config_A_Pos_Standard_Attn_Standard/
Config_B_Pos_Standard_Attn_GQA/
Config_C_Pos_Standard_Attn_Differential/
Config_D_Pos_Standard_Attn_MLA/
Config_E_Pos_Standard_Attn_Dual_Axis/
Config_F_Pos_Standard_Attn_BoltAttention/
Config_G_Pos_PhaseLoom_Attn_BoltAttention/
shared/
inactive/
```

各 Config 子目录中保存对应模型的最佳权重，例如 `best_end2end_finetuned_*.pth`。`shared/` 通常保存共享模块权重，例如 QAM generator。`inactive/` 可用于临时放置不参与当前实验的模型。

### `training_plots/`

保存训练阶段产生的曲线图，例如 loss 曲线、训练过程可视化等。主要由 `SmartOTFS_20260826.py` 输出。

### `metrics_MAT_data/`

保存训练或验证过程中导出的中间指标 `.mat` 文件，便于 MATLAB 后处理或对比分析。该目录主要服务训练脚本中的指标落盘与复查。

### `performance_analysis_result/`

保存完整性能分析结果。每次运行 `analyze_performance_20260827.py` 会生成一个带时间戳的分析目录，例如：

```text
performance_analysis_result/analysis_20260828_195805/
├── performance_figures/
└── performance_mat_data/
```

`performance_figures/` 保存 Python 直接绘制的性能图。

`performance_mat_data/` 保存 MATLAB 后续重绘所需的数据，例如：

```text
cross_correlation_Config_A_to_G.mat
papr_ccdf_Config_A_to_G.mat
dd_error_Config_A_to_G.mat
time_domain_metrics_Config_A_to_G.mat
probe_energy_Config_A_to_G.mat
equivalent_basis_Config_A_to_G.mat
smartotfs_visual_examples_Config_A_to_G.mat
ber_results.mat
```

其中 `ber_results.mat` 包含 AWGN、EVA、水声 WATER 信道下的 BER 数据。MATLAB 绘图脚本会自动读取最新 `analysis_YYYYMMDD_HHMMSS/performance_mat_data`。

### `Plot_Results_Code/`

保存 MATLAB 论文图重绘脚本和导出结果。

主要文件：

```text
Plot_Results.m
save_figure_layout.m
apply_figure_layout.m
figure_layout.mat
```

`Plot_Results.m` 会读取 `performance_analysis_result` 中最新分析结果，并按论文图风格重绘图像。当前脚本会跳过 DD error heatmap、Equivalent Basis 3D 和 Equivalent Basis 2D，只保留需要的图和数据派生图。

每次运行 `Plot_Results.m` 会在 `Plot_Results_Code/` 下创建时间戳输出目录，例如：

```text
Plot_Results_Code/plot_results_20260828_202804/
├── eps/
├── emf/
└── fig/
```

脚本会先生成图窗，再应用 `figure_layout.mat` 中记录的窗口位置和尺寸，最后批量保存所有打开图窗为 `.eps`、`.emf` 和 `.fig`。

### `water_channel_data/`

保存水声信道数据文件。

典型文件：

```text
water_channel_otfs.mat
```

该文件由水声信道生成流程得到，并被 `analyze_performance_20260827.py` 读取，用于 WATER 信道 BER 仿真。

### `generate_bellhop_water_channel/`

保存 Bellhop 水声信道生成相关 MATLAB 代码。

典型文件：

```text
run_water_channel.m
channel_simulator.m
set_channel_params.m
BellhopFunctions/
water_channel_otfs.mat
```

该目录用于生成或复现实验所需的水声信道 `.mat` 文件。生成后的 `water_channel_otfs.mat` 应复制或同步到根目录下的 `water_channel_data/` 中，供 Python 性能分析脚本使用。

### `channel_cache/`

保存 EVA 和 WATER 信道矩阵缓存，避免每次 BER 仿真都重新构造信道矩阵。

典型文件：

```text
eva_M16_N16_frames500_v120_fc4GHz_df15k.npz
water_M16_N16_snap268_df250_fc3000.npz
```

如果信道参数、帧数或系统参数发生变化，可以删除对应缓存文件，让分析脚本重新生成。

### `__pycache__/`、`.idea/`

`__pycache__/` 是 Python 自动生成的字节码缓存目录。

`.idea/` 是 JetBrains/PyCharm 工程配置目录。它们不参与核心算法流程。

## 主要脚本说明

### `DataSetGeneration.py`

用于生成 QAM + IDFT 数据集，并自动拆分训练集和验证集。

主要输出：

```text
DataSet/train_data_4_QAM.pkl
DataSet/val_data_4_QAM.pkl
DataSet/train_data_4_QAM.mat
DataSet/val_data_4_QAM.mat
```

运行方式：

```bash
python DataSetGeneration.py
```

### `SmartOTFS_20260826.py`

核心训练脚本。该脚本定义多种位置编码和注意力机制组合，训练 Config A-G 等模型，并保存权重、训练曲线和指标数据。

主要输入：

```text
DataSet/train_data_4_QAM.pkl
DataSet/val_data_4_QAM.pkl
```

主要输出：

```text
training_models/
training_plots/
metrics_MAT_data/
training.log
```

运行方式：

```bash
python SmartOTFS_20260826.py
```

### `analyze_performance_20260827.py`

性能分析脚本。读取验证集和训练好的模型权重，统一计算并保存：

- 时域互相关；
- PAPR CCDF；
- DD 域误差曲线；
- 时域 MSE、EVM、PSD error；
- 探针能量响应；
- 等效基误差指标；
- AWGN / EVA / WATER 信道 BER。

主要输入：

```text
DataSet/val_data_4_QAM.pkl
training_models/Config_*/
water_channel_data/water_channel_otfs.mat
channel_cache/
```

主要输出：

```text
performance_analysis_result/analysis_YYYYMMDD_HHMMSS/performance_figures/
performance_analysis_result/analysis_YYYYMMDD_HHMMSS/performance_mat_data/
```

运行方式：

```bash
python analyze_performance_20260827.py
```

如果服务器上的水声信道文件不在默认位置，可以通过环境变量指定：

```bash
export SMARTOTFS_WATER_CHANNEL_MAT=/path/to/water_channel_otfs.mat
python analyze_performance_20260827.py
```

### `bellhop_water_channel.py`

水声信道转换工具。该脚本读取 Bellhop/channel simulator 生成的 `water_channel_otfs.mat`，提取多径时延、增益和多普勒参数，并转换为 OTFS 仿真所需的信道抽头格式。

它主要被 `analyze_performance_20260827.py` 调用，不一定需要单独运行。

### `Plot_Results_Code/Plot_Results.m`

MATLAB 论文图重绘脚本。它读取最新性能分析结果中的 `.mat` 文件，重新绘制论文风格图像。

主要输入：

```text
performance_analysis_result/analysis_YYYYMMDD_HHMMSS/performance_mat_data/
Plot_Results_Code/figure_layout.mat
```

主要输出：

```text
Plot_Results_Code/plot_results_YYYYMMDD_HHMMSS/eps/
Plot_Results_Code/plot_results_YYYYMMDD_HHMMSS/emf/
Plot_Results_Code/plot_results_YYYYMMDD_HHMMSS/fig/
```

运行方式：

```matlab
cd('D:\QQ_Files\课题研究工作\SmartOTFS\Plot_Results_Code')
run('Plot_Results.m')
```

## MATLAB 图窗布局复用

如果希望手动调整图窗位置和大小，并在下次自动复用：

1. 运行 `Plot_Results.m`，生成所有图窗；
2. 手动调整图窗大小和位置；
3. 在 MATLAB 命令行执行：

```matlab
save_figure_layout
```

这会生成或覆盖：

```text
Plot_Results_Code/figure_layout.mat
```

以后再次运行：

```matlab
run('Plot_Results.m')
```

脚本会自动：

1. 生成图窗；
2. 应用 `figure_layout.mat` 中保存的窗口尺寸和位置；
3. 将应用布局后的图保存为 `.eps`、`.emf` 和 `.fig`。

## 推荐工作流

### 1. 生成数据集

```bash
python DataSetGeneration.py
```

### 2. 训练模型

```bash
python SmartOTFS_20260826.py
```

训练完成后检查：

```text
training_models/
training_plots/
metrics_MAT_data/
```

### 3. 运行性能分析

```bash
python analyze_performance_20260827.py
```

分析完成后检查：

```text
performance_analysis_result/analysis_YYYYMMDD_HHMMSS/
```

### 4. MATLAB 重绘和导出论文图

```matlab
cd('D:\QQ_Files\课题研究工作\SmartOTFS\Plot_Results_Code')
run('Plot_Results.m')
```

导出结果位于：

```text
Plot_Results_Code/plot_results_YYYYMMDD_HHMMSS/
```

## 注意事项

- MATLAB 绘图脚本会按 `analysis_YYYYMMDD_HHMMSS` 文件夹名称选择最新性能分析结果，而不是按系统修改时间选择。
- 若 `ber_results.mat` 中存在 `ber_*_WATER` 字段，MATLAB 会自动绘制水声信道 BER 曲线。
- 当前 MATLAB 导出顺序是先应用窗口布局，再保存 EPS/EMF/FIG，因此导出的图会保留已记录的窗口大小。
- `.fig` 保存的是 MATLAB 图对象，不依赖 DPI；EPS/EMF 导出中的栅格化部分会受分辨率设置影响。
- 如果更换模型配置、信道参数或数据集，建议重新运行性能分析脚本，生成新的 `performance_analysis_result/analysis_*` 目录。
