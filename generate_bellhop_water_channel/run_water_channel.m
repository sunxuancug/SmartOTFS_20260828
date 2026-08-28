% run_water_channel.m
% 包装脚本：生成水声信道 .mat 数据供 smartOTFS Python BER 测试使用
%
% 用法:
%   run_water_channel                    % 使用默认参数和默认输出路径
%   run_water_channel(output_dir)        % 指定输出目录
%   run_water_channel(output_dir, cfg)   % 指定输出目录和自定义参数结构体
%
% 依赖: channel_simulator.m, set_channel_params.m, BellhopFunctions/

function run_water_channel(output_dir, cfg)
    % === 默认输出路径 ===
    if nargin < 1 || isempty(output_dir)
        script_dir = fileparts(mfilename('fullpath'));
        output_dir = fullfile('..', 'water_channel_data');
    else
        script_dir = fileparts(mfilename('fullpath'));
    end

    % === 默认信道参数（基于 channel_user1 的浅海水声配置）===
    if nargin < 2 || isempty(cfg)
        cfg = struct();
    end

    % 几何参数
    if ~isfield(cfg, 'channel_name'), cfg.channel_name = 'water_channel_otfs'; end
    if ~isfield(cfg, 'h0'),  cfg.h0  = 100;  end  % 水深 [m]
    if ~isfield(cfg, 'ht0'), cfg.ht0 = 50;   end  % TX 深度 [m]
    if ~isfield(cfg, 'hr0'), cfg.hr0 = 50;   end  % RX 深度 [m]
    if ~isfield(cfg, 'd0'),  cfg.d0  = 500;  end  % 通信距离 [m]

    % 信号参数
    if ~isfield(cfg, 'f0'), cfg.f0 = 600;   end  % 最低频率 [Hz]
    if ~isfield(cfg, 'B'),  cfg.B  = 4000;  end  % 带宽 [Hz]

    % 多普勒参数（漂移速度）
    if ~isfield(cfg, 'vtv'),      cfg.vtv      = 0;    end  % TX 航行速度 [m/s]
    if ~isfield(cfg, 'theta_tv'), cfg.theta_tv = 0;    end  % TX 航行角度 [rad]
    if ~isfield(cfg, 'vtd_a'),    cfg.vtd_a    = 0.08; end  % TX 漂移幅度 [m/s]
    if ~isfield(cfg, 'vrv'),      cfg.vrv      = 0;    end  % RX 航行速度 [m/s]
    if ~isfield(cfg, 'theta_rv'), cfg.theta_rv = 0;    end  % RX 航行角度 [rad]
    if ~isfield(cfg, 'vrd_a'),    cfg.vrd_a    = 0.02; end  % RX 漂移幅度 [m/s]

    % === 添加依赖路径 ===
    addpath(script_dir);  % 自身目录 (含 set_channel_params, channel_simulator)
    addpath(fullfile(script_dir, 'BellhopFunctions'));
    addpath(fullfile(script_dir, 'BellhopFunctions', 'Takagi package'));

    % === 切换到脚本所在目录工作 ===
    % channel_simulator.m 内部会 cd 到自身所在目录来读取 .prm/.dop 和保存 .mat
    % 因此所有文件操作都在 script_dir 进行，完成后把 .mat 复制到输出目录
    original_dir = pwd;
    cd(script_dir);

    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    fprintf('=============================================================\n');
    fprintf('  Water Acoustic Channel Generator for smartOTFS\n');
    fprintf('=============================================================\n');
    fprintf('  Channel: %s\n', cfg.channel_name);
    fprintf('  Geometry: depth=%.0fm, range=%.0fm, TX=%.0fm, RX=%.0fm\n', ...
        cfg.h0, cfg.d0, cfg.ht0, cfg.hr0);
    fprintf('  Signal: f_min=%.0fHz, B=%.0fHz, f_center=%.0fHz\n', ...
        cfg.f0, cfg.B, cfg.f0 + cfg.B/2);
    fprintf('  Output: %s\n', fullfile(output_dir, [cfg.channel_name, '.mat']));
    fprintf('=============================================================\n');

    % === 第1步: 生成 .prm 和 .dop 参数文件 ===
    fprintf('\n[Step 1/2] Generating .prm and .dop parameter files...\n');
    Band = set_channel_params(cfg.f0, cfg.B, cfg.channel_name, ...
        cfg.vtv, cfg.theta_tv, cfg.vtd_a, ...
        cfg.vrv, cfg.theta_rv, cfg.vrd_a, ...
        cfg.h0, cfg.ht0, cfg.d0, cfg.hr0);
    close all;  % 关闭 set_channel_params 产生的图形
    fprintf('  Done: %s.prm, %s.dop\n', cfg.channel_name, cfg.channel_name);

    % === 第2步: 运行信道仿真 → 生成 .mat ===
    fprintf('\n[Step 2/2] Running channel simulator...\n');
    channel_simulator(cfg.channel_name);
    close all;  % 关闭 channel_simulator 产生的图形
    fprintf('  Done: %s.mat (in working dir)\n', cfg.channel_name);

    % === 将 .mat 复制到输出目录 ===
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    src_mat = fullfile(script_dir, [cfg.channel_name, '.mat']);
    dst_mat = fullfile(output_dir, [cfg.channel_name, '.mat']);
    [copy_status, copy_msg] = copyfile(src_mat, dst_mat);
    if copy_status
        fprintf('  已复制到: %s\n', dst_mat);
    else
        warning('复制失败: %s', copy_msg);
    end

    % === 清理中间文件 ===
    delete(fullfile(script_dir, [cfg.channel_name, '.prm']));
    delete(fullfile(script_dir, [cfg.channel_name, '.dop']));
    % 保留 script_dir 下的 .mat 作为缓存，下次运行会覆盖

    % === 返回原目录 ===
    cd(original_dir);

    fprintf('\n=============================================================\n');
    fprintf('  Water channel data saved successfully!\n');
    fprintf('  File: %s\n', dst_mat);
    fprintf('=============================================================\n');
end
