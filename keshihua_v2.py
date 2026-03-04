# keshihua.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from scipy.signal import find_peaks
import pandas as pd
import json
import warnings

warnings.filterwarnings('ignore')
matplotlib.use('Agg')

# 确保中文和负号可以正常显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def calculate_clinical_metrics(target_signal, pred_signal, fs=500):
    """
    计算临床医学评估指标（检测用，不参与梯度）
    包括：
    1. R波振幅绝对误差 (R-Peak Amplitude Error)
    2. 平均心率误差 (Heart Rate Error) - 基于R波间隔

    参数:
    - target_signal: 真实信号
    - pred_signal: 预测信号
    - fs: 采样率 (默认500Hz)

    返回:
    - 字典包含 R_Amp_Err, HR_Err
    """

    # 1. 检测R波 (使用scipy)
    # 这里的height和distance是经验值，可能需要根据归一化后的数据调整
    # 假设数据已经Instance Norm过，R波通常>2.0或>1.0，这里设得宽松点
    # distance=fs*0.4 (即200ms)，假设心率不超过300bpm

    # 对归一化数据，很难设定绝对阈值，这里使用相对最大值的比例
    target_max = np.max(target_signal)
    pred_max = np.max(pred_signal)

    target_peaks, target_props = find_peaks(target_signal, height=target_max * 0.5, distance=fs * 0.4)
    pred_peaks, pred_props = find_peaks(pred_signal, height=pred_max * 0.5, distance=fs * 0.4)

    metrics = {
        'R_Amp_Err': 0.0,
        'HR_Err': 0.0
    }

    # --- R波振幅误差 ---
    if len(target_props['peak_heights']) > 0 and len(pred_props['peak_heights']) > 0:
        avg_target_amp = np.mean(target_props['peak_heights'])
        avg_pred_amp = np.mean(pred_props['peak_heights'])
        metrics['R_Amp_Err'] = abs(avg_target_amp - avg_pred_amp)
    else:
        # 如果检测不到波峰，直接用最大值之差
        metrics['R_Amp_Err'] = abs(target_max - pred_max)

    # --- 心率误差 (HR Error) ---
    # 计算RR间隔
    if len(target_peaks) > 1:
        target_rr = np.diff(target_peaks) / fs  # seconds
        target_hr = 60.0 / np.mean(target_rr)
    else:
        target_hr = 0.0

    if len(pred_peaks) > 1:
        pred_rr = np.diff(pred_peaks) / fs  # seconds
        pred_hr = 60.0 / np.mean(pred_rr)
    else:
        pred_hr = 0.0

    if target_hr > 0 and pred_hr > 0:
        metrics['HR_Err'] = abs(target_hr - pred_hr)
    else:
        metrics['HR_Err'] = 0.0  # 无法计算心率时置0，避免NaN

    return metrics


def calculate_metrics(target_signal, pred_signal):
    """
    计算所有评估指标
    参数：
    - target_signal: 目标信号 (1D array)
    - pred_signal: 预测信号 (1D array)
    返回：
    - 包含所有指标的字典
    """

    # 1. PCC (Pearson Correlation Coefficient)
    pcc, _ = pearsonr(target_signal, pred_signal)

    # 2. MSE (Mean Squared Error)
    mse = mean_squared_error(target_signal, pred_signal)

    # 3. RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mse)

    # 4. SNR_dB (Signal-to-Noise Ratio in dB)
    # SNR = 10 * log10(P_signal / P_noise)
    # P_signal = mean(target^2)
    # P_noise = mean((target - pred)^2)
    signal_power = np.mean(target_signal ** 2)
    noise_power = np.mean((target_signal - pred_signal) ** 2)
    snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))

    # 5. PRD (Percentage Root-Mean-Square Difference)
    # PRD = 100 * sqrt(sum((target - pred)^2) / sum(target^2))
    prd_percent = 100 * np.sqrt(np.sum((target_signal - pred_signal) ** 2) /
                                (np.sum(target_signal ** 2) + 1e-10))

    # 6. MAE (Mean Absolute Error) - 额外的鲁棒性指标
    mae = mean_absolute_error(target_signal, pred_signal)

    # 7. 临床医学指标
    clinical_metrics = calculate_clinical_metrics(target_signal, pred_signal)

    return {
        'PCC': pcc,
        'MSE': mse,
        'RMSE': rmse,
        'SNR_dB': snr_db,
        'PRD_%': prd_percent,
        'MAE': mae,
        'R_Amp_Err': clinical_metrics['R_Amp_Err'],
        'HR_Err': clinical_metrics['HR_Err']
    }


def plot_loss_curves(train_losses, val_losses, lead_name, suffix=""):
    """绘制并保存训练和验证损失曲线"""

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label=f'训练损失', color='blue', linewidth=2)
    plt.plot(val_losses, label=f'验证损失', color='orange', linewidth=2)
    plt.title(f'模型训练过程 (输入导联: {lead_name})', fontsize=14, fontweight='bold')
    plt.xlabel('轮次 (Epoch)', fontsize=12)
    plt.ylabel('损失 (Loss)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 创建保存目录
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, f"loss_curve_{lead_name}_{suffix}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Loss curve saved to {save_path}")


def visualize_reconstruction(original_ecg, reconstructed_ecg, target_ecg, input_lead_idx,
                             lead_names, sample_idx, epoch, metrics_dict=None):
    """

    可视化单个样本的重建效果，并显示每个导联的指标
    参数：
    - original_ecg: 原始ECG信号
    - reconstructed_ecg: 重建的ECG信号
    - target_ecg: 目标ECG信号
    - input_lead_idx: 输入导联索引
    - lead_names: 导联名称列表
    - sample_idx: 样本索引
    - epoch: 轮次
    - metrics_dict: 包含各导联指标的字典
    """

    num_leads = len(lead_names)
    fig, axes = plt.subplots(num_leads, 1, figsize=(20, 2 * num_leads), sharex=True)
    fig.suptitle(f'ECG Reconstruction (Input: {lead_names[input_lead_idx]}) - Sample {sample_idx}, Epoch {epoch}',
                 fontsize=16, fontweight='bold')

    # 反归一化以进行可视化 (使用原始信号的统计量)
    mean = original_ecg.mean(axis=-1, keepdims=True)
    std = original_ecg.std(axis=-1, keepdims=True)
    std[std == 0] = 1.0
    recon_denorm = reconstructed_ecg * std + mean
    target_denorm = target_ecg * std + mean

    time_axis = np.arange(original_ecg.shape[1])

    for i in range(num_leads):
        ax = axes[i]

        ax.plot(time_axis, target_denorm[i], color='blue', label='原始信号', linewidth=1.5)
        ax.plot(time_axis, recon_denorm[i], color='red', label='重建信号', linestyle='--', linewidth=1.5)

        title_text = f'导联 {lead_names[i]}'
        if i == input_lead_idx:
            ax.set_facecolor('lightyellow')
            title_text += ' (输入)'

        # 计算当前导联的指标
        # 实时计算一次用于展示，或者从metrics_dict获取（如果是总体平均就不能用）
        # 这里为了展示该样本的特定指标，我们简单重新计算几个关键指标
        pcc, _ = pearsonr(target_denorm[i], recon_denorm[i])
        mse_val = mean_squared_error(target_denorm[i], recon_denorm[i])

        # 临床指标
        clinical = calculate_clinical_metrics(target_denorm[i], recon_denorm[i])

        ax.set_title(
            f'{title_text} | PCC: {pcc:.4f}, MSE: {mse_val:.4f}, R-Amp Err: {clinical["R_Amp_Err"]:.4f}, HR Err: {clinical["HR_Err"]:.1f}')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)

    axes[-1].set_xlabel('时间点', fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    save_path = os.path.join(results_dir,
                             f"reconstruction_{lead_names[input_lead_idx]}_epoch_{epoch}_sample_{sample_idx}.png")

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Reconstruction plot saved to {save_path}")


def plot_metrics_comparison(metrics_dict, lead_names, input_lead_idx, epoch, version="final"):
    """

    绘制所有导联的指标柱状图



    参数：

    - metrics_dict: 包含各导联指标的字典

    - lead_names: 导联名称列表

    - input_lead_idx: 输入导联索引

    - epoch: 轮次

    - version: 模型版本

    """

    # 提取各项指标

    pcc_values = [metrics_dict[lead]['PCC'] for lead in lead_names if lead in metrics_dict]
    mse_values = [metrics_dict[lead]['MSE'] for lead in lead_names if lead in metrics_dict]
    rmse_values = [metrics_dict[lead]['RMSE'] for lead in lead_names if lead in metrics_dict]
    snr_values = [metrics_dict[lead]['SNR_dB'] for lead in lead_names if lead in metrics_dict]
    r_amp_values = [metrics_dict[lead]['R_Amp_Err'] for lead in lead_names if lead in metrics_dict]

    lead_names_filtered = [lead for lead in lead_names if lead in metrics_dict]

    x_pos = np.arange(len(lead_names_filtered))

    # 创建2x3的子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    fig.suptitle(f'Evaluation Metrics for All Leads (Input: {lead_names[input_lead_idx]})',

                 fontsize=16, fontweight='bold')

    # 1. PCC 柱状图

    ax = axes[0, 0]

    bars = ax.bar(x_pos, pcc_values, color='steelblue', alpha=0.7, edgecolor='black')

    # 为输入导联着色
    bars[input_lead_idx].set_color('orange')
    bars[input_lead_idx].set_alpha(0.9)

    ax.axhline(y=np.mean(pcc_values), color='red', linestyle='--', linewidth=2,

               label=f'Mean: {np.mean(pcc_values):.4f}')

    ax.set_ylabel('PCC', fontsize=11, fontweight='bold')

    ax.set_title('Pearson Correlation Coefficient', fontsize=12, fontweight='bold')

    ax.set_xticks(x_pos)

    ax.set_xticklabels(lead_names_filtered, rotation=45)

    ax.set_ylim([0, 1])

    ax.legend()

    ax.grid(axis='y', alpha=0.3)

    # 2. MSE 柱状图

    ax = axes[0, 1]

    bars = ax.bar(x_pos, mse_values, color='coral', alpha=0.7, edgecolor='black')

    # 为输入导联着色
    bars[input_lead_idx].set_color('orange')
    bars[input_lead_idx].set_alpha(0.9)

    ax.axhline(y=np.mean(mse_values), color='red', linestyle='--', linewidth=2,

               label=f'Mean: {np.mean(mse_values):.6f}')

    ax.set_ylabel('MSE', fontsize=11, fontweight='bold')

    ax.set_title('Mean Squared Error', fontsize=12, fontweight='bold')

    ax.set_xticks(x_pos)

    ax.set_xticklabels(lead_names_filtered, rotation=45)

    ax.legend()

    ax.grid(axis='y', alpha=0.3)

    # 3. RMSE 柱状图

    ax = axes[0, 2]

    bars = ax.bar(x_pos, rmse_values, color='lightgreen', alpha=0.7, edgecolor='black')

    # 为输入导联着色
    bars[input_lead_idx].set_color('orange')
    bars[input_lead_idx].set_alpha(0.9)

    ax.axhline(y=np.mean(rmse_values), color='red', linestyle='--', linewidth=2,

               label=f'Mean: {np.mean(rmse_values):.6f}')

    ax.set_ylabel('RMSE', fontsize=11, fontweight='bold')

    ax.set_title('Root Mean Squared Error', fontsize=12, fontweight='bold')

    ax.set_xticks(x_pos)

    ax.set_xticklabels(lead_names_filtered, rotation=45)

    ax.legend()

    ax.grid(axis='y', alpha=0.3)

    # 4. SNR_dB 柱状图

    ax = axes[1, 0]

    bars = ax.bar(x_pos, snr_values, color='skyblue', alpha=0.7, edgecolor='black')

    # 为输入导联着色
    bars[input_lead_idx].set_color('orange')
    bars[input_lead_idx].set_alpha(0.9)

    ax.axhline(y=np.mean(snr_values), color='red', linestyle='--', linewidth=2,

               label=f'Mean: {np.mean(snr_values):.2f}')

    ax.set_ylabel('SNR (dB)', fontsize=11, fontweight='bold')

    ax.set_title('Signal-to-Noise Ratio', fontsize=12, fontweight='bold')

    ax.set_xticks(x_pos)

    ax.set_xticklabels(lead_names_filtered, rotation=45)

    ax.legend()

    ax.grid(axis='y', alpha=0.3)

    # 5. R-Amp Error (Clinical)

    ax = axes[1, 1]

    bars = ax.bar(x_pos, r_amp_values, color='plum', alpha=0.7, edgecolor='black')

    # 为输入导联着色
    bars[input_lead_idx].set_color('orange')
    bars[input_lead_idx].set_alpha(0.9)

    ax.axhline(y=np.mean(r_amp_values), color='red', linestyle='--', linewidth=2,

               label=f'Mean: {np.mean(r_amp_values):.2f}')

    ax.set_ylabel('R-Wave Amp Error', fontsize=11, fontweight='bold')

    ax.set_title('Clinical: R-Peak Amplitude Error', fontsize=12, fontweight='bold')

    ax.set_xticks(x_pos)

    ax.set_xticklabels(lead_names_filtered, rotation=45)

    ax.legend()

    ax.grid(axis='y', alpha=0.3)

    # 6. 综合评分 (归一化指标)

    ax = axes[1, 2]

    # 计算综合评分：(PCC + (1-Normalized_RMSE)) / 2
    # 简单的综合打分逻辑
    comprehensive_scores = []

    for i in range(len(lead_names_filtered)):
        pcc_score = max(0, pcc_values[i])  # 范围[0,1]
        rmse_normalized = 1 - min(rmse_values[i] / (np.max(rmse_values) + 1e-6), 1)  # 越小越好
        score = (pcc_score + rmse_normalized) / 2
        comprehensive_scores.append(score)

    bars = ax.bar(x_pos, comprehensive_scores, color='gold', alpha=0.7, edgecolor='black')

    # 为输入导联着色
    bars[input_lead_idx].set_color('orange')
    bars[input_lead_idx].set_alpha(0.9)

    ax.axhline(y=np.mean(comprehensive_scores), color='red', linestyle='--', linewidth=2,

               label=f'Mean: {np.mean(comprehensive_scores):.4f}')

    ax.set_ylabel('Score', fontsize=11, fontweight='bold')

    ax.set_title('Comprehensive Evaluation Score', fontsize=12, fontweight='bold')

    ax.set_xticks(x_pos)

    ax.set_xticklabels(lead_names_filtered, rotation=45)

    ax.set_ylim([0, 1])

    ax.legend()

    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    results_dir = "results"

    os.makedirs(results_dir, exist_ok=True)

    save_path = os.path.join(results_dir,

                             f"metrics_comparison_{lead_names[input_lead_idx]}_epoch_{epoch}_v{version}.png")

    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()

    print(f"✓ Metrics comparison plot saved to {save_path}")


def test_single_lead_model_with_visualization(model, test_loader, input_lead_idx, device,

                                              lead_names, epoch="final", version="v8"):
    """

    测试模型性能并生成可视化图表，计算详细指标



    参数：

    - model: 训练好的模型

    - test_loader: 测试数据加载器

    - input_lead_idx: 输入导联索引

    - device: 计算设备

    - lead_names: 导联名称列表

    - epoch: 轮次标识

    - version: 模型版本



    返回：

    - avg_metrics: 平均指标字典

    - detail_metrics: 详细指标字典

    """

    model.eval()

    all_targets = []

    all_preds = []

    all_originals = []

    print("Running evaluation and visualization...")

    with torch.no_grad():

        for input_ecg, target_ecg, original_ecg in tqdm(test_loader, desc="Testing"):
            input_ecg, target_ecg = input_ecg.to(device), target_ecg.to(device)

            output_ecg = model(input_ecg)

            all_targets.append(target_ecg.cpu().numpy())

            all_preds.append(output_ecg.cpu().numpy())

            all_originals.append(original_ecg.cpu().numpy())

    all_targets = np.concatenate(all_targets, axis=0)

    all_preds = np.concatenate(all_preds, axis=0)

    all_originals = np.concatenate(all_originals, axis=0)

    # 计算详细指标 - 包含输入导联

    detail_metrics = {}

    avg_metrics = {
        'PCC': [],
        'MSE': [],
        'RMSE': [],
        'SNR_dB': [],
        'PRD_%': [],
        'MAE': [],
        'R_Amp_Err': [],  # 新增
        'HR_Err': []  # 新增
    }

    for i in range(all_targets.shape[1]):

        lead_name = lead_names[i]

        target_lead = all_targets[:, i, :].flatten()

        pred_lead = all_preds[:, i, :].flatten()

        # 计算所有指标
        metrics = calculate_metrics(target_lead, pred_lead)

        detail_metrics[lead_name] = metrics

        # 累积平均值

        for key in avg_metrics.keys():
            avg_metrics[key].append(metrics[key])

    # 计算平均指标

    for key in avg_metrics.keys():

        if avg_metrics[key]:

            avg_metrics[key] = np.mean(avg_metrics[key])

        else:

            avg_metrics[key] = 0.0

    print("\n" + "=" * 80)

    print(f"Evaluation Results for Input Lead: {lead_names[input_lead_idx]} (Epoch: {epoch})")

    print("=" * 80)
    print(f"  Average PCC:       {avg_metrics['PCC']:.6f}")
    print(f"  Average MSE:       {avg_metrics['MSE']:.8f}")
    print(f"  Average RMSE:      {avg_metrics['RMSE']:.8f}")
    print(f"  Average SNR_dB:    {avg_metrics['SNR_dB']:.4f}")
    print(f"  Average PRD_%:     {avg_metrics['PRD_%']:.4f}")
    print(f"  Average R-Amp Err: {avg_metrics['R_Amp_Err']:.4f}")
    print(f"  Average HR Err:    {avg_metrics['HR_Err']:.2f}")

    print("=" * 80)

    print("\nDetailed Metrics for Each Lead:")

    print("-" * 120)

    for lead_name in lead_names:

        if lead_name in detail_metrics:
            metrics = detail_metrics[lead_name]

            indicator = " ← INPUT LEAD" if lead_names.index(lead_name) == input_lead_idx else ""

            print(f"{lead_name:>4s}: PCC={metrics['PCC']:>7.4f}  MSE={metrics['MSE']:>8.6f}  "
                  f"RMSE={metrics['RMSE']:>8.6f}  SNR={metrics['SNR_dB']:>6.2f}dB  "
                  f"R_Err={metrics['R_Amp_Err']:>6.4f}  HR_Err={metrics['HR_Err']:>5.2f} {indicator}")

    print("-" * 120 + "\n")

    # 可视化

    num_samples_to_plot = min(3, all_originals.shape[0])

    for sample_idx in range(num_samples_to_plot):
        visualize_reconstruction(

            original_ecg=all_originals[sample_idx],

            reconstructed_ecg=all_preds[sample_idx],

            target_ecg=all_targets[sample_idx],

            input_lead_idx=input_lead_idx,

            lead_names=lead_names,

            sample_idx=sample_idx,

            epoch=epoch,

            metrics_dict=detail_metrics

        )

    # 绘制指标对比图

    plot_metrics_comparison(detail_metrics, lead_names, input_lead_idx, epoch, version=version)

    return avg_metrics, detail_metrics


def save_metrics_to_file(avg_metrics, detail_metrics, lead_names, input_lead_idx, version="final"):
    """

    将指标保存到文件（JSON和CSV格式）



    参数：

    - avg_metrics: 平均指标字典

    - detail_metrics: 详细指标字典

    - lead_names: 导联名称列表

    - input_lead_idx: 输入导联索引

    - version: 模型版本

    """

    results_dir = "results"

    os.makedirs(results_dir, exist_ok=True)

    # 创建结果汇总字典

    results_summary = {

        'input_lead': lead_names[input_lead_idx],

        'model_version': version,

        'average_metrics': {k: float(v) for k, v in avg_metrics.items()},

        'detailed_metrics': {}

    }

    # 添加详细指标

    for lead_name, metrics in detail_metrics.items():
        results_summary['detailed_metrics'][lead_name] = {k: float(v) for k, v in metrics.items()}

    # 保存为JSON格式

    json_path = os.path.join(results_dir, f"metrics_{lead_names[input_lead_idx]}_v{version}.json")

    with open(json_path, 'w', encoding='utf-8') as f:

        json.dump(results_summary, f, indent=4, ensure_ascii=False)

    print(f"✓ Metrics saved to JSON: {json_path}")

    # 保存为CSV格式（便于分析）

    csv_data = []

    for lead_name in lead_names:

        if lead_name in detail_metrics:

            row = {'Lead': lead_name}

            row.update(detail_metrics[lead_name])

            # 标记输入导联

            if lead_names.index(lead_name) == input_lead_idx:

                row['Type'] = 'INPUT'

            else:

                row['Type'] = 'RECONSTRUCTED'

            csv_data.append(row)

    # 添加平均值行

    avg_row = {'Lead': 'AVERAGE', 'Type': 'ALL'}

    avg_row.update({k: float(v) for k, v in avg_metrics.items()})

    csv_data.append(avg_row)

    df = pd.DataFrame(csv_data)

    csv_path = os.path.join(results_dir, f"metrics_{lead_names[input_lead_idx]}_v{version}.csv")

    df.to_csv(csv_path, index=False, encoding='utf-8')

    print(f"✓ Metrics saved to CSV: {csv_path}")

    # 打印表格

    print("\n" + "=" * 120)

    print("DETAILED METRICS TABLE")

    print("=" * 120)

    print(df.to_string(index=False))

    print("=" * 120 + "\n")


def plot_best_model_final_report(avg_metrics, detail_metrics, lead_names, input_lead_idx):
    """

    绘制最终报告的综合仪表板

    """

    fig = plt.figure(figsize=(20, 12))

    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # 主标题

    fig.suptitle(f'Final Model Evaluation Report (Input Lead: {lead_names[input_lead_idx]})',

                 fontsize=18, fontweight='bold', y=0.98)

    # 1. 平均指标概览 (大文本框)

    ax1 = fig.add_subplot(gs[0, :2])

    ax1.axis('off')

    metrics_text = f"""

    AVERAGE METRICS SUMMARY

    ═══════════════════════════════════════



    Pearson Correlation Coefficient (PCC):    {avg_metrics['PCC']:.6f}

    Mean Squared Error (MSE):                  {avg_metrics['MSE']:.8f}

    Root Mean Squared Error (RMSE):            {avg_metrics['RMSE']:.8f}

    Signal-to-Noise Ratio (SNR_dB):            {avg_metrics['SNR_dB']:.4f} dB

    R-Peak Amplitude Error:                    {avg_metrics['R_Amp_Err']:.4f}

    Heart Rate Error (HR_Err):                 {avg_metrics['HR_Err']:.2f} BPM

    """

    ax1.text(0.05, 0.5, metrics_text, transform=ax1.transAxes, fontsize=11,

             verticalalignment='center', fontfamily='monospace',

             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 2. 综合评分仪表

    ax2 = fig.add_subplot(gs[0, 2])

    # 计算综合评分

    comprehensive_score = (max(0, avg_metrics['PCC']) + (1 - min(avg_metrics['RMSE'], 1))) / 2

    # 绘制仪表盘

    wedges, texts = ax2.pie([comprehensive_score, 1 - comprehensive_score],

                            colors=['lightgreen', 'lightgray'],

                            startangle=90)

    ax2.set_title('Comprehensive Score', fontweight='bold', fontsize=12)

    ax2.text(0, 0, f'{comprehensive_score:.3f}', ha='center', va='center',

             fontsize=16, fontweight='bold')

    # 3-5. PCC, MSE, R-Amp Error 迷你柱状图

    lead_names_filtered = [l for l in lead_names if l in detail_metrics]

    x_pos = np.arange(len(lead_names_filtered))

    # 为输入导联做特殊处理，获取其在过滤列表中的索引

    input_lead_filtered_idx = lead_names_filtered.index(lead_names[input_lead_idx])

    # PCC

    ax3 = fig.add_subplot(gs[1, 0])

    pcc_vals = [detail_metrics[l]['PCC'] for l in lead_names_filtered]

    bars = ax3.bar(x_pos, pcc_vals, color='steelblue', alpha=0.7, edgecolor='black')

    bars[input_lead_filtered_idx].set_color('orange')

    bars[input_lead_filtered_idx].set_alpha(0.9)

    ax3.axhline(y=avg_metrics['PCC'], color='red', linestyle='--', linewidth=2)

    ax3.set_ylabel('PCC', fontweight='bold')

    ax3.set_title('PCC by Lead', fontweight='bold', fontsize=11)

    ax3.set_xticks(x_pos)

    ax3.set_xticklabels(lead_names_filtered, rotation=45, fontsize=9)

    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim([0, 1])

    # MSE

    ax4 = fig.add_subplot(gs[1, 1])

    mse_vals = [detail_metrics[l]['MSE'] for l in lead_names_filtered]

    bars = ax4.bar(x_pos, mse_vals, color='coral', alpha=0.7, edgecolor='black')

    bars[input_lead_filtered_idx].set_color('orange')

    bars[input_lead_filtered_idx].set_alpha(0.9)

    ax4.axhline(y=avg_metrics['MSE'], color='red', linestyle='--', linewidth=2)

    ax4.set_ylabel('MSE', fontweight='bold')

    ax4.set_title('MSE by Lead', fontweight='bold', fontsize=11)

    ax4.set_xticks(x_pos)

    ax4.set_xticklabels(lead_names_filtered, rotation=45, fontsize=9)

    ax4.grid(axis='y', alpha=0.3)

    # R-Amp Error

    ax5 = fig.add_subplot(gs[1, 2])

    r_vals = [detail_metrics[l]['R_Amp_Err'] for l in lead_names_filtered]

    bars = ax5.bar(x_pos, r_vals, color='plum', alpha=0.7, edgecolor='black')

    bars[input_lead_filtered_idx].set_color('orange')

    bars[input_lead_filtered_idx].set_alpha(0.9)

    ax5.axhline(y=avg_metrics['R_Amp_Err'], color='red', linestyle='--', linewidth=2)

    ax5.set_ylabel('Abs Error', fontweight='bold')

    ax5.set_title('R-Wave Amplitude Error', fontweight='bold', fontsize=11)

    ax5.set_xticks(x_pos)

    ax5.set_xticklabels(lead_names_filtered, rotation=45, fontsize=9)

    ax5.grid(axis='y', alpha=0.3)

    # 6-8. SNR_dB, PRD_%, MAE 迷你柱状图

    # SNR_dB

    ax6 = fig.add_subplot(gs[2, 0])

    snr_vals = [detail_metrics[l]['SNR_dB'] for l in lead_names_filtered]

    bars = ax6.bar(x_pos, snr_vals, color='skyblue', alpha=0.7, edgecolor='black')

    bars[input_lead_filtered_idx].set_color('orange')

    bars[input_lead_filtered_idx].set_alpha(0.9)

    ax6.axhline(y=avg_metrics['SNR_dB'], color='red', linestyle='--', linewidth=2)

    ax6.set_ylabel('SNR (dB)', fontweight='bold')

    ax6.set_title('SNR_dB by Lead', fontweight='bold', fontsize=11)

    ax6.set_xticks(x_pos)

    ax6.set_xticklabels(lead_names_filtered, rotation=45, fontsize=9)

    ax6.grid(axis='y', alpha=0.3)

    # HR Error

    ax7 = fig.add_subplot(gs[2, 1])

    hr_vals = [detail_metrics[l]['HR_Err'] for l in lead_names_filtered]

    bars = ax7.bar(x_pos, hr_vals, color='lightgreen', alpha=0.7, edgecolor='black')

    bars[input_lead_filtered_idx].set_color('orange')

    bars[input_lead_filtered_idx].set_alpha(0.9)

    ax7.axhline(y=avg_metrics['HR_Err'], color='red', linestyle='--', linewidth=2)

    ax7.set_ylabel('Error (BPM)', fontweight='bold')

    ax7.set_title('Heart Rate Error (BPM)', fontweight='bold', fontsize=11)

    ax7.set_xticks(x_pos)

    ax7.set_xticklabels(lead_names_filtered, rotation=45, fontsize=9)

    ax7.grid(axis='y', alpha=0.3)

    # MAE

    ax8 = fig.add_subplot(gs[2, 2])

    mae_vals = [detail_metrics[l]['MAE'] for l in lead_names_filtered]
    bars = ax8.bar(x_pos, mae_vals, color='orange', alpha=0.7, edgecolor='black')
    bars[input_lead_filtered_idx].set_color('darkorange')
    bars[input_lead_filtered_idx].set_alpha(0.9)
    ax8.axhline(y=avg_metrics['MAE'], color='red', linestyle='--', linewidth=2)
    ax8.set_ylabel('MAE', fontweight='bold')
    ax8.set_title('MAE by Lead', fontweight='bold', fontsize=11)
    ax8.set_xticks(x_pos)
    ax8.set_xticklabels(lead_names_filtered, rotation=45, fontsize=9)
    ax8.grid(axis='y', alpha=0.3)
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, f"final_report_{lead_names[input_lead_idx]}_best_model.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Final report saved to {save_path}")