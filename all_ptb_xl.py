import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
import warnings
import random
import math
import pickle
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import matplotlib

from keshihua_v2 import (
    plot_loss_curves,
    test_single_lead_model_with_visualization,
    visualize_reconstruction,
    plot_metrics_comparison,
    save_metrics_to_file,
    plot_best_model_final_report,
    calculate_metrics
)

# ---------------------------------------------------------------------------- #
#                           Part 1: 工具函数                                   #
# ---------------------------------------------------------------------------- #

warnings.filterwarnings('ignore')
matplotlib.use('Agg')


def get_cosine_schedule_with_warmup(optimizer, num_warmup_epochs, num_training_epochs, last_epoch=-1):
    def lr_lambda(current_epoch):
        if current_epoch < num_warmup_epochs:
            return float(current_epoch) / float(max(1, num_warmup_epochs))

        progress = float(current_epoch - num_warmup_epochs) / float(max(1, num_training_epochs - num_warmup_epochs))

        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


from mamba_ssm import Mamba



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(42)


class OfficialMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)  # LayerNorm on feature dimension
        x = x.transpose(1, 2)  # (B, L, C) for Mamba
        x = self.mamba(x)
        x = x.transpose(1, 2)  # (B, C, L)
        x = self.dropout(x)
        return x + residual


class SingleLeadECGDataset(Dataset):
    def __init__(self, ecg_data, input_lead_idx=1):
        self.ecg_data = ecg_data
        self.input_lead_idx = input_lead_idx

    def __len__(self):
        return len(self.ecg_data)

    def __getitem__(self, idx):
        full_ecg = torch.FloatTensor(self.ecg_data[idx])

        # Instance normalization
        mean = full_ecg.mean(dim=-1, keepdim=True)
        std = full_ecg.std(dim=-1, keepdim=True)
        std[std == 0] = 1.0
        normalized_ecg = (full_ecg - mean) / std

        # Create input: only the specified lead has signal
        input_ecg = torch.zeros_like(normalized_ecg)
        input_ecg[self.input_lead_idx] = normalized_ecg[self.input_lead_idx]

        target_ecg = normalized_ecg

        # Return original unnormalized ECG for visualization
        return input_ecg, target_ecg, full_ecg


class PhysiologicalConstraintLoss(nn.Module):

    def __init__(self, lead_idx_mapping=None, eps=1e-6):
        super().__init__()
        self.eps = eps

        self.lead_idx_mapping = lead_idx_mapping or {
            'I': 0, 'II': 1, 'III': 2,
            'aVR': 3, 'aVL': 4, 'aVF': 5,
            'V1': 6, 'V2': 7, 'V3': 8, 'V4': 9, 'V5': 10, 'V6': 11
        }

    def einthoven_law_loss(self, y_pred):
        """
        艾因托芬定律约束：
        - II = I + III
        - aVR + aVL + aVF = 0
        """
        I = y_pred[:, self.lead_idx_mapping['I'], :]
        II = y_pred[:, self.lead_idx_mapping['II'], :]
        III = y_pred[:, self.lead_idx_mapping['III'], :]

        aVR = y_pred[:, self.lead_idx_mapping['aVR'], :]
        aVL = y_pred[:, self.lead_idx_mapping['aVL'], :]
        aVF = y_pred[:, self.lead_idx_mapping['aVF'], :]

        einthoven_constraint1 = F.l1_loss(II, I + III)
        einthoven_constraint2 = F.l1_loss(aVR + aVL + aVF, torch.zeros_like(aVR))

        return einthoven_constraint1 + einthoven_constraint2

    def progressive_wave_loss(self, y_pred):
        """
        R波/S波进行性变化约束：
        从V1到V6：
        - R波振幅逐渐增大
        - S波振幅逐渐减小
        """
        chest_leads = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        chest_indices = [self.lead_idx_mapping[lead] for lead in chest_leads]

        chest_signals = y_pred[:, chest_indices, :]

        # 计算R波(最大值)和S波(最小值)
        R_waves = torch.max(chest_signals, dim=2)[0]
        S_waves = torch.min(chest_signals, dim=2)[0]

        total_loss = 0.0

        for i in range(len(chest_indices) - 1):

            r_increase_violation = F.relu(R_waves[:, i] - R_waves[:, i + 1])
            total_loss += torch.mean(r_increase_violation)

        for i in range(len(chest_indices) - 1):

            s_decrease_violation = F.relu(S_waves[:, i] - S_waves[:, i + 1])
            total_loss += torch.mean(s_decrease_violation)

        return total_loss / (2 * (len(chest_indices) - 1))

    def wilson_center_terminal_loss(self, y_pred):
        """
        Wilson中心端约束：
        所有胸导联的负极都是相同的
        """
        chest_leads = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        chest_indices = [self.lead_idx_mapping[lead] for lead in chest_leads]

        chest_signals = y_pred[:, chest_indices, :]
        chest_mean = torch.mean(chest_signals, dim=1, keepdim=True)

        consistency_loss = 0.0
        for i in range(len(chest_indices)):
            residual = chest_signals[:, i, :] - chest_mean.squeeze(1)
            consistency_loss += torch.mean(torch.std(residual, dim=1))

        return consistency_loss / len(chest_indices)

    def forward(self, y_pred, weight_einthoven=1.0, weight_progressive=0.5, weight_wilson=0.3):
        """
        综合所有生理学约束
        """
        loss_einthoven = self.einthoven_law_loss(y_pred)
        loss_progressive = self.progressive_wave_loss(y_pred)
        loss_wilson = self.wilson_center_terminal_loss(y_pred)

        total_loss = (weight_einthoven * loss_einthoven +
                      weight_progressive * loss_progressive +
                      weight_wilson * loss_wilson)

        return total_loss, loss_einthoven, loss_progressive, loss_wilson


class HybridLossV6(nn.Module):
    def __init__(self, lead_weights, alpha=1.0, beta=0.5, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.lead_weights = lead_weights
        self.l1_loss_fn = nn.L1Loss(reduction='none')
        self.lead_I_idx, self.lead_II_idx, self.lead_III_idx = 0, 1, 2

        print(f"HybridLossV6 initialized with lead weights: {self.lead_weights.cpu().numpy()}")

    def pearson_correlation_loss(self, y_pred, y_true):
        y_pred_mean = torch.mean(y_pred, dim=2, keepdim=True)
        y_true_mean = torch.mean(y_true, dim=2, keepdim=True)
        vx = y_pred - y_pred_mean
        vy = y_true - y_true_mean
        corr = torch.sum(vx * vy, dim=2) / (
                torch.sqrt(torch.sum(vx ** 2, dim=2)) * torch.sqrt(torch.sum(vy ** 2, dim=2)) + self.eps)
        loss_per_lead = 1.0 - corr
        weighted_loss = loss_per_lead * self.lead_weights
        return torch.mean(weighted_loss)

    def weighted_l1_loss(self, y_pred, y_true):
        loss_per_sample = self.l1_loss_fn(y_pred, y_true)
        loss_per_lead = torch.mean(loss_per_sample, dim=2)
        weighted_loss = loss_per_lead * self.lead_weights
        return torch.mean(weighted_loss)

    def einthoven_loss(self, y_pred):
        recon_I, recon_II, recon_III = y_pred[:, 0, :], y_pred[:, 1, :], y_pred[:, 2, :]
        einthoven_error = F.l1_loss(recon_I + recon_III, recon_II)
        return einthoven_error

    def forward(self, y_pred, y_true, gamma):
        loss_l1 = self.weighted_l1_loss(y_pred, y_true)
        loss_corr = self.pearson_correlation_loss(y_pred, y_true)
        loss_einthoven = self.einthoven_loss(y_pred)
        total_loss = self.alpha * loss_l1 + self.beta * loss_corr + gamma * loss_einthoven
        unweighted_l1 = F.l1_loss(y_pred, y_true)
        return total_loss, unweighted_l1, loss_corr, loss_einthoven


class CombinedHybridLossV8(nn.Module):
    """
    结合原有HybridLossV6和PhysiologicalConstraintLoss的综合损失函数
    """

    def __init__(self, lead_weights, alpha=1.0, beta=0.5, gamma_einthoven=1.0,
                 lambda_physiological=0.3, eps=1e-6):
        super().__init__()
        self.hybrid_loss = HybridLossV6(lead_weights, alpha, beta, eps)
        self.phys_loss = PhysiologicalConstraintLoss(eps=eps)
        self.lambda_physiological = lambda_physiological

        print(f"CombinedHybridLossV8 initialized with:")
        print(f"  - Hybrid Loss weights: alpha={alpha}, beta={beta}")
        print(f"  - Einthoven weight (gamma): {gamma_einthoven}")
        print(f"  - Physiological constraint weight (lambda): {lambda_physiological}")

    def forward(self, y_pred, y_true, gamma_einthoven,
                weight_einthoven=1.0, weight_progressive=0.5, weight_wilson=0.3):
        """
        综合计算所有损失
        """
        # 原有的混合损失
        loss_hybrid, loss_l1, loss_corr, loss_ein_old = self.hybrid_loss(y_pred, y_true, gamma_einthoven)

        # 新的生理学约束损失
        loss_phys, loss_ein_new, loss_prog, loss_wilson = self.phys_loss(
            y_pred,
            weight_einthoven=weight_einthoven,
            weight_progressive=weight_progressive,
            weight_wilson=weight_wilson
        )

        # 综合总损失
        total_loss = loss_hybrid + self.lambda_physiological * loss_phys

        return {
            'total': total_loss,
            'hybrid': loss_hybrid,
            'l1': loss_l1,
            'corr': loss_corr,
            'phys': loss_phys,
            'einthoven': loss_ein_new,
            'progressive': loss_prog,
            'wilson': loss_wilson
        }


class MultiScaleConvModule(nn.Module):
    """多尺度卷积模块"""

    def __init__(self, in_channels, kernel_sizes=[3, 5, 7, 9]):
        super().__init__()

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, in_channels, kernel_size=k, padding=k // 2, groups=in_channels),
                nn.Conv1d(in_channels, in_channels, kernel_size=1),
                nn.BatchNorm1d(in_channels),
                nn.GELU()
            ) for k in kernel_sizes
        ])

        self.fusion_conv = nn.Conv1d(in_channels * len(kernel_sizes), in_channels, kernel_size=1)
        self.norm = nn.BatchNorm1d(in_channels)
        self.act = nn.GELU()

    def forward(self, x):
        features = [conv(x) for conv in self.convs]
        concatenated = torch.cat(features, dim=1)
        fused = self.fusion_conv(concatenated)
        fused = self.act(self.norm(fused))
        return x + fused


class LeadInteractionModule(nn.Module):
    """导联交互模块"""

    def __init__(self, hidden_dim, num_leads, dropout=0.25):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * num_leads)
        )

    def forward(self, x):
        return self.mlp(x)


class SingleLeadECGMambaReconstructionNet_V7(nn.Module):
    def __init__(self, num_leads=12, hidden_dim=96, num_mamba_layers_per_stage=2,
                 d_state=32, d_conv=4, expand_factor=2, dropout=0.25, input_lead_idx=1):
        super().__init__()

        if not MAMBA_AVAILABLE:
            raise ImportError("Mamba is not available. Please install mamba_ssm first.")

        self.input_lead_idx = input_lead_idx

        # --- Encoder Path ---
        self.initial_conv = nn.Conv1d(num_leads, hidden_dim, kernel_size=21, padding=10)
        self.encoder1_mamba = self._make_mamba_stack(hidden_dim, num_mamba_layers_per_stage, d_state, d_conv,
                                                     expand_factor, dropout)
        self.downsample1 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1)
        self.encoder2_mamba = self._make_mamba_stack(hidden_dim * 2, num_mamba_layers_per_stage, d_state, d_conv,
                                                     expand_factor, dropout)
        self.downsample2 = nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1)

        # --- Bottleneck ---
        self.bottleneck_mamba = self._make_mamba_stack(hidden_dim * 4, num_mamba_layers_per_stage, d_state, d_conv,
                                                       expand_factor, dropout)
        self.bottleneck_multiscale = MultiScaleConvModule(hidden_dim * 4)

        # --- Decoder Path ---
        self.upsample1 = nn.ConvTranspose1d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1)
        self.decoder1_conv = nn.Conv1d(hidden_dim * 4, hidden_dim * 2, kernel_size=3, padding=1)
        self.decoder1_norm = nn.BatchNorm1d(hidden_dim * 2)
        self.decoder1_mamba = self._make_mamba_stack(hidden_dim * 2, num_mamba_layers_per_stage, d_state, d_conv,
                                                     expand_factor, dropout)
        self.upsample2 = nn.ConvTranspose1d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.decoder2_conv = nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1)
        self.decoder2_norm = nn.BatchNorm1d(hidden_dim)
        self.decoder2_mamba = self._make_mamba_stack(hidden_dim, num_mamba_layers_per_stage, d_state, d_conv,
                                                     expand_factor, dropout)

        self.lead_interaction = LeadInteractionModule(hidden_dim, num_leads, dropout)
        self.output_decoder = nn.Sequential(
            nn.Conv1d(hidden_dim * num_leads, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.GELU(),
            nn.Conv1d(hidden_dim * 2, num_leads, kernel_size=1)
        )

    def _make_mamba_stack(self, d_model, num_layers, d_state, d_conv, expand, dropout):
        layers = []
        for _ in range(num_layers):
            layers.append(OfficialMambaBlock(d_model, d_state, d_conv, expand, dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        e1 = self.initial_conv(x)
        e1_mamba = self.encoder1_mamba(e1)
        e2 = self.downsample1(e1_mamba)
        e2_mamba = self.encoder2_mamba(e2)
        b = self.downsample2(e2_mamba)

        # Bottleneck
        b_mamba = self.bottleneck_mamba(b)
        b_enhanced = self.bottleneck_multiscale(b_mamba)

        # Decoder
        d1 = self.upsample1(b_enhanced)
        d1 = torch.cat([d1, e2_mamba], dim=1)
        d1 = F.gelu(self.decoder1_norm(self.decoder1_conv(d1)))
        d1_mamba = self.decoder1_mamba(d1)

        d2 = self.upsample2(d1_mamba)
        d2 = torch.cat([d2, e1_mamba], dim=1)
        d2 = F.gelu(self.decoder2_norm(self.decoder2_conv(d2)))
        d2_mamba = self.decoder2_mamba(d2)

        final_features = d2_mamba.transpose(1, 2)
        lead_features = self.lead_interaction(final_features)
        lead_features_reshaped = lead_features.transpose(1, 2)
        output = self.output_decoder(lead_features_reshaped)

        return output


def train_single_lead_model_v8(model, train_loader, val_loader, test_loader, input_lead_idx,
                               num_epochs=100, warmup_epochs=5, device='cpu',
                               patience=20, gamma_max=1.0, anneal_epochs=50, version="v8",
                               lambda_physiological=0.3,
                               weight_einthoven=1.0,
                               weight_progressive=0.5,
                               weight_wilson=0.3,
                               models_dir="models"):
    """
    改进的训练函数，支持生理学约束
    """

    # 初始化所有导联的权重
    lead_weights = torch.tensor([
        1.5, 1.0, 1.5, 1.5, 1.5, 1.0, 2.5, 2.5, 2.0, 2.0, 1.5, 1.5
    ], device=device)

    # 【关键修改】：将输入导联的权重设为 1.0 或保持原样，而不是 0.0
    # 这样模型在训练时会被惩罚如果它不能正确重建输入信号
    # V1 PCC 负值的原因就是这里原来设为了 0.0，导致模型为了满足生理学约束而任意反转输入导联
    lead_weights[input_lead_idx] = 1.0

    criterion = CombinedHybridLossV8(
        lead_weights=lead_weights,
        alpha=1.0,
        beta=0.5,
        gamma_einthoven=1.0,
        lambda_physiological=lambda_physiological
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_epochs, num_epochs)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    # 创建模型保存目录
    os.makedirs(models_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir,
                                   f'best_single_lead_{lead_names[input_lead_idx]}_mamba_model_{version}.pth')

    print(
        f"Starting single-lead ({lead_names[input_lead_idx]}) to 12-lead Hybrid Mamba-UNet {version.upper()} model training...")
    print(f"  - Lambda (physiological constraint weight): {lambda_physiological}")
    print(f"  - Phys weights: ein={weight_einthoven}, prog={weight_progressive}, wilson={weight_wilson}")

    print(f"  - Input lead weight enforced: {lead_weights[input_lead_idx]}")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Training')
        current_gamma = gamma_max * min(1.0, epoch / anneal_epochs)
        current_lambda = lambda_physiological * min(1.0, epoch / max(1, anneal_epochs // 2))

        for input_ecg, target_ecg, _ in train_bar:
            input_ecg, target_ecg = input_ecg.to(device), target_ecg.to(device)
            optimizer.zero_grad()

            output_ecg = model(input_ecg)

            loss_dict = criterion(
                output_ecg, target_ecg, current_gamma,
                weight_einthoven = weight_einthoven,
                weight_progressive = weight_progressive,
                weight_wilson = weight_wilson
            )

            loss = loss_dict['total']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

            train_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'L1': f'{loss_dict["l1"].item():.4f}',
                'Corr': f'{loss_dict["corr"].item():.4f}',
                'Phys': f'{loss_dict["phys"].item():.4f}',
                'Ein': f'{loss_dict["einthoven"].item():.4f}',
                'Prog': f'{loss_dict["progressive"].item():.4f}'
            })

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for input_ecg, target_ecg, _ in val_loader:
                input_ecg, target_ecg = input_ecg.to(device), target_ecg.to(device)
                output_ecg = model(input_ecg)
                loss_dict = criterion(
                    output_ecg, target_ecg, current_gamma,
                    weight_einthoven=weight_einthoven,
                    weight_progressive=weight_progressive,
                    weight_wilson=weight_wilson
                )

                val_loss += loss_dict['total'].item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f'Epoch {epoch + 1}/{num_epochs}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.6e}, Lambda: {current_lambda:.6f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"✓ Validation loss improved. Saved new best model to {model_save_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 2 == 0:
            print(f"\nRunning periodic test and visualization at epoch {epoch + 1}...")
            test_single_lead_model_with_visualization(model, test_loader, input_lead_idx, device, lead_names,
                                                      epoch=f"{version}_{epoch + 1}", version=version)
            plot_loss_curves(train_losses, val_losses, lead_names[input_lead_idx], f"{version}_epoch_{epoch + 1}")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement.")
            break

        print('-' * 50)

    print("\nTraining finished. Loading best model for final evaluation...")
    model.load_state_dict(torch.load(model_save_path))

    print("\n" + "=" * 100)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 100)
    final_metrics, detail_metrics = test_single_lead_model_with_visualization(
        model, test_loader, input_lead_idx, device, lead_names,
        epoch=f'{version}_final_best', version=version
    )

    plot_loss_curves(train_losses, val_losses, lead_names[input_lead_idx], f"{version}_final")

    # 保存最终指标到文件
    save_metrics_to_file(final_metrics, detail_metrics, lead_names, input_lead_idx, version=version)

    # 绘制最终报告
    plot_best_model_final_report(final_metrics, detail_metrics, lead_names, input_lead_idx)

    return train_losses, val_losses, final_metrics


def load_ptb_xl_data(data_path):
    print(f"Loading PTB-XL data from {data_path}")

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at '{data_path}'")
        return None

    try:
        with open(data_path, 'rb') as f:
            dataset = pickle.load(f)

        print("Data loaded successfully!")
        return dataset

    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def prepare_ecg_data(dataset, max_samples_per_split=None):
    train_data = dataset['train']['ecg_data']
    val_data = dataset['validation']['ecg_data']
    test_data = dataset['test']['ecg_data']

    if max_samples_per_split:
        train_data = train_data[:max_samples_per_split]
        val_data = val_data[:min(max_samples_per_split // 4, len(val_data))]
        test_data = test_data[:min(max_samples_per_split // 4, len(test_data))]

    print(f"Data shapes: Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")

    return train_data, val_data, test_data


def main():
    if not MAMBA_AVAILABLE:
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_file = r'ptb_xl_processed_dataset.pkl'

    dataset = load_ptb_xl_data(data_file)
    if dataset is None:
        return

    # 使用较少数据进行快速测试
    train_ecg, val_ecg, test_ecg = prepare_ecg_data(dataset, max_samples_per_split=None)

    # 定义所有导联
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    num_leads = len(lead_names)

    # 存储所有模型的结果
    all_results = {}

    # 遍历每个导联进行训练
    for input_lead_idx in range(num_leads):
        print("\n" + "=" * 100)
        print(
            f"TRAINING MODEL {input_lead_idx + 1}/{num_leads}: Using lead {lead_names[input_lead_idx]} (index {input_lead_idx})")
        print("=" * 100 + "\n")

        try:
            # 创建该导联的数据集
            train_dataset = SingleLeadECGDataset(train_ecg, input_lead_idx)
            val_dataset = SingleLeadECGDataset(val_ecg, input_lead_idx)
            test_dataset = SingleLeadECGDataset(test_ecg, input_lead_idx)

            batch_size = 24
            num_workers = 0 if os.name == 'nt' else 2

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                      pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                    pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                     pin_memory=True)

            print(
                f"Data loading complete: Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

            # 为该导联创建新的模型
            model = SingleLeadECGMambaReconstructionNet_V7(
                num_leads=12,
                hidden_dim=96,
                num_mamba_layers_per_stage=2,
                d_state=32,
                d_conv=4,
                expand_factor=2,
                dropout=0.25,
                input_lead_idx=input_lead_idx
            ).to(device)

            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(
                f"Hybrid Mamba-UNet V7 model created with {total_params:,} parameters for lead {lead_names[input_lead_idx]}")

            # 训练该导联的模型
            train_losses, val_losses, final_metrics = train_single_lead_model_v8(
                model, train_loader, val_loader, test_loader, input_lead_idx,
                num_epochs=100,
                warmup_epochs=5,
                device=device,
                patience=15,
                version="v8",
                lambda_physiological=0.3
            )

            # 保存结果
            all_results[lead_names[input_lead_idx]] = {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'final_metrics': final_metrics
            }

            print(f"\n✓ Successfully trained model for lead {lead_names[input_lead_idx]}")

        except Exception as e:
            print(f"\n✗ Error occurred while training lead {lead_names[input_lead_idx]}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 训练完所有导联后，生成综合对比报告
    print("\n" + "=" * 100)
    print("GENERATING COMPREHENSIVE COMPARISON REPORT FOR ALL LEADS")
    print("=" * 100 + "\n")

    try:
        plot_metrics_comparison(all_results, lead_names)
        print("✓ Comprehensive comparison report generated successfully!")
    except Exception as e:
        print(f"✗ Error generating comparison report: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 100)
    print("ALL TRAINING COMPLETED!")
    print("=" * 100)


if __name__ == "__main__":
    main()