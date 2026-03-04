import os
import pandas as pd
import numpy as np
import wfdb
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


class PTBXLDataPreprocessor:
    """PTB-XL数据集预处理器"""

    def __init__(self, data_dir="D:/dataset/ptb_xl", sampling_rate=100):
        """
        Args:
            data_dir: PTB-XL数据集根目录
            sampling_rate: 采样率，100Hz或500Hz
        """
        self.data_dir = data_dir
        self.sampling_rate = sampling_rate

        # 根据采样率选择正确的记录目录
        if sampling_rate == 100:
            self.records_dir = "records100"
            self.file_suffix = "lr"
        elif sampling_rate == 500:
            self.records_dir = "records500"
            self.file_suffix = "hr"
        else:
            raise ValueError("采样率必须是100或500Hz")

        # 12导联名称
        self.lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                           'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

        # 加载元数据
        self.database_df = None
        self.scp_statements = None
        self.load_metadata()

    def load_metadata(self):
        """加载数据库元数据"""
        print("Loading metadata...")

        # 加载主数据库文件
        db_path = os.path.join(self.data_dir, 'ptbxl_database.csv')
        self.database_df = pd.read_csv(db_path, index_col='ecg_id')

        # 加载SCP编码说明
        scp_path = os.path.join(self.data_dir, 'scp_statements.csv')
        self.scp_statements = pd.read_csv(scp_path, index_col=0)

        print(f"Loaded {len(self.database_df)} ECG records")
        print(f"Available leads: {self.lead_names}")

    def load_raw_data(self, record_ids=None, max_records=None):
        """
        加载原始ECG数据

        Args:
            record_ids: 指定记录ID列表，None表示加载所有
            max_records: 最大加载记录数，用于测试

        Returns:
            dict: 包含ECG数据和元信息的字典
        """
        if record_ids is None:
            record_ids = self.database_df.index

        if max_records:
            record_ids = record_ids[:max_records]

        ecg_data = []
        valid_records = []
        failed_records = []

        print(f"Loading {len(record_ids)} ECG records...")

        for record_id in tqdm(record_ids):
            try:
                # 构建文件路径
                record_path = self._get_record_path(record_id)

                # 读取ECG信号
                signal, fields = wfdb.rdsamp(record_path)

                # 验证数据质量
                if self._validate_signal(signal):
                    ecg_data.append(signal.T)  # 转置为 (12, length) 格式
                    valid_records.append(record_id)
                else:
                    failed_records.append(record_id)

            except Exception as e:
                print(f"Failed to load record {record_id}: {e}")
                failed_records.append(record_id)

        print(f"Successfully loaded: {len(valid_records)} records")
        print(f"Failed to load: {len(failed_records)} records")

        return {
            'ecg_data': np.array(ecg_data),
            'record_ids': valid_records,
            'failed_records': failed_records,
            'metadata': self.database_df.loc[valid_records]
        }

    def _get_record_path(self, record_id):
        """构建记录文件路径"""
        folder = f"{record_id // 1000:02d}000"
        filename = f"{record_id:05d}_{self.file_suffix}"
        return os.path.join(self.data_dir, self.records_dir, folder, filename)

    def _validate_signal(self, signal, min_length=1000):
        """验证信号质量"""
        if signal is None or signal.shape[0] < min_length:
            return False

        # 检查是否有12导联
        if signal.shape[1] != 12:
            return False

        # 检查是否有异常值
        if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
            return False

        # 检查信号幅度是否合理（ECG通常在-5到5mV范围）
        if np.abs(signal).max() > 20:  # 允许一些余量
            return False

        return True

    def preprocess_signals(self, ecg_data, target_length=5000, normalize=True):
        """
        预处理ECG信号

        Args:
            ecg_data: 原始ECG数据 (N, 12, length)
            target_length: 目标长度
            normalize: 是否标准化

        Returns:
            processed_data: 预处理后的数据
        """
        print("Preprocessing ECG signals...")
        processed_data = []

        for i, signal in enumerate(tqdm(ecg_data)):
            try:
                # 重采样到目标长度
                processed_signal = self._resample_signal(signal, target_length)

                # 去除基线漂移
                processed_signal = self._remove_baseline_drift(processed_signal)

                # 滤波去噪
                processed_signal = self._apply_filters(processed_signal)

                # 标准化
                if normalize:
                    processed_signal = self._normalize_signal(processed_signal)

                processed_data.append(processed_signal)

            except Exception as e:
                print(f"Failed to preprocess signal {i}: {e}")

        return np.array(processed_data)

    def _resample_signal(self, signal, target_length):
        """重采样信号到目标长度"""
        current_length = signal.shape[1]
        if current_length == target_length:
            return signal

        # 使用线性插值重采样
        indices = np.linspace(0, current_length - 1, target_length)
        resampled = np.zeros((12, target_length))

        for lead_idx in range(12):
            resampled[lead_idx] = np.interp(indices, np.arange(current_length),
                                            signal[lead_idx])
        return resampled

    def _remove_baseline_drift(self, signal, cutoff_freq=0.5):
        """去除基线漂移（高通滤波）"""
        from scipy import signal as scipy_signal

        # 设计高通滤波器
        nyquist = self.sampling_rate / 2
        normal_cutoff = cutoff_freq / nyquist
        b, a = scipy_signal.butter(3, normal_cutoff, btype='high')

        # 应用滤波器
        filtered_signal = np.zeros_like(signal)
        for lead_idx in range(12):
            filtered_signal[lead_idx] = scipy_signal.filtfilt(b, a, signal[lead_idx])

        return filtered_signal

    def _apply_filters(self, signal, low_freq=0.5, high_freq=40):
        """应用带通滤波器"""
        from scipy import signal as scipy_signal

        # 设计带通滤波器
        nyquist = self.sampling_rate / 2
        low_normal = low_freq / nyquist
        high_normal = high_freq / nyquist
        b, a = scipy_signal.butter(4, [low_normal, high_normal], btype='band')

        # 应用滤波器
        filtered_signal = np.zeros_like(signal)
        for lead_idx in range(12):
            filtered_signal[lead_idx] = scipy_signal.filtfilt(b, a, signal[lead_idx])

        return filtered_signal

    def _normalize_signal(self, signal):
        """标准化信号"""
        normalized_signal = np.zeros_like(signal)
        for lead_idx in range(12):
            lead_data = signal[lead_idx]
            mean_val = np.mean(lead_data)
            std_val = np.std(lead_data)
            if std_val > 0:
                normalized_signal[lead_idx] = (lead_data - mean_val) / std_val
            else:
                normalized_signal[lead_idx] = lead_data
        return normalized_signal

    def create_lead_reconstruction_dataset(self, ecg_data, record_ids,
                                           validation_split=0.2, test_split=0.1):
        """
        创建导联重建数据集

        Returns:
            dict: 包含训练、验证和测试集的字典
        """
        print("Creating lead reconstruction dataset...")

        # 分割数据集
        train_ids, temp_ids = train_test_split(
            record_ids, test_size=validation_split + test_split,
            random_state=42, stratify=None
        )
        val_ids, test_ids = train_test_split(
            temp_ids, test_size=test_split / (validation_split + test_split),
            random_state=42
        )

        # 获取对应的ECG数据索引
        train_indices = [record_ids.index(rid) for rid in train_ids if rid in record_ids]
        val_indices = [record_ids.index(rid) for rid in val_ids if rid in record_ids]
        test_indices = [record_ids.index(rid) for rid in test_ids if rid in record_ids]

        dataset = {
            'train': {
                'ecg_data': ecg_data[train_indices],
                'record_ids': [record_ids[i] for i in train_indices],
                'metadata': self.database_df.loc[[record_ids[i] for i in train_indices]]
            },
            'validation': {
                'ecg_data': ecg_data[val_indices],
                'record_ids': [record_ids[i] for i in val_indices],
                'metadata': self.database_df.loc[[record_ids[i] for i in val_indices]]
            },
            'test': {
                'ecg_data': ecg_data[test_indices],
                'record_ids': [record_ids[i] for i in test_indices],
                'metadata': self.database_df.loc[[record_ids[i] for i in test_indices]]
            },
            'lead_names': self.lead_names,
            'preprocessing_params': {
                'sampling_rate': self.sampling_rate,
                'target_length': ecg_data.shape[-1],
                'normalized': True
            }
        }

        print(f"Train samples: {len(dataset['train']['ecg_data'])}")
        print(f"Validation samples: {len(dataset['validation']['ecg_data'])}")
        print(f"Test samples: {len(dataset['test']['ecg_data'])}")

        return dataset

    def save_processed_dataset(self, dataset, output_path):
        """保存预处理后的数据集"""
        print(f"Saving dataset to {output_path}")

        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 保存数据集
        with open(output_path, 'wb') as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

        print("Dataset saved successfully!")

    def load_processed_dataset(self, dataset_path):
        """加载预处理后的数据集"""
        print(f"Loading dataset from {dataset_path}")

        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)

        print("Dataset loaded successfully!")
        return dataset

    def visualize_samples(self, ecg_data, record_ids, num_samples=3, save_path=None):
        """可视化ECG样本"""
        fig, axes = plt.subplots(num_samples, 1, figsize=(15, 4 * num_samples))
        if num_samples == 1:
            axes = [axes]

        for i in range(min(num_samples, len(ecg_data))):
            ax = axes[i]
            signal = ecg_data[i]  # (12, length)

            # 绘制所有12导联
            for lead_idx, lead_name in enumerate(self.lead_names):
                ax.plot(signal[lead_idx] + lead_idx * 2,
                        label=lead_name, linewidth=0.8)

            ax.set_title(f'ECG Record ID: {record_ids[i]}')
            ax.set_xlabel('Sample Points')
            ax.set_ylabel('Amplitude (normalized)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")

        plt.show()

    def get_statistics(self, dataset):
        """获取数据集统计信息"""
        stats = {}

        for split in ['train', 'validation', 'test']:
            if split in dataset:
                ecg_data = dataset[split]['ecg_data']
                metadata = dataset[split]['metadata']

                stats[split] = {
                    'num_samples': len(ecg_data),
                    'signal_shape': ecg_data.shape,
                    'signal_mean': np.mean(ecg_data),
                    'signal_std': np.std(ecg_data),
                    'signal_min': np.min(ecg_data),
                    'signal_max': np.max(ecg_data),
                }

                # 添加元数据统计
                if 'age' in metadata.columns:
                    stats[split]['age_mean'] = metadata['age'].mean()
                    stats[split]['age_std'] = metadata['age'].std()

                if 'sex' in metadata.columns:
                    stats[split]['gender_distribution'] = metadata['sex'].value_counts().to_dict()

        return stats


def main():
    """主函数：完整的数据预处理流程"""

    # 初始化预处理器 - 使用100Hz数据（与mam12_t3.py一致）
    preprocessor = PTBXLDataPreprocessor(
        data_dir="D:/dataset/ptb_xl",
        sampling_rate=100  # 使用100Hz采样率，与mam12_t3.py保持一致
    )

    # 设置输出路径
    output_dir = "D:/project/mamba_12/data"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("PTB-XL数据集预处理开始")
    print("使用完整21837条记录")
    print("=" * 60)

    # 步骤1：加载原始数据（使用完整数据集）
    print("\n步骤1：加载原始ECG数据")
    raw_data = preprocessor.load_raw_data(max_records=None)  # 加载所有记录

    # 步骤2：预处理信号
    print("\n步骤2：预处理ECG信号")
    processed_ecg = preprocessor.preprocess_signals(
        raw_data['ecg_data'],
        target_length=5000,  # 50秒 * 100Hz = 5000个采样点
        normalize=True
    )

    # 步骤3：创建数据集
    print("\n步骤3：创建训练/验证/测试数据集")
    dataset = preprocessor.create_lead_reconstruction_dataset(
        processed_ecg,
        raw_data['record_ids'],
        validation_split=0.15,
        test_split=0.15
    )

    # 步骤4：保存数据集
    print("\n步骤4：保存预处理后的数据集")
    dataset_path = os.path.join(output_dir, "ptb_xl_processed_dataset.pkl")
    preprocessor.save_processed_dataset(dataset, dataset_path)

    # 步骤5：生成统计信息
    print("\n步骤5：生成数据集统计信息")
    stats = preprocessor.get_statistics(dataset)

    # 打印统计信息
    for split, split_stats in stats.items():
        print(f"\n{split.upper()} SET:")
        for key, value in split_stats.items():
            print(f"  {key}: {value}")

    # 步骤6：可视化样本
    print("\n步骤6：可视化ECG样本")
    viz_path = os.path.join(output_dir, "ecg_samples_visualization.png")
    preprocessor.visualize_samples(
        dataset['train']['ecg_data'][:3],
        dataset['train']['record_ids'][:3],
        save_path=viz_path
    )

    # 保存统计信息
    stats_path = os.path.join(output_dir, "dataset_statistics.pkl")
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    print("=" * 60)
    print("数据预处理完成！")
    print(f"成功处理了 {len(raw_data['ecg_data'])} 条ECG记录")
    print(f"处理后的数据集保存在: {dataset_path}")
    print(f"统计信息保存在: {stats_path}")
    print(f"可视化图片保存在: {viz_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()