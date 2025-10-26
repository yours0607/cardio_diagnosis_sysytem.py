import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import time
import os
import warnings
import zipfile
import pickle
import json
import h5py
from scipy import io
from tqdm import tqdm
import glob
import wfdb
import ast
from cardio_diagnosis_system import (
    HybridCardiovascularSystem, 
    get_device, 
    train_model,
    create_dataloader
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve

warnings.filterwarnings('ignore')

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class PTBXLDataProcessor:
    """PTB-XL心电图数据集处理器，修复属性与数据逻辑"""
    def __init__(self, data_path: str, seq_len: int = 1000, fs: int = 100):
        self.data_path = data_path  # 数据集根目录
        self.seq_len = seq_len      # 目标序列长度
        self.fs = fs                # 目标采样率（100Hz匹配records100）
        self.metadata = None        # 元数据（ptbxl_database.csv）
        self.valid_ids = []         # 有效ECG样本ID列表
        self.scaler = StandardScaler()  # 标准化器（内置，避免外部调用混乱）

    def load_metadata(self):
        """加载并验证元数据，确保关键字段存在"""
        metadata_path = os.path.join(self.data_path, 'ptbxl_database.csv')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"元数据文件缺失: {metadata_path}")
        
        # 读取元数据并解析标签
        self.metadata = pd.read_csv(metadata_path, index_col='ecg_id')
        # 检查关键字段（避免解析失败）
        required_cols = ['filename_hr', 'scp_codes']
        if not all(col in self.metadata.columns for col in required_cols):
            raise ValueError(f"元数据缺少关键字段，需包含{required_cols}")
        
        # 解析scp_codes（标签：0=正常，1=异常）
        self.metadata['scp_codes'] = self.metadata['scp_codes'].apply(ast.literal_eval)
        self.metadata['label'] = self.metadata['scp_codes'].apply(
            lambda x: 0 if 'NORM' in x else 1
        )
        
        # 筛选有效样本ID（排除标签为空的样本）
        self.valid_ids = self.metadata[self.metadata['label'].notna()].index.tolist()
        if len(self.valid_ids) == 0:
            raise ValueError("无有效样本ID，元数据标签解析失败")
        
        print(f"元数据加载完成：有效样本数={len(self.valid_ids)}")
        return self

    def load_ecg_data(self, ecg_id):
        """读取单个ECG信号，修复路径拼接与维度适配"""
        if self.metadata is None:
            raise RuntimeError("请先调用load_metadata()加载元数据")
        
        # 1. 拼接ECG文件路径（基于元数据的filename_hr字段）
        hr_filename = self.metadata.loc[ecg_id, 'filename_hr']  # 如"00000/00000_hr"
        record_path = os.path.join(self.data_path, hr_filename)  # 完整路径
        
        # 2. 检查文件完整性（WFDB需.dat和.hea文件）
        required_files = [f"{record_path}.dat", f"{record_path}.hea"]
        for file in required_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"ECG文件缺失: {file}")
        
        # 3. 读取ECG信号（10秒数据，采样率500Hz→降采样到100Hz）
        record = wfdb.rdrecord(record_path, sampto=int(500 * 10))  # 原始采样率500Hz
        signal = record.p_signal.T  # 转置为 (n_channels, n_samples)
        
        # 4. 降采样（500Hz→100Hz，步长=5）
        signal = signal[:, ::5]  # 每5个点取1个，长度从5000→1000
        
        # 5. 统一序列长度（补零或截断）
        if signal.shape[1] > self.seq_len:
            signal = signal[:, :self.seq_len]
        else:
            pad_length = self.seq_len - signal.shape[1]
            signal = np.pad(signal, ((0, 0), (0, pad_length)), mode='constant')
        
        return signal.astype(np.float32)

    def create_dataset(self, max_samples: int = 21799, normalize: bool = True):
        """创建完整数据集（含标准化），返回(signals, labels)，不保存为self.data"""
        if len(self.valid_ids) == 0:
            raise RuntimeError("无有效样本ID，无法创建数据集")
        
        # 限制样本数量（避免内存不足）
        sample_ids = self.valid_ids[:max_samples]
        signals, labels = [], []
        
        # 批量读取ECG信号
        print(f"加载{len(sample_ids)}个ECG样本...")
        for ecg_id in tqdm(sample_ids, desc="读取ECG数据"):
            try:
                signal = self.load_ecg_data(ecg_id)
                label = self.metadata.loc[ecg_id, 'label']
                signals.append(signal)
                labels.append(label)
            except Exception as e:
                print(f"跳过ECG {ecg_id}: {str(e)}")
        
        # 验证数据集有效性
        if len(signals) == 0:
            raise ValueError("未加载到任何有效ECG信号")
        
        # 转换为numpy数组
        signals = np.array(signals)  # shape: (n_samples, n_channels, seq_len)
        labels = np.array(labels, dtype=np.int64)  # shape: (n_samples,)
        
        # 标准化（内置逻辑，避免外部访问scaler）
        if normalize:
            print("标准化ECG信号...")
            original_shape = signals.shape
            # 重塑为2D（n_samples×n_channels, seq_len）以便标准化
            signals_2d = signals.reshape(-1, original_shape[-1])
            signals_2d = self.scaler.fit_transform(signals_2d)
            # 重塑回原始3D形状
            signals = signals_2d.reshape(original_shape)
            # 裁剪极端值（±3σ，避免异常值影响训练）
            signals = np.clip(signals, -3.0, 3.0)
        
        # 打印数据集信息
        print(f"数据集创建完成：")
        print(f"  信号形状: {signals.shape}")
        print(f"  类别分布: 正常={np.sum(labels==0)}, 异常={np.sum(labels==1)}")
        return signals, labels

class PTBXLDataset(torch.utils.data.Dataset):
    """PTB-XL数据集类，修复属性访问与数据增强逻辑"""
    def __init__(self, data_path: str, seq_len: int = 1000, fs: int = 100, 
                 max_samples: int = 21799, training: bool = True, normalize: bool = True):
        self.data_path = data_path
        self.seq_len = seq_len
        self.fs = fs
        self.training = training  # 是否为训练集（控制数据增强）
        self.normalize = normalize
        self.max_samples = max_samples
        self.data = None  # 数据集信号（3D: n_samples×n_channels×seq_len）
        self.labels = None  # 数据集标签（1D: n_samples）

        try:
            # 1. 初始化处理器并加载数据
            self.processor = PTBXLDataProcessor(
                data_path=data_path,
                seq_len=seq_len,
                fs=fs
            )
            self.processor.load_metadata()  # 加载元数据
            # 2. 创建数据集（通过返回值获取data和labels，而非访问processor.data）
            self.data, self.labels = self.processor.create_dataset(
                max_samples=max_samples,
                normalize=normalize
            )
            print(f"PTB-XL真实数据加载成功：{len(self.data)}个样本")

        except Exception as e:
            # 加载失败时使用合成数据作为后备
            print(f"真实数据加载失败：{str(e)}")
            print("使用合成ECG数据继续训练...")
            self.data, self.labels = self._generate_synthetic_data()

    def _generate_synthetic_data(self):
        """生成合成ECG数据（后备方案）"""
        n_samples = 1000
        n_channels = 12  # 模拟12导联ECG
        signals = np.random.randn(n_samples, n_channels, self.seq_len) * 0.1
        labels = np.random.randint(0, 2, n_samples)  # 平衡类别
        
        # 添加ECG特征（模拟正常/异常波形）
        t = np.linspace(0, 4*np.pi, self.seq_len)
        for i in range(n_samples):
            for ch in range(n_channels):
                if labels[i] == 0:  # 正常心律（正弦波模拟）
                    ecg_wave = np.sin(t) + 0.5*np.sin(2*t) + 0.2*np.sin(3*t)
                else:  # 异常心律（不规则波形）
                    ecg_wave = 0.8*np.sin(1.5*t) + 0.6*np.sin(3*t) + 0.1*np.random.randn(self.seq_len)
                # 通道特异性调整
                channel_factor = 0.5 + 0.5*np.sin(2*np.pi*ch/n_channels)
                signals[i, ch] += ecg_wave * channel_factor
        
        return signals.astype(np.float32), labels.astype(np.int64)

    def __getitem__(self, idx):
        """获取单个样本，修复数据增强中的维度错误"""
        # 读取数据并转换为Tensor
        data = torch.FloatTensor(self.data[idx])  # (n_channels, seq_len)
        label = torch.LongTensor([self.labels[idx]])[0]  # 单个标签
        
        # 训练时添加数据增强（仅对训练集生效）
        if self.training:
            # 1. 添加轻微噪声（模拟信号干扰）
            noise = torch.randn_like(data) * 0.01
            data += noise
            
            # 2. 随机幅度缩放（±10%，避免信号幅值异常）
            scale = torch.rand(1) * 0.2 + 0.9  # 缩放范围：0.9~1.1
            data *= scale
            
            # 3. 随机裁剪（保持90%长度，避免维度不匹配）
            crop_len = int(self.seq_len * 0.9)
            start = torch.randint(0, self.seq_len - crop_len + 1, (1,)).item()
            data = data[..., start:start + crop_len]  # 裁剪时间维度
            
            # 4. 补零到原始长度（确保输入维度一致）
            if data.shape[-1] < self.seq_len:
                pad_length = self.seq_len - data.shape[-1]
                data = torch.nn.functional.pad(data, (0, pad_length))  # 仅补时间维度
        
        return data, label

    def __len__(self):
        """返回样本数量"""
        return len(self.data)
    
    def __getitem__(self, idx):
        data = torch.FloatTensor(self.data[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
         # 训练时添加数据增强
        if self.training:
            # 1. 添加轻微噪声（模拟信号干扰）
            noise = torch.randn_like(data) * 0.01
            data += noise
            
            # 2. 随机幅度缩放（±10%）
            scale = torch.rand(1) * 0.2 + 0.9  # 0.9-1.1之间的缩放因子
            data *= scale
            
            # 3. 随机裁剪（保持90%长度）
            crop_len = int(data.shape[-1] * 0.9)
            start = torch.randint(0, data.shape[-1] - crop_len, (1,)).item()
            data = data[..., start:start+crop_len]
            # 若需要固定长度，可补零
            if data.shape[-1] < self.seq_len:
                data = torch.nn.functional.pad(data, (0, self.seq_len - data.sh6ape[-1]))
        
        return data, label
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.LongTensor([self.labels[idx]])[0]

# 其余类保持不变 (DataAnalyzer)
class DataAnalyzer:
    """数据分析器，用于探索和理解数据集"""
    
    def __init__(self, dataset, output_dir='results'):
        self.dataset = dataset
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def analyze_dataset(self):
        """全面分析数据集"""
        print("开始数据集分析...")
        
        data, labels = [], []
        for i in tqdm(range(min(1000, len(self.dataset))), desc="收集数据"):
            sample_data, sample_label = self.dataset[i]
            data.append(sample_data.numpy())
            labels.append(sample_label.numpy())
        
        data = np.array(data)
        labels = np.array(labels)
        
        # 基本统计
        self._print_basic_stats(data, labels)
        
        # 绘制分析图表
        self._plot_data_analysis(data, labels)
        
        return data, labels
    
    def _print_basic_stats(self, data, labels):
        """打印基本统计信息"""
        print("\n" + "="*50)
        print("数据集统计分析")
        print("="*50)
        
        print(f"数据形状: {data.shape}")
        print(f"样本数量: {len(data)}")
        print(f"通道数量: {data.shape[1]}")
        print(f"序列长度: {data.shape[2]}")
        
        # 类别分布
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\n类别分布:")
        for cls, count in zip(unique, counts):
            print(f"  类别 {cls}: {count} 样本 ({count/len(labels)*100:.1f}%)")
        
        # 数据统计
        print(f"\n数据统计:")
        print(f"  全局均值: {np.mean(data):.4f}")
        print(f"  全局标准差: {np.std(data):.4f}")
        print(f"  最小值: {np.min(data):.4f}")
        print(f"  最大值: {np.max(data):.4f}")
        
        # 通道统计
        print(f"\n各通道统计:")
        for ch in range(min(6, data.shape[1])):
            channel_data = data[:, ch, :].flatten()
            print(f"  通道 {ch}: 均值={np.mean(channel_data):.4f}, 标准差={np.std(channel_data):.4f}")
    
    def _plot_data_analysis(self, data, labels):
        """绘制数据分析图表"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 类别分布
        unique, counts = np.unique(labels, return_counts=True)
        axes[0, 0].bar(unique, counts, color=['lightblue', 'lightcoral'], alpha=0.7)
        axes[0, 0].set_title('类别分布', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('类别')
        axes[0, 0].set_ylabel('样本数量')
        for i, v in enumerate(counts):
            axes[0, 0].text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
        
        # 2. 数据分布直方图
        axes[0, 1].hist(data.flatten(), bins=50, alpha=0.7, color='green', density=True)
        axes[0, 1].set_title('数据值分布', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('数值')
        axes[0, 1].set_ylabel('密度')
        
        # 3. 通道均值比较
        channel_means = [np.mean(data[:, i, :]) for i in range(min(6, data.shape[1]))]
        axes[0, 2].bar(range(len(channel_means)), channel_means, alpha=0.7, color='orange')
        axes[0, 2].set_title('各通道均值', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('通道')
        axes[0, 2].set_ylabel('均值')
        
        # 4. 样本可视化（正常 vs 异常）
        normal_idx = np.where(labels == 0)[0]
        abnormal_idx = np.where(labels == 1)[0]
        
        if len(normal_idx) > 0 and len(abnormal_idx) > 0:
            normal_sample = data[normal_idx[0], 0, :]
            abnormal_sample = data[abnormal_idx[0], 0, :]
            
            axes[1, 0].plot(normal_sample, 'b-', alpha=0.8, label='正常')
            axes[1, 0].plot(abnormal_sample, 'r-', alpha=0.8, label='异常')
            axes[1, 0].set_title('正常 vs 异常样本 (通道 0)', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('时间点')
            axes[1, 0].set_ylabel('振幅')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 相关性热图（通道间）
        if data.shape[1] > 1:
            channel_corr = np.corrcoef(data.reshape(data.shape[0], -1).T)
            display_size = min(20, channel_corr.shape[0])
            im = axes[1, 1].imshow(channel_corr[:display_size, :display_size], cmap='coolwarm', aspect='auto')
            axes[1, 1].set_title('通道间相关性', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('特征维度')
            axes[1, 1].set_ylabel('特征维度')
            plt.colorbar(im, ax=axes[1, 1])
        else:
            axes[1, 1].text(0.5, 0.5, '单通道数据\n无法计算相关性', 
                           ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].set_title('通道间相关性', fontsize=14, fontweight='bold')
        
        # 6. 统计摘要
        axes[1, 2].axis('off')
        stats_text = f"""数据集统计摘要:
        总样本数: {len(data)}
        数据形状: {data.shape}
        类别数: {len(unique)}
        数据范围: [{np.min(data):.3f}, {np.max(data):.3f}]
        数据均值: {np.mean(data):.3f}
        数据标准差: {np.std(data):.3f}
        
        类别平衡性:
        {'平衡' if max(counts)/min(counts) < 2 else '不平衡'}
        最大类别比例: {max(counts)/len(labels)*100:.1f}%
        """
        axes[1, 2].text(0.1, 0.9, stats_text, fontsize=11, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'dataset_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()

def validate_model_dimensions(model, input_channels=12, seq_len=1000, device=None):
    """验证模型各层的维度是否匹配"""
    print("=== 模型维度验证 ===")
    
    # 确定设备 - 使用模型所在的设备或指定的设备
    if device is None:
        # 获取模型所在的设备
        device = next(model.parameters()).device
    
    # 创建测试输入并移动到正确的设备
    test_input = torch.randn(2, input_channels, seq_len).to(device)
    print(f"输入维度: {test_input.shape}, 设备: {test_input.device}")
    
    # 逐层验证
    with torch.no_grad():
        # 贝叶斯CNN
        cnn_features = model.bayesian_cnn(test_input)
        print(f"贝叶斯CNN输出: {cnn_features.shape}")
        
        if model.use_decoupler:
            # 特征解耦
            shared_features, specific_features = model.feature_decoupler(
                cnn_features.unsqueeze(1)
            )
            print(f"共享特征: {shared_features.shape}")
            print(f"特定特征: {specific_features.shape}")
            
            # Transformer
            try:
                transformer_out = model.transformer(shared_features)
                print(f"Transformer输出: {transformer_out.shape}")
            except Exception as e:
                print(f"Transformer错误: {e}")
            
            # LSTM
            try:
                lstm_out, _ = model.lstm(specific_features)
                print(f"LSTM输出: {lstm_out.shape}")
            except Exception as e:
                print(f"LSTM错误: {e}")
    
    print("=== 维度验证完成 ===\n")

def main():
    """主训练函数"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设备配置
    device = get_device(use_xpu=False)
    print(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = 'ptbxl_training_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 配置数据集参数 - 使用PTB-XL数据集
    data_config = {
        'data_path': r"D:\廖国栋\violet\A\study\Python\CABLETS\Dataset\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
        'seq_len': 1000,
        'fs': 100,
        'max_samples': 21799,  # 限制样本数量以加快训练
        'normalize': True
    }
    
    # 2. 加载PTB-XL数据集
    print("加载PTB-XL心血管疾病数据集...")
    try:
        dataset = PTBXLDataset(**data_config)
        
        # 3. 数据分析
        print("进行数据集分析...")
        analyzer = DataAnalyzer(dataset, output_dir)
        data_array, labels_array = analyzer.analyze_dataset()
        
    except Exception as e:
        print(f"PTB-XL数据加载失败: {e}")
        print("使用合成数据继续训练...")
        dataset = PTBXLDataset('')  # 这会触发合成数据生成
    
    # 4. 数据集划分
    print("划分数据集...")
    
    # 获取标签
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(label.item())
    labels = np.array(labels)
    
    train_idx, test_idx = train_test_split(
        range(len(dataset)), 
        test_size=0.1, 
        random_state=42,
        stratify=labels
    )
    train_idx, val_idx = train_test_split(
        train_idx, 
        test_size=0.11, 
        random_state=42,
        stratify=labels[train_idx]
    )
    
    # 创建子集
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本")
    
    # 5. 创建数据加载器
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    # 6. 初始化混合模型
    print("初始化混合心血管疾病预测系统...")
    
    # 获取数据形状信息
    sample_data, _ = dataset[0]
    input_channels = sample_data.shape[0]
    
    hybrid_model = HybridCardiovascularSystem(
        input_channels=input_channels,
        base_dim=128,
        use_decoupler=False,
        xgb_n_estimators=100,
        xgb_max_depth=10,
        xgb_learning_rate=0.05,
        use_ensemble=True,
        ensemble_method='majority_voting'
    )
    
    # 验证模型维度
    validate_model_dimensions(hybrid_model.feature_extractor, input_channels, data_config['seq_len'])
    
    # 7. 训练特征提取器
    print("开始训练特征提取器...")
    feature_extractor_path = os.path.join(output_dir, 'best_feature_extractor.pth')
    best_val_acc = hybrid_model.train_feature_extractor(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,  # 减少轮数以加快训练
        lr=1.7e-3,
        save_path=feature_extractor_path
    )
    
    # 8. 训练分类器
    print("开始训练分类器...")
    classifier_accuracy = hybrid_model.train_classifiers(
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    # 9. 在测试集上评估
    print("在测试集上评估模型...")
    y_pred, y_true = hybrid_model.predict(test_loader)
    
    # 计算测试准确率
    test_accuracy = np.mean(y_pred == y_true)
    print(f"测试集准确率: {test_accuracy:.4f}")
    
    # 分类报告
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=['正常', '异常']))
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    print("混淆矩阵:")
    print(cm)
    
    # 10. 绘制评估图表
    print("生成评估图表...")
    plt.figure(figsize=(15, 10))
    
    # 混淆矩阵热图
    plt.subplot(2, 3, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['正常', '异常'], 
                yticklabels=['正常', '异常'])
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    
    # ROC曲线
    plt.subplot(2, 3, 2)
    y_proba, _ = hybrid_model.predict_proba(test_loader)
    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率')
    plt.ylabel('真正率')
    plt.title('ROC曲线')
    plt.legend(loc="lower right")
    
    # 精确率-召回率曲线
    plt.subplot(2, 3, 3)
    precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
    pr_auc = auc(recall, precision)
    
    plt.plot(recall, precision, color='green', lw=2, label=f'PR曲线 (AUC = {pr_auc:.2f})')
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.title('精确率-召回率曲线')
    plt.legend(loc="lower left")
    
    # 分类器权重（如果使用集成）
    plt.subplot(2, 3, 4)
    weights = hybrid_model.get_classifier_weights()
    plt.bar(weights.keys(), weights.values(), color=['skyblue', 'lightcoral', 'lightgreen'])
    plt.title('分类器权重')
    plt.ylabel('权重')
    plt.xticks(rotation=45)
    
    # 各分类器性能比较
    plt.subplot(2, 3, 5)
    individual_accuracies = hybrid_model.evaluate_individual_classifiers(test_loader)
    classifiers = list(individual_accuracies.keys())
    accuracies = list(individual_accuracies.values())
    
    colors = ['lightblue'] * (len(classifiers) - 1) + ['gold']
    bars = plt.bar(classifiers, accuracies, color=colors, alpha=0.7)
    plt.title('各分类器性能比较')
    plt.ylabel('准确率')
    plt.xticks(rotation=45)
    
    # 在柱状图上添加数值
    for bar, acc in zip(bars, accuracies):
        if acc is not None:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.3f}', ha='center', va='bottom')
    
    # 分类器一致性
    plt.subplot(2, 3, 6)
    if hasattr(hybrid_model.ensemble, 'get_agreement_rate'):
        agreement_rate = hybrid_model.ensemble.get_agreement_rate(
            hybrid_model.extract_features(test_loader)[0]
        )
        plt.pie([agreement_rate, 1-agreement_rate], 
                labels=['一致', '不一致'], 
                autopct='%1.1f%%', 
                colors=['lightgreen', 'lightcoral'])
        plt.title(f'分类器一致性: {agreement_rate:.3f}')
    else:
        plt.text(0.5, 0.5, '一致性分析\n不适用于当前集成方法', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('分类器一致性')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_evaluation.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 11. 保存模型和结果
    print("保存模型和结果...")
    model_save_path = os.path.join(output_dir, 'hybrid_cardio_model')
    hybrid_model.save(model_save_path)
    
    # 保存训练结果
    results = {
        'feature_extractor_val_accuracy': best_val_acc,
        'classifier_val_accuracy': classifier_accuracy,
        'test_accuracy': test_accuracy,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm.tolist(),
        'individual_accuracies': individual_accuracies,
        'classifier_weights': weights
    }
    
    with open(os.path.join(output_dir, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # 12. 输出总结报告
    print("\n" + "="*60)
    print("训练总结报告")
    print("="*60)
    print(f"特征提取器最佳验证准确率: {best_val_acc:.2f}%")
    print(f"分类器验证准确率: {classifier_accuracy:.4f}" if classifier_accuracy else "N/A")
    print(f"测试集准确率: {test_accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    
    print("\n各分类器性能:")
    for name, acc in individual_accuracies.items():
        if acc is not None:
            print(f"  {name}: {acc:.4f}")
    
    print(f"\n分类器权重:")
    for name, weight in weights.items():
        print(f"  {name}: {weight:.4f}")
    
    print(f"\n所有结果已保存到: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()