import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import optuna
import time
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import joblib
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

# 尝试导入Intel XPU扩展
try:
    import intel_extension_for_pytorch as ipex
    has_ipex = True
except ImportError:
    has_ipex = False
    print("Intel Extension for PyTorch not found. XPU acceleration will be disabled.")


def get_device(use_xpu: bool = True) -> torch.device:
    """获取最佳可用设备（优先XPU，其次CUDA，最后CPU）"""
    if use_xpu and has_ipex and torch.xpu.is_available():
        return torch.device("xpu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def create_dataloader(dataset, batch_size: int, shuffle: bool = True, num_workers: int = None):
    """创建数据加载器，自动适配环境设置num_workers"""
    if num_workers is None:
        num_workers = 0 if torch.utils.data.get_worker_info() is None else 4
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True  
    )

# 贝叶斯CNN用于动态特征提取
class BayesianCNN(nn.Module):
    """贝叶斯卷积神经网络，增强特征提取能力"""
    def __init__(self, input_channels: int = 3, output_dim: int = 128):
        super(BayesianCNN, self).__init__()
        self.output_dim = output_dim
        
        # 增加卷积层深度，使用更大卷积核捕捉长时特征
        self.conv1_mu = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2)  # 卷积核从5→7
        self.conv1_sigma = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2)
        
        self.conv2_mu = nn.Conv1d(64, 128, kernel_size=5, stride=2)  # 卷积核从3→5
        self.conv2_sigma = nn.Conv1d(64, 128, kernel_size=5, stride=2)
        
        self.conv3_mu = nn.Conv1d(128, 640, kernel_size=3, stride=1)
        self.conv3_sigma = nn.Conv1d(128, 640, kernel_size=3, stride=1)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)  # dropout抑制过拟合
        self.adaptive_pool = nn.AdaptiveAvgPool1d(7)
        self.adaptive_layer = nn.Linear(640 * 7, output_dim)  # 输入维度随conv3调整

    def forward(self, x):
        """改进贝叶斯采样，降低噪声权重"""
        def bayesian_layer(x, mu_layer, sigma_layer):
            mu = mu_layer(x)
            # 改进sigma计算，增加数值稳定性
            sigma = F.softplus(sigma_layer(x)) + 1e-4  # 增加偏置值，确保sigma更大更稳定
            # 限制sigma的范围，防止过小
            sigma = torch.clamp(sigma, min=1e-4, max=1.0)
            # 对mu进行裁剪，防止极端值
            mu = torch.clamp(mu, min=-10.0, max=10.0)
            dist = Normal(mu, sigma)
            return dist.rsample()      # 使用重参数化采样  
            
        x = F.relu(bayesian_layer(x, self.conv1_mu, self.conv1_sigma))
        x = self.pool(x)
        x = self.dropout(x)  # dropout
        
        x = F.relu(bayesian_layer(x, self.conv2_mu, self.conv2_sigma))
        x = self.pool(x)
        x = self.dropout(x)  # dropout
        
        # 卷积层前向传播
        x = F.relu(bayesian_layer(x, self.conv3_mu, self.conv3_sigma))
        x = self.adaptive_pool(x)
        
        x = x.view(x.size(0), -1)
        x = self.adaptive_layer(x)
        
        return x


class EnhancedFeatureDecoupler(nn.Module):
    """增强的特征解耦模块"""
    def __init__(self, input_dim: int, hidden_dim: int = None, output_dim: int = None):
        super(EnhancedFeatureDecoupler, self).__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim * 2  # 默认隐藏层是输入的两倍
        if output_dim is None:
            output_dim = input_dim // 2  # 默认输出是输入的一半
        
        self.output_dim = output_dim
        
        # 增强的特征提取器
        self.shared_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),  
            nn.ReLU()
        )
        
        self.modal_specific = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),  
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
        
        # 增强的判别器（输入维度与输出维度匹配）
        self.discriminator = nn.Sequential(
            nn.Linear(output_dim, output_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_dim//2, output_dim//4),
            nn.ReLU(),
            nn.Linear(output_dim//4, 1)
        )
        
        # 正交性约束
        self.ortho_loss_coeff = 0.1
        
    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        x_flat = x.view(-1, input_dim)
        
        shared_flat = self.shared_extractor(x_flat)
        specific_flat = self.modal_specific(x_flat)
        
        # 确保输出维度正确
        shared_features = shared_flat.view(batch_size, seq_len, self.output_dim)
        specific_features = specific_flat.view(batch_size, seq_len, self.output_dim)
        
        return shared_features, specific_features
    
    def compute_decouple_loss(self, shared_features, specific_features, epoch):
        """增强的解耦损失计算"""
        batch_size, seq_len, feat_dim = shared_features.shape
        
        # 向量化处理
        shared_flat = shared_features.view(-1, feat_dim)
        specific_flat = specific_features.view(-1, feat_dim)
        
        # 判别器损失
        pred_shared = self.discriminator(shared_flat)
        pred_specific = self.discriminator(specific_flat)
        
        # 使用标签平滑
        real_labels = torch.ones_like(pred_shared) * 0.9
        fake_labels = torch.ones_like(pred_specific) * 0.1
        
        loss_shared = F.binary_cross_entropy_with_logits(pred_shared, real_labels)
        loss_specific = F.binary_cross_entropy_with_logits(pred_specific, fake_labels)
        
        discriminator_loss = (loss_shared + loss_specific) / 2
        
        # 正交性约束：鼓励共享特征和特定特征相互独立
        shared_norm = F.normalize(shared_flat, p=2, dim=1)
        specific_norm = F.normalize(specific_flat, p=2, dim=1)
        ortho_loss = torch.mean(torch.abs(torch.sum(shared_norm * specific_norm, dim=1)))
        
        # 多样性约束：鼓励特定特征在不同样本间有区分度
        diversity_loss = -torch.std(specific_flat, dim=0).mean()
        
        # 渐进式训练：随着训练进行逐渐增加正交性约束
        current_ortho_coeff = min(self.ortho_loss_coeff * (epoch / 50), self.ortho_loss_coeff)
        
        total_loss = (discriminator_loss + 
                     current_ortho_coeff * ortho_loss + 
                     0.01 * diversity_loss)
        
        return total_loss, {
            'discriminator': discriminator_loss.item(),
            'orthogonal': ortho_loss.item(),
            'diversity': diversity_loss.item()
        }


# 注意力机制模块
class AttentionMechanism(nn.Module):
    """自注意力机制，捕捉特征序列中的依赖关系"""
    def __init__(self, feature_dim: int):
        super(AttentionMechanism, self).__init__()
        self.feature_dim = feature_dim
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.scale = torch.sqrt(torch.tensor(feature_dim, dtype=torch.float32)).item()
        
    def forward(self, x):
        """
        Args:
            x: 输入序列，形状为(batch_size, seq_len, feature_dim)
        Returns:
            output: 注意力加权后的序列，形状同上
            attention_weights: 注意力权重，形状为(batch_size, seq_len, seq_len)
        """
        batch_size, seq_len, feature_dim = x.size()
        if feature_dim != self.feature_dim:
            raise ValueError(
                f"注意力输入维度不匹配: 期望 {self.feature_dim}, 实际 {feature_dim}"
            )
        
        Q = self.query(x)  # (batch, seq_len, feature_dim)
        K = self.key(x)    # (batch, seq_len, feature_dim)
        V = self.value(x)  # (batch, seq_len, feature_dim)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.permute(0, 2, 1)) / self.scale  # (batch, seq_len, seq_len)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)  # (batch, seq_len, feature_dim)
        
        return output, attention_weights


class XGBoostClassifier:
    """XGBoost分类器包装类"""
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42):
        self.model = None
        self.scaler = StandardScaler()
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'random_state': random_state,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss'
        }
        
    def fit(self, X, y):
        """训练XGBoost模型"""
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 创建DMatrix
        dtrain = xgb.DMatrix(X_scaled, label=y)
        
        # 训练模型
        self.model = xgb.train(
            self.params, 
            dtrain,
            num_boost_round=self.params['n_estimators']
        )
        return self
    
    def predict(self, X):
        """预测类别"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        X_scaled = self.scaler.transform(X)
        dtest = xgb.DMatrix(X_scaled)
        probs = self.model.predict(dtest)
        return (probs > 0.5).astype(int)
    
    def predict_proba(self, X):
        """预测概率"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        X_scaled = self.scaler.transform(X)
        dtest = xgb.DMatrix(X_scaled)
        probs = self.model.predict(dtest)
        # 转换为二分类概率格式
        return np.column_stack([1 - probs, probs])
    
    def save(self, path):
        """保存模型和标准化器"""
        if self.model is None:
            raise ValueError("没有可保存的模型")
        
        # 保存XGBoost模型
        self.model.save_model(f"{path}_xgb.model")
        # 保存标准化器
        joblib.dump(self.scaler, f"{path}_scaler.pkl")
        
    def load(self, path):
        """加载模型和标准化器"""
        # 加载XGBoost模型
        self.model = xgb.Booster()
        self.model.load_model(f"{path}_xgb.model")
        # 加载标准化器
        self.scaler = joblib.load(f"{path}_scaler.pkl")
        return self


class MajorityVotingEnsemble:
    """多数投票集成分类器"""
    def __init__(self, classifiers=None, weights=None):
        """
        Args:
            classifiers: 分类器列表，每个分类器需要有predict方法
            weights: 每个分类器的权重，如果为None则使用等权重
        """
        self.classifiers = classifiers if classifiers is not None else []
        self.weights = weights
        self.n_classifiers = len(self.classifiers)
        
        # 如果未提供权重，使用等权重
        if self.weights is None and self.n_classifiers > 0:
            self.weights = [1.0 / self.n_classifiers] * self.n_classifiers
        elif self.weights is None:
            self.weights = []  # 初始化为空列表
    
    def add_classifier(self, classifier, weight=None):
        """添加分类器到集成"""
        self.classifiers.append(classifier)
        
        if weight is None:
            if self.n_classifiers == 0:
                # 第一个分类器
                weight = 1.0
                self.weights = [weight]
            else:
                weight = 1.0 / (self.n_classifiers + 1)
                # 重新调整所有权重
                self.weights = [w * self.n_classifiers / (self.n_classifiers + 1) for w in self.weights]
                self.weights.append(weight)
        else:
            if self.n_classifiers == 0:
                self.weights = [weight]
            else:
                self.weights.append(weight)
                # 归一化权重
                total_weight = sum(self.weights)
                self.weights = [w / total_weight for w in self.weights]
        
        self.n_classifiers = len(self.classifiers)
    
    def predict(self, X):
        """多数投票预测"""
        if self.n_classifiers == 0:
            raise ValueError("集成中没有分类器")
        
        # 收集所有分类器的预测
        all_predictions = []
        for classifier in self.classifiers:
            predictions = classifier.predict(X)
            all_predictions.append(predictions)
        
        # 转换为numpy数组
        all_predictions = np.array(all_predictions)
        
        # 加权多数投票
        final_predictions = []
        for i in range(X.shape[0]):
            votes = all_predictions[:, i]
            # 计算加权投票
            weighted_votes = {}
            for j, vote in enumerate(votes):
                weighted_votes[vote] = weighted_votes.get(vote, 0) + self.weights[j]
            
            # 选择得票最高的类别
            final_pred = max(weighted_votes.items(), key=lambda x: x[1])[0]
            final_predictions.append(final_pred)
        
        return np.array(final_predictions)
    
    def predict_proba(self, X):
        """预测概率（加权平均）"""
        if self.n_classifiers == 0:
            raise ValueError("集成中没有分类器")
        
        # 收集所有分类器的概率预测
        all_probabilities = []
        for classifier in self.classifiers:
            if hasattr(classifier, 'predict_proba'):
                probabilities = classifier.predict_proba(X)
                all_probabilities.append(probabilities)
            else:
                # 如果没有predict_proba方法，使用硬投票转换为概率
                predictions = classifier.predict(X)
                probas = np.zeros((len(predictions), 2))
                for i, pred in enumerate(predictions):
                    probas[i, pred] = 1.0
                all_probabilities.append(probas)
        
        # 加权平均概率
        weighted_probas = np.zeros_like(all_probabilities[0])
        for i, proba in enumerate(all_probabilities):
            weighted_probas += proba * self.weights[i]
        
        return weighted_probas
    
    def get_agreement_rate(self, X):
        """计算分类器之间的一致性率"""
        if self.n_classifiers < 2:
            return 1.0
        
        # 收集所有分类器的预测
        all_predictions = []
        for classifier in self.classifiers:
            predictions = classifier.predict(X)
            all_predictions.append(predictions)
        
        # 计算一致性
        n_samples = X.shape[0]
        agreement_count = 0
        
        for i in range(n_samples):
            votes = [pred[i] for pred in all_predictions]
            # 如果所有分类器预测相同，则认为一致
            if len(set(votes)) == 1:
                agreement_count += 1
        
        return agreement_count / n_samples


class EnhancedCardiovascularModel(nn.Module):
    def __init__(self, input_channels: int = 3, seq_len: int = 10, 
                 base_dim: int = 128, num_classes: int = 2,
                 lstm_layers: int = 2, use_decoupler: bool = True,
                 use_xgboost: bool = True):
        super(EnhancedCardiovascularModel, self).__init__()
        self.use_decoupler = use_decoupler
        self.use_xgboost = use_xgboost
        self.base_dim = base_dim
        
        # 贝叶斯CNN特征提取
        self.bayesian_cnn = BayesianCNN(input_channels, output_dim=base_dim)
        
        if use_decoupler:
            # 使用增强的特征解耦
            self.feature_decoupler = EnhancedFeatureDecoupler(
                input_dim=base_dim, 
                hidden_dim=base_dim,
                output_dim=base_dim//2
            )
            
            # Transformer用于处理共享特征
            transformer_layer = nn.TransformerEncoderLayer(
                d_model=base_dim//2,
                nhead=4,
                dim_feedforward=base_dim,
                dropout=0.3,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=2)
            
            # LSTM用于处理特定特征
            self.lstm = nn.LSTM(
                input_size=base_dim//2,
                hidden_size=base_dim//4,
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=True,
                dropout=0.3 if lstm_layers > 1 else 0.0
            )
            
            # 特征融合维度
            fusion_dim = (base_dim//2) + (base_dim//2)
        else:
            # 如果不使用特征解耦，直接处理原始特征
            fusion_dim = base_dim
        
        self.attention = AttentionMechanism(feature_dim=fusion_dim)
        
        # 如果使用XGBoost，则只输出特征；否则使用神经网络分类器
        if not use_xgboost:
            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim, base_dim//2),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(base_dim//2, base_dim//4),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(base_dim//4, num_classes)
            )
        
        # 自适应损失权重
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 分类损失权重
        self.beta = nn.Parameter(torch.tensor(0.5))   # 解耦损失权重
        
    def forward(self, x, epoch=0):
        batch_size, channels, seq_len = x.shape[:3]
        
        # 贝叶斯CNN特征提取
        cnn_features = self.bayesian_cnn(x)
        
        if self.use_decoupler:
            # 使用特征解耦
            shared_features, specific_features = self.feature_decoupler(
                cnn_features.unsqueeze(1)
            )
            
            # 分别处理共享和特定特征
            transformer_out = self.transformer(shared_features)
            lstm_out, _ = self.lstm(specific_features)
            
            # 特征融合
            combined_features = torch.cat([transformer_out, lstm_out], dim=2)
            
            # 计算解耦损失
            decouple_loss, loss_components = self.feature_decoupler.compute_decouple_loss(
                shared_features, specific_features, epoch
            )
        else:
            # 不使用特征解耦，直接处理
            combined_features = cnn_features.unsqueeze(1)
            decouple_loss = torch.tensor(0.0).to(x.device)
            loss_components = {}
        
        # 注意力机制
        attention_out, attn_weights = self.attention(combined_features)
        time_aggregated = attention_out.mean(dim=1)
        
        # 如果使用XGBoost，返回特征；否则返回分类结果
        if self.use_xgboost:
            # 返回特征用于XGBoost训练
            return time_aggregated, attn_weights, decouple_loss, loss_components
        else:
            # 使用神经网络分类器
            logits = self.classifier(time_aggregated)
            return logits, attn_weights, decouple_loss, loss_components
        
    def compute_loss(self, logits, labels, decouple_loss, attn_weights, loss_components, epoch):
        """优化类别不平衡处理"""
        num_classes = logits.size(1)
        
        # 更激进的类别权重计算（加重异常类权重）
        class_counts = torch.zeros(num_classes, device=labels.device)
        existing_counts = torch.bincount(labels, minlength=num_classes)
        class_counts[:] = existing_counts
        class_counts = class_counts.clamp(min=1e-6)
        
        # 异常类（假设索引1为异常）权重额外放大2.5倍
        weights = 1.0 / class_counts
        weights[1] *= 2.5  # 异常类错分代价更高
        weights = weights / weights.sum() * num_classes
        
        # 分类损失
        classification_loss = F.cross_entropy(logits, labels, weight=weights)
        
        # 注意力正则化
        attention_reg = -torch.mean(torch.log(torch.mean(attn_weights, dim=1) + 1e-8))
        
        # 自适应损失权重
        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta)
        
        if self.use_decoupler:
            # 渐进式训练：早期关注分类，后期关注解耦
            progress_factor = min(epoch / 30, 1.0)
            decouple_weight = beta * progress_factor
            classification_weight = alpha * (1 - 0.5 * progress_factor)
            
            total_loss = (classification_weight * classification_loss + 
                         decouple_weight * decouple_loss + 
                         0.01 * attention_reg)
        else:
            total_loss = classification_loss + 0.01 * attention_reg
        
        loss_details = {
            'classification': classification_loss.item(),
            'attention_reg': attention_reg.item(),
            'total': total_loss.item()
        }
        
        if self.use_decoupler:
            loss_details.update(loss_components)
            loss_details['decouple'] = decouple_loss.item()
            loss_details['alpha'] = alpha.item()
            loss_details['beta'] = beta.item()
        
        return total_loss, loss_details


class HybridCardiovascularSystem:
    def __init__(self, input_channels=3, base_dim=128, use_decoupler=True,
                 xgb_n_estimators=250, xgb_max_depth=12, xgb_learning_rate=0.03,
                 lgbm_n_estimators=250,   # 新增LightGBM参数
                 use_ensemble=True, ensemble_method='majority_voting'):
        
        # 神经网络特征提取器
        self.feature_extractor = EnhancedCardiovascularModel(
            input_channels=input_channels,
            base_dim=base_dim,
            use_decoupler=use_decoupler,
            use_xgboost=True  # 设置为True以输出特征
        )
        
        # 分类器集成
        self.use_ensemble = use_ensemble
        self.ensemble_method = ensemble_method
        
        if use_ensemble:
            self.ensemble = MajorityVotingEnsemble()
            
            # 替换SVM为LightGBM（提升多样性）
            classifiers = {
                'xgb': XGBoostClassifier(
                    n_estimators=xgb_n_estimators,
                    max_depth=xgb_max_depth,
                    learning_rate=xgb_learning_rate
                ),
                'rf': RandomForestClassifier(
                    n_estimators=250,
                    max_depth=12,
                    random_state=42
                ),
                'lgbm': LGBMClassifier(  # 新增LightGBM
                    n_estimators=lgbm_n_estimators,
                    max_depth=12,
                    learning_rate=0.03,
                    random_state=42,
                    reg_alpha=0.025,     # 增加L1正则化，防止过拟合
                    reg_lambda=0.025    # 增加L2正则化，防止过拟合
                )
            }
            
            for name, classifier in classifiers.items():
                self.ensemble.add_classifier(classifier)
                
            self.classifiers = classifiers
        else:
            # 只使用XGBoost
            self.xgb_classifier = XGBoostClassifier(
                n_estimators=xgb_n_estimators,
                max_depth=xgb_max_depth,
                learning_rate=xgb_learning_rate
            )
            self.classifiers = {'xgb': self.xgb_classifier}
        
        self.device = get_device()
        self.feature_extractor.to(self.device)
        
    def extract_features(self, dataloader):
        """从数据加载器中提取特征"""
        self.feature_extractor.eval()
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in dataloader:
                # 数据预处理 - 确保维度顺序为 [batch, channels, seq_len]
                if data.dim() == 4:
                    # 处理4维数据：假设输入为 [batch, 1, seq_len, channels]
                    data = data.squeeze(1)  # 移除多余的维度
                    data = data.permute(0, 2, 1)  # 转换为 [batch, channels, seq_len]
                elif data.dim() == 3:
                    # 处理3维数据：检查并转换为 [batch, channels, seq_len]
                    # 关键修正：如果通道数大于序列长度，可能是维度颠倒了
                    if data.shape[1] > data.shape[2] or data.shape[1] != self.feature_extractor.bayesian_cnn.conv1_mu.in_channels:
                        # 交换通道和序列长度维度
                        data = data.permute(0, 2, 1)
                
                # 确保通道数正确
                if data.shape[1] != self.feature_extractor.bayesian_cnn.conv1_mu.in_channels:
                    # 尝试强制调整通道数（如果差异较小）
                    if abs(data.shape[1] - self.feature_extractor.bayesian_cnn.conv1_mu.in_channels) < 5:
                        # 取最接近的通道数
                        data = data[:, :self.feature_extractor.bayesian_cnn.conv1_mu.in_channels, :]
                    else:
                        raise ValueError(f"通道数不匹配: 期望 {self.feature_extractor.bayesian_cnn.conv1_mu.in_channels}, 实际 {data.shape[1]}")
                
                data = data.to(self.device)
                
                # 提取特征
                features, _, _, _ = self.feature_extractor(data)
                all_features.append(features.cpu().numpy())
                all_labels.append(labels.numpy())
        
        return np.vstack(all_features), np.hstack(all_labels)
    
    def train_feature_extractor(self, train_loader, val_loader, epochs=30, lr=5e-5, save_path=None):
        """训练神经网络特征提取器"""
        print("训练神经网络特征提取器...")
        
        # 临时使用神经网络分类器进行特征提取器训练
        temp_model = EnhancedCardiovascularModel(
            input_channels=self.feature_extractor.bayesian_cnn.conv1_mu.in_channels,
            base_dim=self.feature_extractor.base_dim,
            use_decoupler=self.feature_extractor.use_decoupler,
            use_xgboost=False  # 使用神经网络分类器进行训练
        ).to(self.device)
        
        # 复制权重
        temp_model.load_state_dict(self.feature_extractor.state_dict(), strict=False)
        
        # 训练模型
        trained_model, best_val_acc = train_model(
            temp_model, train_loader, val_loader, 
            epochs=epochs, lr=lr, save_path=save_path
        )
        
        # 将训练好的权重复制回特征提取器
        self.feature_extractor.load_state_dict(trained_model.state_dict(), strict=False)
        
        print(f"特征提取器训练完成，最佳验证准确率: {best_val_acc:.2f}%")
        return best_val_acc
    
    def train_classifiers(self, train_loader, val_loader=None):
        """根据验证集性能设置动态权重"""
        X_train, y_train = self.extract_features(train_loader)
        
        if self.use_ensemble and self.ensemble_method == 'majority_voting':
            # 训练并评估每个分类器的验证性能
            clf_val_accs = []
            for name, classifier in self.classifiers.items():
                print(f"训练 {name} 分类器...")
                classifier.fit(X_train, y_train)
                
                # 验证集评估
                if val_loader:
                    X_val, y_val = self.extract_features(val_loader)
                    y_pred = classifier.predict(X_val)
                    acc = np.mean(y_pred == y_val)
                    clf_val_accs.append(acc)
                    print(f"{name} 验证准确率: {acc:.4f}")
            
            # 根据验证准确率设置动态权重（性能越好权重越高）
            if val_loader and clf_val_accs:
                weights = [acc / sum(clf_val_accs) for acc in clf_val_accs]
                self.ensemble.weights = weights  # 更新集成权重
                print(f"分类器动态权重: {dict(zip(self.classifiers.keys(), weights))}")
               
                 # 计算分类器一致性
                if self.ensemble_method == 'majority_voting':
                    agreement_rate = self.ensemble.get_agreement_rate(X_val)
                    print(f"分类器一致性率: {agreement_rate:.4f}")
                
        
        else:
            # 训练单个XGBoost分类器
            self.xgb_classifier.fit(X_train, y_train)
            
            # 验证集评估（如果有）
            if val_loader is not None:
                X_val, y_val = self.extract_features(val_loader)
                y_pred = self.xgb_classifier.predict(X_val)
                accuracy = np.mean(y_pred == y_val)
                print(f"XGBoost验证集准确率: {accuracy:.4f}")
                return accuracy
        
        return None
    
    def predict(self, dataloader):
        """预测"""
        X, y_true = self.extract_features(dataloader)
        
        if self.use_ensemble:
            y_pred = self.ensemble.predict(X)
        else:
            y_pred = self.xgb_classifier.predict(X)
        
        return y_pred, y_true
    
    def predict_proba(self, dataloader):
        """预测概率"""
        X, y_true = self.extract_features(dataloader)
        
        if self.use_ensemble:
            if hasattr(self.ensemble, 'predict_proba'):
                y_proba = self.ensemble.predict_proba(X)
            else:
                # 如果没有概率预测方法，使用硬投票
                y_pred = self.ensemble.predict(X)
                y_proba = np.zeros((len(y_pred), 2))
                for i, pred in enumerate(y_pred):
                    y_proba[i, pred] = 1.0
        else:
            y_proba = self.xgb_classifier.predict_proba(X)
        
        return y_proba, y_true
    
    def get_classifier_weights(self):
        """获取分类器权重（仅对多数投票集成有效）"""
        if self.use_ensemble and self.ensemble_method == 'majority_voting':
            weights = {}
            for i, (name, _) in enumerate(self.classifiers.items()):
                weights[name] = self.ensemble.weights[i]
            return weights
        else:
            return {"single_classifier": 1.0}
    
    def evaluate_individual_classifiers(self, dataloader):
        """评估集成中每个单独分类器的性能"""
        if not self.use_ensemble:
            return {"xgb": None}  # 只有单个分类器
        
        X, y_true = self.extract_features(dataloader)
        individual_accuracies = {}
        
        for name, classifier in self.classifiers.items():
            y_pred = classifier.predict(X)
            accuracy = np.mean(y_pred == y_true)
            individual_accuracies[name] = accuracy
        
        # 添加集成性能
        if self.ensemble_method == 'majority_voting':
            y_pred_ensemble = self.ensemble.predict(X)
        else:
            y_pred_ensemble = self.ensemble.predict(X)
        
        ensemble_accuracy = np.mean(y_pred_ensemble == y_true)
        individual_accuracies['ensemble'] = ensemble_accuracy
        
        return individual_accuracies
    
    def save(self, path):
        """保存整个系统"""
        # 保存特征提取器
        torch.save(self.feature_extractor.state_dict(), f"{path}_feature_extractor.pth")
        
        # 保存分类器
        if self.use_ensemble:
            if self.ensemble_method == 'majority_voting':
                # 保存每个分类器
                for name, classifier in self.classifiers.items():
                    if hasattr(classifier, 'save'):
                        classifier.save(f"{path}_{name}")
                    else:
                        # 对于sklearn分类器
                        joblib.dump(classifier, f"{path}_{name}.pkl")
                # 保存集成权重
                ensemble_info = {
                    'weights': self.ensemble.weights,
                    'classifier_names': list(self.classifiers.keys())
                }
                joblib.dump(ensemble_info, f"{path}_ensemble_info.pkl")
            else:
                # 保存sklearn VotingClassifier
                joblib.dump(self.ensemble, f"{path}_ensemble.pkl")
        else:
            # 保存单个XGBoost分类器
            self.xgb_classifier.save(path)
        
    def load(self, path):
        """加载整个系统"""
        # 加载特征提取器
        self.feature_extractor.load_state_dict(torch.load(f"{path}_feature_extractor.pth"))
        
        # 加载分类器
        if self.use_ensemble:
            if self.ensemble_method == 'majority_voting':
                # 加载每个分类器
                ensemble_info = joblib.load(f"{path}_ensemble_info.pkl")
                for name in ensemble_info['classifier_names']:
                    if name in self.classifiers:
                        if hasattr(self.classifiers[name], 'load'):
                            self.classifiers[name].load(f"{path}_{name}")
                        else:
                            # 对于sklearn分类器
                            self.classifiers[name] = joblib.load(f"{path}_{name}.pkl")
                # 重新创建集成
                self.ensemble = MajorityVotingEnsemble()
                for name, weight in zip(ensemble_info['classifier_names'], ensemble_info['weights']):
                    self.ensemble.add_classifier(self.classifiers[name], weight)
            else:
                # 加载sklearn VotingClassifier
                self.ensemble = joblib.load(f"{path}_ensemble.pkl")
        else:
            # 加载单个XGBoost分类器
            self.xgb_classifier.load(path)
        
        return self


def train_model(model, train_loader, val_loader, epochs: int = 30, lr: float = 5e-5,
                use_xpu: bool = True, early_stop_patience: int = 10,  # 早停patience从5→3
                save_path: str = None):
    """
    训练模型并支持XPU加速、混合精度和早停机制
    """
    device = get_device(use_xpu)
    print(f"使用设备: {device.type}")

    model = model.to(dtype=torch.float32, device=device)

    # 优化器与调度器
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=1e-5  # 增加权重衰减抑制过拟合
    )
     # 学习率调度改为余弦退火
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs  # 随epoch余弦衰减
    )
    # 混合精度配置
    scaler = torch.amp.GradScaler() if device.type in ("cuda", "xpu") else None
    if device.type == "xpu" and has_ipex:
        model, optimizer = ipex.optimize(model, optimizer=optimizer)
    
    best_val_accuracy = 0.0
    early_stop_counter = 0
    
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        # 训练循环
        for batch_idx, (data, labels) in enumerate(train_loader):
            # 数据预处理
            if data.dim() == 4:
                data = data.permute(0, 2, 1, 3)
                data = data.reshape(data.shape[0], data.shape[1], data.shape[2] * data.shape[3])
            elif data.dim() == 3:
                if data.shape[1] == 12 and data.shape[2] == 1000:
                    pass
                elif data.shape[1] == 1000 and data.shape[2] == 12:
                    data = data.permute(0, 2, 1)
                else:
                    data = data.permute(0, 2, 1)
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # 前向传播与损失计算
            with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
                logits, attn_weights, decouple_loss, loss_components = model(data, epoch=epoch)
                total_loss, loss_details = model.compute_loss(
                    logits, labels, decouple_loss, attn_weights, loss_components, epoch
                )
            
            # 反向传播与优化
            if scaler is not None:
                scaler.scale(total_loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 关键！
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # 统计训练指标
            train_loss += total_loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 打印批次信息
            if (batch_idx + 1) % 10 == 0:
                print(f'[Epoch {epoch+1}/{epochs}] Batch {batch_idx+1}, '
                      f'Loss: {total_loss.item():.4f}, Cls Loss: {loss_details["classification"]:.4f}, '
                      f'Dec Loss: {loss_details.get("decouple", 0):.4f}')
        
        # 训练集指标
        train_accuracy = 100 * correct / total
        avg_train_loss = train_loss / len(train_loader)
        epoch_time = time.time() - start_time
        print(f'\nEpoch {epoch+1} 耗时: {epoch_time:.2f}秒')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
        
        # 验证集评估
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in val_loader:
                if data.dim() == 4:
                    data = data.permute(0, 2, 1, 3)
                    data = data.reshape(data.shape[0], data.shape[1], data.shape[2] * data.shape[3])
                elif data.dim() == 3:
                    if data.shape[1] == 12 and data.shape[2] == 1000:
                        pass
                    elif data.shape[1] == 1000 and data.shape[2] == 12:
                        data = data.permute(0, 2, 1)
                    else:
                        data = data.permute(0, 2, 1)
                data, labels = data.to(device), labels.to(device)
                
                with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
                    logits, attn_weights, decouple_loss, loss_components = model(data, epoch=epoch)
                    total_loss, _ = model.compute_loss(
                        logits, labels, decouple_loss, attn_weights, loss_components, epoch
                    )
                
                val_loss += total_loss.item()
                _, predicted = torch.max(logits.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 验证集指标
        val_accuracy = 100 * correct_val / total_val
        avg_val_loss = val_loss / len(val_loader)
        print(f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
        
        # 计算混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        print(f'混淆矩阵:\n{cm}\n')
        
        # 学习率调度
        scheduler.step(avg_val_loss)
        
        # 保存最佳模型
        save_dir = os.path.dirname(save_path)
        if save_dir:  # 如果存在父目录
            os.makedirs(save_dir, exist_ok=True)
    
        torch.save(model.state_dict(), save_path)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            if save_path is not None:
                torch.save(model.state_dict(), save_path)
                print(f'已保存最佳模型 (准确率: {best_val_accuracy:.2f}%)\n')
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print(f'早停触发: 连续{early_stop_patience}轮无提升')
                break
    
    return model, best_val_accuracy


# 导出所有需要的类和函数
__all__ = [
    'EnhancedCardiovascularModel',
    'HybridCardiovascularSystem',
    'XGBoostClassifier',
    'MajorityVotingEnsemble',
    'EnhancedFeatureDecoupler',
    'create_dataloader',
    'train_model',
    'get_device'
]