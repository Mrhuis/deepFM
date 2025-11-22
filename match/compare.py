import torch
import numpy as np
import json
import os
from typing import Dict
from getModel.DeepFM import DeepFM
from config import DEVICE, EMBEDDING_DIM, DNN_HIDDEN, DROPOUT, CONTINUOUS_COLS, CATEGORICAL_COLS
from config import MODEL_CONFIG_DIR, TRAINING_MODELS_DIR,USED_MODEL
from datetime import datetime
from modelsContainer import TrainingSet
import glob

class ModelPredictor:
    def __init__(self, model_path: str = None):
        """
        初始化模型预测器
        
        Args:
            model_path: 训练好的模型文件路径
        """
        # 从特征信息文件中加载特征维度信息
        current_date = datetime.now().strftime("%Y%m%d")  # 格式化日期为"YYYYMMDD"
        feature_info_filename = f"{current_date}.json"  # 构造json文件名
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        target_folder_path = os.path.join(base_dir, MODEL_CONFIG_DIR)
        feature_info_path = os.path.join(target_folder_path, feature_info_filename)

        
        if os.path.exists(feature_info_path):
            with open(feature_info_path, 'r', encoding='utf-8') as f:
                feature_info = json.load(f)
            self.cat_dims = feature_info["cat_dims"]
            self.continuous_dim = feature_info["continuous_dim"]
            # 加载训练时目标变量的最值信息
            self.target_min = feature_info.get("target_min", -1.0)
            self.target_max = feature_info.get("target_max", 1.0)
            # 加载训练时连续特征的均值和标准差
            self.continuous_mean = feature_info.get("continuous_mean", None)
            self.continuous_std = feature_info.get("continuous_std", None)
        else:
            # 如果特征信息文件不存在，查找最新的特征文件
            feature_files = glob.glob(os.path.join(target_folder_path, "*.json"))
            if feature_files:
                # 按文件名排序，获取最新的特征文件
                latest_feature_file = max(feature_files, key=os.path.getctime)
                print(f"未找到当日特征文件，使用最新特征文件: {latest_feature_file}")
                with open(latest_feature_file, 'r', encoding='utf-8') as f:
                    feature_info = json.load(f)
                self.cat_dims = feature_info["cat_dims"]
                self.continuous_dim = feature_info["continuous_dim"]
                # 加载训练时目标变量的最值信息
                self.target_min = feature_info.get("target_min", -1.0)
                self.target_max = feature_info.get("target_max", 1.0)
                # 加载训练时连续特征的均值和标准差
                self.continuous_mean = feature_info.get("continuous_mean", None)
                self.continuous_std = feature_info.get("continuous_std", None)
            else:
                # 如果没有任何特征信息文件，使用默认值并给出警告
                print("警告：未找到任何特征信息文件，使用默认配置")
                self.cat_dims = {
                    "resource_form": 10,        # 资源形式类别数
                    "resource_knowledges": 2,   # 知识点相关性（0/1）
                    "resource_tags": 2,         # 标签相关性（0/1）
                    "is_first_submit_24h": 2,   # 是否24小时内首次提交（0/1）
                    "is_complete": 2,           # 是否完成（0/1）
                    "is_correct": 3,            # 是否正确（-1/0/1）
                    "is_view_analysis": 2,      # 是否观看解析（0/1）
                    "is_pause": 2,              # 是否暂停（0/1）
                    "is_replay": 2              # 是否重播（0/1）
                }
                self.continuous_dim = len(CONTINUOUS_COLS)
                # 使用训练数据中的实际范围作为默认值
                self.target_min = -1.0
                self.target_max = 1.0
                # 没有保存的统计信息
                self.continuous_mean = None
                self.continuous_std = None
        
        # 初始化模型
        self.model = DeepFM(
            continuous_dim=self.continuous_dim,
            cat_dims=self.cat_dims,
            EMBEDDING_DIM=EMBEDDING_DIM,
            DNN_HIDDEN=DNN_HIDDEN,
            DROPOUT=DROPOUT
        ).to(DEVICE)
        
        # 加载预训练模型权重
        # 如果没有提供模型路径，则尝试使用默认路径
        if not model_path:
            # 检查USED_MODEL配置
            if USED_MODEL == "DEFAULT":
                # 保持原有逻辑
                model_filename = f"{current_date}.pth"
                model_dir = os.path.join(base_dir, TRAINING_MODELS_DIR)
                model_path = os.path.join(model_dir, model_filename)
            else:
                # 使用USED_MODEL作为文件名
                model_filename = f"{USED_MODEL}.pth"
                model_dir = os.path.join(base_dir, TRAINING_MODELS_DIR)
                model_path = os.path.join(model_dir, model_filename)
            
        if os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print(f"警告：未找到模型文件 {model_path}，将在 {TRAINING_MODELS_DIR} 目录中查找最新模型...")
            # 在模型目录中查找最新日期命名的模型文件
            model_files = glob.glob(os.path.join(model_dir, "*.pth"))
            if model_files:
                # 按文件名排序，获取最新的模型文件
                latest_model_file = max(model_files, key=os.path.getctime)
                print(f"使用最新模型文件: {latest_model_file}")
                self.load_model(latest_model_file)
            else:
                print(f"警告：{TRAINING_MODELS_DIR} 目录中未找到任何模型文件，将使用随机初始化的模型权重")
    
    def load_model(self, model_path: str):
        """
        加载预训练模型
        
        Args:
            model_path: 模型文件路径
        """
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        self.model.eval()
    
    def predict(self, training_set: TrainingSet) -> float:
        """
        根据训练集数据预测 correct_rate_change 值
        
        Args:
            training_set: 包含所有特征的 TrainingSet 对象
            
        Returns:
            float: 预测的 correct_rate_change 值，在-1到1范围内
        """
        # 将输入数据转换为模型所需的格式
        continuous_tensor, categorical_dict = self._prepare_input(training_set)
        
        # 使用模型进行预测
        with torch.no_grad():
            prediction = self.model(continuous_tensor, categorical_dict)
            # 将结果从张量转换为浮点数
            raw_result = float(prediction.item())
            
            # 将预测值映射到-1到1范围内
            # 使用训练时目标变量的统计信息进行反归一化
            mapped_result = self._map_to_range(raw_result)
            
            # 打印预测值用于调试
            # print(f"原始预测值: {raw_result:.4f}, 映射后: {mapped_result:.4f}")
            # 如果预测值异常（绝对值过大），给出警告
            if abs(raw_result) > 10:
                print(f"警告：预测值异常大: {raw_result}")
            return mapped_result
    
    def _map_to_range(self, value: float) -> float:
        """
        将预测值映射到-1到1范围内
        
        Args:
            value: 原始预测值
            
        Returns:
            float: 映射到-1到1范围内的值
        """
        # 使用训练时目标变量的最值信息进行映射
        # 如果最值信息有效，则进行映射
        if self.target_min < self.target_max:
            # 先将值限制在训练时目标变量的范围内
            clamped_value = max(self.target_min, min(self.target_max, value))
            # 线性映射到-1到1范围
            normalized_value = (clamped_value - self.target_min) / (self.target_max - self.target_min)
            mapped_value = normalized_value * 2 - 1  # 映射到-1到1
            return mapped_value
        else:
            # 如果最值信息无效，则直接返回clamp到-1到1的值
            return max(-1.0, min(1.0, value))
    
    def _prepare_input(self, training_set: TrainingSet) -> tuple:
        """
        将 TrainingSet 对象转换为模型输入格式
        
        Args:
            training_set: TrainingSet 对象
            
        Returns:
            tuple: (continuous_tensor, categorical_dict)
        """
        # 处理连续特征
        # 根据模型期望的维度(continuous_dim)来构造输入特征
        continuous_features_by_col = []
        
        for col in CONTINUOUS_COLS:
            value = getattr(training_set, col, 0)
            # 如果值是列表，直接使用列表中的值
            if isinstance(value, list):
                continuous_features_by_col.append([float(v) for v in value])
            else:
                # 单个值也要作为列表处理
                continuous_features_by_col.append([float(value)])
        
        # 对每列分别进行归一化（使用训练时的统计信息）
        normalized_features = []
        for col_idx, col_features in enumerate(continuous_features_by_col):
            for val in col_features:
                # 使用训练时保存的均值和标准差进行归一化
                if self.continuous_mean is not None and self.continuous_std is not None and col_idx < len(self.continuous_mean):
                    mean = self.continuous_mean[col_idx]
                    std = self.continuous_std[col_idx]
                    # 如果标准差为0，说明该列所有值相同，不需要归一化，直接减去均值
                    if std == 0:
                        normalized_val = val - mean
                    else:
                        normalized_val = (val - mean) / std
                else:
                    # 如果没有保存的统计信息，使用原始值（不推荐）
                    normalized_val = val
                    if col_idx == 0:  # 只打印一次警告
                        print("警告：未找到训练时保存的均值和标准差，使用原始值进行预测，结果可能不准确")
                normalized_features.append(normalized_val)
        
        # 如果特征数量不足continuous_dim，用0填充；如果超过，则截断
        if len(normalized_features) < self.continuous_dim:
            normalized_features.extend([0.0] * (self.continuous_dim - len(normalized_features)))
        elif len(normalized_features) > self.continuous_dim:
            normalized_features = normalized_features[:self.continuous_dim]
        
        # 转换为张量
        continuous_array = np.array(normalized_features, dtype=np.float32)
        continuous_tensor = torch.tensor(continuous_array, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        
        # 处理类别特征
        # 模型期望每个类别特征的形状为 [BATCH_SIZE, feature_len]，这里BATCH_SIZE为1
        categorical_dict = {}
        for col in CATEGORICAL_COLS:
            value = getattr(training_set, col, 0)
            # 如果值是列表，转换为numpy数组再转为张量
            if isinstance(value, list):
                if len(value) > 0:
                    # 确保转换为numpy数组后再创建张量，避免性能警告
                    np_array = np.array(value, dtype=np.int64)
                    # 形状应为 [1, len(value)]，表示batch_size=1，feature_len=len(value)
                    categorical_dict[col] = torch.from_numpy(np_array).unsqueeze(0).to(DEVICE)
                else:
                    # 空列表用0填充
                    categorical_dict[col] = torch.tensor([[0]], dtype=torch.long, device=DEVICE)
            else:
                # 单个值需要扩展为二维张量 [1, 1]
                categorical_dict[col] = torch.tensor([[int(value)]], dtype=torch.long, device=DEVICE)
        
        return continuous_tensor, categorical_dict


def predict_correct_rate_change(training_set: TrainingSet, model_path: str = None) -> float:
    """
    根据训练集数据预测 correct_rate_change 值的便捷函数
    
    Args:
        training_set: 包含所有特征的 TrainingSet 对象
        model_path: 模型文件路径（可选）
        
    Returns:
        float: 预测的 correct_rate_change 值，在-1到1范围内
    """
    predictor = ModelPredictor(model_path)
    return predictor.predict(training_set)