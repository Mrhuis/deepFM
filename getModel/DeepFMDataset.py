import os
import sys
import pandas as pd
import numpy as np
import torch
import json
from typing import Tuple, Optional, Dict

# 添加项目根目录到sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import CATEGORICAL_COLS,CONTINUOUS_COLS,TARGET_COL,DEVICE,MODEL_CONFIG_DIR,TRAINING_SETS_DIR
from datetime import datetime

class DeepFMDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path: str, train_num: Optional[int] = None):

        current_date = datetime.now().strftime("%Y%m%d")  # 格式化日期为"YYYYMMDD"
        csv_filename = f"{current_date}.csv"  # 构造CSV文件名

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        target_folder_path = os.path.join(base_dir, csv_path)
        full_csv_path = os.path.join(target_folder_path, csv_filename)

        self.df = pd.read_csv(full_csv_path, sep="|", dtype=str)
        if train_num is not None and train_num < len(self.df):
            self.df = self.df.head(train_num)

        # 预处理连续特征和类别特征
        self.continuous_data, self.categorical_data, self.cat_dims = self._preprocess_features()
        self.target_data, self.target_min, self.target_max = self._preprocess_target()

        # 记录特征维度
        self.continuous_dim = self.continuous_data.shape[1]
        self.categorical_dim = len(CATEGORICAL_COLS)
        
        # 保存特征维度信息到文件
        self._save_feature_info()

    def _preprocess_features(self) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, int]]:
        """预处理连续特征和类别特征"""
        # 1. 处理连续特征（多值字段拆分+归一化）
        continuous_list = []
        self.continuous_mean = []  # 保存每列的均值
        self.continuous_std = []   # 保存每列的标准差
        
        for col in CONTINUOUS_COLS:
            col_data = self.df[col].fillna("").apply(
                lambda x: [float(val) for val in x.split(",") if val.strip()] if x.strip() else []
            )
            max_len = max(len(vals) for vals in col_data) if col_data.any() else 1
            padded = np.array([vals + [0.0] * (max_len - len(vals)) for vals in col_data])
            
            # 计算并保存均值和标准差
            col_mean = padded.mean()
            col_std = padded.std()
            self.continuous_mean.append(col_mean)
            self.continuous_std.append(col_std)
            
            # 归一化：如果标准差为0（所有值相同），只减去均值
            if col_std == 0:
                padded = padded - col_mean
            else:
                padded = (padded - col_mean) / col_std
            continuous_list.append(padded)
        continuous_matrix = np.hstack(continuous_list)

        # 2. 处理类别特征
        categorical_dict = {}
        cat_dims = {}
        for col in CATEGORICAL_COLS:
            col_data = self.df[col].fillna("").apply(
                lambda x: [int(val) for val in x.split(",") if val.strip()] if x.strip() else []
            )
            max_len = max(len(vals) for vals in col_data) if col_data.any() else 1
            padded = np.array([vals + [0] * (max_len - len(vals)) for vals in col_data])
            max_val = padded.max() if len(padded) > 0 else 0
            cat_dims[col] = int(max_val + 1)  # 索引从0开始，维度需+1
            categorical_dict[col] = padded

        return continuous_matrix, categorical_dict, cat_dims

    def _preprocess_target(self) -> Tuple[np.ndarray, float, float]:
        """处理目标变量"""
        target_data = self.df[TARGET_COL].fillna("0").apply(
            lambda x: float(x) if x.strip() else 0.0
        ).values.reshape(-1, 1)
        
        # 获取目标变量的最小值和最大值，用于预测时的映射
        target_min = float(target_data.min())
        target_max = float(target_data.max())
        
        # 打印目标变量的统计信息，用于调试
        print(f"目标变量统计信息 - 最小值: {target_min:.4f}, 最大值: {target_max:.4f}, 平均值: {target_data.mean():.4f}")
        
        return target_data, target_min, target_max
        
    def _save_feature_info(self):
        """保存特征维度信息到文件"""
        current_date = datetime.now().strftime("%Y%m%d")  # 格式化日期为"YYYYMMDD"
        feature_info_filename = f"{current_date}.json"  # 构造json文件名
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        target_folder_path = os.path.join(base_dir, MODEL_CONFIG_DIR)
        feature_info_path = os.path.join(target_folder_path, feature_info_filename)
        
        feature_info = {
            "cat_dims": self.cat_dims,
            "continuous_dim": self.continuous_dim,
            "categorical_cols": CATEGORICAL_COLS,
            "continuous_cols": CONTINUOUS_COLS,
            "target_min": self.target_min,  # 保存目标变量最小值
            "target_max": self.target_max,  # 保存目标变量最大值
            "continuous_mean": [float(m) for m in self.continuous_mean],  # 保存连续特征的均值
            "continuous_std": [float(s) for s in self.continuous_std]     # 保存连续特征的标准差
        }
        
        with open(feature_info_path, 'w', encoding='utf-8') as f:
            json.dump(feature_info, f, ensure_ascii=False, indent=2)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """返回：连续特征、类别特征字典、目标变量"""
        continuous = torch.tensor(
            self.continuous_data[idx], dtype=torch.float32, device=DEVICE
        )
        categorical = {
            col: torch.tensor(
                self.categorical_data[col][idx], dtype=torch.long, device=DEVICE
            ) for col in CATEGORICAL_COLS
        }
        target = torch.tensor(
            self.target_data[idx], dtype=torch.float32, device=DEVICE
        )
        return continuous, categorical, target