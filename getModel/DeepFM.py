import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict

# 添加项目根目录到sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import DEVICE

class DeepFM(nn.Module):
    def __init__(
            self,
            continuous_dim: int,  #连续特征的维度数量，如果有8个连续特征（如知识准确率、总练习数、资源偏好等），则该值为8
            cat_dims: Dict[str, int],  # 每个类别特征的维度，类别特征的维度字典，键为特征名，值为该特征的类别数量
            EMBEDDING_DIM: int,  #嵌入向量的维度大小，如果设置为16，则每个类别特征的每个取值都会被表示为一个16维的向量
            DNN_HIDDEN: List[int],  #DNN（深度神经网络）隐藏层的结构配置
            DROPOUT: float  #Dropout层的丢弃率，用于防止过拟合
    ):
        super().__init__()
        self.continuous_dim = continuous_dim
        self.cat_dims = cat_dims
        self.EMBEDDING_DIM = EMBEDDING_DIM

        # 1. 连续特征一阶线性层
        #输入维度：continuous_dim（连续特征的数量）
        #输出维度：1（因为最终要预测一个数值）
        #这是因子分解机(FM)
        #中线性部分的实现，用于捕捉每个连续特征的独立影响
        self.continuous_first = nn.Linear(continuous_dim, 1)

        # 2. 类别特征一阶线性层（关键修正：确保输出形状对齐）
        #使用ModuleDict为每个类别特征创建一个Embedding层
        #每个Embedding层的输出维度是1，表示该类别特征的权重
        #例如：如果resource_form有10个类别，就创建一个Embedding(10, 1)层
        self.categorical_first = nn.ModuleDict({
            col: nn.Embedding(cat_dim, 1)  # 嵌入维度为1，输出形状[batch, len, 1]
            for col, cat_dim in cat_dims.items()
        })

        # 3. 类别特征嵌入层（用于二阶交互和DNN）
        #为每个类别特征创建一个高维Embedding层（维度为EMBEDDING_DIM）
        #这些Embedding向量用于：
        #FM部分的二阶特征交互计算
        #DNN部分的深度神经网络输入
        self.categorical_embedding = nn.ModuleDict({
            col: nn.Embedding(cat_dim, EMBEDDING_DIM)
            for col, cat_dim in cat_dims.items()
        })
        # 初始化嵌入层
        for emb in self.categorical_embedding.values():
            nn.init.xavier_uniform_(emb.weight)

        # 4. DNN层（输入：连续特征 + 所有类别特征的嵌入向量）
        # 计算DNN输入维度：连续特征维度 + 每个类别特征的嵌入维度之和
        cat_emb_total_dim = sum([EMBEDDING_DIM for _ in cat_dims])
        dnn_input_dim = continuous_dim + cat_emb_total_dim
        self.dnn = self._build_dnn(dnn_input_dim, DNN_HIDDEN, DROPOUT)

        # 5. 输出层（融合FM和DNN结果）
        #输入维度：DNN_HIDDEN[-1] + 1（DNN的输出维度 + FM部分的输出维度）
        #输出维度：1（最终的预测值）
        #将FM部分和DNN部分的结果融合，产生最终预测
        self.output = nn.Linear(DNN_HIDDEN[-1] + 1, 1)

    # 构建DNN
    def _build_dnn(self, in_dim: int, hidden_units: List[int], DROPOUT: float) -> nn.Sequential:
        layers = []
        current_dim = in_dim

        for units in hidden_units:
            layers.append(nn.Linear(current_dim, units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(DROPOUT))
            current_dim = units
        return nn.Sequential(*layers)

    # 前向传播
    # 定义前向传播方法（每个隐藏层有个输出方法），接收连续特征张量和类别特征字典，返回预测结果张量。
    def forward(
            self,
            continuous: torch.Tensor,  # 连续特征 (BATCH_SIZE, continuous_dim)
            categorical: Dict[str, torch.Tensor]  # 类别特征字典 (每个值形状[BATCH_SIZE, feature_len])
    ) -> torch.Tensor:
        #获取批次大小，即一次处理多少个样本。continuous.shape[0]表示张量第一维的大小。
        BATCH_SIZE = continuous.shape[0]

        # -------------------------- FM一阶部分（核心修正：形状对齐） --------------------------
        # 1. 连续特征一阶输出：(BATCH_SIZE, 1)
        fm1_continuous = self.continuous_first(continuous)  # 形状[batch, 1]

        # 2. 类别特征一阶输出：每个类别特征先求和压缩为[batch, 1]，再累加
        fm1_categorical = torch.zeros((BATCH_SIZE, 1), device=DEVICE)  # 初始化形状[batch, 1]
        for col in categorical:
            # 步骤1：类别特征通过嵌入层，输出形状[batch, feature_len, 1]
            cat_emb = self.categorical_first[col](categorical[col])
            # 步骤2：在特征长度维度（dim=1）求和，压缩为[batch, 1]
            cat_sum = torch.sum(cat_emb, dim=1)  # 关键修正：求和后形状[batch, 1]
            # 步骤3：累加所有类别特征的一阶输出
            fm1_categorical += cat_sum

        # FM一阶总输出：(BATCH_SIZE, 1)
        fm1_out = fm1_continuous + fm1_categorical  # 形状一致，可直接相加

        # -------------------------- FM二阶部分（修正嵌入向量堆叠逻辑） --------------------------
        # 1. 获取每个类别特征的嵌入向量，并压缩为[batch, EMBEDDING_DIM]
        cat_emb_list = []
        for col in categorical:
            # 嵌入输出形状[batch, feature_len, EMBEDDING_DIM]
            emb = self.categorical_embedding[col](categorical[col])
            # 在特征长度维度求和，压缩为[batch, EMBEDDING_DIM]
            emb_sum = torch.sum(emb, dim=1)  # 形状[batch, EMBEDDING_DIM]
            cat_emb_list.append(emb_sum)

        # 2. 计算二阶交互：仅对类别特征的嵌入向量进行交互
        if len(cat_emb_list) == 0:
            fm2_out = torch.zeros((BATCH_SIZE, 1), device=DEVICE)
        else:
            # 堆叠所有类别特征的嵌入向量：[batch, num_cat, EMBEDDING_DIM]
            cat_emb_stack = torch.stack(cat_emb_list, dim=1)
            # FM二阶公式：0.5 * sum[(sum(emb))² - sum(emb²)]
            sum_emb = torch.sum(cat_emb_stack, dim=1)  # [batch, EMBEDDING_DIM]
            square_sum = torch.square(sum_emb)  # [batch, EMBEDDING_DIM]
            sum_square = torch.sum(torch.square(cat_emb_stack), dim=1)  # [batch, EMBEDDING_DIM]
            fm2_out = 0.5 * torch.sum(square_sum - sum_square, dim=1, keepdim=True)  # [batch, 1]

        # FM总输出：(BATCH_SIZE, 1)
        fm_out = fm1_out + fm2_out

        # -------------------------- DNN部分（修正输入拼接逻辑） --------------------------
        # 1. 类别嵌入向量拼接：[batch, total_emb_dim]
        cat_emb_flat = torch.cat(cat_emb_list, dim=1)  # 关键修正：用cat替代flatten
        # 2. 拼接连续特征和类别嵌入：[batch, continuous_dim + total_emb_dim]
        dnn_in = torch.cat([continuous, cat_emb_flat], dim=1)
        # 3. DNN前向传播：[batch, DNN_HIDDEN[-1]]
        dnn_out = self.dnn(dnn_in)

        # -------------------------- 最终输出 --------------------------
        fusion = torch.cat([fm_out, dnn_out], dim=1)  # [batch, 1 + DNN_HIDDEN[-1]]
        final_out = self.output(fusion)  # [batch, 1]

        return final_out