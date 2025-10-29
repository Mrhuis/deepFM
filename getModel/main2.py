import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict

# 添加项目根目录到sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import TRAINING_SETS_DIR,TRAINING_MODELS_DIR,DEFAULT_TRAIN_NUM,BATCH_SIZE,EMBEDDING_DIM,DNN_HIDDEN,DROPOUT,EPOCHS,LR,DEVICE
from getModel.DeepFM import DeepFM
from getModel.DeepFMDataset import DeepFMDataset
from datetime import datetime

def train_deepfm(
        csv_path: str ,
        train_num: Optional[int]
):
    # 加载数据
    print(f"[1/4] 加载数据：{csv_path}，训练条数：{train_num}")
    dataset = DeepFMDataset(csv_path, train_num)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"[数据信息] 连续特征维度：{dataset.continuous_dim}，类别特征数：{dataset.categorical_dim}，设备：{DEVICE}")

    # 初始化模型
    model = DeepFM(
        continuous_dim=dataset.continuous_dim,
        cat_dims=dataset.cat_dims,
        EMBEDDING_DIM=EMBEDDING_DIM,
        DNN_HIDDEN=DNN_HIDDEN,
        DROPOUT=DROPOUT
    ).to(DEVICE)

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 训练循环
    print(f"[2/4] 开始训练（共{EPOCHS}轮）")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch_cont, batch_cat, batch_y in dataloader:
            # 前向传播
            pred = model(batch_cont, batch_cat)
            # 计算损失
            loss = criterion(pred, batch_y)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 累计损失
            total_loss += loss.item() * batch_cont.shape[0]

        # 打印日志
        avg_loss = total_loss / len(dataset)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[Epoch {epoch + 1}/{EPOCHS}] 平均MSE损失：{avg_loss:.6f}")
        else:
            print(f"[Epoch {epoch + 1}/{EPOCHS}] 平均MSE损失：{avg_loss:.6f}", end="\r")

    # 保存模型
    current_date = datetime.now().strftime("%Y%m%d")  # 格式化日期为"YYYYMMDD"
    model_filename = f"{current_date}.pth"  # 构造CSV文件名

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    target_folder_path = os.path.join(base_dir, TRAINING_MODELS_DIR)
    save_path = os.path.join(target_folder_path, model_filename)

    torch.save(model.state_dict(), save_path)
    print(f"[4/4] 训练完成！模型保存至：{save_path}")


if __name__ == "__main__":
    train_deepfm(
        csv_path=TRAINING_SETS_DIR,
        train_num=DEFAULT_TRAIN_NUM
    )