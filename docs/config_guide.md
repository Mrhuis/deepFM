# 配置管理指南

本文档介绍如何使用配置管理系统来管理推荐系统的各项参数。

## 配置文件格式

系统使用JSON格式的配置文件，示例如下：

```json
{
    "DB_CONFIG": {
        "host": "localhost",
        "database": "lrp",
        "user": "root",
        "password": "13579"
    },
    "TRAINING_SETS_DIR": "trainingSets",
    "TRAINING_MODELS_DIR": "models",
    "MODEL_CONFIG_DIR": "featureInfoConfigs",
    "USER_FEATURE_DIR": "userFeatures",
    "CONTENT_FEATURE_DIR": "contentFeatures",
    "USER_RECOMMEND_RESOURCE_DIR": "userRecommendResources",
    "HIGH_MATCHING_CANDIDATE_RESOURCE_COUNT": 50,
    "RECOMMEND_USED_RESOURCES": -1,
    "USED_MODEL": "DEFAULT",
    "CONTINUOUS_COLS": [
        "knowledge_accuracy",
        "knowledge_total_count",
        "resource_preference",
        "post_3d_correct_rate",
        "post_practice_count",
        "watch_rate",
        "active_days",
        "resource_time"
    ],
    "CATEGORICAL_COLS": [
        "resource_form",
        "resource_knowledges",
        "resource_tags",
        "is_first_submit_24h",
        "is_complete",
        "is_correct",
        "is_view_analysis",
        "is_pause",
        "is_replay"
    ],
    "TARGET_COL": "correct_rate_change",
    "DEFAULT_TRAIN_NUM": 2000,
    "BATCH_SIZE": 32,
    "EMBEDDING_DIM": 16,
    "DNN_HIDDEN": [
        128,
        64,
        32
    ],
    "DROPOUT": 0.2,
    "EPOCHS": 50,
    "LR": 0.001,
    "DEVICE": "cuda"
}
```

## 使用方法

### 1. 通过配置文件启动系统

```bash
python main.py --config-file system_config.json
```

### 2. 通过命令行参数覆盖配置

```bash
python main.py --train-num 3000 --batch-size 64 --epochs 100
```

### 3. 通过API动态修改配置

获取当前配置：
```
GET /config
```

更新配置：
```
PUT /config
Content-Type: application/json

{
    "BATCH_SIZE": 64,
    "EPOCHS": 100,
    "DROPOUT": 0.3
}
```

导出当前配置：
```
GET /config/export
```

导入配置：
```
POST /config/import
Content-Type: application/json

{
    "config_file": "system_config.json"
}
```

## 配置项说明

### 数据配置
- `DEFAULT_TRAIN_NUM`: 训练数据条数，默认2000
- `BATCH_SIZE`: 批次大小，默认32

### 模型配置
- `EMBEDDING_DIM`: 类别特征嵌入维度，默认16
- `DNN_HIDDEN`: 隐藏层结构，默认[128, 64, 32]
- `DROPOUT`: Dropout比例，默认0.2

### 训练配置
- `EPOCHS`: 训练轮数，默认50
- `LR`: 学习率，默认0.001
- `DEVICE`: 训练设备，默认自动检测是否有CUDA

### 其他配置
- `HIGH_MATCHING_CANDIDATE_RESOURCE_COUNT`: 高匹配备选资源个数，默认50
- `RECOMMEND_USED_RESOURCES`: 是否推荐使用过的资源，默认-1（不推荐使用过的）
- `USED_MODEL`: 使用的模型，默认"DEFAULT"