import torch

# 数据库配置
DB_CONFIG = {
    "host": "localhost",
    "database": "lrp",
    "user": "root",
    "password": "13579"
}

# 训练集输出位置
TRAINING_SETS_DIR = "trainingSets"
# 训练集模型出位置
TRAINING_MODELS_DIR = "models"
# 模型特征配置文件位置
MODEL_CONFIG_DIR = "featureInfoConfigs"
# 用户动态特征文件存贮位置
USER_FEATURE_DIR = "userFeatures"
# 内容静态特征文件存贮位置
CONTENT_FEATURE_DIR = "contentFeatures"
# 用户推荐资源文件位置
USER_RECOMMEND_RESOURCE_DIR = "userRecommendResources"
# 高匹配备选资源个数
HIGH_MATCHING_CANDIDATE_RESOURCE_COUNT = 50
#是否推荐使用过后的资源-1：不使用过。0：无限制。1：使用过
RECOMMEND_USED_RESOURCES = -1
# 使用的模型
USED_MODEL = "DEFAULT"

#推荐个数（暂未启用）
# 是否需要每次推荐前重新判断是否需要再次调整匹配前后顺序（暂未启用）

# 特征类型定义（区分连续/类别特征）
CONTINUOUS_COLS = [
    "knowledge_accuracy", "knowledge_total_count", "resource_preference",
    "post_3d_correct_rate", "post_practice_count", "watch_rate", "active_days", "resource_time"
]  # 连续特征


CATEGORICAL_COLS = [
    "resource_form", "resource_knowledges", "resource_tags",
    "is_first_submit_24h", "is_complete", "is_correct",
    "is_view_analysis", "is_pause", "is_replay"
]  # 类别特征


TARGET_COL = "correct_rate_change"

# 数据配置
DEFAULT_TRAIN_NUM = 2000  # 训练数据条数,数据集个数
BATCH_SIZE = 32 # 批次大小：每次送入模型训练的样本数量



# 模型配置
# 类别特征嵌入维度，
# 用于将高维离散的类别特征（如is_correct、resource_tags）映射为低维连续向量的长度（此处为 16 维）。
# 避免维度爆炸，内存开销剧增，高维度下数据会呈现 “稀疏性”，过拟合严重，高维度下，有用特征的权重会被大量 “无效稀疏特征” 稀释。
EMBEDDING_DIM = 16
# 隐藏层结构，列表中每个数字代表对应隐藏层的神经元数量（此处为 3 层：128→64→32）。
DNN_HIDDEN = [128, 64, 32]
# 训练时随机 “关闭”（丢弃）神经元的比例（此处为 20%）。
# 提升预测新数据的泛化能力
DROPOUT = 0.2


# 训练配置
# 训练轮数
EPOCHS = 50
# 学习率
LR = 0.001
# 训练设备(gpu/cpu)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")