# 数据模型类定义
from dataclasses import dataclass
from typing import Optional
import datetime


@dataclass
class ProcessedUserResourceInteraction:
    """用户资源交互记录处理后的数据模型"""
    id: int  # 主键ID
    user_key: str  # 用户标识
    form_key: str  # 资源类型标识
    resource_key: str  # 资源唯一标识
    post_3d_correct_rate: Optional[float]  # 交互后3天正确率
    post_practice_count: Optional[int]  # 交互后3天总练习量
    is_first_submit_24h: Optional[int]  # 24小时内是否首次提交
    correct_rate_change: Optional[float]  # 正确率变化（目标变量）
    is_complete: int  # 是否完成该题（习题特有）
    is_correct: int  # 是否做对该题（习题特有）
    is_view_analysis: Optional[int]  # 是否观看解析（习题特有）
    watch_rate: float  # 视频观看率（视频特有）
    is_pause: Optional[int]  # 是否暂停（视频特有）
    is_replay: Optional[int]  # 是否反复观看（视频特有）
    interaction_time: datetime.datetime  # 交互时间
    effect_calc_time: Optional[datetime.datetime]  # 滞后特征计算时间

@dataclass
class TrainingSet:
    """用户资源交互记录处理后的数据模型"""
    knowledge_accuracy: Optional[float]  # 各知识点正确率
    knowledge_total_count: Optional[int]  # 各知识点做题总数
    resource_preference: Optional[float]  # 近 7 天题型偏好
    active_days: int  # 用户活跃天数
    resource_form: int  # 资源形式
    resource_knowledges: Optional[int]  # 资源涉及知识(全都是0/getModel)
    resource_tags: Optional[int]  # 资源涉及标签(全都是0/getModel)
    resource_time: int  #视频时长，习题默认0
    post_3d_correct_rate: Optional[float]  # 交互后3天各知识点正确率
    post_practice_count: Optional[int]  # 交互后3天各个知识点总练习量
    is_first_submit_24h: Optional[int]  # 24小时内是否首次提交
    is_complete: int  # 是否完成该题（习题特有）
    is_correct: int  # 是否做对该题（习题特有）
    is_view_analysis: Optional[int]  # 是否观看解析（习题特有）
    watch_rate: float  # 视频观看率（视频特有）
    is_pause: Optional[int]  # 是否暂停（视频特有）
    is_replay: Optional[int]  # 是否反复观看（视频特有）
    correct_rate_change: float  # 有涉及的知识点正确率变化平均值（目标变量）

