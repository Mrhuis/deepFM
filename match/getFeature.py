from modelsContainer import TrainingSet
import json
import os
from datetime import datetime
from config import CONTENT_FEATURE_DIR, USER_FEATURE_DIR
from getTrainingSet.utils.trainingSet_utils import (
    get_user_knowledge_stats, 
    get_user_resource_preference, 
    get_user_active_days,
    get_resource_form_by_resource_key,
    get_resource_knowledges,
    get_resource_tags,
    get_resource_time
)


def get_training_set_by_user_resource(user_key: str, resource_key: str) -> TrainingSet:
    """
    根据用户和资源信息获取训练集对象
    
    参数:
        user_key: 用户标识
        resource_key: 资源标识
        form_type: 资源类型
    
    返回:
        TrainingSet: 训练集对象
    """
    # 获取用户知识点统计数据
    knowledgeResult = get_user_knowledge_stats(user_key, datetime.now())
    
    # 获取用户资源偏好数据
    preferenceResult = get_user_resource_preference(user_key, datetime.now())
    form_type = get_resource_form_by_resource_key(resource_key)
    
    # 从contentFeatures目录的JSON文件中查找resource_key匹配的资源特征
    content_features = None
    current_date = datetime.now().strftime("%Y%m%d")
    
    # 构建content features文件路径
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    content_features_dir = os.path.join(base_dir, CONTENT_FEATURE_DIR)
    content_file_path = os.path.join(content_features_dir, f"{current_date}.json")
    
    # 从content features文件中查找匹配的资源特征
    if os.path.exists(content_file_path):
        try:
            with open(content_file_path, 'r', encoding='utf-8') as f:
                content_data = json.load(f)
                if isinstance(content_data, list):
                    for item in content_data:
                        if item.get("resource_key") == resource_key and item.get("form_type") == form_type:
                            content_features = item
                            break
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    
    # 如果未找到匹配的资源特征，则使用实时查询
    if content_features is None:
        resource_form = get_resource_form_by_resource_key(resource_key)
        resource_knowledges = get_resource_knowledges(resource_key)
        resource_tags = get_resource_tags(resource_key)
        resource_time = get_resource_time(resource_key)
    else:
        resource_form = content_features.get("form_type", 0)
        resource_knowledges = content_features.get("resource_knowledges", [])
        resource_tags = content_features.get("resource_tags", [])
        resource_time = content_features.get("resource_time", [])
    
    # 从userFeatures目录的JSON文件中查找user_key匹配的用户交互数据
    user_features = None
    user_features_dir = os.path.join(base_dir, USER_FEATURE_DIR)
    user_file_path = os.path.join(user_features_dir, f"{current_date}.json")
    
    # 从user features文件中查找匹配的用户特征
    if os.path.exists(user_file_path):
        try:
            with open(user_file_path, 'r', encoding='utf-8') as f:
                user_data = json.load(f)
                if isinstance(user_data, dict):
                    user_features = user_data.get(user_key)
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    
    # 如果未找到匹配的用户特征，则使用默认值
    if user_features is None:
        post_3d_correct_rate = [0.0] * len(knowledgeResult['accuracy'])  # 默认值
        post_practice_count = [0] * len(knowledgeResult['accuracy'])     # 默认值
    else:
        post_3d_correct_rate = user_features.get("post_3d_correct_rate", [])
        post_practice_count = user_features.get("post_practice_count", [])


    # 实例化数据模型
    return TrainingSet(
        knowledge_accuracy=knowledgeResult['accuracy'],
        knowledge_total_count=knowledgeResult['totalCount'],
        resource_preference=preferenceResult['preference'],
        active_days=get_user_active_days(user_key),
        resource_form=resource_form,
        resource_knowledges=resource_knowledges,
        resource_tags=resource_tags,
        resource_time=resource_time[0] if resource_time else 0,
        post_3d_correct_rate=post_3d_correct_rate,
        post_practice_count=post_practice_count,
        is_first_submit_24h=0,
        is_complete=0,
        is_correct=0,
        is_view_analysis=0,
        watch_rate=0.0,
        is_pause=0,
        is_replay=0,
        correct_rate_change=0.0
    )

