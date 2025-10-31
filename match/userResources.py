import os
import json
import logging
from typing import List, Dict, Any
from datetime import datetime
from config import USER_RECOMMEND_RESOURCE_DIR, RECOMMEND_USED_RESOURCES, TRAINING_MODELS_DIR, \
    HIGH_MATCHING_CANDIDATE_RESOURCE_COUNT, DEVICE
from getTrainingSet.utils.db_utils import close_db_connection, get_db_connection
from match.compare import predict_correct_rate_change
from match.getFeature import get_training_set_by_user_resource

# 创建日志记录器
logger = logging.getLogger(__name__)

def get_all_user_keys() -> List[str]:
    """
    从数据库中获取所有用户标识(user_key)
    
    Returns:
        List[str]: 所有用户标识列表
    """
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 获取所有user_key
        cursor.execute("SELECT user_key FROM users")
        user_keys = [row[0] for row in cursor.fetchall()]
        
        return user_keys
    except Exception as e:
        print(f"获取用户标识时出错: {e}")
        return []
    finally:
        close_db_connection(conn, cursor)


def get_all_resource_keys() -> List[str]:
    """
    从数据库中获取所有资源标识(resource_key)
    
    Returns:
        List[str]: 所有资源标识列表
    """
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 获取所有item_key
        cursor.execute("SELECT item_key FROM items")
        item_keys = [row[0] for row in cursor.fetchall()]
        
        # 获取所有media_key
        cursor.execute("SELECT media_key FROM media_assets")
        media_keys = [row[0] for row in cursor.fetchall()]
        
        # 合并所有resource_key
        resource_keys = item_keys + media_keys
        return resource_keys
    except Exception as e:
        print(f"获取资源标识时出错: {e}")
        return []
    finally:
        close_db_connection(conn, cursor)


def generate_user_recommendations(model_name: str = None):
    """
    为每个用户生成推荐资源列表，包含预测的correct_rate_change值
    并将结果保存到以日期命名的文件夹中的JSON文件里
    
    Args:
        model_name: 可选的模型文件名，如果提供则使用该模型而不是默认模型
    """
    # 创建当日日期文件夹
    current_date_dir = datetime.now().strftime("%Y%m%d")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    training_sets_dir = os.path.join(base_dir, USER_RECOMMEND_RESOURCE_DIR)
    output_path = os.path.join(training_sets_dir, current_date_dir)
    
    # 如果目录不存在则创建
    os.makedirs(output_path, exist_ok=True)
    
    # 构造模型路径
    if model_name:
        model_path = os.path.join(base_dir, TRAINING_MODELS_DIR, model_name)
    else:
        model_path = os.path.join(base_dir, TRAINING_MODELS_DIR, f"{current_date_dir}.pth")
        
    if not os.path.exists(model_path):
        print(f"警告：模型文件 {model_path} 不存在，将使用随机初始化的模型权重")
        model_path = None
    
    # 获取所有用户和资源标识
    user_keys = get_all_user_keys()
    resource_keys = get_all_resource_keys()
    
    print(f"获取到 {len(user_keys)} 个用户和 {len(resource_keys)} 个资源")
    
    # 为每个用户生成推荐资源
    for i, user_key in enumerate(user_keys):
        print(f"处理用户 {i+1}/{len(user_keys)}: {user_key}")
        
        # 存储每个资源的预测结果 (correct_rate_change, resource_key, form_type)
        predictions = []
        
        # 为当前用户预测所有资源的correct_rate_change值
        for resource_key in resource_keys:
            try:
                # 获取训练集对象
                training_set = get_training_set_by_user_resource(user_key, resource_key)
                
                # 预测correct_rate_change值
                correct_rate_change = predict_correct_rate_change(training_set, model_path)
                
                # 获取资源类型
                form_type = training_set.resource_form
                
                # 添加到预测结果列表
                predictions.append((correct_rate_change, resource_key, form_type))
            except Exception as e:
                print(f"处理用户 {user_key} 和资源 {resource_key} 时出错: {e}")
                continue
        
        # 按correct_rate_change值降序排序，取前200个
        predictions.sort(key=lambda x: x[0], reverse=True)
        top_predictions = predictions[:HIGH_MATCHING_CANDIDATE_RESOURCE_COUNT]
        
        # 构造要保存的数据
        result_data = [
            {
                "resource_key": resource_key,
                "form_type": form_type,
                "correct_rate_change": correct_rate_change,
                "is_use": 0
            }
            for correct_rate_change, resource_key, form_type in top_predictions
        ]
        
        # 保存到以user_key命名的JSON文件中
        user_file_path = os.path.join(output_path, f"{user_key}.json")
        try:
            with open(user_file_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存用户 {user_key} 的推荐结果时出错: {e}")
            continue
    
    print("所有用户的推荐资源生成完成")


def generate_single_user_recommendations(user_key: str, directory_name: str = None, model_name: str = None, type_filter: int = 0):
    """
    为单个用户重新生成推荐资源列表，包含预测的correct_rate_change值
    并将结果保存到以日期命名的文件夹中的JSON文件里
    
    Args:
        user_key: 用户标识
        directory_name: 可选的目录名称，如果提供则使用该目录而不是最新目录
        model_name: 可选的模型文件名，如果提供则使用该模型而不是默认模型
        type_filter: 资源类型过滤器
            -1: 只计算is_use为0的资源
             0: 无条件计算所有资源（is_use为0和1的资源）
             1: 只计算is_use为1的资源
    """
    # 创建当日日期文件夹
    current_date_dir = datetime.now().strftime("%Y%m%d")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    training_sets_dir = os.path.join(base_dir, USER_RECOMMEND_RESOURCE_DIR)
    
    # 添加打印语句查看 DEVICE 的值
    print(f"当前使用的设备: {DEVICE}")
    
    # 如果提供了目录名称，使用指定目录，否则使用当前日期目录
    if directory_name:
        output_path = os.path.join(training_sets_dir, directory_name)
    else:
        output_path = os.path.join(training_sets_dir, current_date_dir)
    
    # 如果目录不存在则使用USER_RECOMMEND_RESOURCE_DIR下最新的文件夹
    if not os.path.exists(output_path):
        # 如果提供了目录名称但目录不存在，则创建该目录
        if directory_name:
            os.makedirs(output_path, exist_ok=True)
        else:
            # 获取USER_RECOMMEND_RESOURCE_DIR下的所有文件夹
            all_dirs = [d for d in os.listdir(training_sets_dir) if os.path.isdir(os.path.join(training_sets_dir, d))]
            if all_dirs:
                # 按名称排序，取最新的文件夹
                all_dirs.sort()
                latest_dir = all_dirs[-1]
                output_path = os.path.join(training_sets_dir, latest_dir)
                print(f"指定日期目录不存在，使用最新的目录: {output_path}")
            else:
                # 如果没有文件夹，创建当日目录
                os.makedirs(output_path, exist_ok=True)
    
    # 构造模型路径
    if model_name:
        model_path = os.path.join(base_dir, TRAINING_MODELS_DIR, model_name)
    else:
        model_path = os.path.join(base_dir, TRAINING_MODELS_DIR, f"{current_date_dir}.pth")
        
    if not os.path.exists(model_path):
        print(f"警告：模型文件 {model_path} 不存在，将使用随机初始化的模型权重")
        model_path = None
    
    # 获取用户当前的推荐资源列表
    user_file_path = os.path.join(output_path, f"{user_key}.json")
    
    # 如果用户文件不存在，尝试从S0000.json复制
    if not os.path.exists(user_file_path):
        s0000_file_path = os.path.join(output_path, "S0000.json")
        if os.path.exists(s0000_file_path):
            try:
                # 复制S0000.json文件并重命名为user_key.json
                with open(s0000_file_path, 'r', encoding='utf-8') as src_file:
                    user_recommendations = json.load(src_file)
                
                with open(user_file_path, 'w', encoding='utf-8') as dst_file:
                    json.dump(user_recommendations, dst_file, ensure_ascii=False, indent=2)
                    
                print(f"用户 {user_key} 的文件不存在，已从 S0000.json 复制并重命名为 {user_file_path}")
            except Exception as e:
                print(f"复制 S0000.json 文件时出错: {e}")
                user_recommendations = []
        else:
            # 如果S0000.json也不存在，则创建空列表
            user_recommendations = []
    else:
        # 读取现有的用户推荐资源文件
        try:
            with open(user_file_path, 'r', encoding='utf-8') as f:
                user_recommendations = json.load(f)
        except Exception as e:
            print(f"读取用户 {user_key} 的推荐资源文件时出错: {e}")
            user_recommendations = []
    
    # 根据type_filter参数过滤资源
    if type_filter == -1:
        # 只处理is_use为0的资源
        filtered_resources = [item for item in user_recommendations if item.get("is_use", 0) == 0]
        filter_desc = "is_use=0"
    elif type_filter == 1:
        # 只处理is_use为1的资源
        filtered_resources = [item for item in user_recommendations if item.get("is_use", 0) == 1]
        filter_desc = "is_use=1"
    else:
        # 处理所有资源 (type_filter == 0)
        filtered_resources = user_recommendations
        filter_desc = "all resources"
    
    # 提取资源标识
    resource_keys = [item["resource_key"] for item in filtered_resources]
    
    print(f"为用户 {user_key} 处理 {len(resource_keys)} 个资源 ({filter_desc})")
    
    # 存储每个资源的预测结果 (correct_rate_change, resource_key, form_type)
    predictions = []
    
    # 为当前用户预测过滤后资源的correct_rate_change值
    for item in filtered_resources:
        resource_key = item["resource_key"]
        form_type = item["form_type"]
        try:
            # 获取训练集对象
            training_set = get_training_set_by_user_resource(user_key, resource_key)
            
            # 预测correct_rate_change值
            correct_rate_change = predict_correct_rate_change(training_set, model_path)
            
            # 添加到预测结果列表
            predictions.append((correct_rate_change, resource_key, form_type))
        except Exception as e:
            print(f"处理用户 {user_key} 和资源 {resource_key} 时出错: {e}")
            # 保留原始值
            correct_rate_change = item.get("correct_rate_change", 0.0)
            predictions.append((correct_rate_change, resource_key, form_type))
            continue
    
    # 对于未被处理的资源，保留原始值
    unprocessed_predictions = []
    if type_filter == -1:
        # 保留is_use为1的资源
        unprocessed_resources = [item for item in user_recommendations if item.get("is_use", 0) == 1]
        for item in unprocessed_resources:
            resource_key = item["resource_key"]
            form_type = item["form_type"]
            correct_rate_change = item.get("correct_rate_change", 0.0)
            unprocessed_predictions.append((correct_rate_change, resource_key, form_type))
    elif type_filter == 1:
        # 保留is_use为0的资源
        unprocessed_resources = [item for item in user_recommendations if item.get("is_use", 0) == 0]
        for item in unprocessed_resources:
            resource_key = item["resource_key"]
            form_type = item["form_type"]
            correct_rate_change = item.get("correct_rate_change", 0.0)
            unprocessed_predictions.append((correct_rate_change, resource_key, form_type))
    # type_filter == 0 时不需要保留未处理的资源，因为所有资源都会被处理
    
    # 合并处理过的和未处理过的预测结果
    predictions.extend(unprocessed_predictions)
    
    # 按correct_rate_change值降序排序，取前200个
    predictions.sort(key=lambda x: x[0], reverse=True)
    top_predictions = predictions[:HIGH_MATCHING_CANDIDATE_RESOURCE_COUNT]
    
    # 构造要保存的数据
    result_data = [
        {
            "resource_key": resource_key,
            "form_type": form_type,
            "correct_rate_change": correct_rate_change,
            "is_use": 0  # 默认设置为0，保持与原始逻辑一致
        }
        for correct_rate_change, resource_key, form_type in top_predictions
    ]
    
    # 保持原有的is_use值
    for new_item in result_data:
        for old_item in user_recommendations:
            if new_item["resource_key"] == old_item["resource_key"] and new_item["form_type"] == old_item["form_type"]:
                new_item["is_use"] = old_item.get("is_use", 0)
                break
    
    # 保存到以user_key命名的JSON文件中
    try:
        with open(user_file_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        print(f"用户 {user_key} 的推荐结果已保存到 {user_file_path}")
    except Exception as e:
        print(f"保存用户 {user_key} 的推荐结果时出错: {e}")


def get_user_recommendations(user_key: str, directory_name: str = None, model_name: str = None) -> List[Dict[str, Any]]:
    """
    接收user_key,获取并返回6个其对应的json文件内前6个的资源信息的resource_key和form_type

    Args:
        user_key: 用户标识
        directory_name: 可选的目录名称，如果提供则使用该目录而不是最新目录

    Returns:
        List[Dict[str, Any]]: 包含resource_key和form_type的字典列表，最多返回6个
    """
    try:
        # 首先生成或更新用户的推荐资源
        if RECOMMEND_USED_RESOURCES == -1:
            # 只返回is_use为0的资源
            generate_single_user_recommendations(user_key, directory_name, model_name,RECOMMEND_USED_RESOURCES)
        elif RECOMMEND_USED_RESOURCES == 1:
            # 只返回is_use为1的资源
            generate_single_user_recommendations(user_key, directory_name, model_name,RECOMMEND_USED_RESOURCES)
        else:
            # 无限制，返回所有资源
            generate_single_user_recommendations(user_key, directory_name, model_name,RECOMMEND_USED_RESOURCES)



        # 获取推荐资源目录
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        recommend_dir = os.path.join(base_dir, USER_RECOMMEND_RESOURCE_DIR)
        
        # 如果提供了目录名称，使用指定目录
        if directory_name:
            target_dir = os.path.join(recommend_dir, directory_name)
            # 确保目录存在
            os.makedirs(target_dir, exist_ok=True)
            user_file_path = os.path.join(target_dir, f"{user_key}.json")
            
            # 如果用户文件不存在，尝试从同目录的S0000.json复制
            if not os.path.exists(user_file_path):
                default_file_path = os.path.join(target_dir, "S0000.json")
                if os.path.exists(default_file_path):
                    import shutil
                    shutil.copy(default_file_path, user_file_path)
        else:
            # 获取最新的日期目录
            date_dirs = [d for d in os.listdir(recommend_dir) if os.path.isdir(os.path.join(recommend_dir, d))]
            if not date_dirs:
                raise FileNotFoundError("未找到任何推荐资源目录")

            # 按名称排序，获取最新的日期目录
            date_dirs.sort()
            latest_date_dir = date_dirs[-1]
            user_file_path = os.path.join(recommend_dir, latest_date_dir, f"{user_key}.json")

            # 检查用户文件是否存在
            if not os.path.exists(user_file_path):
                # 如果用户文件不存在，尝试使用S0000.json作为默认文件
                default_file_path = os.path.join(recommend_dir, latest_date_dir, "S0000.json")
                if os.path.exists(default_file_path):
                    user_file_path = default_file_path
                else:
                    raise FileNotFoundError(f"未找到用户 {user_key} 的推荐资源文件")

        # 读取JSON文件
        with open(user_file_path, 'r', encoding='utf-8') as f:
            user_recommendations = json.load(f)

        # 根据RECOMMEND_USED_RESOURCES值过滤结果
        filtered_recommendations = []
        if RECOMMEND_USED_RESOURCES == -1:
            # 只返回is_use为0的资源
            filtered_recommendations = [item for item in user_recommendations if item.get("is_use", 0) == 0]
        elif RECOMMEND_USED_RESOURCES == 1:
            # 只返回is_use为1的资源
            filtered_recommendations = [item for item in user_recommendations if item.get("is_use", 0) == 1]
        else:
            # 无限制，返回所有资源
            filtered_recommendations = user_recommendations

        # 提取前6个资源的resource_key和form_type
        result = []
        for i, item in enumerate(filtered_recommendations):
            if i >= 6:  # 只取前6个
                break
            result.append({
                "resource_key": item["resource_key"],
                "form_id": item["form_type"]
            })

        return result

    except Exception as e:
        print(f"获取用户 {user_key} 的推荐资源时出错: {e}")
        # 返回空列表或默认值
        return []


def get_user_recommendations_test(user_key: str) -> List[Dict[str, Any]]:
    """
    测试方法，用于获取指定用户的推荐资源，但使用固定的测试文件路径
    D:\\javacode\\big_project\\lrp-hybrid\\py\\deepFM\\userRecommendResources\\20251029\\text.json
    
    Args:
        user_key: 用户标识
        
    Returns:
        List[Dict[str, Any]]: 包含resource_key和form_type的字典列表，最多返回6个
    """
    try:
        # 固定的测试文件路径
        test_file_path = r"D:\javacode\big_project\lrp-hybrid\py\deepFM\userRecommendResources\20251029\text.json"
        
        # 直接读取JSON文件
        with open(test_file_path, 'r', encoding='utf-8') as f:
            user_recommendations = json.load(f)
        
        # 提取前6个资源的resource_key和form_type
        result = []
        for i, item in enumerate(user_recommendations):
            if i >= 6:  # 只取前6个
                break
            result.append({
                "resource_key": item["resource_key"],
                "form_id": item["form_type"]
            })

        return result

    except Exception as e:
        logger.error(f"获取用户 {user_key} 的测试推荐资源时出错: {e}")
        # 返回空列表或默认值
        return []


def mark_resource_as_used(user_key: str, resource_key: str, form_id: int, directory_name: str = None) -> bool:
    """
    将指定用户和资源的is_use字段设置为1
    
    Args:
        user_key: 用户标识
        resource_key: 资源标识
        directory_name: 可选的目录名称，如果提供则使用该目录而不是最新目录
        
    Returns:
        bool: 操作是否成功
    """
    try:
        # 获取推荐资源目录
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        recommend_dir = os.path.join(base_dir, USER_RECOMMEND_RESOURCE_DIR)
        
        # 如果提供了目录名称，使用指定目录
        if directory_name:
            target_dir = os.path.join(recommend_dir, directory_name)
            # 确保目录存在
            os.makedirs(target_dir, exist_ok=True)
            user_file_path = os.path.join(target_dir, f"{user_key}.json")
            
            # 如果用户文件不存在，尝试从同目录的S0000.json复制
            if not os.path.exists(user_file_path):
                default_file_path = os.path.join(target_dir, "S0000.json")
                if os.path.exists(default_file_path):
                    import shutil
                    shutil.copy(default_file_path, user_file_path)
        else:
            # 获取最新的日期目录
            date_dirs = [d for d in os.listdir(recommend_dir) if os.path.isdir(os.path.join(recommend_dir, d))]
            if not date_dirs:
                print("未找到任何推荐资源目录")
                return False

            # 按名称排序，获取最新的日期目录
            date_dirs.sort()
            latest_date_dir = date_dirs[-1]
            user_file_path = os.path.join(recommend_dir, latest_date_dir, f"{user_key}.json")

            # 检查用户文件是否存在
            if not os.path.exists(user_file_path):
                # 如果用户文件不存在，尝试使用S0000.json作为默认文件
                default_file_path = os.path.join(recommend_dir, latest_date_dir, "S0000.json")
                if os.path.exists(default_file_path):
                    # 复制默认文件
                    import shutil
                    shutil.copy(default_file_path, user_file_path)
                else:
                    print(f"未找到用户 {user_key} 的推荐资源文件")
                    return False

        # 读取JSON文件
        with open(user_file_path, 'r', encoding='utf-8') as f:
            user_recommendations = json.load(f)

        # 查找对应的资源并更新is_use字段
        resource_found = False
        for item in user_recommendations:
            if item.get("resource_key") == resource_key and item.get("form_type") == form_id:
                item["is_use"] = 1
                resource_found = True
                break

        if not resource_found:
            print(f"在用户 {user_key} 的推荐资源中未找到资源 {resource_key}")
            return False

        # 写回文件
        with open(user_file_path, 'w', encoding='utf-8') as f:
            json.dump(user_recommendations, f, ensure_ascii=False, indent=2)

        print(f"已将用户 {user_key} 的资源 {resource_key} 的 is_use 字段设置为 1")
        return True

    except Exception as e:
        print(f"更新用户 {user_key} 的资源 {resource_key} 状态时出错: {e}")
        return False


def reset_user_resources_usage(user_key: str, directory_name: str = None) -> bool:
    """
    将指定用户的所有资源的is_use字段设置为0
    
    Args:
        user_key: 用户标识
        directory_name: 可选的目录名称，如果提供则使用该目录而不是最新目录
        
    Returns:
        bool: 操作是否成功
    """
    try:
        # 获取推荐资源目录
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        recommend_dir = os.path.join(base_dir, USER_RECOMMEND_RESOURCE_DIR)
        
        # 如果提供了目录名称，使用指定目录
        if directory_name:
            target_dir = os.path.join(recommend_dir, directory_name)
            # 确保目录存在
            os.makedirs(target_dir, exist_ok=True)
            user_file_path = os.path.join(target_dir, f"{user_key}.json")
            
            # 如果用户文件不存在，尝试从同目录的S0000.json复制
            if not os.path.exists(user_file_path):
                default_file_path = os.path.join(target_dir, "S0000.json")
                if os.path.exists(default_file_path):
                    import shutil
                    shutil.copy(default_file_path, user_file_path)
        else:
            # 获取最新的日期目录
            date_dirs = [d for d in os.listdir(recommend_dir) if os.path.isdir(os.path.join(recommend_dir, d))]
            if not date_dirs:
                print("未找到任何推荐资源目录")
                return False

            # 按名称排序，获取最新的日期目录
            date_dirs.sort()
            latest_date_dir = date_dirs[-1]
            user_file_path = os.path.join(recommend_dir, latest_date_dir, f"{user_key}.json")

            # 检查用户文件是否存在
            if not os.path.exists(user_file_path):
                # 如果用户文件不存在，尝试使用S0000.json作为默认文件
                default_file_path = os.path.join(recommend_dir, latest_date_dir, "S0000.json")
                if os.path.exists(default_file_path):
                    # 复制默认文件
                    import shutil
                    shutil.copy(default_file_path, user_file_path)
                else:
                    print(f"未找到用户 {user_key} 的推荐资源文件")
                    return False

        # 读取JSON文件
        with open(user_file_path, 'r', encoding='utf-8') as f:
            user_recommendations = json.load(f)

        # 将所有资源的is_use字段设置为0
        for item in user_recommendations:
            item["is_use"] = 0

        # 写回文件
        with open(user_file_path, 'w', encoding='utf-8') as f:
            json.dump(user_recommendations, f, ensure_ascii=False, indent=2)

        print(f"已将用户 {user_key} 的所有资源的 is_use 字段设置为 0")
        return True

    except Exception as e:
        print(f"重置用户 {user_key} 的资源使用状态时出错: {e}")
        return False
