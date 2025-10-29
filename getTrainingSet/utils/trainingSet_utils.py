import sys
import os as sys_os
sys.path.append(sys_os.path.join(sys_os.path.dirname(__file__), '..', '..'))
from config import DB_CONFIG, USER_FEATURE_DIR, CONTENT_FEATURE_DIR
import mysql.connector
from datetime import datetime, timedelta
from typing import Dict, List
from mysql.connector import Error
import json
import os

# 缓存字典，用于存储函数调用结果，避免重复的数据库查询
# _user_knowledge_stats_cache: 存储用户知识点统计数据，键为"user_key_日期"格式字符串
_user_knowledge_stats_cache: Dict[str, Dict] = {}
# _user_resource_preference_cache: 存储用户资源偏好数据，键为"user_key_日期"格式字符串
_user_resource_preference_cache: Dict[str, dict] = {}  # 修正：原返回为dict，此处类型同步
# _user_active_days_cache: 存储用户活跃天数，键为user_key
_user_active_days_cache: Dict[str, int] = {}
# _resource_form_cache: 存储资源类型ID（resource_form表id），键为resource_key
_resource_form_cache: Dict[str, int] = {}
# _get_resource_knowledges_cache: 存储资源关联知识点的1/0列表，键为resource_key
_get_resource_knowledges_cache: Dict[str, List[int]] = {}  # 修正：值类型为函数返回的List[int]
# _get_resource_tags_cache: 存储资源关联标签的1/0列表，键为resource_key
_get_resource_tags_cache: Dict[str, List[int]] = {}  # 修正：值类型为函数返回的List[int]
# _get_resource_time_cache: 存储资源时长列表，键为resource_key
_get_resource_time_cache: Dict[str, List[int]] = {}  # 修正：值类型为函数返回的List[int]


def clear_cache():
    """
    清空所有缓存数据
    """
    _user_knowledge_stats_cache.clear()
    _user_resource_preference_cache.clear()
    _user_active_days_cache.clear()
    _resource_form_cache.clear()
    _get_resource_knowledges_cache.clear()
    _get_resource_tags_cache.clear()
    _get_resource_time_cache.clear()


def save_user_interaction_data(user_key: str, post_3d_correct_rate: List[float], post_practice_count: List[int], output_path: str = None):
    """
    保存用户交互数据到JSON文件，按用户键存储，避免重复
    
    参数:
        user_key: 用户唯一标识
        post_3d_correct_rate: 用户3天后正确率列表
        post_practice_count: 用户3天后练习次数列表
        output_path: 输出文件路径，如果为None则使用默认路径（当前日期命名）
    """
    try:
        # 确定输出路径
        if output_path is None:
            # 使用当前日期生成文件名
            current_date_file = f"{datetime.now().strftime("%Y%m%d")}.json"
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
            training_sets_dir = os.path.join(base_dir, USER_FEATURE_DIR)
            output_path = os.path.join(training_sets_dir, current_date_file)

        
        # 如果文件存在，读取现有数据
        user_data = {}
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                    # 确保loaded_data是字典类型，如果不是则初始化为空字典
                    if isinstance(loaded_data, dict):
                        user_data = loaded_data
                    else:
                        user_data = {}
            except (json.JSONDecodeError, FileNotFoundError):
                user_data = {}
        
        # 创建或更新用户数据
        user_data[user_key] = {
            "user_key": user_key,
            "post_3d_correct_rate": post_3d_correct_rate,
            "post_practice_count": post_practice_count
        }
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(user_data, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"保存用户交互数据时出错: {e}")
        raise


def generate_resource_info_json(output_path: str = None):
    """
    生成包含所有资源信息的JSON文件
    
    从items表和media_assets表中获取所有resource_key，
    然后为每个resource_key获取对应的form_type, resource_knowledges, 
    resource_tags, resource_time信息，并保存到JSON文件中
    
    参数：
        output_path: 输出文件路径，如果为None则使用默认路径
    """
    conn = None
    cursor = None
    try:
        # 建立数据库连接
        conn = mysql.connector.connect(**DB_CONFIG)
        if not conn.is_connected():
            raise Error("数据库连接失败")
        
        cursor = conn.cursor()
        
        # 获取所有item_key
        cursor.execute("SELECT item_key FROM items")
        item_keys = [row[0] for row in cursor.fetchall()]
        
        # 获取所有media_key
        cursor.execute("SELECT media_key FROM media_assets")
        media_keys = [row[0] for row in cursor.fetchall()]
        
        # 合并所有resource_key
        resource_keys = item_keys + media_keys
        
        # 为每个resource_key构建信息
        resources_info = []
        for resource_key in resource_keys:
            # 获取资源类型ID
            try:
                form_type = get_resource_form_by_resource_key(resource_key)
            except (ValueError, Error):
                form_type = 0  # 默认值
            
            # 获取知识点信息
            try:
                resource_knowledges = get_resource_knowledges(resource_key)
            except (ValueError, Error):
                resource_knowledges = []  # 默认值
            
            # 获取标签信息
            try:
                resource_tags = get_resource_tags(resource_key)
            except (ValueError, Error):
                resource_tags = []  # 默认值
            
            # 获取资源时长信息
            try:
                resource_time_list = get_resource_time(resource_key)
            except (ValueError, Error):
                resource_time_list = []  # 默认值
            
            # 构建资源信息字典
            resource_info = {
                "resource_key": resource_key,
                "form_type": form_type,
                "resource_knowledges": resource_knowledges,
                "resource_tags": resource_tags,
                "resource_time": resource_time_list
            }
            
            resources_info.append(resource_info)
        
        # 确定输出路径
        if output_path is None:
            # 使用当前日期生成文件名
            current_date_file = f"{datetime.now().strftime("%Y%m%d")}.json"
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
            training_sets_dir = os.path.join(base_dir, CONTENT_FEATURE_DIR)
            output_path = os.path.join(training_sets_dir, current_date_file)
        # 保存到JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(resources_info, f, ensure_ascii=False, indent=2)
        
        print(f"资源信息已保存到 {output_path}，共处理 {len(resources_info)} 个资源")
        return output_path
        
    except Error as e:
        print(f"数据库错误: {e}")
        raise
    except Exception as e:
        print(f"处理错误: {e}")
        raise
    finally:
        # 确保资源关闭
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()


def get_user_knowledge_stats(user_key: str, current_time: datetime) -> Dict[str, List]:
    """
    根据用户key和指定时间，获取该用户在指定时间前30天内的知识点正确率和总题数列表

    方法逻辑：
        getModel. 关联user_knowledge_stats_20d(t1)和knowledges(t2)，筛选条件为：
           - t1.user_key与传入参数一致
           - t1.record_time在[current_time - 30天, current_time)范围内
        2. 确保包含所有knowledges(t2)数据（即使t1中无对应记录）
        3. 按t2.id升序排列结果，计算每个知识点的正确率（correct_count/total_count）和总题数
        4. 无数据时正确率和总题数默认0

    参数：
        user_key: 目标用户的user_key
        current_time: 时间临界点，仅统计record_time在该时间之前的数据

    返回：
        字典对象，包含两个键：
            - accuracy: List[double]，按t2.id升序排列的正确率列表，长度等于t2总数据量
            - totalCount: List[int]，按t2.id升序排列的总题数列表，长度等于t2总数据量
    """
    # 构造缓存键，格式为"user_key_年月日"
    cache_key = f"{user_key}_{current_time.strftime('%Y%m%d')}"
    # 检查是否已有缓存结果，如果有则直接返回缓存数据，避免重复数据库查询
    if cache_key in _user_knowledge_stats_cache:
        return _user_knowledge_stats_cache[cache_key]

    conn = None
    cursor = None
    try:
        # getModel. 连接数据库
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)

        # 2. 获取所有知识点(knowledges t2)数据，按id升序排列（确定列表顺序和长度）
        cursor.execute("SELECT id, knowledge_key FROM knowledges t2 ORDER BY id ASC")
        t2_data = cursor.fetchall()
        total_knowledges = len(t2_data)
        if total_knowledges == 0:
            raise ValueError("knowledges表中无数据，无法生成结果列表")

        # 提取知识点顺序（按t2.id升序）和knowledge_key映射
        knowledge_order = [item["knowledge_key"] for item in t2_data]  # 按t2.id升序的knowledge_key列表

        # 3. 计算时间范围：[current_time - 30天, current_time)
        start_time = (current_time - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
        end_time = current_time.strftime('%Y-%m-%d %H:%M:%S')

        # 4. 查询用户在时间范围内的知识点数据（左连接确保t2全包含）
        query = """
        SELECT 
            t2.knowledge_key,
            COALESCE(t1.correct_count, 0) AS correct_count,
            COALESCE(t1.total_count, 0) AS total_count
        FROM 
            knowledges t2
        LEFT JOIN 
            user_knowledge_stats_20d t1 
            ON t1.knowledge_key = t2.knowledge_key 
            AND t1.user_key = %s 
            AND t1.record_time >= %s 
            AND t1.record_time < %s  -- 严格小于current_time
        ORDER BY 
            t2.id ASC  -- 按t2.id升序排列，与knowledge_order顺序一致
        """
        cursor.execute(query, (user_key, start_time, end_time))
        raw_data = cursor.fetchall()

        # 5. 构建知识点数据映射（knowledge_key -> (correct_count, total_count)）
        knowledge_stats = {}
        for row in raw_data:
            key = row["knowledge_key"]
            knowledge_stats[key] = (row["correct_count"], row["total_count"])

        # 6. 按t2顺序构建accuracy和totalCount列表
        accuracy: List[float] = []
        total_count: List[int] = []
        for key in knowledge_order:
            correct, total = knowledge_stats.get(key, (0, 0))
            # 处理除数为0的情况，默认正确率0.0
            acc = correct / total if total != 0 else 0.0
            accuracy.append(acc)
            total_count.append(total)

        # 验证列表长度是否正确
        if len(accuracy) != total_knowledges or len(total_count) != total_knowledges:
            raise RuntimeError(
                f"结果列表长度错误（应为{total_knowledges}，实际accuracy:{len(accuracy)}, totalCount:{len(total_count)}）")

        result = {
            "accuracy": accuracy,
            "totalCount": total_count
        }

        # 将查询结果存入缓存，下次相同参数调用时可直接返回
        _user_knowledge_stats_cache[cache_key] = result
        return result

    except mysql.connector.Error as e:
        print(f"数据库错误: {e}")
        raise
    except Exception as e:
        print(f"处理错误: {e}")
        raise
    finally:
        # 确保资源关闭
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()


def get_user_resource_preference(user_key: str, current_time: datetime) -> dict:
    """
    根据用户key和指定时间，获取该用户在指定时间前30天内的资源偏好比例列表及资源映射哈希表

    方法逻辑：
        getModel. 关联user_resource_preference_7d(t1)和resource_form(t3)，筛选条件为：
           - t1.user_key与传入参数一致
           - t1.record_time在[current_time - 30天, current_time)范围内
        2. 确保包含所有resource_form(t3)数据（即使t1中无对应记录）
        3. 按t3.id升序排列结果，计算每个资源的点击占比（当前资源click_count / 该用户总click_count）
        4. 无数据或总点击为0时，占比默认0.0，结果保留两位小数

    参数：
        user_key: 目标用户的user_key
        current_time: 时间临界点，仅统计record_time在该时间之前的数据

    返回：
        字典包含：
            - preference: List[double]，按t3.id升序排列的资源偏好比例列表（保留两位小数），长度等于t3总数据量
    """
    # 构造缓存键，格式为"user_key_年月日"
    cache_key = f"{user_key}_{current_time.strftime('%Y%m%d')}"
    # 检查是否已有缓存结果，如果有则直接返回缓存数据，避免重复数据库查询
    if cache_key in _user_resource_preference_cache:
        return _user_resource_preference_cache[cache_key]

    conn = None
    cursor = None
    try:
        # getModel. 连接数据库
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)

        # 2. 获取所有resource_form(t3)数据，按id升序排列（确定列表顺序、长度及哈希映射）
        cursor.execute("SELECT id, form_key FROM resource_form t3 ORDER BY id ASC")
        t3_data = cursor.fetchall()
        total_resources = len(t3_data)
        if total_resources == 0:
            raise ValueError("resource_form表中无数据，无法生成结果列表")

        # 构建资源顺序列表（按t3.id升序）
        form_key_order: List[str] = [item["form_key"] for item in t3_data]

        # 3. 计算时间范围：[current_time - 30天, current_time)
        start_time = (current_time - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
        end_time = current_time.strftime('%Y-%m-%d %H:%M:%S')

        # 4. 查询用户在时间范围内的资源点击数据（左连接确保t3全包含）
        query = """
        SELECT 
            t3.form_key,
            COALESCE(t1.click_count, 0) AS click_count
        FROM 
            resource_form t3
        LEFT JOIN 
            user_resource_preference_7d t1 
            ON t1.form_key = t3.form_key 
            AND t1.user_key = %s 
            AND t1.record_time >= %s 
            AND t1.record_time < %s  -- 严格小于currentTime
        ORDER BY 
            t3.id ASC  -- 按t3.id升序，与form_key_order顺序一致
        """
        cursor.execute(query, (user_key, start_time, end_time))
        raw_data = cursor.fetchall()

        # 5. 计算该用户的总click_count（用于占比分母）
        total_click = sum(row["click_count"] for row in raw_data)

        # 6. 构建资源偏好比例列表（preference），保留两位小数
        preference: List[float] = []
        form_click_map = {row["form_key"]: row["click_count"] for row in raw_data}
        for form_key in form_key_order:
            click = form_click_map.get(form_key, 0)
            ratio = click / total_click if total_click != 0 else 0.0
            # 保留两位小数
            preference.append(round(ratio, 2))

        # 验证列表长度
        if len(preference) != total_resources:
            raise RuntimeError(f"preference列表长度错误（应为{total_resources}，实际{len(preference)}）")

        result = {
            "preference": preference
        }
        # 将查询结果存入缓存，下次相同参数调用时可直接返回
        _user_resource_preference_cache[cache_key] = result
        return result

    except mysql.connector.Error as e:
        print(f"数据库错误: {e}")
        raise
    except Exception as e:
        print(f"处理错误: {e}")
        raise
    finally:
        # 确保资源关闭
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()


def get_user_active_days(user_key: str) -> int:
    """
    根据用户的user_key查询其总活跃天数（total_active_days）

    参数：
        user_key: 用户唯一标识（关联users表的user_key字段）

    返回：
        若查询到用户且字段有值，返回对应的total_active_days（整数）；
        若用户不存在、字段为NULL或查询异常，返回0（确保返回int类型）
    """
    # 检查是否已有缓存结果，如果有则直接返回缓存数据，避免重复数据库查询
    if user_key in _user_active_days_cache:
        return _user_active_days_cache[user_key]

    conn = None
    cursor = None
    try:
        # 建立数据库连接
        conn = mysql.connector.connect(**DB_CONFIG)
        if not conn.is_connected():
            raise Error("数据库连接失败")

        # 创建游标并执行查询（参数化查询防止SQL注入）
        cursor = conn.cursor(dictionary=True)
        query = "SELECT total_active_days FROM users WHERE user_key = %s"
        cursor.execute(query, (user_key,))  # 参数用元组传递

        # 获取查询结果
        result = cursor.fetchone()

        if result and result["total_active_days"] is not None:
            # 正常情况：返回查询到的整数（强制转换为int确保类型正确）
            value = int(result["total_active_days"])
            # 将查询结果存入缓存，下次相同参数调用时可直接返回
            _user_active_days_cache[user_key] = value
            return value
        else:
            # 用户不存在或字段为空：返回0
            _user_active_days_cache[user_key] = 0
            return 0

    except mysql.connector.Error as e:
        print(f"数据库错误: {e}")
        # 发生错误时存入缓存（0），避免重复异常
        _user_active_days_cache[user_key] = 0
        return 0  # 数据库异常时返回0
    except Exception as e:
        print(f"处理错误: {e}")
        # 发生错误时存入缓存（0），避免重复异常
        _user_active_days_cache[user_key] = 0
        return 0  # 其他异常时返回0
    finally:
        # 确保资源关闭
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()


# def get_resource_form(form_key: str) -> int:
#     """
#     根据form_key查询resource_form表中的id并返回（整数类型）
#
#     参数：
#         form_key: 资源类型标识（如"qa"、"video"等）
#
#     返回：
#         对应的id（整数）
#
#     异常：
#         若查询不到对应form_key的记录，抛出ValueError；
#         数据库操作失败时，抛出mysql.connector.Error或通用Exception
#     """
#     # 检查缓存：存在则直接返回
#     if form_key in _resource_form_cache:
#         return _resource_form_cache[form_key]
#
#     conn = None
#     cursor = None
#     try:
#         # getModel. 建立数据库连接（使用导入的全局DB_CONFIG，删除函数内重复定义）
#         conn = mysql.connector.connect(**DB_CONFIG)
#         if not conn.is_connected():
#             raise Error("数据库连接失败")
#
#         # 2. 执行参数化查询（防止SQL注入）
#         cursor = conn.cursor()
#         query = "SELECT id FROM resource_form WHERE form_key = %s"
#         cursor.execute(query, (form_key,))  # 参数用元组传递
#
#         # 3. 获取查询结果
#         result = cursor.fetchone()  # 最多一条记录（form_key唯一）
#         if not result:
#             # 未查询到对应记录，不存入缓存（下次查询仍需验证）
#             raise ValueError(f"resource_form表中未找到form_key为'{form_key}'的记录")
#
#         # 4. 转换结果为int并存入缓存
#         form_id = int(result[0])
#         _resource_form_cache[form_key] = form_id
#         return form_id
#
#     except Error as e:
#         print(f"数据库错误: {e}")
#         raise  # 抛出数据库异常供上层处理
#     except ValueError as e:
#         print(f"查询错误: {e}")
#         raise  # 抛出业务异常（未找到记录）
#     except Exception as e:
#         print(f"处理错误: {e}")
#         raise  # 抛出其他异常
#     finally:
#         # 5. 确保资源关闭
#         if cursor:
#             cursor.close()
#         if conn and conn.is_connected():
#             conn.close()


def get_resource_form_by_resource_key(resource_key: str) -> int:
    """
    根据resource_key查询resource_form表中的id并返回（整数类型）
    先在items表中查找，如果没找到再到media_assets表中查找

    参数：
        resource_key: 资源唯一标识

    返回：
        resource_form表中对应的id（整数）

    异常：
        若查询不到对应resource_key的记录，抛出ValueError；
        数据库操作失败时，抛出mysql.connector.Error或通用Exception
    """
    # 检查缓存：存在则直接返回
    if resource_key in _resource_form_cache:
        return _resource_form_cache[resource_key]

    conn = None
    cursor = None
    try:
        # 建立数据库连接
        conn = mysql.connector.connect(**DB_CONFIG)
        if not conn.is_connected():
            raise Error("数据库连接失败")

        # 1. 先在items和resource_form表中查找
        cursor = conn.cursor()
        query = """
            SELECT t2.id 
            FROM items t1, resource_form t2 
            WHERE t1.item_key = %s AND t1.form_key = t2.form_key
        """
        cursor.execute(query, (resource_key,))
        result = cursor.fetchone()
        
        # 2. 如果没查到，再到media_assets和resource_form表中查找
        if not result:
            query = """
                SELECT t2.id 
                FROM media_assets t3, resource_form t2 
                WHERE t3.media_key = %s AND t2.form_key = t3.form_key
            """
            cursor.execute(query, (resource_key,))
            result = cursor.fetchone()

        # 3. 获取查询结果
        if not result:
            # 未查询到对应记录，不存入缓存（下次查询仍需验证）
            raise ValueError(f"未找到resource_key为'{resource_key}'的记录")

        # 4. 转换结果为int并存入缓存
        form_id = int(result[0])
        _resource_form_cache[resource_key] = form_id
        return form_id

    except Error as e:
        print(f"数据库错误: {e}")
        raise  # 抛出数据库异常供上层处理
    except ValueError as e:
        print(f"查询错误: {e}")
        raise  # 抛出业务异常（未找到记录）
    except Exception as e:
        print(f"处理错误: {e}")
        raise  # 抛出其他异常
    finally:
        # 5. 确保资源关闭
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()


def get_resource_knowledges(resource_key: str) -> List[int]:
    """
    根据resource_key查询关联表，返回整数列表：
    getModel. 列表长度 = knowledges表（t2）的总数据数（按t2.id升序排列）
    2. 对于t2中每条数据：
       - 若其id在关联查询结果中（即满足resource_key条件），列表对应位置添加1
       - 否则添加0

    关联逻辑：
       内连接knowledge_resources（t1）和knowledges（t2），条件t1.knowledge_key = t2.knowledge_key
       筛选t1.resource_key = 参数resource_key的记录，取t2.id

    参数：
        resource_key: 资源唯一标识（如"C02_1"）

    返回：
        按t2.id升序排列的int列表（1或0）

    异常：
        数据库操作失败时抛出mysql.connector.Error或通用Exception
    """
    # 检查缓存：存在则直接返回
    if resource_key in _get_resource_knowledges_cache:
        return _get_resource_knowledges_cache[resource_key]

    conn = None
    cursor = None
    try:
        # 建立连接（使用缓冲游标避免多轮查询的未读结果错误）
        conn = mysql.connector.connect(**DB_CONFIG)
        if not conn.is_connected():
            raise Error("数据库连接失败")
        cursor = conn.cursor(buffered=True)

        # 步骤1：查询knowledges表（t2）所有id，按id升序（确定列表长度和顺序）
        cursor.execute("SELECT id FROM knowledges ORDER BY id ASC")
        t2_all_ids = [row[0] for row in cursor.fetchall()]  # 提取t2所有id，按升序排列
        total_t2_count = len(t2_all_ids)
        if total_t2_count == 0:
            empty_result = []
            _get_resource_knowledges_cache[resource_key] = empty_result  # 空结果也存入缓存
            return empty_result

        # 步骤2：执行关联查询，获取符合条件的t2.id（内连接确保t2数据都存在）
        join_query = """
            SELECT t2.id 
            FROM knowledge_resources t1
            INNER JOIN knowledges t2 
              ON t1.knowledge_key = t2.knowledge_key  # 内连接条件
            WHERE t1.resource_key = %s  # 筛选目标resource_key
            ORDER BY t2.id ASC  # 按t2.id升序，与步骤1的顺序一致
        """
        cursor.execute(join_query, (resource_key,))
        # 用集合存储符合条件的t2.id，加速后续判断
        target_t2_ids = {row[0] for row in cursor.fetchall()}

        # 步骤3：生成结果列表（1表示匹配，0表示不匹配）
        result = [1 if t2_id in target_t2_ids else 0 for t2_id in t2_all_ids]

        # 存入缓存后返回
        _get_resource_knowledges_cache[resource_key] = result
        return result

    except Error as e:
        print(f"数据库错误: {e}")
        raise
    except Exception as e:
        print(f"处理错误: {e}")
        raise
    finally:
        # 确保资源关闭
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()


def get_resource_tags(resource_key: str) -> List[int]:
    """
    根据resource_key查询标签关联关系，返回整数列表：
    getModel. 列表长度 = tags表（t2）的总数据数（按t2.id升序排列）
    2. 对于tags表每条数据：
       - 若其id在「resource_key匹配的关联结果」中，列表对应位置添加1
       - 否则添加0

    关联逻辑：
       内连接tag_resource（t1）和tags（t2），条件t1.tag_id = t2.id（确保t2数据存在）
       筛选t1.resource_key = 参数resource_key的记录，取t2.id

    参数：
        resource_key: 资源唯一标识（如"C02_1"）

    返回：
        按tags表id升序排列的int列表（1表示关联，0表示未关联）

    异常：
        数据库操作失败时抛出mysql.connector.Error或通用Exception
    """
    # 检查缓存：存在则直接返回
    if resource_key in _get_resource_tags_cache:
        return _get_resource_tags_cache[resource_key]

    conn = None
    cursor = None
    try:
        # 建立数据库连接（删除函数内重复定义的DB_CONFIG，使用全局导入的配置）
        conn = mysql.connector.connect(**DB_CONFIG)
        if not conn.is_connected():
            raise Error("数据库连接失败")
        cursor = conn.cursor(buffered=True)

        # --------------------------
        # 步骤1：获取tags表（t2）所有id，按id升序（确定列表长度和顺序）
        # --------------------------
        cursor.execute("SELECT id FROM tags ORDER BY id ASC")
        tags_all_ids = [row[0] for row in cursor.fetchall()]  # 提取所有tags的id（升序）
        total_tags_count = len(tags_all_ids)
        if total_tags_count == 0:
            empty_result = []
            _get_resource_tags_cache[resource_key] = empty_result  # 空结果存入缓存
            return empty_result

        # --------------------------
        # 步骤2：关联查询，获取resource_key匹配的tags.id
        # --------------------------
        join_query = """
            SELECT t2.id 
            FROM tag_resource t1
            INNER JOIN tags t2 
              ON t1.tag_id = t2.id  # 连接条件：t1的tag_id关联t2的id
            WHERE t1.resource_key = %s  # 筛选目标资源的关联记录
            ORDER BY t2.id ASC
        """
        cursor.execute(join_query, (resource_key,))
        # 用集合存储匹配的tags.id（O(getModel)查询效率）
        matched_tags_ids = {row[0] for row in cursor.fetchall()}

        # --------------------------
        # 步骤3：生成1/0结果列表
        # --------------------------
        result = [1 if tag_id in matched_tags_ids else 0 for tag_id in tags_all_ids]

        # 存入缓存后返回
        _get_resource_tags_cache[resource_key] = result
        return result

    except Error as e:
        print(f"数据库错误: {e}")
        raise  # 抛出数据库异常供上层处理
    except Exception as e:
        print(f"处理错误: {e}")
        raise  # 抛出其他异常
    finally:
        # 确保资源关闭
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()


def get_resource_time(resource_key: str) -> List[int]:
    """
    根据resource_key查询media_assets表的duration，返回整数列表

    处理逻辑：
        getModel. 查询media_assets表中media_key = 参数resource_key的所有duration
        2. 将查询到的duration转换为int类型，组成列表返回
        3. 若未查询到数据，返回默认列表[0]

    参数：
        resource_key: 资源唯一标识（对应media_assets表的media_key）

    返回：
        包含duration的int列表（无数据时返回[0]）

    异常：
        数据库操作失败时抛出mysql.connector.Error或通用Exception
    """
    # 检查缓存：存在则直接返回
    if resource_key in _get_resource_time_cache:
        return _get_resource_time_cache[resource_key]

    conn = None
    cursor = None
    try:
        # 建立数据库连接（使用缓冲游标）
        conn = mysql.connector.connect(**DB_CONFIG)
        if not conn.is_connected():
            raise Error("数据库连接失败")
        cursor = conn.cursor(buffered=True)

        # 执行查询：获取匹配resource_key的所有duration
        query = "SELECT duration FROM media_assets WHERE media_key = %s"
        cursor.execute(query, (resource_key,))
        results = cursor.fetchall()  # 获取所有匹配记录

        # 处理查询结果：转换为int列表
        duration_list = []
        for row in results:
            duration = row[0]  # 提取duration字段
            try:
                # 转换为int（支持字符串或数值类型的duration）
                duration_int = int(duration)
                duration_list.append(duration_int)
            except (ValueError, TypeError):
                # 若转换失败（如非数值格式），跳过该条记录
                continue

        # 若无有效数据，返回默认[0]
        if not duration_list:
            duration_list = [0]

        # 存入缓存后返回
        _get_resource_time_cache[resource_key] = duration_list
        return duration_list

    except Error as e:
        print(f"数据库错误: {e}")
        raise  # 抛出数据库异常
    except Exception as e:
        print(f"处理错误: {e}")
        raise  # 抛出其他异常
    finally:
        # 确保资源关闭
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()