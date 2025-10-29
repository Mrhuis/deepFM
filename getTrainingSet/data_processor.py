# 数据处理核心逻辑
from typing import Optional, Dict, Set
from mysql.connector import Error
from datetime import datetime  # 正确：导入datetime类
import os

# 导入自定义模块
from getTrainingSet.utils.db_utils import get_db_connection, close_db_connection
from getTrainingSet.utils.trainingSet_utils import (get_user_knowledge_stats, get_user_resource_preference,
                                                  get_user_active_days, get_resource_form_by_resource_key,
                                     get_resource_knowledges, get_resource_tags, get_resource_time,generate_resource_info_json,clear_cache,
                                     save_user_interaction_data)

from getTrainingSet.utils.csv_utils import save_processed_result
from getTrainingSet.utils.common_utils import str_to_intlist, str_to_floatlist
from modelsContainer import ProcessedUserResourceInteraction, TrainingSet
from config import DEFAULT_TRAIN_NUM, TRAINING_SETS_DIR


def clear_today_csv_file():
    """
    清除当天的CSV文件，为覆盖写入做准备
    """
    # 获取当天CSV文件路径
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    training_sets_dir = os.path.join(base_dir, TRAINING_SETS_DIR)
    today_csv_path = os.path.join(training_sets_dir, f"{datetime.now().strftime('%Y%m%d')}.csv")
    
    # 检查文件是否存在
    if os.path.exists(today_csv_path):
        existing_lines = 0
        try:
            # 读取现有文件行数（不包括表头）
            with open(today_csv_path, 'r', encoding='utf-8') as f:
                existing_lines = sum(1 for _ in f) - 1  # 减去表头行
            
            # 删除旧文件
            os.remove(today_csv_path)
            print(f"🗑️  已删除旧CSV文件: {today_csv_path}")
            print(f"   原有数据行数: {existing_lines}")
            return existing_lines
        except Exception as e:
            print(f"⚠️  删除旧CSV文件失败: {e}")
            return 0
    else:
        print(f"📝 当天CSV文件不存在，将创建新文件: {today_csv_path}")
        return 0


def process_user_resource_interaction():
    """
    逐行读取user_resource_interaction表，加工后返回数据模型列表
    每次运行都会覆盖当天的CSV文件
    """
    conn = None
    cursor = None

    try:
        # 1. 先清除已存在的CSV文件（覆盖模式）
        old_count = clear_today_csv_file()
        
        # 获取数据库连接和游标
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)  # 字典游标（按字段名访问）

        # 执行查询（按主键顺序读取）
        query = "SELECT * FROM user_resource_interaction WHERE effect_calc_time IS NOT NULL ORDER BY id ASC"
        cursor.execute(query)

        # 逐行处理数据，限制处理数量不超过DEFAULT_TRAIN_NUM
        row: Optional[Dict] = cursor.fetchone()
        row_count = 0

        while row is not None and row_count < DEFAULT_TRAIN_NUM:
            row_count += 1
            print(f"开始处理第{row_count}行数据（id: {row['id']}）")

            # 加工单行数据为模型实例
            processed_row = process_single_row(row)
            save_processed_result(processed_row)

            # 读取下一行
            row = cursor.fetchone()

        generate_resource_info_json()

        print(f"\n" + "="*60)
        print(f"✅ 数据处理完成！")
        print(f"   原有数据: {old_count} 行（已覆盖）")
        print(f"   新写入数据: {row_count} 行")
        print("="*60 + "\n")
        clear_cache()

    except Error as e:
        print(f"数据库操作错误: {e}")
        raise
    except Exception as e:
        print(f"数据处理异常: {e}")
        raise
    finally:
        # 确保所有结果都被消费，防止出现"Unread result found"错误
        if cursor:
            # 消费掉剩余的所有结果
            cursor.fetchall()
        # 关闭连接
        close_db_connection(conn, cursor)


def process_single_row(row: Dict) -> ProcessedUserResourceInteraction:
    """将单行字典数据转换为数据模型实例（含加工逻辑）"""
    # getModel. 必选字段（非空）
    user_key = row['user_key']
    form_key = row['form_key']
    resource_key = row['resource_key']


    # 3. 滞后特征字段（字符串转数值）
    post_3d_correct_rate = str_to_floatlist(row.get('post_3d_correct_rate'))
    post_practice_count = str_to_intlist(row.get('post_practice_count'))
    is_first_submit_24h = row['is_first_submit_24h']
    correct_rate_change = row['correct_rate_change']

    # 4. 习题/视频特有字段（空值处理）
    is_complete = row['is_complete'] if row['is_complete'] is not None else 0
    is_correct = row['is_correct'] if row['is_correct'] is not None else -1
    is_view_analysis = row['is_view_analysis']
    watch_rate = row['watch_rate'] if row['watch_rate'] is not None else 0.0
    is_pause = row['is_pause']
    is_replay = row['is_replay']

    knowledgeResult = get_user_knowledge_stats(user_key, datetime.now())
    preferenceResult = get_user_resource_preference(user_key, datetime.now())

    # 保存用户交互数据到JSON文件
    # 数据库中仍会存在用户，但是这些用户没有可以用于推荐训练的行为记录，故直接用全0。
    # 故不需要遍历用户表来产生有所有用户信息的json,更何况，前脚训练完，后脚又有新用户加入，这种事情基本无解。
    save_user_interaction_data(user_key, post_3d_correct_rate, post_practice_count)

    # 实例化数据模型
    return TrainingSet(
        knowledge_accuracy=knowledgeResult['accuracy'],
        knowledge_total_count=knowledgeResult['totalCount'],
        resource_preference=preferenceResult['preference'],
        active_days=get_user_active_days(user_key),
        resource_form=get_resource_form_by_resource_key(resource_key),
        resource_knowledges=get_resource_knowledges(resource_key),
        resource_tags=get_resource_tags(resource_key),
        resource_time=get_resource_time(resource_key),
        post_3d_correct_rate=post_3d_correct_rate,
        post_practice_count=post_practice_count,
        is_first_submit_24h=is_first_submit_24h,
        is_complete=is_complete,
        is_correct=is_correct,
        is_view_analysis=is_view_analysis,
        watch_rate=watch_rate,
        is_pause=is_pause,
        is_replay=is_replay,
        correct_rate_change=correct_rate_change
    )