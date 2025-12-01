# 定时任务：计算用户资源交互后3天的正确率
import schedule
import time
import mysql.connector
from datetime import datetime, timedelta
from mysql.connector import Error
from config import DB_CONFIG
from getTrainingSet.utils.db_utils import get_db_connection, close_db_connection


def calculate_post_3d_correct_rate():
    """
    每天凌晨1点执行的任务：
    1. 查找user_resource_interaction表中interaction_time是3天前的数据
    2. 根据user_knowledge_stats_20d表计算post_3d_correct_rate
    3. 更新user_resource_interaction表的effect_calc_time字段
    """
    conn = None
    cursor = None
    
    try:
        print(f"[{datetime.now()}] 开始执行定时任务：计算3天后正确率")
        
        # 获取数据库连接
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # 计算3天前的日期
        three_days_ago = (datetime.now() - timedelta(days=3)).date()
        three_days_ago_str = three_days_ago.strftime('%Y-%m-%d')
        
        # 查找3天前的用户资源交互记录
        query_interactions = """
            SELECT id, user_key, interaction_time 
            FROM user_resource_interaction 
            WHERE DATE(interaction_time) = %s AND effect_calc_time IS NULL
        """
        cursor.execute(query_interactions, (three_days_ago_str,))
        interactions = cursor.fetchall()
        
        print(f"找到 {len(interactions)} 条3天前的交互记录")
        
        # 遍历每条交互记录
        for interaction in interactions:
            interaction_id = interaction['id']
            user_key = interaction['user_key']
            
            # 计算该用户在交互时间点之后3天内的知识点正确率
            # 这里我们假设需要计算的是用户在这段时间内所有知识点的平均正确率
            query_knowledge_stats = """
                SELECT 
                    SUM(correct_count) as total_correct,
                    SUM(total_count) as total_questions
                FROM user_knowledge_stats_20d
                WHERE user_key = %s 
                AND record_time >= %s 
                AND record_time < %s
            """
            
            # 计算时间范围：交互时间点之后的3天
            start_time = interaction['interaction_time']
            end_time = start_time + timedelta(days=3)
            
            cursor.execute(query_knowledge_stats, (user_key, start_time, end_time))
            stats_result = cursor.fetchone()
            
            if stats_result and stats_result['total_questions'] and stats_result['total_questions'] > 0:
                correct_rate = stats_result['total_correct'] / stats_result['total_questions']
                print(f"用户 {user_key} 在交互后3天内的正确率: {correct_rate:.4f}")
                
                # 更新user_resource_interaction表的post_3d_correct_rate和effect_calc_time字段
                update_query = """
                    UPDATE user_resource_interaction 
                    SET post_3d_correct_rate = %s, effect_calc_time = %s
                    WHERE id = %s
                """
                cursor.execute(update_query, (correct_rate, datetime.now(), interaction_id))
                conn.commit()
                print(f"已更新记录 ID {interaction_id}")
            else:
                # 即使没有数据，也设置effect_calc_time，避免重复处理
                update_query = """
                    UPDATE user_resource_interaction 
                    SET effect_calc_time = %s
                    WHERE id = %s
                """
                cursor.execute(update_query, (datetime.now(), interaction_id))
                conn.commit()
                print(f"用户 {user_key} 在交互后3天内无答题记录，已标记为已处理 ID {interaction_id}")
        
        print(f"[{datetime.now()}] 定时任务执行完成")
        
    except Error as e:
        print(f"数据库错误: {e}")
        if conn:
            conn.rollback()
    except Exception as e:
        print(f"处理错误: {e}")
        if conn:
            conn.rollback()
    finally:
        close_db_connection(conn, cursor)


def run_scheduled_tasks():
    """
    运行定时任务调度器
    """
    # 设置每天凌晨1点执行任务
    schedule.every().day.at("01:00").do(calculate_post_3d_correct_rate)
    
    print("定时任务调度器已启动，将在每天凌晨1点执行任务...")
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # 每分钟检查一次


if __name__ == "__main__":
    run_scheduled_tasks()