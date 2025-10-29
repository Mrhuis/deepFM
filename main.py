import time
from datetime import datetime
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 用于记录上次执行流水线任务的时间
last_pipeline_run = None

def run_daily_pipeline():
    """
    按顺序执行三个主要模块的函数：
    1. main1 - 处理用户资源交互数据
    2. main2 - 训练DeepFM模型
    3. main3 - 生成用户推荐资源
    """
    global last_pipeline_run
    
    logger.info("开始执行每日推荐系统流水线任务")
    
    try:
        # 步骤1: 执行main1 - 处理用户资源交互数据
        logger.info("步骤1: 执行数据处理 (main1)")
        from getTrainingSet.data_processor import process_user_resource_interaction
        process_user_resource_interaction()
        logger.info("步骤1完成: 数据处理完成")
        
        # 步骤2: 执行main2 - 训练DeepFM模型
        logger.info("步骤2: 执行模型训练 (main2)")
        from getModel.main2 import train_deepfm
        from config import TRAINING_SETS_DIR, DEFAULT_TRAIN_NUM
        train_deepfm(
            csv_path=TRAINING_SETS_DIR,
            train_num=DEFAULT_TRAIN_NUM
        )
        logger.info("步骤2完成: 模型训练完成")
        
        # 步骤3: 执行main3 - 生成用户推荐资源
        logger.info("步骤3: 执行推荐生成 (main3)")
        from match.userResources import generate_user_recommendations
        from getTrainingSet.utils.trainingSet_utils import clear_cache
        generate_user_recommendations()
        clear_cache()
        logger.info("步骤3完成: 推荐生成完成")
        
        # 更新最后执行时间
        last_pipeline_run = datetime.now()
        
        logger.info("每日推荐系统流水线任务执行完成")
        
    except Exception as e:
        logger.error(f"执行过程中发生错误: {str(e)}")
        raise

def schedule_daily_task():
    """
    设置每天凌晨3点执行任务
    如果24小时内已经执行过立即任务，则跳过本次执行
    """
    global last_pipeline_run
    
    logger.info("任务调度已启动，将在每天凌晨3点自动执行")
    
    while True:
        # 获取当前时间
        now = datetime.now()
        # 计算下次运行时间（今天或明天的凌晨3点）
        next_run = now.replace(hour=3, minute=0, second=0, microsecond=0)
        if now >= next_run:
            # 如果当前时间已经超过今天的3点，则下次运行是明天的3点
            next_run = next_run.replace(day=next_run.day + 1)
        
        # 计算距离下次运行的秒数
        sleep_seconds = (next_run - now).total_seconds()
        logger.info(f"下次运行时间: {next_run}, 等待 {sleep_seconds} 秒")
        
        # 等待到下次运行时间
        time.sleep(sleep_seconds)
        
        # 检查是否在24小时内执行过立即任务
        if last_pipeline_run:
            time_diff = datetime.now() - last_pipeline_run
            if time_diff.total_seconds() < 24 * 60 * 60:  # 24小时
                logger.info("24小时内已执行过流水线任务，跳过本次定时执行")
                continue
        
        # 执行任务
        run_daily_pipeline()

if __name__ == "__main__":
    # 立即执行一次任务
    logger.info("立即执行一次推荐系统流水线任务")
    run_daily_pipeline()
    
    # 启动定时任务
    logger.info("启动定时任务调度器")
    schedule_daily_task()