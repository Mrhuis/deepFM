import time
from datetime import datetime
import logging
import argparse
import os

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

def parse_arguments():
    """
    解析命令行参数并更新配置
    """
    parser = argparse.ArgumentParser(description='推荐系统主程序')
    
    # 数据配置参数
    parser.add_argument('--train-num', type=int, help='训练数据条数')
    parser.add_argument('--batch-size', type=int, help='批次大小')
    
    # 模型配置参数
    parser.add_argument('--embedding-dim', type=int, help='类别特征嵌入维度')
    parser.add_argument('--dnn-hidden', nargs='+', type=int, help='隐藏层结构')
    parser.add_argument('--dropout', type=float, help='Dropout比例')
    
    # 训练配置参数
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--lr', type=float, help='学习率')
    
    # 其他配置参数
    parser.add_argument('--high-matching-count', type=int, help='高匹配备选资源个数')
    parser.add_argument('--recommend-used-resources', type=int, choices=[-1, 0, 1], 
                        help='是否推荐使用过后的资源 -1: 不使用过, 0: 无限制, 1: 使用过')
    parser.add_argument('--used-model', type=str, help='使用的模型')
    
    # 配置文件选项
    parser.add_argument('--config-file', type=str, help='配置文件路径')
    
    # 运行选项
    parser.add_argument('--mode', choices=['once', 'schedule'], default='once',
                        help='运行模式: once(执行一次后退出), schedule(按计划执行)')
    
    return parser.parse_args()

def update_config_with_args(args):
    """
    根据命令行参数更新配置
    """
    import config
    
    # 如果指定了配置文件，优先从文件加载
    if args.config_file and os.path.exists(args.config_file):
        try:
            from config_manager import apply_config_from_file
            if apply_config_from_file(args.config_file):
                print(f"已从配置文件 {args.config_file} 加载配置")
        except ImportError:
            print("警告: config_manager 模块不可用，无法加载配置文件")
        except Exception as e:
            print(f"警告: 从配置文件加载配置失败: {e}")
    
    # 更新数据配置
    if args.train_num is not None:
        config.DEFAULT_TRAIN_NUM = args.train_num
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    
    # 更新模型配置
    if args.embedding_dim is not None:
        config.EMBEDDING_DIM = args.embedding_dim
    if args.dnn_hidden is not None:
        config.DNN_HIDDEN = args.dnn_hidden
    if args.dropout is not None:
        config.DROPOUT = args.dropout
    
    # 更新训练配置
    if args.epochs is not None:
        config.EPOCHS = args.epochs
    if args.lr is not None:
        config.LR = args.lr
    
    # 更新其他配置
    if args.high_matching_count is not None:
        config.HIGH_MATCHING_CANDIDATE_RESOURCE_COUNT = args.high_matching_count
    if args.recommend_used_resources is not None:
        config.RECOMMEND_USED_RESOURCES = args.recommend_used_resources
    if args.used_model is not None:
        config.USED_MODEL = args.used_model

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_arguments()
    
    # 根据参数更新配置
    update_config_with_args(args)
    
    # 立即执行一次任务
    logger.info("立即执行一次推荐系统流水线任务")
    run_daily_pipeline()
    
    # 根据模式决定是否启动定时任务
    if args.mode == 'schedule':
        logger.info("启动定时任务调度器")
        schedule_daily_task()
    else:
        logger.info("程序执行完毕，退出")