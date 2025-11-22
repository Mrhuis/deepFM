import os
import sys
import json
import logging
import glob
from typing import List, Dict, Any
from datetime import datetime
from flask import Flask, request, jsonify
from config import USER_RECOMMEND_RESOURCE_DIR, TRAINING_MODELS_DIR
from match.userResources import generate_single_user_recommendations, get_user_recommendations, mark_resource_as_used,get_user_recommendations_test, reset_user_resources_usage


app = Flask(__name__)

def configure_logging():
    """配置日志系统"""
    # 配置根日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)  # 强制输出到stdout
        ],
        force=True  # 强制重新配置日志
    )
    
    # 配置Flask应用的日志
    app.logger.setLevel(logging.INFO)
    # 让Flask使用我们配置的日志系统
    app.logger.handlers = []
    app.logger.propagate = True
    
    # 配置werkzeug日志（Flask的HTTP服务器）
    werkzeug_logger = logging.getLogger('werkzeug')
    werkzeug_logger.setLevel(logging.INFO)
    
    return logging.getLogger(__name__)

# 初始化日志
logger = configure_logging()

# 用于记录上次执行流水线任务的时间和状态
last_pipeline_run = None
is_pipeline_running = False

# Flask请求钩子 - 记录所有请求
@app.before_request
def log_request():
    """在每个请求前记录请求信息"""
    logger.info(f"收到请求: {request.method} {request.path}")
    logger.info(f"请求参数: {dict(request.args)}")
    if request.method in ['POST', 'PUT', 'PATCH']:
        logger.info(f"请求体: {request.get_data(as_text=True)[:200]}")  # 限制长度

@app.after_request
def log_response(response):
    """在每个请求后记录响应信息"""
    logger.info(f"响应状态: {response.status_code}")
    return response

@app.route('/recommendations/<user_key>', methods=['GET'])
def get_recommendations(user_key: str):
    """
    获取指定用户的推荐资源列表
    
    Args:
        user_key: 用户标识
        
    Returns:
        JSON格式的推荐资源列表
    """
    print(f"\n{'='*60}")
    print(f"[DEBUG] API调用开始: 获取用户 {user_key} 的推荐资源")
    print(f"{'='*60}\n")
    logger.info(f"=== API调用开始: 获取用户 {user_key} 的推荐资源 ===")
    try:
        # 根据USED_MODEL的值决定是否传递model_name参数
        # print("[DEBUG] 调用 get_user_recommendations_test 方法")
        # logger.info("调用 get_user_recommendations_test 方法")
        # recommendations = get_user_recommendations_test(user_key)
        # print(f"[DEBUG] 方法返回 {len(recommendations)} 条推荐资源")
        # logger.info(f"方法返回 {len(recommendations)} 条推荐资源")


        from config import USED_MODEL
        if USED_MODEL == "DEFAULT":
            # 调用get_user_recommendations函数获取推荐资源
            recommendations = get_user_recommendations(user_key)
        else:
            # 调用get_user_recommendations函数获取推荐资源，并传递model_name参数
            recommendations = get_user_recommendations(user_key, model_name=USED_MODEL)
        
        # 返回JSON响应
        print(f"[DEBUG] API调用结束: 成功获取用户 {user_key} 的推荐资源\n")
        logger.info(f"=== API调用结束: 成功获取用户 {user_key} 的推荐资源 ===")
        return jsonify({
            "success": True,
            "data": recommendations,
            "message": f"成功获取用户 {user_key} 的推荐资源"
        })
    except Exception as e:
        print(f"[DEBUG ERROR] 获取用户 {user_key} 的推荐资源时出错: {str(e)}\n")
        logger.error(f"=== API调用异常: 获取用户 {user_key} 的推荐资源时出错: {str(e)} ===")
        return jsonify({
            "success": False,
            "data": [],
            "message": f"获取用户 {user_key} 的推荐资源时出错: {str(e)}"
        }), 500


@app.route('/recommendations/<user_key>/<resource_key>/<int:form_id>/mark_used', methods=['POST'])
def mark_resource_used(user_key: str, resource_key: str, form_id: int):
    """
    标记用户推荐资源为已正确
    
    Args:
        user_key: 用户标识
        resource_key: 资源标识
        form_id: 资源类型标识
        
    Returns:
        JSON格式的响应结果
    """
    try:

        # 调用mark_resource_as_used函数标记资源为已正确
        success = mark_resource_as_used(user_key, resource_key, form_id)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"成功将用户 {user_key} 的资源 {resource_key} 标记为已使用"
            })
        else:
            return jsonify({
                "success": False,
                "message": f"未能将用户 {user_key} 的资源 {resource_key} 标记为已使用"
            }), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"标记用户 {user_key} 的资源 {resource_key} 时出错: {str(e)}"
        }), 500


@app.route('/recommendations/<user_key>/reset_usage', methods=['POST'])
def reset_user_resources(user_key: str):
    """
    重置用户所有推荐资源的使用状态（将所有资源的is_use字段设置为0）
    
    Args:
        user_key: 用户标识
        
    Returns:
        JSON格式的响应结果
    """
    try:
        # 调用reset_user_resources_usage函数重置用户资源使用状态
        success = reset_user_resources_usage(user_key)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"成功重置用户 {user_key} 的所有资源使用状态"
            })
        else:
            return jsonify({
                "success": False,
                "message": f"未能重置用户 {user_key} 的资源使用状态"
            }), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"重置用户 {user_key} 的资源使用状态时出错: {str(e)}"
        }), 500


@app.route('/run-pipeline', methods=['POST'])
def run_pipeline_immediately():
    """
    立即执行推荐系统流水线任务
    如果任务正在执行，则拒绝新的请求
    任务执行完成后，24小时内定时任务不会触发
    """
    global last_pipeline_run, is_pipeline_running
    
    try:
        # 检查是否正在执行中
        if is_pipeline_running:
            return jsonify({
                "success": False,
                "message": "流水线任务正在执行中，请稍后再试"
            }), 409  # Conflict
        
        # 检查是否在24小时内已经执行过
        if last_pipeline_run:
            time_diff = datetime.now() - last_pipeline_run
            if time_diff.total_seconds() < 24 * 60 * 60:  # 24小时
                remaining_time = 24 * 60 * 60 - time_diff.total_seconds()
                hours, remainder = divmod(remaining_time, 3600)
                minutes, _ = divmod(remainder, 60)
                return jsonify({
                    "success": False,
                    "message": f"24小时内只能执行一次流水线任务，请在 {int(hours)}小时{int(minutes)}分钟 后再试"
                }), 429  # Too Many Requests
        
        # 设置任务执行状态
        is_pipeline_running = True
        
        # 执行流水线任务
        from main import run_daily_pipeline
        run_daily_pipeline()
        
        # 更新最后执行时间和状态
        last_pipeline_run = datetime.now()
        is_pipeline_running = False
        
        return jsonify({
            "success": True,
            "message": "推荐系统流水线任务执行完成",
            "last_run": last_pipeline_run.isoformat()
        })
        
    except Exception as e:
        # 确保即使出现异常也重置运行状态
        is_pipeline_running = False
        return jsonify({
            "success": False,
            "message": f"执行流水线任务时出错: {str(e)}"
        }), 500


# 添加一个新的API端点用于更新配置参数
@app.route('/config', methods=['PUT'])
def update_config():
    """
    动态更新配置参数
    
    Args:
        JSON对象包含要更新的配置项
        
    Returns:
        JSON格式的响应结果
    """
    try:
        # 获取请求中的JSON数据
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "message": "请求体必须包含JSON数据"
            }), 400
            
        # 定义允许通过API修改的配置项
        allowed_configs = {
            "HIGH_MATCHING_CANDIDATE_RESOURCE_COUNT": int,
            "RECOMMEND_USED_RESOURCES": int,
            "USED_MODEL": str,
            "DEFAULT_TRAIN_NUM": int,
            "BATCH_SIZE": int,
            "EMBEDDING_DIM": int,
            "DNN_HIDDEN": list,
            "DROPOUT": float,
            "EPOCHS": int,
            "LR": float
        }
            
        # 导入配置模块
        import config
        
        # 更新配置项
        updated_configs = []
        invalid_configs = []
        
        for key, value in data.items():
            # 检查配置项是否允许修改
            if key in allowed_configs:
                # 检查配置项类型是否正确
                if isinstance(value, allowed_configs[key]):
                    # 更新配置项
                    setattr(config, key, value)
                    updated_configs.append(key)
                else:
                    invalid_configs.append(f"{key}(类型错误)")
            else:
                invalid_configs.append(key)
                
        # 构造响应消息
        message_parts = []
        if updated_configs:
            message_parts.append(f"成功更新配置项: {', '.join(updated_configs)}")
        if invalid_configs:
            message_parts.append(f"无效配置项: {', '.join(invalid_configs)}")
            
        return jsonify({
            "success": True,
            "message": "; ".join(message_parts),
            "updated_configs": updated_configs,
            "invalid_configs": invalid_configs
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"更新配置时出错: {str(e)}"
        }), 500


@app.route('/config', methods=['GET'])
def get_config():
    """
    获取当前配置参数
    
    Returns:
        JSON格式的所有配置项
    """
    try:
        import config
        import inspect
        
        # 定义允许通过API获取的配置项
        allowed_configs = {
            "HIGH_MATCHING_CANDIDATE_RESOURCE_COUNT": int,
            "RECOMMEND_USED_RESOURCES": int,
            "USED_MODEL": str,
            "DEFAULT_TRAIN_NUM": int,
            "BATCH_SIZE": int,
            "EMBEDDING_DIM": int,
            "DNN_HIDDEN": list,
            "DROPOUT": float,
            "EPOCHS": int,
            "LR": float
        }
        
        # 获取config模块中允许的配置项
        config_items = {}
        for name, value in inspect.getmembers(config):
            if not name.startswith('__') and name.isupper() and name in allowed_configs:
                config_items[name] = value
                
        return jsonify({
            "success": True,
            "data": config_items
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"获取配置时出错: {str(e)}"
        }), 500


@app.route('/config/export', methods=['GET'])
def export_config():
    """
    导出当前配置到文件
    
    Returns:
        JSON格式的配置数据
    """
    try:
        from config_manager import export_current_config
        
        config_data = export_current_config()
        
        # 定义允许通过API导出的配置项
        allowed_configs = {
            "HIGH_MATCHING_CANDIDATE_RESOURCE_COUNT": int,
            "RECOMMEND_USED_RESOURCES": int,
            "USED_MODEL": str,
            "DEFAULT_TRAIN_NUM": int,
            "BATCH_SIZE": int,
            "EMBEDDING_DIM": int,
            "DNN_HIDDEN": list,
            "DROPOUT": float,
            "EPOCHS": int,
            "LR": float
        }
        
        # 过滤配置项
        filtered_config_data = {
            key: value for key, value in config_data.items() 
            if key in allowed_configs
        }
        
        return jsonify({
            "success": True,
            "data": filtered_config_data
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"导出配置时出错: {str(e)}"
        }), 500


@app.route('/config/import', methods=['POST'])
def import_config():
    """
    从文件导入配置
    
    Returns:
        JSON格式的响应结果
    """
    try:
        # 获取请求中的JSON数据
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "message": "请求体必须包含JSON数据"
            }), 400
            
        # 定义允许通过API导入的配置项
        allowed_configs = {
            "HIGH_MATCHING_CANDIDATE_RESOURCE_COUNT": int,
            "RECOMMEND_USED_RESOURCES": int,
            "USED_MODEL": str,
            "DEFAULT_TRAIN_NUM": int,
            "BATCH_SIZE": int,
            "EMBEDDING_DIM": int,
            "DNN_HIDDEN": list,
            "DROPOUT": float,
            "EPOCHS": int,
            "LR": float
        }
            
        # 检查是否提供了配置文件路径
        config_file = data.get('config_file')
        if config_file:
            # 从文件加载配置
            from config_manager import apply_config_from_file
            if apply_config_from_file(config_file):
                return jsonify({
                    "success": True,
                    "message": f"成功从文件 {config_file} 导入配置"
                })
            else:
                return jsonify({
                    "success": False,
                    "message": f"从文件 {config_file} 导入配置失败"
                }), 500
        else:
            # 直接使用请求中的配置数据
            # 过滤允许的配置项
            filtered_data = {
                key: value for key, value in data.items() 
                if key in allowed_configs and isinstance(value, allowed_configs[key])
            }
            
            if not filtered_data:
                return jsonify({
                    "success": False,
                    "message": "没有提供有效的配置项"
                }), 400
            
            from config_manager import update_runtime_config
            update_runtime_config(filtered_data)
            
            return jsonify({
                "success": True,
                "message": "成功导入配置",
                "imported_configs": list(filtered_data.keys())
            })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"导入配置时出错: {str(e)}"
        }), 500


@app.route('/models', methods=['GET'])
def list_models():
    """
    获取models目录下所有.pth模型文件的文件名列表
    
    Returns:
        JSON格式的模型文件名列表
    """
    try:
        # 获取项目根目录
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
        # 构造models目录路径
        models_dir = os.path.join(base_dir, TRAINING_MODELS_DIR)
        
        # 检查models目录是否存在
        if not os.path.exists(models_dir):
            return jsonify({
                "success": False,
                "message": f"模型目录 {models_dir} 不存在"
            }), 404
        
        # 查找所有.pth文件
        pattern = os.path.join(models_dir, "*.pth")
        model_files = glob.glob(pattern)
        
        # 提取文件名（不含路径）
        model_names = [os.path.basename(f) for f in model_files]
        # 按文件名排序
        model_names.sort()
        
        return jsonify({
            "success": True,
            "data": model_names,
            "count": len(model_names)
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"获取模型文件列表时出错: {str(e)}"
        }), 500


# 启动Flask应用的函数
def run_api():
    """启动Flask API服务"""
    # 确保输出不被缓冲，让日志立即显示
    os.environ['PYTHONUNBUFFERED'] = '1'
    sys.stdout.flush()
    sys.stderr.flush()
    
    # 重新配置日志（因为debug模式会重启）
    configure_logging()
    
    logger.info("=" * 60)
    logger.info("Flask API 服务启动中...")
    logger.info(f"监听地址: http://0.0.0.0:5000")
    logger.info(f"本地访问: http://127.0.0.1:5000")
    logger.info("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=True)


# 示例调用
if __name__ == "__main__":
    # 如果直接运行此脚本，则启动Flask应用
    if os.getenv('FLASK_RUN_FROM_CLI') != 'true':
        run_api()
    else:
        # 示例：获取用户S0001的推荐资源
        recommendations = get_user_recommendations("S0001")
        print(f"用户 S0001 的推荐资源:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. resource_key: {rec['resource_key']}, form_type: {rec['form_type']}")