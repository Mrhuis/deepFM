import os
import json
from typing import List, Dict, Any
from datetime import datetime
from flask import Flask, request, jsonify
from config import USER_RECOMMEND_RESOURCE_DIR
from match.userResources import generate_single_user_recommendations, get_user_recommendations, mark_resource_as_used


app = Flask(__name__)

# 用于记录上次执行流水线任务的时间和状态
last_pipeline_run = None
is_pipeline_running = False

@app.route('/recommendations/<user_key>', methods=['GET'])
def get_recommendations(user_key: str):
    """
    获取指定用户的推荐资源列表
    
    Args:
        user_key: 用户标识
        
    Returns:
        JSON格式的推荐资源列表
    """
    try:
        # 根据USED_MODEL的值决定是否传递model_name参数
        from config import USED_MODEL
        if USED_MODEL == "DEFAULT":
            # 调用get_user_recommendations函数获取推荐资源
            recommendations = get_user_recommendations(user_key)
        else:
            # 调用get_user_recommendations函数获取推荐资源，并传递model_name参数
            recommendations = get_user_recommendations(user_key, model_name=USED_MODEL)
        
        # 返回JSON响应
        return jsonify({
            "success": True,
            "data": recommendations,
            "message": f"成功获取用户 {user_key} 的推荐资源"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "data": [],
            "message": f"获取用户 {user_key} 的推荐资源时出错: {str(e)}"
        }), 500


@app.route('/recommendations/<user_key>/<resource_key>/mark_used', methods=['POST'])
def mark_resource_used(user_key: str, resource_key: str):
    """
    标记用户推荐资源为已使用
    
    Args:
        user_key: 用户标识
        resource_key: 资源标识
        
    Returns:
        JSON格式的响应结果
    """
    try:
        # 调用mark_resource_as_used函数标记资源为已使用
        success = mark_resource_as_used(user_key, resource_key)
        
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


# 启动Flask应用的函数
def run_api():
    app.run(host='0.0.0.0', port=5000, debug=True)


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