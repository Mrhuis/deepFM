import json
import os
from typing import Dict, Any

def load_config_from_file(config_file: str) -> Dict[str, Any]:
    """
    从JSON文件加载配置
    
    Args:
        config_file: 配置文件路径
        
    Returns:
        包含配置项的字典
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"配置文件 {config_file} 不存在")
        
    with open(config_file, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
        
    return config_data

def update_runtime_config(config_data: Dict[str, Any]) -> None:
    """
    更新运行时配置
    
    Args:
        config_data: 配置数据字典
    """
    import config
    
    # 遍历配置项并更新
    for key, value in config_data.items():
        if hasattr(config, key):
            setattr(config, key, value)
            print(f"已更新配置项 {key} = {value}")
        else:
            print(f"警告: 配置项 {key} 在config模块中不存在")

def save_config_to_file(config_file: str, config_data: Dict[str, Any]) -> None:
    """
    将配置保存到JSON文件
    
    Args:
        config_file: 配置文件路径
        config_data: 配置数据字典
    """
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=4, ensure_ascii=False)

def export_current_config() -> Dict[str, Any]:
    """
    导出当前运行时配置
    
    Returns:
        包含当前配置项的字典
    """
    import config
    import inspect
    
    # 获取config模块中所有大写的配置项
    config_items = {}
    for name, value in inspect.getmembers(config):
        if not name.startswith('__') and name.isupper():
            config_items[name] = value
            
    return config_items

def apply_config_from_file(config_file: str) -> bool:
    """
    从文件应用配置
    
    Args:
        config_file: 配置文件路径
        
    Returns:
        应用是否成功
    """
    try:
        config_data = load_config_from_file(config_file)
        update_runtime_config(config_data)
        return True
    except Exception as e:
        print(f"从文件应用配置失败: {e}")
        return False