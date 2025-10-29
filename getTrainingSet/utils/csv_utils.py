import csv
import os
from dataclasses import dataclass, asdict, fields
from datetime import datetime
# 添加导入config模块
import sys
import os as sys_os
sys.path.append(sys_os.path.join(sys_os.path.dirname(__file__), '../..'))
from modelsContainer import TrainingSet
from config import TRAINING_SETS_DIR


def save_processed_result(processed_row: TrainingSet, file_path: str = None):
    """
    将TrainingSet对象写入CSV文件，支持重复调用追加数据，首次写入自动添加表头

    格式规则：
        - 字段分隔符：竖线 |
        - 列表类型属性：元素用逗号 , 分隔（如 [0.8, 0.9] → "0.8,0.9"）
        - 空值处理：None 转为空字符串
        - 写入模式：首次写入含表头，后续调用追加数据行

    参数：
        processed_row: 待保存的TrainingSet实例
        file_path: 输出CSV文件路径，默认为当前日期命名的文件，格式为YYYYMMDD.csv
    """
    # 如果未指定文件路径，则使用当前日期作为文件名
    if file_path is None:
        # 确保目录存在
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        training_sets_dir = os.path.join(base_dir, TRAINING_SETS_DIR)
        os.makedirs(training_sets_dir, exist_ok=True)
        file_path = os.path.join(training_sets_dir, f"{datetime.now().strftime('%Y%m%d')}.csv")
    
    # getModel. 获取类属性相关信息（保证表头和数据顺序一致）
    # 表头：类的属性名列表（按类定义顺序）
    headers = [field.name for field in fields(TrainingSet)]
    # 数据：对象属性值字典（按属性名映射）
    row_dict = asdict(processed_row)

    # 2. 处理数据行：转换各属性值为字符串，适配格式要求
    processed_values = []
    for header in headers:
        value = row_dict[header]

        # 处理列表类型：元素转字符串，用逗号连接
        if isinstance(value, list):
            list_str = ",".join([str(item) if item is not None else "" for item in value])
            processed_values.append(list_str)
        # 处理非列表类型：None转空字符串，其他直接转字符串
        else:
            str_val = str(value) if value is not None else ""
            processed_values.append(str_val)

    # 3. 拼接数据行（用|分隔字段）
    data_line = "|".join(processed_values)

    # 4. 写入文件：判断是否首次写入（决定是否加表头）
    # 检查文件是否存在且非空
    is_first_write = not (os.path.exists(file_path) and os.path.getsize(file_path) > 0)

    with open(file_path, "a", encoding="utf-8", newline="") as f:
        # 首次写入：先写表头
        if is_first_write:
            header_line = "|".join(headers)  # 表头用|分隔
            f.write(header_line + "\n")
        # 追加数据行
        f.write(data_line + "\n")