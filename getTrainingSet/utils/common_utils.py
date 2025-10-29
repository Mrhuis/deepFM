
from typing import Optional, List


# 内部工具函数：将逗号分隔的字符串（如"getModel.23， 2.33，3.01"）转为float列表（无效元素忽略）
def str_to_floatlist(value: Optional[str]) -> Optional[List[float]]:
    if not value:  # 空字符串或None直接返回None
        return None

    # 统一替换中文逗号为英文逗号，再按英文逗号分割
    # 处理可能的空格（如"getModel.23,  2.33"中的空格）
    elements = [elem.strip() for elem in value.replace('，', ',').split(',')]

    # 过滤空字符串（如连续逗号导致的空元素）
    valid_elements = [elem for elem in elements if elem]

    # 转换每个元素为float，忽略转换失败的元素
    float_list = []
    for elem in valid_elements:
        try:
            float_list.append(float(elem))
        except ValueError:
            continue  # 跳过无效数值

    # 若没有有效元素，返回None；否则返回列表
    return float_list if float_list else None


# 内部工具函数：将逗号分隔的字符串（如"getModel， 2，3，4"）转为int列表（无效元素忽略）
def str_to_intlist(value: Optional[str]) -> Optional[List[int]]:
    if not value:  # 空字符串或None直接返回None
        return None

    # 统一替换中文逗号为英文逗号，再按英文逗号分割
    # 处理可能的空格（如"getModel,  2, 3"中的空格）
    elements = [elem.strip() for elem in value.replace('，', ',').split(',')]

    # 过滤空字符串（如连续逗号导致的空元素）
    valid_elements = [elem for elem in elements if elem]

    # 转换每个元素为int，忽略转换失败的元素
    int_list = []
    for elem in valid_elements:
        try:
            int_list.append(int(elem))
        except ValueError:
            continue  # 跳过无效数值

    # 若没有有效元素，返回None；否则返回列表
    return int_list if int_list else None