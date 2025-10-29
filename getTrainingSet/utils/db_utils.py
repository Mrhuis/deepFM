# 数据库通用工具
import mysql.connector
from mysql.connector import Error, cursor
# 修改导入路径，指向新的config文件位置
import sys
import os as sys_os
sys.path.append(sys_os.path.join(sys_os.path.dirname(__file__), '..', '..'))
from config import DB_CONFIG


def get_db_connection():
    """获取数据库连接"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        if conn.is_connected():
            return conn
        raise Error("数据库连接失败")
    except Error as e:
        print(f"获取连接错误: {e}")
        raise


def close_db_connection(conn, cursor):
    """关闭数据库连接和游标"""
    if cursor:
        cursor.close()
    if conn and conn.is_connected():
        conn.close()
        print("数据库连接已关闭")