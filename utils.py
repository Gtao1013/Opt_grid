from datetime import datetime


def safe_filename(base_name, extension="", include_timestamp=True, iteration=None):
    """生成安全的文件名，避免使用非法字符"""
    parts = [base_name]

    if iteration is not None:
        parts.append(f"iter_{iteration}")

    if include_timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        parts.append(timestamp)

    filename = "_".join(parts)

    if extension:
        filename = f"{filename}.{extension}"

    # 移除任何可能的非法字符
    illegal_chars = '<>:"/\\|?*'
    for char in illegal_chars:
        filename = filename.replace(char, '_')

    return filename