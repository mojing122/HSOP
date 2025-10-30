import os
import pandas as pd
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def process_and_save_batch(output_csv_path: str, batch_df: pd.DataFrame, first_batch: bool):
    """保存批次数据并处理文件头"""
    try:
        batch_df.to_csv(
            output_csv_path,
            mode='a',
            header=first_batch,
            index=False
        )
        logging.info(f"成功保存 {len(batch_df)} 条记录到 {output_csv_path}")
    except Exception as e:
        logging.error(f"保存数据时发生错误: {str(e)}")


def load_processed_progress(output_csv_path: str, total_rows: int) -> int:
    """加载处理进度，返回已处理的行数"""
    try:
        if os.path.exists(output_csv_path):
            processed_df = pd.read_csv(output_csv_path, names=['tweet_text', 'cyberbullying_type'])
            return len(processed_df)
        return 0
    except Exception as e:
        logging.error(f"读取进度时发生错误: {str(e)}")
        return 0


def clean_commas_except_last(line: str) -> str:
    line = line.rstrip("\n\r")
    last_comma_pos = line.rfind(",")
    if last_comma_pos == -1:
        return line
    first_part = line[:last_comma_pos]
    last_field = line[last_comma_pos + 1:]
    cleaned_first_part = first_part.replace(",", "，")
    return cleaned_first_part + "," + last_field