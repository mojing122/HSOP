from data_utils import process_and_save_batch, load_processed_progress, clean_commas_except_last
from model_factory import get_model_wrapper
import pandas as pd
from tqdm import tqdm
import logging
from src.hallucination_metrics import HallucinationDetector
import argparse

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_samples_with_metrics(texts, model_wrapper, detector, n_samples=5, batch_size=8, lang="en"):
    """
    为每个文本生成n个采样，并计算幻觉指标
    
    Args:
        texts: 输入文本列表
        model_wrapper: 模型包装器
        detector: 幻觉检测器
        n_samples: 每个文本生成的样本数
        batch_size: 批处理大小
        dataset: 处理数据库
    
    Returns:
        List[str]: 每个输入文本的最佳样本列表
    """
    # 使用模型包装器内置的方法生成样本和计算指标
    sample_results = model_wrapper.generate_with_metrics(
        texts, 
        detector, 
        n_samples=n_samples,
        batch_size=batch_size,
        lang=lang
    )
    
    # 提取最佳样本
    best_samples = [result['best_sample'] for result in sample_results]
    
    # 记录详细信息（可选）
    for result in sample_results:
        logging.info(f"原始文本: {result['input_text']}")
        logging.info(f"生成了 {len(result['samples'])} 个样本")
        logging.info(f"最佳样本 (SVD分数: {result['samples'][0]['svd_score']:.4f}): {result['best_sample']}")
    
    return best_samples


def main(input_csv_path: str,
         output_csv_path: str,
         model_type: str = "huggingface",
         model_name: str = "./Qwen3-4B",
         batch_size: int = 8,
         n_samples: int = 5,  # 每个文本生成的样本数
         lang: str = None,
         **model_kwargs):
    # 初始化模型
    model_wrapper = get_model_wrapper(model_type, model_name, **model_kwargs)
    
    # 使用模型包装器中的模型和tokenizer初始化幻觉检测器
    logging.info("初始化幻觉检测器...")
    detector = HallucinationDetector(model_wrapper.model, model_wrapper.tokenizer)

    # 清理CSV
    clean_path = input_csv_path.replace(".csv", "_clean.csv")
    with open(input_csv_path, "r", encoding="utf-8") as infile, \
            open(clean_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            cleaned = clean_commas_except_last(line)
            outfile.write(cleaned + "\n")

    if True:
        # 加载数据
        df = pd.read_csv(input_csv_path, names=["cyberbullying_type", "tweet_text"], header=None)
        total_rows = len(df)
    else:
        df = pd.read_csv(clean_path, names=["tweet_text", "cyberbullying_type"], header=0)
        total_rows = len(df)

    # 加载处理进度
    processed_count = load_processed_progress(output_csv_path, total_rows)

    # 初始化输出文件
    if processed_count == 0:
        pd.DataFrame(columns=['tweet_text', 'cyberbullying_type']).to_csv(output_csv_path, index=False, header=False)

    # 分批处理
    start_index = processed_count
    for i in tqdm(range(start_index, total_rows, batch_size),
                  desc=f"Processing ({model_type.upper()})",
                  total=(total_rows - start_index + batch_size - 1) // batch_size,
                  initial=start_index // batch_size,
                  unit='batch'):
        try:
            end_index = min(i + batch_size, total_rows)
            batch_df = df.iloc[i:end_index].copy()
            texts = batch_df["tweet_text"].astype(str).tolist()

            # 生成增强文本（使用幻觉指标选择最佳样本）
            new_texts = generate_samples_with_metrics(
                texts, 
                model_wrapper, 
                detector, 
                n_samples=n_samples,
                batch_size=batch_size,
                lang=lang
            )

            # 更新数据
            batch_df["tweet_text"] = batch_df["tweet_text"] + " " + new_texts
            batch_df.dropna(inplace=True)

            # 实时保存
            first_batch = (i == 0) and (processed_count == 0)
            process_and_save_batch(output_csv_path, batch_df, first_batch)

        except Exception as e:
            logging.error(f"处理批次 {i // batch_size} 时发生错误: {str(e)}")
            continue

    logging.info(f"处理完成！最终结果保存至 {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model generation and metric calculation.")

    # Required Arguments
    parser.add_argument("--dataset", type=str, required=True, help="The name of the dataset, e.g., 'ToxiGen'.")
    parser.add_argument("--model_name",  type=str, required=True, help="Hugging Face model path or name, e.g., '../models/Qwen3-4B'.")

    # Optional Arguments
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for generation (Default: 16).")
    parser.add_argument("--n_samples", type=int, default=5, help="Number of samples to generate per input text (Default: 5).")
    parser.add_argument("--lang", type=str, default="en", help="Language setting ('en' or 'zh') (Default: 'en').")
    parser.add_argument("--model_type", type=str, default="huggingface", help="Model type (Default: 'huggingface').")

    args = parser.parse_args()

    # Construct input and output file paths based on command-line arguments
    input_file = f"../soft_prompt/datasets/{args.dataset}/test_no_aug.csv"
    # Keep the output file naming convention
    output_file = f"../soft_prompt/datasets/{args.dataset}/test.csv"

    print(f"--- Run Parameters ---")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model_name}")
    print(f"Number of Samples (n_samples): {args.n_samples}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Language: {args.lang}")
    print(f"Input File: {input_file}")
    print(f"Output File: {output_file}")
    print(f"----------------------")

    # Assuming 'main' is the function that orchestrates the generation process
    main(input_file,
         output_file,
         model_type=args.model_type,
         model_name=args.model_name,
         batch_size=args.batch_size,
         n_samples=args.n_samples,
         lang=args.lang)