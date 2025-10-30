import torch
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple, Dict
from model_wrapper import ModelWrapper
import numpy as np
from sklearn.neighbors import KernelDensity

class HuggingFaceModel(ModelWrapper):
    def __init__(self, model_name: str, device: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._configure_tokenizer()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # 使用float16减少内存使用
            device_map="auto",
            attn_implementation = "eager"
        )
        self.device = device

    def _configure_tokenizer(self):
        """自动配置tokenizer参数"""
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"

        # 设置默认聊天模板（兼容Qwen、Llama等主流模型）
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = (
                "{% for message in messages %}"
                "{{ message.role }}\n{{ message.content }}"
                "{% endfor %}"
                "{% if add_generation_prompt %}assistant\n{% endif %}"
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _build_chat_messages(self, input_text: str, lang: str) -> List[dict]:
        """构建符合Hugging Face标准的消息结构"""
        if lang == "cn":
            system = "你是一个文本增强工具。将输入的中文网络攻击性内容中使用的谐音梗、反讽、表情符号等替代性表达，全部转换为对应的正常文字表达。在保留原意、攻击类型的前提下，强化原有情绪，使其更直接、更有冲击力。不要分析，不要解释，直接输出改写后的句子。"
        else:
            system = "You are a text enhancement tool. Generate a paraphrased version of the input tweet in the same language, limiting its length to around 50 characters and reinforcing the emotions it contains, while retaining its meaning and the type of cyberbullying. Don't think, just give the sentence."
        return [
            {
                "role": "system",
                "content": system
            },
            *self._get_few_shot_examples(),
            {"role": "user", "content": input_text}
        ]

    def _get_few_shot_examples(self) -> List[dict]:
        """返回few-shot示例"""
        examples = [
            {"input": "@risqiaditya f*** hate you", "output": "I fucking hate you so much"},
            {"input": "I fucking love you!", "output": "I love you very much"},
            {"input": "i fucking hate school so much", "output": "I don't want to go to school"},
            {"input": "辣鸡快递", "output": "垃圾快递，太废物了这个快递"},
            {"input": "NMSL", "output": "你妈死了"}
        ]
        return [
            {"role": "user", "content": ex["input"]} if i % 2 == 0 else {"role": "assistant", "content": ex["output"]}
            for ex in examples
            for i in range(2)
        ]

    def generate(self, prompts: List[str], lang: str, batch_size: int = 8, **kwargs) -> List[str]:
        all_outputs = []
        self.model.eval()

        with torch.no_grad():
            for i in tqdm(range(0, len(prompts), batch_size),
                          desc="Processing (HF)",
                          unit="batch"):
                # 生成符合模板的prompt
                batch_prompts = [
                    self.tokenizer.apply_chat_template(
                        self._build_chat_messages(text, lang=lang),
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False
                    ) for text in prompts[i:i + batch_size]
                ]

                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048,
                    add_special_tokens=False,
                    pad_to_multiple_of=8 if "cuda" in self.device else None,
                ).to(self.device)

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=kwargs.get("max_new_tokens", 128),
                    temperature=kwargs.get("temperature", 0.7),
                    top_p=kwargs.get("top_p", 0.9),
                    do_sample=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                # 仅解码新生成的文本
                decoded = self.tokenizer.batch_decode(
                    outputs[:, inputs.input_ids.shape[1]:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                processed = [self._postprocess_output(text) for text in decoded]
                all_outputs.extend(processed)

        return all_outputs

    def generate_with_metrics(self, texts: List[str], detector, n_samples: int = 5, batch_size: int = 8, lang = "en", **kwargs) -> List[Dict]:
        """
        为每个文本生成n个采样，并计算幻觉指标，返回包含指标的样本信息
        
        Args:
            texts: 输入文本列表
            detector: 幻觉检测器
            n_samples: 每个文本生成的样本数
            batch_size: 批处理大小
            **kwargs: 其他生成参数
            
        Returns:
            List[Dict]: 每个输入文本的样本信息列表，包含生成文本和指标
        """
        all_sample_metrics = []
        
        # 批量生成所有样本
        # 构建所有prompts (texts * n_samples)
        all_prompts = []
        prompt_indices = []  # 记录每个原始文本对应的样本索引范围
        for idx, text in enumerate(texts):
            # 截取前500字符
            truncated_text = text[:500] if len(text) > 500 else text

            prompts = [truncated_text] * n_samples
            start_idx = len(all_prompts)
            all_prompts.extend(prompts)
            prompt_indices.append((start_idx, start_idx + n_samples))
        
        # 批量生成样本
        all_samples = self.generate(all_prompts, batch_size=batch_size, lang=lang,  **kwargs)
        
        # 按原始文本分组处理
        for idx, text in enumerate(texts):
            start_idx, end_idx = prompt_indices[idx]
            samples = all_samples[start_idx:end_idx]
            
            # 计算每个样本的幻觉指标
            sample_metrics = []
            for sample in samples:
                try:
                    # 构建完整的对话历史用于指标计算
                    # 截取前500字符
                    truncated_text = text[:500] if len(text) > 500 else text
                    chat_messages = self._build_chat_messages(truncated_text, lang=lang)
                    # 添加生成的回复
                    chat_messages.append({"role": "assistant", "content": sample})
                    
                    # 构建完整的prompt
                    full_prompt = self.tokenizer.apply_chat_template(
                        chat_messages,
                        tokenize=False,
                        add_generation_prompt=False  # 不再添加生成提示，因为我们已经有了回复
                    )
                    
                    # 编码输入
                    inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
                    # 计算提示长度（不包括生成的回复）
                    prompt_only = self.tokenizer.apply_chat_template(
                        self._build_chat_messages(truncated_text, lang=lang),
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    prompt_length = len(self.tokenizer(prompt_only, return_tensors="pt")['input_ids'][0])
                    total_length = inputs['input_ids'].shape[1]
                    
                    # 运行前向传递以获取logits、hidden_states和attentions
                    with torch.no_grad():
                        full_outputs = self.model(
                            inputs.input_ids,
                            output_hidden_states=True,
                            output_attentions=True,
                            return_dict=True
                        )
                    
                    # 检查输出是否包含必要的属性
                    if not hasattr(full_outputs, 'logits') or full_outputs.logits is None:
                        raise ValueError("Model output does not contain logits")
                    
                    # 如果模型不支持output_hidden_states或output_attentions，设置为None
                    if not hasattr(full_outputs, 'hidden_states') or full_outputs.hidden_states is None:
                        full_outputs.hidden_states = None
                    
                    if not hasattr(full_outputs, 'attentions') or full_outputs.attentions is None:
                        full_outputs.attentions = None
                    
                    # 转换数据类型为float32以避免BFloat16不支持的问题
                    logits_float32 = full_outputs.logits.to(torch.float32) if full_outputs.logits.dtype == torch.bfloat16 else full_outputs.logits
                    hidden_states_float32 = [
                        tuple(h.to(torch.float32) if h.dtype == torch.bfloat16 else h for h in layer) 
                        for layer in full_outputs.hidden_states
                    ] if full_outputs.hidden_states else None
                    attentions_float32 = [
                        tuple(a.to(torch.float32) if a.dtype == torch.bfloat16 else a for a in layer) 
                        for layer in full_outputs.attentions
                    ] if full_outputs.attentions else None
                    
                    # 创建兼容的输出对象
                    from transformers.utils import ModelOutput
                    from dataclasses import dataclass
                    
                    @dataclass
                    class GeneratorOutputs(ModelOutput):
                        logits: torch.FloatTensor = None
                        hidden_states: torch.FloatTensor = None
                        attentions: torch.FloatTensor = None
                    
                    compatible_outputs = GeneratorOutputs(
                        logits=logits_float32,
                        hidden_states=hidden_states_float32,
                        attentions=attentions_float32
                    )
                    
                    # 使用中间层计算指标
                    num_layers = self.model.config.num_hidden_layers
                    target_layer = num_layers // 2
                    
                    metrics = detector.calculate_metrics(
                        compatible_outputs,
                        tok_lens=[prompt_length, total_length],
                        layer_num=target_layer,
                        top_k=50
                    )
                    
                    # 使用SVD分数作为排序指标（分数越低幻觉越少）
                    sample_metrics.append({
                        'text': sample,
                        'svd_score': metrics['svd_score'][0],
                        'attention_eigenvalue_product': metrics['attention_eigenvalue_product'][0],
                        'logit_entropy': metrics['logit_entropy'][0],
                        'full_prompt': full_prompt
                    })
                except Exception as e:
                    print(f"计算样本指标时出错: {str(e)}")
                    # 如果计算失败，使用默认值
                    sample_metrics.append({
                        'text': sample,
                        'svd_score': float('inf'),
                        'attention_eigenvalue_product': 0.0,
                        'logit_entropy': 0.0,
                        'full_prompt': ''
                    })
            
            # 根据SVD分数排序
            sorted_samples = sorted(sample_metrics, key=lambda x: x['svd_score'])

            # --- KDE 修改开始 ---
            # 默认选择 SVD 分数最低的样本（作为回退）
            best_sample_text = sorted_samples[0]['text'] if sorted_samples else ""

            # 提取所有有效的 SVD 分数
            svd_scores = [s['svd_score'] for s in sample_metrics if s['svd_score'] != float('inf')]

            # 至少需要2个点才能进行有意义的KDE
            if len(svd_scores) > 1:
                try:
                    scores_np = np.array(svd_scores).reshape(-1, 1)

                    # 动态计算带宽 (使用 Silverman's rule of thumb)
                    std_dev = np.std(scores_np)
                    n = len(scores_np)

                    # 避免 std_dev 或 n 为 0 导致的问题
                    if std_dev > 1e-6 and n > 0:
                        bandwidth = (4 * (std_dev ** 5) / (3 * n)) ** (1 / 5)
                    else:
                        bandwidth = 0.1  # 如果标准差为0或点太少，则使用一个小的默认值

                    # 确保带宽不是0
                    if bandwidth < 1e-6:
                        bandwidth = 0.1

                    # 1. 拟合 KDE
                    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(scores_np)

                    # 2. 在一个精细的网格上评估密度以找到峰值
                    # 覆盖从略低于最小值到略高于最大值的范围
                    grid_min = np.min(scores_np) - std_dev * 0.5
                    grid_max = np.max(scores_np) + std_dev * 0.5
                    search_grid = np.linspace(grid_min, grid_max, 500).reshape(-1, 1)

                    log_density = kde.score_samples(search_grid)

                    # 3. 找到密度峰值对应的 SVD 分数
                    peak_index = np.argmax(log_density)
                    density_peak_score = search_grid[peak_index][0]

                    # 4. 找到最接近密度峰值分数的原始样本
                    best_sample_by_kde = min(sample_metrics,
                                             key=lambda s: abs(s['svd_score'] - density_peak_score)
                                             if s['svd_score'] != float('inf')
                                             else float('inf'))

                    best_sample_text = best_sample_by_kde['text']

                except Exception as e:
                    print(f"KDE 计算失败: {str(e)}，将回退到最低 SVD 分数。")
                    # best_sample_text 已经设置为回退值，无需操作
                    pass

            all_sample_metrics.append({
                'input_text': text,
                'samples': sorted_samples,
                'best_sample': best_sample_text  # 使用KDE选择的样本或回退的样本
            })
            
        return all_sample_metrics

    def _postprocess_output(self, text: str) -> str:
        """后处理生成的文本"""
        # 移除可能的模板残留
        text = re.sub(r"<\|im_end\|>", "", text).strip()
        # 提取第一个句子（可选）
        return text.split("\n")[0].split(".")[0] + "." if "." in text else text