import openai
from tqdm import tqdm
from typing import List
import logging
from model_wrapper import ModelWrapper
from prompt_utils import extract_augmented_text

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class OpenAIModel(ModelWrapper):
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: str = None, base_url: str = None):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        all_outputs = []
        for prompt in tqdm(prompts, desc="Processing (OpenAI)"):
            try:
                messages = self._parse_prompt(prompt)
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=kwargs.get("temperature", 0.7),
                    top_p=kwargs.get("top_p", 0.9),
                    max_tokens=kwargs.get("max_new_tokens", 128)
                )
                processed = extract_augmented_text(response.choices[0].message.content)
                all_outputs.append(processed)
            except Exception as e:
                logging.error(f"Error generating text: {str(e)}")
                all_outputs.append("")
        return all_outputs

    # _parse_prompt 方法保持不变...
    def _parse_prompt(self, prompt: str) -> List[dict]:
        """解析原始提示为OpenAI API所需的格式"""
        # 这里需要根据实际prompt格式实现解析逻辑
        # 例如，如果prompt是简单的字符串，则直接作为user消息
        return [{"role": "user", "content": prompt}]