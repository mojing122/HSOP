import ollama
from tqdm import tqdm
from typing import List
import logging
from model_wrapper import ModelWrapper
from prompt_utils import extract_augmented_text

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class OllamaModel(ModelWrapper):
    def __init__(self, model_name: str = "llama2", base_url: str = "http://localhost:11434"):
        self.client = ollama.Client(host=base_url)
        self.model_name = model_name

    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        all_outputs = []
        for prompt in tqdm(prompts, desc="Processing (Ollama)"):
            try:
                response = self.client.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={
                        "temperature": kwargs.get("temperature", 0.7),
                        "top_p": kwargs.get("top_p", 0.9),
                        "max_tokens": kwargs.get("max_new_tokens", 128)
                    }
                )
                processed = extract_augmented_text(response["response"])
                all_outputs.append(processed)
            except Exception as e:
                logging.error(f"Error generating text: {str(e)}")
                all_outputs.append("")
        return all_outputs