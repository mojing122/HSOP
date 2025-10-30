import re
from typing import List


def build_few_shot_prompt(input_text: str) -> str:
    few_shot_examples = [
        {"input": "@risqiaditya f*** hate you", "output": "I fucking hate you so much"},
        {"input": "I fucking love you!", "output": "I love you very much"},
        {"input": "i fucking hate school so much", "output": "I don't want to go to school"},
        {"input": "辣鸡快递", "output": "垃圾快递，太废物了这个快递"},
        {"input": "NMSL", "output": "你妈死了"}
    ]

    prompt = "<|begin_of_text|>"
    prompt += "<|start_header_id|>system<|end_header_id|>\n\n"
    prompt += "You are a text enhancement tool. Generate a paraphrased version of the input tweet in the same language, limiting its length to around 50 characters and reinforcing the emotions it contains, while retaining its meaning and the type of cyberbullying. Don't think, just give the sentence.<|eot_id|>\n"

    for ex in few_shot_examples:
        prompt += f"<|start_header_id|>user<|end_header_id|>\n\nInput: {ex['input']}<|eot_id|>\n"
        prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\nAugmented: {ex['output']}<|eot_id|>\n"

    prompt += f"<|start_header_id|>user<|end_header_id|>\n\nInput: {input_text}<|eot_id|>\n"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\nAugmented: "
    return prompt


def extract_augmented_text(output: str) -> str:
    start_marker = "Augmented: "
    end_marker = "<|eot_id|>"

    start_idx = output.rfind(start_marker)
    if start_idx == -1:
        return output.strip()

    start_idx += len(start_marker)
    end_idx = output.find(end_marker, start_idx)
    return output[start_idx:end_idx].strip() if end_idx != -1 else output[start_idx:].strip()


def batch_process_texts(texts: List[str], model_wrapper, batch_size: int = 8) -> List[str]:
    prompts = [build_few_shot_prompt(text) for text in texts]
    return model_wrapper.generate(prompts, batch_size=batch_size)