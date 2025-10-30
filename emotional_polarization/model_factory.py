from huggingface_model import HuggingFaceModel
from ollama_model import OllamaModel
from openai_model import OpenAIModel


def get_model_wrapper(model_type: str, model_name: str, **kwargs):
    """根据模型类型获取对应的模型包装器实例"""
    if model_type == "huggingface":
        return HuggingFaceModel(model_name=model_name, **kwargs)
    elif model_type == "ollama":
        return OllamaModel(model_name=model_name, **kwargs)
    elif model_type == "openai":
        return OpenAIModel(model_name=model_name, **kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")