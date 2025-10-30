from typing import List


class ModelWrapper:
    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        raise NotImplementedError