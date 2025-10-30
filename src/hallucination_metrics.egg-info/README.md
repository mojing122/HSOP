# Hallucination Metrics

A Python library to calculate hallucination metrics from the outputs of large language models (LLMs), such as logits, hidden states, and attentions.

## Usage

```python
import torch
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.hallucination_metrics import HallucinationDetector
from fastchat.model import get_conversation_template

def run_test(model_path: str, device: str = "cpu"):
    """
    This function loads a pre-trained model, generates text,
    and then calculates hallucination metrics.
    """
    # 1. Load a pre-trained model and tokenizer from a local path
    model_path = os.path.abspath(model_path)

    if not os.path.isdir(model_path):
        print(f"Error: Model path not found: {model_path}")
        return

    print(f"Loading model from: {model_path}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="eager").to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model from local path. Please ensure the directory contains all necessary model files (e.g., config.json, pytorch_model.bin).")
        print(e)
        return

    # Ensure the model is in evaluation mode
    model.eval()
    # 2. Prepare the input prompt
    system_prompt = ""
    prompt_text = "曾国藩是中国人吗？"
    print(f"Prompt: {prompt_text}")
    chat_template = get_conversation_template(model_path)
    chat_template.set_system_message(system_prompt.strip())
    chat_template.append_message(chat_template.roles[0], prompt_text.strip())
    full_prompt = chat_template.get_prompt()

    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    prompt_length = inputs['input_ids'].shape[1]

    # 3. Generate text and get outputs
    print("Generating text...")
    with torch.no_grad():
        # Generate based on the input IDs
        generated_ids = model.generate(
            inputs.input_ids,
            max_new_tokens=500,
            do_sample=True
        )

    generated_length = generated_ids.shape[-1]
    generated_text = tokenizer.decode(generated_ids[0][prompt_length:], skip_special_tokens=True)
    print(f"Generated Text: {generated_text}")

    # The detector requires logits, hidden_states, and attentions.
    # We need to run a forward pass on the generated sequence to get them.
    # print("Running forward pass to get logits, attentions, and hidden states for the full sequence...")
    with torch.no_grad():
        full_outputs = model(
            generated_ids,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True
        )

    # 4. Initialize the Hallucination Detector
    detector = HallucinationDetector(model, tokenizer)

    # 5. Calculate hallucination metrics
    print("Calculating hallucination metrics...")

    # We need to manually create a compatible output object for the detector
    from transformers.utils import ModelOutput
    from dataclasses import dataclass

    @dataclass
    class GeneratorOutputs(ModelOutput):
        logits: torch.FloatTensor = None
        hidden_states: torch.FloatTensor = None
        attentions: torch.FloatTensor = None

    compatible_outputs = GeneratorOutputs(
        logits=full_outputs.logits,
        hidden_states=full_outputs.hidden_states,
        attentions=full_outputs.attentions
    )

    # Use a middle layer for calculation
    num_layers = model.config.num_hidden_layers
    target_layer = num_layers // 2

    metrics = detector.calculate_metrics(
        compatible_outputs,
        tok_lens=[prompt_length,generated_length],
        layer_num=target_layer,
        top_k=50
    )

    # 6. Print the results
    print("\n--- Hallucination Metrics ---")
    print(f"Calculated for layer: {target_layer}")
    print(f"Attention Eigenvalue Product: {metrics['attention_eigenvalue_product'][0]:.4f}")
    print(f"SVD Score: {metrics['svd_score'][0]:.4f}")
    print(f"Logit Entropy: {metrics['logit_entropy'][0]:.4f}")
    print("---------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hallucination metric test on a local model.")
    parser.add_argument("--model_path", type=str, default="/home/mojing/models/Qwen3-1.7B", help="Path to the local Hugging Face model directory.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to load the model on (e.g., 'cpu', 'cuda:0').")
    args = parser.parse_args()
    run_test(args.model_path, args.device)
```
