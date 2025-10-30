# 1. Generating diverse augmented samples

(1) Download the [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) model (or other LLMs) file to the ./models folder.

(2) Run the data augment script.

```bash
python ./emotional_polarization/main.py --dataset ToxiGen --model_name ../models/Qwen3-4B
```

# 2. Soft Prompt-tuning classification

(1)Download the [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) model (or other BERT models) file to the ./soft_prompt/models folder.

(2) Run the Soft Prompt-tuning classification script.

```bash
cd soft_prompt
python fewshot.py --result_file ./output.txt --dataset tweets --template_id 0 --seed 142 --kptw_lr 3e-05 --verbalizer manual --template_type soft --batch_size 32 --model_name_or_path models/bert-base-uncased --shot 20
```
