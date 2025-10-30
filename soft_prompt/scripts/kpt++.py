# -*- coding:utf-8 -*-
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import statistics

# 初始化 BERT tokenizer 和模型
tokenizer = BertTokenizer.from_pretrained('/home/wsco/szq/code/promptLearning/models/bert-base-uncased')
model = BertModel.from_pretrained('/home/wsco/szq/code/promptLearning/models/bert-base-uncased')

label0 = 'offensive'
label1 = 'friendly'
verbalizer = 'attacking,loathsome,offense,vile,foul,horrific,aggressive,sickening,unsavory,unsavoury,nauseating,offence,nauseous,noisome,violative,objectionable,obscene,distasteful,horrid,abhorrent,abusive,obnoxious,disgusting,outrageous,repugnant,repulsive,detestable,revolting,hideous,scurrilous,wicked,marauding,repellent,offending,ghastly,unpleasant,invading,predatory,opprobrious'
# 使用逗号分割字符串，生成列表
verbalizer_list = verbalizer.split(',')
# 循环遍历列表中的每个元素
devs = {}
for word in verbalizer_list:
    # 使用 tokenizer 对两个字符串进行编码
    encoded_word = tokenizer(word, return_tensors='pt', padding=True, truncation=True)
    encoded_label0 = tokenizer(label0, return_tensors='pt', padding=True, truncation=True)
    encoded_label1 = tokenizer(label1, return_tensors='pt', padding=True, truncation=True)

    # 获取编码后的输入 ID
    input_ids_word = encoded_word['input_ids']
    input_ids_label0 = encoded_label0['input_ids']
    input_ids_label1 = encoded_label1['input_ids']

    # 使用模型获取词向量表示
    with torch.no_grad():
        outputs_word = model(input_ids_word)
        outputs_label0 = model(input_ids_label0)
        outputs_label1 = model(input_ids_label1)

    # 获取词向量表示的最后一层 hidden states
    word_embeddings_word = outputs_word.last_hidden_state
    word_embeddings_label0 = outputs_label0.last_hidden_state
    word_embeddings_label1 = outputs_label1.last_hidden_state

    # 计算平均向量
    average_vector_word = torch.mean(word_embeddings_word, dim=1)
    average_vector_label0 = torch.mean(word_embeddings_label0, dim=1)
    average_vector_label1 = torch.mean(word_embeddings_label1, dim=1)

    # 将 PyTorch 张量转换为 NumPy 数组
    average_vector_word = average_vector_word.detach().numpy()
    average_vector_label0 = average_vector_label0.detach().numpy()
    average_vector_label1 = average_vector_label1.detach().numpy()


    cos0 = cosine_similarity(average_vector_word, average_vector_label0)
    cos1 = cosine_similarity(average_vector_word, average_vector_label1)
    data = [float(cos0), float(cos1)]  # 两个数作为数据集的一部分
    std_dev = statistics.stdev(data)
    devs[word] = std_dev

sorted_dict = dict(sorted(devs.items(), key=lambda x: x[1],reverse=True))

keys_to_keep = list(sorted_dict.keys())[:30]  # 保留前30个键
output = ','.join(keys_to_keep)
print(output)