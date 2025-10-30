# encoding: utf-8
import tqdm
from openprompt.data_utils.data_processor import DataProcessor
import torch
from openprompt.data_utils.utils import InputExample
import argparse
import numpy as np
import math
from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer, KnowledgeableVerbalizer, SoftVerbalizer, AutomaticVerbalizer, \
    PtuningTemplate
from openprompt.prompts import ManualTemplate, SoftTemplate, MixedTemplate
from openprompt.utils import metrics
import os
import csv

from sklearn.metrics import precision_recall_fscore_support

class CyberbullyingProcessor2(DataProcessor):

    def __init__(self):
        super().__init__()
        self.labels = ["cyberbullying", "not_cyberbullying"]

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.csv".format(split))
        examples = []
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)  # 跳过第一行（标题行）
            for idx, row in enumerate(reader):
                label, text_a = row
                example = InputExample(guid=str(idx), text_a=text_a, label=int(label))
                examples.append(example)
        return examples

class WikiProcessor(DataProcessor):

    def __init__(self):
        super().__init__()
        self.labels = ["not_cyberbullying", "cyberbullying"]

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.csv".format(split))
        examples = []
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, text_a = row
                example = InputExample(guid=str(idx), text_a=text_a, label=int(label))
                examples.append(example)
        return examples

class WeiBoProcessor(DataProcessor):

    def __init__(self):
        super().__init__()
        self.labels = ["非暴力", "暴力"]

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.csv".format(split))
        examples = []
        with open(path, encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, text_a = row
                text_a = text_a.replace('USERNAME', '某人')
                example = InputExample(guid=str(idx), text_a=text_a, label=int(label))
                examples.append(example)
        return examples

class CyberbullyingProcessor(DataProcessor):

    def __init__(self):
        super().__init__()
        self.labels = ["cyberbullying", "not_cyberbullying"]

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.csv".format(split))
        examples = []
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)  # 跳过第一行（标题行）
            for idx, row in enumerate(reader):
                text_a, label = row
                if type(label) == str:
                    label = self.labels.index(label)
                if label != 0 and label != 1:
                    print(label)
                example = InputExample(guid=str(idx), text_a=text_a, label=int(label))
                examples.append(example)
        return examples

parser = argparse.ArgumentParser("")
parser.add_argument("--shot", type=int, default=10)
parser.add_argument("--seed", type=int, default=144)

parser.add_argument("--plm_eval_mode", action="store_true")
parser.add_argument("--model", type=str, default='bert')
parser.add_argument("--model_name_or_path", default='models/bert-base-chinese')
parser.add_argument("--verbalizer", type=str)
parser.add_argument("--calibration", action="store_true")
parser.add_argument("--filter", default="none", type=str)
parser.add_argument("--template_type", type=str)
parser.add_argument("--template_id", type=int)
parser.add_argument("--dataset",type=str)
parser.add_argument("--result_file", type=str, default="../sfs_scripts/results_fewshot_manual_kpt.txt")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--max_epochs", type=int, default=5)
parser.add_argument("--kptw_lr", default=0.06, type=float)
parser.add_argument("--pred_temp", default=1.0, type=float)
parser.add_argument("--max_token_split", default=-1, type=int)
parser.add_argument("--uid", default=-1, type=int)
parser.add_argument("--rate_num", default=-1, type=int)
parser.add_argument("--soft_token_num", default=4, type=int)
parser.add_argument("--init_from_vocab", action="store_false")
parser.add_argument("--batch_size", default=30, type=int)
parser.add_argument("--learning_rate", default=4e-5, type=str)
args = parser.parse_args()

import random
this_run_unicode = str(random.randint(0, 1e10))

from openprompt.utils.reproduciblity import set_seed
set_seed(args.seed)

from openprompt.plms import load_plm
plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)

dataset = {}

if args.dataset == "tweets":
    dataset['train'] = CyberbullyingProcessor().get_train_examples("./datasets/Cyberbullying/tweets/")
    dataset['test'] = CyberbullyingProcessor().get_test_examples("./datasets/Cyberbullying/tweets/")
    dataset['validation'] = CyberbullyingProcessor().get_validation_examples("./datasets/Cyberbullying/tweets/")
    class_labels =CyberbullyingProcessor().get_labels()
    scriptsbase = "Cyberbullying/tweets"
    scriptformat = "txt"
    cutoff=0.5
    max_seq_l = 128
    batch_s = args.batch_size
elif args.dataset == "ToxiGen":
    dataset['train'] = WikiProcessor().get_train_examples("./datasets/Cyberbullying/ToxiGen/")
    dataset['test'] = WikiProcessor().get_test_examples("./datasets/Cyberbullying/ToxiGen/")
    #dataset['validation'] = CyberbullyingProcessor().get_validation_examples("./datasets/Cyberbullying/tweets2/")
    class_labels =WikiProcessor().get_labels()
    scriptsbase = "Cyberbullying/tweets"
    scriptformat = "txt"
    cutoff=0.5
    max_seq_l = 128
    batch_s = args.batch_size
elif args.dataset == "LHd":
    dataset['train'] = WikiProcessor().get_train_examples("./datasets/Cyberbullying/LHd/")
    dataset['test'] = WikiProcessor().get_test_examples("./datasets/Cyberbullying/LHd/")
    #dataset['validation'] = CyberbullyingProcessor().get_validation_examples("./datasets/Cyberbullying/tweets2/")
    class_labels =WikiProcessor().get_labels()
    scriptsbase = "Cyberbullying/tweets"
    scriptformat = "txt"
    cutoff=0.5
    max_seq_l = 128
    batch_s = args.batch_size
else:
    raise NotImplementedError

if args.template_type == "soft":
    mytemplate = SoftTemplate(model=plm, tokenizer=tokenizer, num_tokens=args.soft_token_num, initialize_from_vocab=args.init_from_vocab).from_file(f"scripts/{scriptsbase}/soft_template.txt", choice=args.template_id)
    with open("log.txt", "a") as f:
        f.write(str(mytemplate.text)+"\n")
elif args.template_type == 'ptuning':
    mytemplate = PtuningTemplate(model=plm, tokenizer=tokenizer, prompt_encoder_type="bilstm").from_file(f"./scripts/{scriptsbase}/ptuning_template.txt", choice=args.template_id)
elif args.template_type == "manual":
    mytemplate = ManualTemplate(tokenizer=tokenizer).from_file(f"./scripts/{scriptsbase}/manual_template.txt", choice=args.template_id)
elif args.template_type == "mixed":
    mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer).from_file(f"scripts/{scriptsbase}/mixed_template.txt", choice=args.template_id)
elif args.template_type == "cot":
    examples_with_label = open("../scripts/CoT/csqa.txt").read()
    template_text = examples_with_label + """\n\nQ: {\"placeholder\":\"text_a\"}\nAnswer Choices:\n{\"meta\":\"choices\", \"post_processing\": 'lambda x:\"|\".join([\"(\"+i[\"label\"]+\")\" + \" \" + i[\"text\"] for i in x])'}\nA: {\"mask\"}"""
    mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer,)
else:
    mytemplate = ManualTemplate(tokenizer=tokenizer).from_file(f"./scripts/{scriptsbase}/manual_template.txt", choice=args.template_id)

if args.verbalizer == "kpt":
    myverbalizer = KnowledgeableVerbalizer(tokenizer, classes=class_labels, candidate_frac=cutoff, pred_temp=args.pred_temp, max_token_split=args.max_token_split).from_file(f"scripts/{scriptsbase}/knowledgeable_verbalizer.{scriptformat}")
elif args.verbalizer == "kpt++":
    myverbalizer = ManualVerbalizer(tokenizer, classes=class_labels).from_file(f"scripts/{scriptsbase}/kpt++_verbalizer.{scriptformat}")
elif args.verbalizer == "manual":
    myverbalizer = ManualVerbalizer(tokenizer, classes=class_labels).from_file(f"scripts/{scriptsbase}/manual_verbalizer.{scriptformat}")
elif args.verbalizer == "soft":
    myverbalizer = SoftVerbalizer(tokenizer, plm=plm, classes=class_labels).from_file(f"scripts/{scriptsbase}/manual_verbalizer.{scriptformat}")
elif args.verbalizer == "auto":
    myverbalizer = AutomaticVerbalizer(tokenizer, classes=class_labels)
elif args.verbalizer == "adj":
    myverbalizer = ManualVerbalizer(tokenizer, classes=class_labels).from_file(
        f"./scripts/{scriptsbase}/adj_verbalizer.{scriptformat}")


# (contextual) calibration
if args.verbalizer in ["kpt","manual"]:
    if args.calibration or args.filter != "none":
        from openprompt.data_utils.data_sampler import FewShotSampler
        support_sampler = FewShotSampler(num_examples_total=200, also_sample_dev=False)
        dataset['support'] = support_sampler(dataset['train'], seed=args.seed)

        # for example in dataset['support']:
        #     example.label = -1 # remove the labels of support set for clarification
        support_dataloader = PromptDataLoader(dataset=dataset["support"], template=mytemplate, tokenizer=tokenizer, 
            tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3, 
            batch_size=batch_s,shuffle=False, teacher_forcing=False, predict_eos_token=False,
            truncate_method="tail")


from openprompt import PromptForClassification
use_cuda = True
prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False, plm_eval_mode=args.plm_eval_mode)
if use_cuda:
    prompt_model=  prompt_model.cuda()



# HP
# if args.calibration:
if args.verbalizer in ["kpt","manual"]:
    if args.calibration or args.filter != "none":
        org_label_words_num = [len(prompt_model.verbalizer.label_words[i]) for i in range(len(class_labels))]
        from openprompt.utils.calibrate import calibrate
        # calculate the calibration logits
        cc_logits = calibrate(prompt_model, support_dataloader)
        print("the calibration logits is", cc_logits)
        print("origial label words num {}".format(org_label_words_num))

    if args.calibration:
        myverbalizer.register_calibrate_logits(cc_logits)
        new_label_words_num = [len(myverbalizer.label_words[i]) for i in range(len(class_labels))]
        print("After filtering, number of label words per class: {}".format(new_label_words_num))



    
    # register the logits to the verbalizer so that the verbalizer will divide the calibration probability in producing label logits
    # currently, only ManualVerbalizer and KnowledgeableVerbalizer support calibration.

from openprompt.data_utils.data_sampler import FewShotSampler
sampler = FewShotSampler(num_examples_per_label=args.shot, also_sample_dev=True, num_examples_per_label_dev=args.shot)
dataset['train'], dataset['validation'] = sampler(dataset['train'], seed=args.seed)
#dataset['test'], dataset['validation'] = sampler(dataset['test'], seed=args.seed)
#dataset['test'] = MovielensProcessor().get_test_examples("./datasets/RecommendationSystem/movielens/")

train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3, 
    batch_size=batch_s,shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")

validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3, 
    batch_size=batch_s,shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")

# zero-shot test
test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3, 
    batch_size=batch_s,shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")


def evaluate(prompt_model, dataloader, desc):
    prompt_model.eval()
    allpreds = []
    alllabels = []
    pbar = tqdm.tqdm(dataloader, desc=desc)
    for step, inputs in enumerate(pbar):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
    RMSEacc = math.sqrt(sum([(i - j) ** 2 for i, j in zip(allpreds, alllabels)]) / len(allpreds))
    MAEacc = sum([abs(i - j) for i, j in zip(allpreds, alllabels)]) / len(allpreds)
    # Micro_F1 = metrics.classification_metrics(allpreds,alllabels,metric="micro-f1")
    precision, recall, f1, _ = precision_recall_fscore_support(alllabels, allpreds, average='binary')

    acclist=[acc,RMSEacc,MAEacc, recall, precision, f1]
    return acclist

############
#############
###############

from transformers import  AdamW, get_linear_schedule_with_warmup
loss_func = torch.nn.CrossEntropyLoss()


def prompt_initialize(verbalizer, prompt_model, init_dataloader):
    dataloader = init_dataloader
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Init_using_{}".format("train")):
            batch = batch.cuda()
            logits = prompt_model(batch)
        verbalizer.optimize_to_initialize()
   

if args.verbalizer == "soft":


    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters

    optimizer_grouped_parameters2 = [
        {'params': prompt_model.verbalizer.group_parameters_1, "lr":3e-5},
        {'params': prompt_model.verbalizer.group_parameters_2, "lr":3e-4},
    ]


    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)
    optimizer2 = AdamW(optimizer_grouped_parameters2)

    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1, 
        num_warmup_steps=0, num_training_steps=tot_step)

    scheduler2 = get_linear_schedule_with_warmup(
        optimizer2, 
        num_warmup_steps=0, num_training_steps=tot_step)

elif args.verbalizer == "auto":
    prompt_initialize(myverbalizer, prompt_model, train_dataloader)

    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters

    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)

    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1, 
        num_warmup_steps=0, num_training_steps=tot_step)
    
    optimizer2 = None
    scheduler2 = None

elif args.verbalizer == "kpt":
    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters

    # optimizer_grouped_parameters2 = [
    #     {'params': , "lr":1e-1},
    # ]
    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)

    optimizer2 = AdamW(prompt_model.verbalizer.parameters(), lr=args.kptw_lr)
    # print(optimizer_grouped_parameters2)

    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=0, num_training_steps=tot_step)

    # scheduler2 = get_linear_schedule_with_warmup(
    #     optimizer2,
    #     num_warmup_steps=0, num_training_steps=tot_step)
    scheduler2 = None

elif args.verbalizer == "manual" or args.verbalizer == ("kpt++"):
    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters

    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=args.kptw_lr)

    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1, 
        num_warmup_steps=0, num_training_steps=tot_step)
    
    optimizer2 = None
    scheduler2 = None
elif args.verbalizer == "adj":
    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters

    # optimizer_grouped_parameters2 = [
    #     {'params': , "lr":1e-1},
    # ]
    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=float(args.learning_rate))
    optimizer2 = AdamW(prompt_model.verbalizer.parameters(), lr=args.kptw_lr)
    # print(optimizer_grouped_parameters2)

    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=0, num_training_steps=tot_step)

    # scheduler2 = get_linear_schedule_with_warmup(
    #     optimizer2,
    #     num_warmup_steps=0, num_training_steps=tot_step)
    scheduler2 = None

tot_loss = 0 
log_loss = 0
best_val_acc = 0
for epoch in range(args.max_epochs):
    tot_loss = 0 
    prompt_model.train()
    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 1.0)
        tot_loss += loss.item()
        optimizer1.step()
        scheduler1.step()
        optimizer1.zero_grad()
        if optimizer2 is not None:
            optimizer2.step()
            optimizer2.zero_grad()
        if scheduler2 is not None:
            scheduler2.step()
            
    val_acc = evaluate(prompt_model, validation_dataloader, desc="Valid")
    if val_acc[0]>=best_val_acc:
        torch.save(prompt_model.state_dict(),f"./ckpts/{this_run_unicode}.ckpt")
        best_val_acc = val_acc[0]
    print("Epoch {}, val_acc {}, RMSE {}, MAE {}".format(epoch, val_acc[0], val_acc[1], val_acc[2]), flush=True)

prompt_model.load_state_dict(torch.load(f"./ckpts/{this_run_unicode}.ckpt"))
# prompt_model = prompt_model.cuda()

with torch.no_grad():
    test_acc = evaluate(prompt_model, test_dataloader, desc="Test")


    content_write = "="*20+"\n"
    content_write += f"dataset {args.dataset}\t"
    content_write += f"temp {args.template_id}\t"
    content_write += f"seed {args.seed}\t"
    content_write += f"shot {args.shot}\t"
    content_write += f"verb {args.verbalizer}\t"
    content_write += f"cali {args.calibration}\t"
    content_write += f"filt {args.filter}\t"
    content_write += f"maxsplit {args.max_token_split}\t"
    content_write += f"kptw_lr {args.kptw_lr}\t"
    content_write += " \n"
    content_write += f"Acc: {test_acc[0]}\t"
    content_write += f"RMSE: {test_acc[1]}\t"
    content_write += f"MAE: {test_acc[2]}\t"
    content_write += f"PRECISION: {test_acc[3]}\t"
    content_write += f"RECALL: {test_acc[4]}\t"
    content_write += f"F1 Score: {test_acc[5]}"
    content_write += "\n\n"

print(content_write)

os.remove(f"./ckpts/{this_run_unicode}.ckpt")