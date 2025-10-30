import torch
import numpy as np
from .utils import centered_svd_val

class HallucinationDetector:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def get_attn_eig_prod(self, attns, layer_num=15, tok_lens=[], use_toklens=True):
        """Compute an eigenvalue-based attention score by analyzing attention matrices.

            This function takes the attention matrices of a given layer and for each sample,
            computes the mean log of the diagonal elements (assumed to be eigenvalues) across
            all attention heads. Slices are applied if `tok_lens` is used.

            Args:
                attns (list): A list of tuples, each containing attention matrices for all layers
                    and heads for a single sample.
                layer_num (int, optional): The layer index to evaluate. Defaults to 15.
                tok_lens (list, optional): A list of (start, end) indices for each sample to slice
                    the attention matrices. Defaults to [].
                use_toklens (bool, optional): Whether to slice the attention matrices using `tok_lens`.
                    Defaults to True.

            Returns:
                np.array: An array of computed attention-based eigenvalue scores for each sample.
            """
        attn_scores = []

        # 检查attns是否为None或attns[layer_num]是否为None
        if attns is None or layer_num >= len(attns) or attns[layer_num] is None:
            return np.array([0.0])

        eigscore = 0.0
        for attn_head_num in range(len(attns[layer_num])):  # iterating over number of attn heads
            # attns[i][layer_num][j] is of size seq_len x seq_len
            Sigma = attns[layer_num][attn_head_num]

            if use_toklens and tok_lens:
                i1, i2 = tok_lens[0], tok_lens[1]
                Sigma = Sigma[i1:i2, i1:i2]

            eigscore += torch.log(torch.diagonal(Sigma, 0)).mean()
        attn_scores.append(eigscore.item())
        return np.stack(attn_scores)

    def print_attn_head_diag(self, attns, sample_idx=0, layer_num=15, head_num=20, start=0, end=10):
        """
        打印指定样本、层和注意力头的注意力矩阵对角线中 start 到 end 范围的对数值。

        Args:
            attns (list): attention 数据，结构为 List[sample][layer][head] -> Tensor(seq_len, seq_len)
            sample_idx (int): 样本索引
            layer_num (int): 层号
            head_num (int): 头号（第20个头为 head_num=20）
            start (int): 起始下标（inclusive）
            end (int): 结束下标（exclusive）

        Returns:
            Tensor: 指定范围内对角线取对数后的值
        """
        # 取出注意力矩阵
        attn_matrix = attns[sample_idx][layer_num][head_num]  # shape: [seq_len, seq_len]

        # 取出对角线
        diag_vals = torch.diagonal(attn_matrix, 0)

        # 截取范围
        selected_vals = diag_vals[start:end]

        # 计算对数，避免 log(0) 报错，加一个很小的 epsilon
        epsilon = 1e-12
        log_vals = torch.log(selected_vals + epsilon)

        # 打印
        print(f"Log of diagonal values from head {head_num}, sample {sample_idx}, layer {layer_num} [{start}:{end}]:")
        print(log_vals)

        return log_vals

    def get_svd_eval(self, hidden_acts, layer_num=15, tok_lens=[], use_toklens=True):
        """Evaluate hidden states at a given layer using SVD-based scoring.

            For each sample, this function extracts the hidden states at a specified layer,
            optionally slices them according to `tok_lens`, and computes the SVD-based score.

            Args:
                hidden_acts (list): A list of tuples, each containing hidden states for all layers
                    for a single sample.
                layer_num (int, optional): The layer index to evaluate. Defaults to 15.
                tok_lens (list, optional): A list of (start, end) indices for each sample to slice
                    the hidden states. Defaults to [].
                use_toklens (bool, optional): Whether to slice the hidden states using `tok_lens`.
                    Defaults to True.

            Returns:
                np.array: An array of SVD-based scores for each sample.
            """
        svd_scores = []

        # 检查hidden_acts是否为None或hidden_acts[layer_num]是否为None
        if hidden_acts is None or layer_num >= len(hidden_acts) or hidden_acts[layer_num] is None:
            return np.array([0.0])

        Z = hidden_acts[layer_num]

        if use_toklens and tok_lens:
            i1, i2 = tok_lens[0], tok_lens[1]
            Z = Z[i1:i2, :]

        Z = torch.transpose(Z, 0, 1)
        svd_scores.append(centered_svd_val(Z).item())
        # print("Sigma matrix shape:",Z.shape[1])
        return np.stack(svd_scores)

    def logit_entropy(self, logits, tok_lens, top_k=None):
        """Compute the entropy of the model's output distribution over tokens.

            For each sample, this function computes the entropy of the softmax distribution
            over predicted tokens. If `top_k` is provided, only the top K predictions are considered
            when computing entropy.

            Args:
                logits: A list or array of model logits (samples x seq_len x vocab_size).
                tok_lens (list): A list of (start, end) indices specifying the portion of the
                    sequence to evaluate.
                top_k (int, optional): Number of top tokens to consider for computing the entropy.
                    If None, considers all tokens.

            Returns:
                np.array: An array of entropy values for each sample.
            """
        softmax = torch.nn.Softmax(dim=-1)
        scores = []


        i1, i2 = tok_lens[0], tok_lens[1]
        if top_k is None:
            l = softmax(torch.tensor(logits))[i1:i2]
            scores.append((-l * torch.log(l)).mean())
        else:
            l = logits[i1:i2]
            l = softmax(torch.topk(l, top_k, 1).values)
            scores.append((-l * torch.log(l)).mean())

        return np.stack(scores)

    def calculate_metrics(self, outputs, tok_lens, layer_num=15, top_k=None):
        logits = outputs.logits
        hidden_states = outputs.hidden_states
        attentions = outputs.attentions

        # 添加空值检查
        if logits is None:
            raise ValueError("Logits cannot be None")
        
        logit = logits[0].cpu()
        
        # 处理hidden_states，如果为None则使用空列表
        if hidden_states is None:
            hidden_act = None
        else:
            hidden_act = [x[0].to(torch.float32).detach().cpu() for x in hidden_states]
        
        # 处理attentions，如果为None则使用空列表
        if attentions is None:
            attn = None
        else:
            attn = [x[0].to(torch.float32).detach().cpu() for x in attentions]

        tok_lens = tok_lens

        attn_scores = self.get_attn_eig_prod(attn, layer_num=layer_num, tok_lens=tok_lens)
        svd_scores = self.get_svd_eval(hidden_act, layer_num=layer_num, tok_lens=tok_lens)
        entropy_scores = self.logit_entropy(logit, tok_lens, top_k=top_k)

        return {
            "attention_eigenvalue_product": attn_scores,
            "svd_score": svd_scores,
            "logit_entropy": entropy_scores
        }
