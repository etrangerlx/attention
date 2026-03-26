# 导入相关需要的包
import math
import torch
import torch.nn as nn

import warnings
warnings.filterwarnings(action="ignore")


class SelfAttV1(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttV1, self).__init__()
        self.hidden_dim = hidden_dim
        # 一般 Linear 都是默认有 bias
        # 一般来说， input dim 的 hidden dim
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X):
        # X shape is: (batch, seq_len, hidden_dim)， 一般是和 hidden_dim 相同
        # 但是 X 的 final dim 可以和 hidden_dim 不同
        Q = self.query_proj(X)
        K = self.key_proj(X)
        V = self.value_proj(X)

        # shape is: (batch, seq_len, seq_len)
        # torch.matmul 可以改成 Q @ K.T
        # 其中 K 需要改成 shape 为： (batch, hidden_dim, seq_len)
        attention_value = torch.matmul(Q, K.transpose(-1, -2))
        attention_wight = torch.softmax(
            attention_value / math.sqrt(self.hidden_dim), dim=-1
        )
        # print(attention_wight)
        # shape is: (batch, seq_len, hidden_dim)
        output = torch.matmul(attention_wight, V)
        return output

class SelfAttV2(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim
        # 这样可以进行加速, 那么为什么现在 Llama, qwen, gpt 等
        self.proj = nn.Linear(dim, dim * 3)

        self.output_proj = nn.Linear(dim, dim)

    def forward(self, X):
        # X shape is: (batch, seq, dim)

        QKV = self.proj(X)  # (batch, seq, dim * 3)
        # reshape 从希望的 q, k, 的形式
        Q, K, V = torch.split(QKV, self.dim, dim=-1)

        # print(x)
        att_weight = torch.softmax(
            Q @ K.transpose(-1, -2) / math.sqrt(self.dim), dim=-1
        )
        output = att_weight @ V
        return self.output_proj(output)

class SelfAttV3(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim
        # 这样可以进行加速
        self.proj = nn.Linear(dim, dim * 3)
        # 一般是 0.1 的 dropout，一般写作 config.attention_probs_dropout_prob
        # hidden_dropout_prob 一般也是 0.1
        self.att_drop = nn.Dropout(0.1)

        # 不写这个应该也没人怪，应该好像是 MultiHeadAttention 中的产物，这个留给 MultiHeadAttention 也没有问题；
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, X, attention_mask=None):
        # attention_mask shape is: (batch, seq)
        # X shape is: (batch, seq, dim)

        QKV = self.proj(X)  # (batch, seq, dim * 3)
        # reshape 从希望的 q, k, 的形式
        Q, K, V = torch.split(QKV, self.dim, dim=-1)

        att_weight = Q @ K.transpose(-1, -2) / math.sqrt(self.dim)
        if attention_mask is not None:
            # 给 weight 填充一个极小的值
            att_weight = att_weight.masked_fill(attention_mask == 0, float("-1e20"))

        att_weight = torch.softmax(att_weight, dim=-1)

        # 这里在 BERT中的官方代码也说很奇怪，但是原文中这么用了，所以继承了下来
        # （用于 output 后面会更符合直觉？）
        att_weight = self.att_drop(att_weight)

        output = att_weight @ V
        ret = self.output_proj(output)
        return ret

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, nums_head) -> None:
        super().__init__()
        self.nums_head = nums_head

        # 一般来说，
        self.head_dim = hidden_dim // nums_head
        self.hidden_dim = hidden_dim

        # 一般默认有 bias，需要时刻注意，hidden_dim = head_dim * nums_head，所以最终是可以算成是 n 个矩阵
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # gpt2 和 bert 类都有，但是 llama 其实没有
        self.att_dropout = nn.Dropout(0.1)
        # 输出时候的 proj
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X, attention_mask=None):
        # 需要在 mask 之前 masked_fill
        # X shape is (batch, seq, hidden_dim)
        # attention_mask shape is (batch, seq)

        batch_size, seq_len, _ = X.size()

        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)

        # shape 变成 （batch_size, num_head, seq_len, head_dim）
        q_state = Q.view(batch_size, seq_len, self.nums_head, self.head_dim).permute(
            0, 2, 1, 3
        )
        k_state = K.view(batch_size, seq_len, self.nums_head, self.head_dim).transpose(
            1, 2
        )
        v_state = V.view(batch_size, seq_len, self.nums_head, self.head_dim).transpose(
            1, 2
        )
        # 注意：这里需要用 head_dim，而不是 hidden_dim
        attention_weight = (
            q_state @ k_state.transpose(-1, -2) / math.sqrt(self.head_dim)
        )
        print(type(attention_mask))
        if attention_mask is not None:
            attention_weight = attention_weight.masked_fill(
                attention_mask == 0, float("-1e20")
            )

        # 第四个维度 softmax
        attention_weight = torch.softmax(attention_weight, dim=3)
        print(attention_weight)

        attention_weight = self.att_dropout(attention_weight)
        output_mid = attention_weight @ v_state

        # 重新变成 (batch, seq_len, num_head, head_dim)
        # 这里的 contiguous() 是相当于返回一个连续内存的 tensor，一般用了 permute/tranpose 都要这么操作
        # 如果后面用 Reshape 就可以不用这个 contiguous()，因为 view 只能在连续内存中操作
        output_mid = output_mid.transpose(1, 2).contiguous()

        # 变成 (batch, seq, hidden_dim),
        output = output_mid.view(batch_size, seq_len, -1)
        output = self.o_proj(output)
        return output
# 忽略了 attention_mask, attention_dropout; 
class GroupQueryAttention(nn.Module):
    def __init__(self, hidden_dim, nums_head, nums_key_value_head):
        super().__init__()
        assert hidden_dim % nums_head == 0 # 可以整除
        assert nums_head % nums_key_value_head == 0  # N 个 query head 为一组

        self.hidden_dim = hidden_dim
        self.nums_head = nums_head
        self.nums_key_value_head = nums_key_value_head
        self.head_dim = hidden_dim // nums_head

        # 初始化 qkv o
        self.q_proj = nn.Linear(hidden_dim, nums_head * self.head_dim)  # out feature_size (nums_head * head_dim)
        # k v out shape (nums_key_value_head * head_dim)
        self.k_proj = nn.Linear(hidden_dim, nums_key_value_head * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, nums_key_value_head * self.head_dim)

        self.o_proj = nn.Linear(hidden_dim, hidden_dim) # input_size nums_head * head_dim

    def forward(self, X, attention_mask=None):
        # X shape (batch, seq, hidden_dim)
        batch_size, seq, _ = X.size()

        # qkv projection
        q = self.q_proj(X)  # （batch, seq, hidden_dim)
        k = self.k_proj(X)
        v = self.v_proj(X) 

        # attention_weight 目标shape 是 (batch, nums_head, seq, seq)
        q = q.view(batch_size, seq, self.nums_head, self.head_dim)
        k = k.view(batch_size, seq, self.nums_key_value_head, self.head_dim)
        v = v.view(batch_size, seq, self.nums_key_value_head, self.head_dim)

        # 关注: nums_head 和 nums_key_value_head 的关系
        q = q.transpose(1, 2) # (b, nums_head, seq, head_dim)
        k = k.transpose(1, 2) # (b, nums_key_value_head, seq, head_dim)
        v = v.transpose(1, 2)  # (b, nums_key_value_head, seq, head_dim)

        # k v repeat； （广播操作）
        k = k.repeat_interleave(self.nums_head // self.nums_key_value_head, dim=1)
        v = v.repeat_interleave(self.nums_head // self.nums_key_value_head, dim=1)

        attention_score = (q @ k.transpose(2, 3)) / math.sqrt(self.head_dim)

        attention_weight = torch.softmax(attention_score, dim=-1)
        # （attention_mask 忽略） # 可以看前面的视频

        output = attention_weight @ v  # (b, nums_head, seq, head_dim)

        # output projection 变成 (b, seq, hidden_dim)
        output = output.transpose(1, 2).contiguous()
        final_output = self.o_proj(output.view(batch_size, seq, -1))

        return final_output

# Multi Query Attention
# 由于 MQA 是 GQA 的一种特殊形式，因此只要在参数设置的时候将 nums_key_value_head = 1 就是 Multi Query Self-Attention。
if __name__ == "__main__":
    X = torch.rand(3, 2, 4)
    net = SelfAttV1(4) 
    print(net(X).shape)
    
    net = SelfAttV2(4)
    print(net(X).shape)

    X = torch.rand(3, 4, 2)
    b = torch.tensor(
        [
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 0],
        ]
    )
    print(b.shape)
    mask = b.unsqueeze(dim=1).repeat(1, 4, 1)

    net = SelfAttV3(2)
    print(net(X, mask).shape)

    attention_mask = (
    torch.tensor(
        [
            [0, 1],
            [0, 0],
            [1, 0],
        ]
    ).unsqueeze(1)
    .unsqueeze(2)
    .expand(3, 8, 2, 2))

    x = torch.rand(3, 2, 128)
    net = MultiHeadAttention(128, 8)
    print(net(x, attention_mask).shape)
    # 测试
    x = torch.rand(3, 2, 128)
    net = GroupQueryAttention(128, 8, 4)
    net(x).shape