import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # 创建一个足够长的位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 计算位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # 注册为缓冲区，不作为模型参数
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]

# 多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert self.head_dim * num_heads == d_model, "d_model必须能被num_heads整除"
        
        # 定义线性变换层
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 线性变换
        q = self.q_linear(q)  # [batch_size, q_len, d_model]
        k = self.k_linear(k)  # [batch_size, k_len, d_model]
        v = self.v_linear(v)  # [batch_size, v_len, d_model]
        
        # 分割多头
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, q_len, head_dim]
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, k_len, head_dim]
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, v_len, head_dim]
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [batch_size, num_heads, q_len, k_len]
        
        # 应用mask（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 应用softmax获取注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        context = torch.matmul(attn_weights, v)  # [batch_size, num_heads, q_len, head_dim]
        
        # 拼接多头结果
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # [batch_size, q_len, d_model]
        
        # 最终线性变换
        output = self.out_linear(context)  # [batch_size, q_len, d_model]
        
        return output, attn_weights

# 前馈神经网络
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# Encoder层
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 自注意力机制
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈神经网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

# Decoder层
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # 自注意力机制
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 交叉注意力机制
        attn_output, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # 前馈神经网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

# Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

# Decoder
class TransformerDecoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x

# 完整的Transformer模型
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 d_ff=2048, num_layers=6, dropout=0.1, max_seq_len=5000):
        super(Transformer, self).__init__()
        
        # 嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Encoder和Decoder
        self.encoder = TransformerEncoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = TransformerDecoder(d_model, num_heads, d_ff, num_layers, dropout)
        
        # 输出层
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        
        # 初始化参数
        self._init_parameters()
        
        self.d_model = d_model
        
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # src: [batch_size, src_len]
        # tgt: [batch_size, tgt_len]
        
        # 嵌入和位置编码
        src_embedded = self.positional_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_embedded = self.positional_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        
        # 编码
        enc_output = self.encoder(src_embedded, src_mask)
        
        # 解码
        dec_output = self.decoder(tgt_embedded, enc_output, src_mask, tgt_mask)
        
        # 输出层
        output = self.output_layer(dec_output)
        
        return output

# 生成方形后续遮罩（用于解码器中的自注意力）
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# 生成填充遮罩
def create_padding_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]

# Decoder-only Transformer（类似GPT）
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, d_ff=2048, 
                 num_layers=6, dropout=0.1, max_seq_len=5000):
        super(DecoderOnlyTransformer, self).__init__()
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # 使用Decoder层，但只用自注意力部分
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # 初始化参数
        self._init_parameters()
        
        self.d_model = d_model
        
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, mask=None):
        # x: [batch_size, seq_len]
        
        # 嵌入和位置编码
        x = self.positional_encoding(self.embedding(x) * math.sqrt(self.d_model))
        
        # 通过所有层
        for layer in self.layers:
            x = layer(x, mask)
        
        # 输出层
        output = self.output_layer(x)
        
        return output

# 数据集类
class MathDataset(Dataset):
    def __init__(self, equations, results, src_vocab, tgt_vocab, max_len=100):
        self.equations = equations
        self.results = results
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.equations)
    
    def __getitem__(self, idx):
        equation = self.equations[idx]
        result = self.results[idx]
        
        # 将方程和结果转换为索引
        src_indices = [self.src_vocab.get(token, self.src_vocab['<unk>']) for token in equation]
        tgt_indices = [self.tgt_vocab['<sos>']] + [self.tgt_vocab.get(token, self.tgt_vocab['<unk>']) for token in result] + [self.tgt_vocab['<eos>']]
        
        return {
            'src': torch.tensor(src_indices, dtype=torch.long),
            'tgt': torch.tensor(tgt_indices[:-1], dtype=torch.long),  # 输入到decoder
            'tgt_y': torch.tensor(tgt_indices[1:], dtype=torch.long)  # 期望输出
        }

# 文本数据集类
class TextDataset(Dataset):
    def __init__(self, texts, vocab, seq_len=50):
        self.texts = texts
        self.vocab = vocab
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # 将文本转换为索引
        indices = [self.vocab.get(token, self.vocab['<unk>']) for token in text]
        
        # 确保长度为seq_len+1（用于输入和目标）
        if len(indices) > self.seq_len + 1:
            start_idx = np.random.randint(0, len(indices) - self.seq_len - 1)
            indices = indices[start_idx:start_idx + self.seq_len + 1]
        else:
            # 填充
            indices = indices + [self.vocab['<pad>']] * (self.seq_len + 1 - len(indices))
        
        return {
            'input': torch.tensor(indices[:-1], dtype=torch.long),
            'target': torch.tensor(indices[1:], dtype=torch.long)
        }

# 生成数学数据集
def generate_math_data(num_samples, digits_range=(3, 5)):
    equations = []
    results = []
    
    for _ in range(num_samples):
        # 随机选择位数
        digits1 = np.random.randint(digits_range[0], digits_range[1] + 1)
        digits2 = np.random.randint(digits_range[0], digits_range[1] + 1)
        
        # 生成随机数
        num1 = np.random.randint(10**(digits1-1), 10**digits1)
        num2 = np.random.randint(10**(digits2-1), 10**digits2)
        
        # 计算结果
        result = num1 + num2
        
        # 转换为字符列表
        equation = list(str(num1) + '+' + str(num2))
        result = list(str(result))
        
        equations.append(equation)
        results.append(result)
    
    return equations, results

# 构建词汇表
def build_vocab(texts, special_tokens=None):
    if special_tokens is None:
        special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
    
    vocab = {token: idx for idx, token in enumerate(special_tokens)}
    idx = len(special_tokens)
    
    for text in texts:
        for token in text:
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    
    return vocab

# 训练函数
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        # 对于标准Transformer
        if isinstance(model, Transformer):
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            tgt_y = batch['tgt_y'].to(device)
            
            # 创建mask
            src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(device)  # [batch_size, 1, 1, src_len]
            tgt_mask = generate_square_subsequent_mask(tgt.size(1)).to(device)
            
            optimizer.zero_grad()
            
            output = model(src, tgt, src_mask, tgt_mask)
            loss = criterion(output.contiguous().view(-1, output.size(-1)), tgt_y.contiguous().view(-1))
            
        # 对于Decoder-only Transformer
        else:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            # 创建mask
            mask = generate_square_subsequent_mask(inputs.size(1)).to(device)
            
            optimizer.zero_grad()
            
            output = model(inputs, mask)
            loss = criterion(output.contiguous().view(-1, output.size(-1)), targets.contiguous().view(-1))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

# 评估函数
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # 对于标准Transformer
            if isinstance(model, Transformer):
                src = batch['src'].to(device)
                tgt = batch['tgt'].to(device)
                tgt_y = batch['tgt_y'].to(device)
                
                # 创建mask
                src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(device)
                tgt_mask = generate_square_subsequent_mask(tgt.size(1)).to(device)
                
                output = model(src, tgt, src_mask, tgt_mask)
                loss = criterion(output.contiguous().view(-1, output.size(-1)), tgt_y.contiguous().view(-1))
                
            # 对于Decoder-only Transformer
            else:
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)
                
                # 创建mask
                mask = generate_square_subsequent_mask(inputs.size(1)).to(device)
                
                output = model(inputs, mask)
                loss = criterion(output.contiguous().view(-1, output.size(-1)), targets.contiguous().view(-1))
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

# 预测函数（用于数学任务）
def predict_math(model, equation, src_vocab, tgt_vocab, device, max_len=100):
    model.eval()
    
    # 将方程转换为索引
    src_indices = [src_vocab.get(token, src_vocab['<unk>']) for token in equation]
    src = torch.tensor([src_indices], dtype=torch.long).to(device)
    
    # 初始化目标序列
    tgt = torch.tensor([[tgt_vocab['<sos>']]], dtype=torch.long).to(device)
    
    with torch.no_grad():
        for i in range(max_len):
            # 创建mask
            src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(device)
            tgt_mask = generate_square_subsequent_mask(tgt.size(1)).to(device)
            
            # 预测下一个token
            output = model(src, tgt, src_mask, tgt_mask)
            next_token = output[:, -1].argmax(dim=1).unsqueeze(1)
            
            # 添加到目标序列
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # 如果预测到<eos>，则停止
            if next_token.item() == tgt_vocab['<eos>']:
                break
    
    # 将索引转换回token
    idx_to_tgt = {idx: token for token, idx in tgt_vocab.items()}
    result = [idx_to_tgt[idx] for idx in tgt[0].tolist()[1:]]  # 跳过<sos>
    
    # 如果结果中有<eos>，则截断
    if '<eos>' in result:
        result = result[:result.index('<eos>')]
    
    return result

# 生成文本（用于语言模型任务）
def generate_text(model, prompt, vocab, device, max_len=100, temperature=1.0):
    model.eval()
    
    # 将提示转换为索引
    idx_to_token = {idx: token for token, idx in vocab.items()}
    prompt_indices = [vocab.get(token, vocab['<unk>']) for token in prompt]
    input_seq = torch.tensor([prompt_indices], dtype=torch.long).to(device)
    
    generated = list(prompt)
    
    with torch.no_grad():
        for _ in range(max_len):
            # 创建mask
            mask = generate_square_subsequent_mask(input_seq.size(1)).to(device)
            
            # 预测下一个token
            output = model(input_seq, mask)
            logits = output[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            # 添加到输入序列
            input_seq = torch.cat([input_seq, torch.tensor([[next_token]], device=device)], dim=1)
            
            # 添加到生成的文本
            generated.append(idx_to_token[next_token])
            
            # 如果预测到<eos>，则停止
            if next_token == vocab.get('<eos>', -1):
                break
    
    return generated
