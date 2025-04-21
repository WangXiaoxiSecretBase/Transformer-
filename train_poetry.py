import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import os
from transformer import DecoderOnlyTransformer, TextDataset, build_vocab, train, evaluate, generate_text

def load_poetry_data(file_path, max_poems=None):
    """加载唐诗数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        poems = json.load(f)
    
    if max_poems:
        poems = poems[:max_poems]
    
    # 提取诗的内容并分字
    processed_poems = []
    for poem in poems:
        content = poem.get('paragraphs', [])
        if content:
            # 将段落合并，并分成字符列表
            text = ''.join(content)
            processed_poems.append(list(text))
    
    return processed_poems

def train_poetry_model(data_path, model_save_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # 加载数据
    poems = load_poetry_data(data_path)
    print(f"加载了 {len(poems)} 首诗")
    
    # 构建词汇表
    vocab = build_vocab(poems, special_tokens=['<pad>', '<unk>', '<sos>', '<eos>'])
    vocab_size = len(vocab)
    print(f"词汇表大小: {vocab_size}")
    
    # 创建数据集和数据加载器
    dataset = TextDataset(poems, vocab, seq_len=128)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 创建模型
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=256,
        num_heads=4,
        d_ff=512,
        num_layers=4,
        dropout=0.1
    ).to(device)
    
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    
    # 训练模型
    epochs = 10
    best_loss = float('inf')
    
    for epoch in range(epochs):
        train_loss = train(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{epochs}, 训练损失: {train_loss:.4f}")
        
        # 保存最佳模型
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab': vocab,
            }, model_save_path)
            print(f"模型已保存到 {model_save_path}")
        
        # 生成一首示例诗
        prompt = ['春', '日', '和', '风', '拂', '山', '岗']
        generated = generate_text(model, prompt, vocab, device, max_len=100, temperature=0.7)
        print("生成的诗:")
        print(''.join(generated))
        print("-" * 30)
    
    return model, vocab

def generate_poetry(model_path, prompt, max_len=100, temperature=0.7, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    vocab = checkpoint['vocab']
    vocab_size = len(vocab)
    
    # 创建模型并加载参数
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=256,
        num_heads=4,
        d_ff=512,
        num_layers=4,
        dropout=0.1
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 将提示转换为字符列表
    prompt_chars = list(prompt)
    
    # 生成诗
    generated = generate_text(model, prompt_chars, vocab, device, max_len, temperature)
    return ''.join(generated)

if __name__ == "__main__":
    # 设置路径
    data_path = "poetry_data.json"  # 唐诗数据文件路径
    model_save_path = "poetry_model.pt"
    
    # 检查文件是否存在
    if not os.path.exists(data_path):
        print(f"错误: 找不到数据文件 {data_path}")
        print("请确保数据文件位于正确位置，或更新代码中的文件路径。")
        exit(1)
    
    # 训练模型
    model, vocab = train_poetry_model(data_path, model_save_path)
    
    # 测试生成
    prompts = [
        "春日和风拂山岗",
        "明月几时有",
        "床前明月光",
        "飞流直下三千尺"
    ]
    
    print("\n生成多首诗:")
    for prompt in prompts:
        print(f"\n提示: {prompt}")
        poem = generate_poetry(model_save_path, prompt)
        print(poem) 