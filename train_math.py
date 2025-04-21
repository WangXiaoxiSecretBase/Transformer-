import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from transformer import Transformer, MathDataset, build_vocab, generate_math_data, train, evaluate, predict_math

def train_math_model(data_path, model_save_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # 生成数学数据
    num_samples = 10000
    equations, results = generate_math_data(num_samples)
    print(f"生成了 {num_samples} 个数学样本")
    
    # 构建词汇表
    src_vocab = build_vocab(equations)
    tgt_vocab = build_vocab(results)
    print(f"源词汇表大小: {len(src_vocab)}, 目标词汇表大小: {len(tgt_vocab)}")
    
    # 创建数据集和数据加载器
    dataset = MathDataset(equations, results, src_vocab, tgt_vocab)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32)
    
    # 创建模型
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=256,
        num_heads=4,
        d_ff=512,
        num_layers=4,
        dropout=0.1
    ).to(device)
    
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 是 <pad> 的索引
    
    # 训练模型
    epochs = 10
    best_loss = float('inf')
    
    for epoch in range(epochs):
        train_loss = train(model, train_dataloader, optimizer, criterion, device)
        eval_loss = evaluate(model, test_dataloader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs}, 训练损失: {train_loss:.4f}, 验证损失: {eval_loss:.4f}")
        
        # 保存最佳模型
        if eval_loss < best_loss:
            best_loss = eval_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'src_vocab': src_vocab,
                'tgt_vocab': tgt_vocab,
            }, model_save_path)
            print(f"模型已保存到 {model_save_path}")
        
        # 测试一些示例
        test_equations = ['123+456', '789+321', '111+222']
        for eq in test_equations:
            equation = list(eq)
            result = predict_math(model, equation, src_vocab, tgt_vocab, device)
            print(f"方程: {''.join(equation)}, 预测结果: {''.join(result)}, 实际结果: {eval(eq)}")
    
    return model, src_vocab, tgt_vocab

def solve_math_problem(model_path, equation, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    src_vocab = checkpoint['src_vocab']
    tgt_vocab = checkpoint['tgt_vocab']
    
    # 创建模型并加载参数
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=256,
        num_heads=4,
        d_ff=512,
        num_layers=4,
        dropout=0.1
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 将方程转换为字符列表
    equation_chars = list(equation)
    
    # 预测结果
    result = predict_math(model, equation_chars, src_vocab, tgt_vocab, device)
    return ''.join(result)

if __name__ == "__main__":
    # 设置路径
    model_save_path = "math_model.pt"
    
    # 训练模型
    model, src_vocab, tgt_vocab = train_math_model(None, model_save_path)
    
    # 测试预测
    test_equations = [
        "1234+5678",
        "9876+1234",
        "5432+9876",
        "12345+67890"
    ]
    
    print("\n测试预测:")
    for equation in test_equations:
        print(f"方程: {equation}")
        result = solve_math_problem(model_save_path, equation)
        print(f"预测结果: {result}")
        print(f"实际结果: {eval(equation)}")
        print("-" * 30) 