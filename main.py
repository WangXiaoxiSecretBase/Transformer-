import argparse
import os
from train_math import train_math_model, solve_math_problem
from train_poetry import train_poetry_model, generate_poetry
import torch

def main():
    parser = argparse.ArgumentParser(description="Transformer模型训练和预测程序")
    parser.add_argument('--task', type=str, required=True, choices=['math', 'poetry'], 
                       help='任务类型: math (数学) 或 poetry (唐诗)')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict'],
                       help='模式: train (训练) 或 predict (预测)')
    parser.add_argument('--data', type=str, help='数据文件路径 (仅对唐诗任务的训练模式有效)')
    parser.add_argument('--model', type=str, help='模型文件路径')
    parser.add_argument('--input', type=str, help='输入内容 (预测模式下需要)')
    
    args = parser.parse_args()
    
    # 默认模型路径
    math_model_path = args.model if args.model else "math_model.pt"
    poetry_model_path = args.model if args.model else "poetry_model.pt"
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 数学任务
    if args.task == 'math':
        if args.mode == 'train':
            print("开始训练数学模型...")
            train_math_model(None, math_model_path, device)
        elif args.mode == 'predict':
            if not args.input:
                print("错误: 预测模式需要提供输入方程，使用 --input 参数")
                return
            
            if not os.path.exists(math_model_path):
                print(f"错误: 找不到模型文件 {math_model_path}")
                return
                
            equation = args.input
            print(f"求解方程: {equation}")
            result = solve_math_problem(math_model_path, equation, device)
            print(f"预测结果: {result}")
            print(f"实际结果: {eval(equation)}")
    
    # 唐诗任务
    elif args.task == 'poetry':
        if args.mode == 'train':
            data_path = args.data if args.data else "poetry_data.json"
            if not os.path.exists(data_path):
                print(f"错误: 找不到数据文件 {data_path}")
                return
                
            print("开始训练唐诗模型...")
            train_poetry_model(data_path, poetry_model_path, device)
        elif args.mode == 'predict':
            if not args.input:
                print("错误: 预测模式需要提供输入提示，使用 --input 参数")
                return
                
            if not os.path.exists(poetry_model_path):
                print(f"错误: 找不到模型文件 {poetry_model_path}")
                return
                
            prompt = args.input
            print(f"生成唐诗，提示: {prompt}")
            poem = generate_poetry(poetry_model_path, prompt, device=device)
            print("生成的诗:")
            print(poem)

if __name__ == "__main__":
    main() 
