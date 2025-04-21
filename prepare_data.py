import json
import os
import urllib.request
import zipfile
import argparse

def download_chinese_poetry_dataset(save_path="poetry_data.json", sample_size=None):
    """
    下载中国古典诗词数据集并处理为唐诗数据集
    
    参数:
        save_path: 保存数据的路径
        sample_size: 要保存的样本数量，None表示保存全部
    """
    print("开始下载中国古典诗词数据集...")
    
    # 创建临时目录
    if not os.path.exists("temp"):
        os.makedirs("temp")
    
    # 下载数据集
    url = "https://github.com/chinese-poetry/chinese-poetry/archive/refs/heads/master.zip"
    zip_path = "temp/chinese-poetry.zip"
    
    try:
        urllib.request.urlretrieve(url, zip_path)
        print("数据集下载完成")
    except Exception as e:
        print(f"下载失败: {e}")
        return False
    
    # 解压数据集
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("temp")
        print("数据集解压完成")
    except Exception as e:
        print(f"解压失败: {e}")
        return False
    
    # 处理唐诗数据
    print("开始处理唐诗数据...")
    
    tang_path = "temp/chinese-poetry-master/全唐诗"
    all_poems = []
    
    try:
        # 读取所有唐诗文件
        for file in os.listdir(tang_path):
            if file.startswith("poet") and file.endswith(".json"):
                with open(os.path.join(tang_path, file), 'r', encoding='utf-8') as f:
                    poems = json.load(f)
                    all_poems.extend(poems)
    except Exception as e:
        print(f"处理唐诗数据失败: {e}")
        return False
    
    print(f"共读取到 {len(all_poems)} 首唐诗")
    
    # 筛选具有完整内容的唐诗
    filtered_poems = []
    for poem in all_poems:
        if 'paragraphs' in poem and poem['paragraphs']:
            filtered_poems.append(poem)
    
    print(f"筛选后剩余 {len(filtered_poems)} 首唐诗")
    
    # 如果指定了样本大小，则进行随机抽样
    if sample_size and sample_size < len(filtered_poems):
        import random
        filtered_poems = random.sample(filtered_poems, sample_size)
        print(f"随机选择了 {sample_size} 首唐诗")
    
    # 保存处理后的数据
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_poems, f, ensure_ascii=False, indent=2)
        print(f"数据已保存到 {save_path}")
    except Exception as e:
        print(f"保存数据失败: {e}")
        return False
    
    # 清理临时文件
    import shutil
    # shutil.rmtree("temp")
    print("临时文件已清理")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载和处理中国古典诗词数据集")
    parser.add_argument('--output', type=str, default="poetry_data.json", help='输出文件路径')
    parser.add_argument('--samples', type=int, default=None, help='要保存的样本数量，默认为全部')
    
    args = parser.parse_args()
    
    success = download_chinese_poetry_dataset(args.output, args.samples)
    
    if success:
        print("数据准备完成！")
    else:
        print("数据准备失败！")
        
    # 显示如何使用此数据训练模型
    print("\n要训练唐诗生成模型，请运行:")
    print(f"python main.py --task poetry --mode train --data {args.output}") 