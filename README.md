# Transformer实现与应用

这个项目基于Transformer架构实现了两个任务：
1. 数学计算：使用标准的Encoder-Decoder Transformer结构进行简单加法运算
2. 唐诗生成：使用Decoder-only Transformer（类似GPT）生成唐诗

## 环境要求

- Python 3.6+
- PyTorch 1.0+
- NumPy

安装依赖：
```bash
pip install torch numpy
```

## 数据准备

对于唐诗生成任务，需要准备一个JSON格式的唐诗数据集。JSON文件结构应为：
```json
[
  {
    "title": "诗的标题",
    "author": "作者",
    "paragraphs": ["诗的第一行", "诗的第二行", ...]
  },
  ...
]
```

将此文件保存为`poetry_data.json`，并放在项目根目录。

## 使用方法

### 训练模型

数学模型训练：
```bash
python main.py --task math --mode train
```

唐诗模型训练：
```bash
python main.py --task poetry --mode train --data poetry_data.json
```

### 预测/生成

数学计算：
```bash
python main.py --task math --mode predict --input "123+456"
```

唐诗生成：
```bash
python main.py --task poetry --mode predict --input "春日和风"
```

## 模型配置

两个模型都使用了较小的配置以便快速训练：
- d_model: 256
- num_heads: 4
- num_layers: 4
- d_ff: 512

您可以通过修改源代码来调整这些参数。

## 模型文件说明

- `transformer.py`: 包含Transformer相关组件的实现，包括注意力机制、编码器、解码器等
- `train_math.py`: 数学模型的训练和预测代码
- `train_poetry.py`: 唐诗模型的训练和生成代码
- `main.py`: 命令行入口程序
- `prepare_data.py`: 用于下载和处理中国古典诗词数据集的脚本

## 示例

数学计算示例：
```
$ python main.py --task math --mode predict --input "123+456"
求解方程: 123+456
预测结果: 579
实际结果: 579
```

唐诗生成示例：
```
$ python main.py --task poetry --mode predict --input "春日和风"
生成唐诗，提示: 春日和风
生成的诗:
春日和风满袖香，花开不尽水流长。
绿阴深处闻啼鸟，白云飘处见斜阳。
细雨轻尘洒碧苔，远山如黛近如黛。
相思一夜无人语，独倚栏杆看水来。
``` 
