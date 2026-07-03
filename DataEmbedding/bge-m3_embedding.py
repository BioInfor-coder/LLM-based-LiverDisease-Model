# -*- coding: utf-8 -*-

# 标准库
import argparse
import os
from typing import Dict, List

# 第三方库
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


# 归一化时使用的向量范数阶数（L2归一化）
NORM_ORDER = 2


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数

    参数:
        无

    返回值:
        args: 包含model_path、input_path、output_path、max_length、
            device等字段的命名空间对象

    示例:
        >>> args = parse_args()
        >>> args.model_path
        './bge-m3'
    """
    parser = argparse.ArgumentParser(description="BGE-M3批量编码脚本")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="BGE-M3模型权重所在的路径（本地目录或HuggingFace模型ID）",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="输入样本文件路径，每行一个样本",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="输出向量文件路径，每行一个样本的向量",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="tokenizer截断的最大长度，默认None表示不限制（沿用模型默认行为）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="推理设备，如'cuda'/'cuda:0'/'cpu'；不指定则自动检测GPU可用性",
    )
    return parser.parse_args()


def mean_pooling(model_output: tuple, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    对Transformer输出做考虑attention mask的均值池化

    参数:
        model_output: 模型前向输出，model_output[0]为所有token的embeddings
        attention_mask: 输入对应的attention mask，形状[batch, seq_len]

    返回值:
        句向量，形状[batch, hidden_size]
    """
    token_embeddings = model_output[0]  # 所有token的embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def load_model_and_tokenizer(model_path: str, device: str) -> tuple:
    """
    加载BGE-M3模型及对应的tokenizer

    参数:
        model_path: 模型权重所在的本地路径
        device: 模型加载到的目标设备

    返回值:
        model: 加载完成并置于eval模式的模型
        tokenizer: 与模型匹配的分词器

    异常:
        OSError: 当模型路径不存在或权重加载失败时
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def encode_sentence(
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    max_length: int = None,
) -> List[float]:
    """
    将单条文本编码为归一化后的句向量

    参数:
        text: 待编码的单条文本
        model: 已加载的BGE-M3模型
        tokenizer: 与模型匹配的分词器
        device: 输入张量应搬运到的设备
        max_length: tokenizer截断的最大长度，None表示不限制

    返回值:
        rep_list: 归一化后的句向量，以浮点数列表形式返回

    异常:
        RuntimeError: 当模型前向推理失败时
    """
    tokenize_kwargs: Dict[str, object] = {
        "padding": True,
        "truncation": True,
        "return_tensors": "pt",
    }
    if max_length is not None:
        tokenize_kwargs["max_length"] = max_length

    encoded_input = tokenizer(text, **tokenize_kwargs)
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    sentence_embeddings = F.normalize(sentence_embeddings, p=NORM_ORDER, dim=1)

    return sentence_embeddings[0].cpu().tolist()


def run_batch_encoding(
    input_path: str,
    output_path: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    max_length: int = None,
) -> None:
    """
    批量读取输入文件并逐行编码，将结果写入输出文件

    参数:
        input_path: 输入样本文件路径，每行一个样本
        output_path: 输出向量文件路径，每行一个样本的向量
        model: 已加载的BGE-M3模型
        tokenizer: 与模型匹配的分词器
        device: 输入张量应搬运到的设备
        max_length: tokenizer截断的最大长度，None表示不限制

    返回值:
        无

    异常:
        FileNotFoundError: 当输入文件不存在时
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    with open(input_path, "r", encoding="utf-8") as fin, \
            open(output_path, "w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue

            rep_list = encode_sentence(line, model, tokenizer, device, max_length)

            if idx == 0:
                print(f"Embedding 维度: {len(rep_list)}")

            fout.write(" ".join(f"{x:.6f}" for x in rep_list) + "\n")
            print(f"[✓] 已处理样本 {idx + 1}")


def main() -> None:
    """
    脚本主入口：解析参数、加载模型并执行批量编码

    参数:
        无

    返回值:
        无
    """
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_model_and_tokenizer(args.model_path, device)
    run_batch_encoding(
        args.input_path, args.output_path, model, tokenizer, device, args.max_length
    )


if __name__ == "__main__":
    main()