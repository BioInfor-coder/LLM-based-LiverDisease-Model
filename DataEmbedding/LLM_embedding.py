# -*- coding: utf-8 -*-

# 标准库
import argparse
import os
from typing import Dict, List, Optional

# 第三方库
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


# 归一化时使用的向量范数阶数（L2归一化）
NORM_ORDER = 2
# 隐藏状态在序列维度上做均值池化时的维度索引
SEQ_MEAN_DIM = 1

# 各模型默认的可见GPU编号（不涉及任何路径，仅作为--cuda_visible_devices的默认值）
DEFAULT_CUDA_VISIBLE_DEVICES: Dict[str, str] = {
    "qwen3": "0,1,2,3",
    "huatuo": "4,5,6,7",
    "iimed": "0,1,2,3",
}


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数

    参数:
        无

    返回值:
        args: 包含model_name、model_path、input_path、output_path、
            cuda_visible_devices等字段的命名空间对象

    示例:
        >>> args = parse_args()
        >>> args.model_path
        './Qwen3-8B'
    """
    parser = argparse.ArgumentParser(description="通用LLM批量编码脚本")
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="模型别名（如qwen3/huatuo/iimed），仅用于在未指定"
             "--cuda_visible_devices时查找默认GPU配置，可不传",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="模型权重所在的路径（本地目录或HuggingFace模型ID）",
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
        "--cuda_visible_devices",
        type=str,
        default=None,
        help="指定CUDA_VISIBLE_DEVICES，例如'0,1,2,3'；不指定则尝试根据"
             "--model_name查找默认配置，都未提供则不设置该环境变量",
    )
    return parser.parse_args()


def load_model_and_tokenizer(model_path: str) -> tuple:
    """
    加载因果语言模型及对应的分词器

    参数:
        model_path: 模型权重所在的本地路径

    返回值:
        model: 加载完成的因果语言模型
        tokenizer: 与模型匹配的分词器
        first_device: 模型第一个模块所在的设备，用于输入张量搬运

    异常:
        OSError: 当模型路径不存在或权重加载失败时
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 确定模型的第一个模块所在设备，用于后续输入张量搬运
    first_device = list(model.hf_device_map.values())[0]

    return model, tokenizer, first_device


def encode_sentence(
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
) -> List[float]:
    """
    将单条文本编码为归一化后的句向量

    通过chat模板构造prompt，取模型最后一层hidden state在序列维度上
    做均值池化，再做L2归一化，得到定长的句向量表示。

    参数:
        text: 待编码的单条文本
        model: 已加载的因果语言模型
        tokenizer: 与模型匹配的分词器
        device: 输入张量应搬运到的设备

    返回值:
        rep_list: 归一化后的句向量，以浮点数列表形式返回

    异常:
        RuntimeError: 当模型前向推理失败时
    """
    # 构造prompt（用于chat模型）
    messages = [{"role": "user", "content": text}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 推理（仅提取表征，不生成文本）
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    # 获取最后一层hidden state，并在序列维度上均值池化
    hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden]
    rep = hidden_states.mean(dim=SEQ_MEAN_DIM)  # [batch, hidden_size]
    rep = F.normalize(rep, p=NORM_ORDER, dim=1)  # 单位向量

    return rep[0].cpu().tolist()


def run_batch_encoding(
    input_path: str,
    output_path: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
) -> None:
    """
    批量读取输入文件并逐行编码，将结果写入输出文件

    参数:
        input_path: 输入样本文件路径，每行一个样本
        output_path: 输出向量文件路径，每行一个样本的向量
        model: 已加载的因果语言模型
        tokenizer: 与模型匹配的分词器
        device: 输入张量应搬运到的设备

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

            rep_list = encode_sentence(line, model, tokenizer, device)
            print(len(rep_list))

            # 写入文件，一行一个样本向量
            fout.write(" ".join(f"{x:.6f}" for x in rep_list) + "\n")

            print(f"[✓] 已处理样本 {idx + 1}")


def main() -> None:
    """
    脚本主入口：解析参数、设置GPU、加载模型并执行批量编码

    参数:
        无

    返回值:
        无
    """
    args = parse_args()

    # 优先使用命令行显式传入的CUDA_VISIBLE_DEVICES；
    # 若未传入，则尝试根据model_name查找默认GPU配置；两者都没有则不设置该变量
    cuda_visible_devices = args.cuda_visible_devices
    if cuda_visible_devices is None and args.model_name is not None:
        cuda_visible_devices = DEFAULT_CUDA_VISIBLE_DEVICES.get(args.model_name)

    if cuda_visible_devices is not None:
        # 必须在任何CUDA上下文初始化之前设置，否则不生效
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    model, tokenizer, first_device = load_model_and_tokenizer(args.model_path)
    run_batch_encoding(args.input_path, args.output_path, model, tokenizer, first_device)


if __name__ == "__main__":
    main()