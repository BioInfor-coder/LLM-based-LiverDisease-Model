# -*- coding: utf-8 -*-

# 标准库
import argparse
import os
from typing import List, Optional

# 第三方库
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


# 默认最小词频阈值
DEFAULT_MIN_COUNT = 2
# 默认训练轮数
DEFAULT_EPOCHS = 40
# 默认上下文窗口大小
DEFAULT_WINDOW = 5
# 默认训练算法：1=PV-DM（推荐），0=PV-DBOW
DEFAULT_DM = 1
# 默认训练并行线程数
DEFAULT_WORKERS = 4
# 默认推断向量时的迭代轮数
DEFAULT_INFER_EPOCHS = 100
# 默认随机种子
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数

    参数:
        无

    返回值:
        args: 包含输入/输出路径、模型保存/加载路径及全部Doc2Vec
            超参数字段的命名空间对象

    示例:
        >>> args = parse_args()
        >>> args.vector_size
        384
    """
    parser = argparse.ArgumentParser(description="Doc2Vec批量文本向量化脚本")
    parser.add_argument(
        "--input_path", type=str, required=True,
        help="输入样本文件路径，每行一个样本",
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="输出向量文件路径，每行一个样本的向量",
    )
    parser.add_argument(
        "--model_save_path", type=str, default=None,
        help="训练完成后模型的保存路径，默认None表示不保存",
    )
    parser.add_argument(
        "--model_load_path", type=str, default=None,
        help="若指定且文件存在，则跳过训练直接加载该模型",
    )
    parser.add_argument(
        "--vector_size", type=int, default=None,
        help=f"输出向量维度",
    )
    parser.add_argument(
        "--min_count", type=int, default=DEFAULT_MIN_COUNT,
        help=f"最小词频阈值，默认{DEFAULT_MIN_COUNT}",
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"Doc2Vec训练轮数，默认{DEFAULT_EPOCHS}",
    )
    parser.add_argument(
        "--window", type=int, default=DEFAULT_WINDOW,
        help=f"上下文窗口大小，默认{DEFAULT_WINDOW}",
    )
    parser.add_argument(
        "--dm", type=int, default=DEFAULT_DM, choices=[0, 1],
        help="训练算法，1=PV-DM（推荐），0=PV-DBOW，默认1",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"训练并行线程数，默认{DEFAULT_WORKERS}",
    )
    parser.add_argument(
        "--infer_epochs", type=int, default=DEFAULT_INFER_EPOCHS,
        help=f"推断向量时的迭代轮数，越大越稳定，默认{DEFAULT_INFER_EPOCHS}",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"随机种子，默认{DEFAULT_SEED}",
    )
    parser.add_argument(
        "--no_normalize", action="store_true",
        help="关闭输出向量的L2归一化（默认开启归一化）",
    )
    return parser.parse_args()


def load_texts(input_path: str) -> List[str]:
    """
    从文本文件逐行加载非空文本

    参数:
        input_path: 输入文本文件路径

    返回值:
        texts: 非空文本行列表

    异常:
        FileNotFoundError: 当输入文件不存在时
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    texts = []
    with open(input_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                texts.append(line)
    print(f"[✓] 共加载 {len(texts)} 条非空文本")
    return texts


def build_tagged_documents(texts: List[str]) -> List[TaggedDocument]:
    """
    将文本列表转换为gensim TaggedDocument格式

    参数:
        texts: 文本字符串列表

    返回值:
        tagged_docs: TaggedDocument列表，tag为样本索引
    """
    tagged_docs = [
        TaggedDocument(words=text.split(), tags=[str(i)])
        for i, text in enumerate(texts)
    ]
    return tagged_docs


def train_doc2vec(
    tagged_docs: List[TaggedDocument],
    vector_size: int = 384,
    min_count: int = DEFAULT_MIN_COUNT,
    epochs: int = DEFAULT_EPOCHS,
    window: int = DEFAULT_WINDOW,
    dm: int = DEFAULT_DM,
    workers: int = DEFAULT_WORKERS,
    seed: int = DEFAULT_SEED,
    save_path: Optional[str] = None,
) -> Doc2Vec:
    """
    训练Doc2Vec模型

    参数:
        tagged_docs: TaggedDocument列表
        vector_size: 输出向量维度
        min_count: 最小词频阈值
        epochs: 训练轮数
        window: 上下文窗口大小
        dm: 训练算法，1=PV-DM，0=PV-DBOW
        workers: 并行线程数
        seed: 随机种子
        save_path: 模型保存路径，None则不保存

    返回值:
        model: 训练完成的Doc2Vec模型
    """
    print(f"[...] 开始训练 Doc2Vec: vector_size={vector_size}, "
          f"dm={dm}, epochs={epochs}, window={window}")

    model = Doc2Vec(
        vector_size=vector_size,
        min_count=min_count,
        epochs=epochs,
        window=window,
        dm=dm,
        workers=workers,
        seed=seed,
    )

    # 构建词汇表
    model.build_vocab(tagged_docs)
    print(f"[✓] 词汇表大小: {len(model.wv)}")

    # 训练
    model.train(
        tagged_docs,
        total_examples=model.corpus_count,
        epochs=model.epochs,
    )
    print("[✓] Doc2Vec 训练完成")

    # 保存模型（可选）
    if save_path is not None:
        model.save(save_path)
        print(f"[✓] 模型已保存: {save_path}")

    return model


def infer_and_save_embeddings(
    model: Doc2Vec,
    texts: List[str],
    output_path: str,
    infer_epochs: int = DEFAULT_INFER_EPOCHS,
    normalize: bool = True,
) -> None:
    """
    对每条文本推断向量并按行写入输出文件（格式与Transformer脚本一致）

    参数:
        model: 训练好的Doc2Vec模型
        texts: 原始文本列表
        output_path: 输出文件路径
        infer_epochs: 推断时迭代轮数，越大越稳定
        normalize: 是否L2归一化

    返回值:
        无
    """
    with open(output_path, "w", encoding="utf-8") as fout:
        for idx, text in enumerate(texts):
            words = text.split()

            # 推断向量
            vec = model.infer_vector(words, epochs=infer_epochs)

            # L2归一化
            if normalize:
                norm = np.linalg.norm(vec)
                if norm > 1e-9:
                    vec = vec / norm

            # 首行打印维度信息
            if idx == 0:
                print(f"Embedding 维度: {len(vec)}")

            # 写入文件：空格分隔，6位小数
            fout.write(" ".join(f"{x:.6f}" for x in vec) + "\n")
            print(f"[✓] 已处理样本 {idx + 1}")

    print(f"\n[✓] 全部 embedding 已保存: {output_path}")


def load_existing_model(model_path: str) -> Doc2Vec:
    """
    加载已保存的Doc2Vec模型

    参数:
        model_path: 模型文件路径

    返回值:
        加载的Doc2Vec模型

    异常:
        FileNotFoundError: 当模型文件不存在时
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    model = Doc2Vec.load(model_path)
    print(f"[✓] 模型已加载: {model_path}, vector_size={model.vector_size}")
    return model


def main() -> None:
    """
    脚本主入口：解析参数、加载文本、训练或加载模型、推断并保存embedding

    参数:
        无

    返回值:
        无
    """
    args = parse_args()

    # 1. 加载文本
    all_texts = load_texts(args.input_path)

    # 2. 训练或加载模型
    if args.model_load_path is not None and os.path.exists(args.model_load_path):
        d2v_model = load_existing_model(args.model_load_path)
    else:
        tagged = build_tagged_documents(all_texts)
        d2v_model = train_doc2vec(
            tagged,
            vector_size=args.vector_size,
            min_count=args.min_count,
            epochs=args.epochs,
            window=args.window,
            dm=args.dm,
            workers=args.workers,
            seed=args.seed,
            save_path=args.model_save_path,
        )

    # 3. 推断embedding并保存
    infer_and_save_embeddings(
        model=d2v_model,
        texts=all_texts,
        output_path=args.output_path,
        infer_epochs=args.infer_epochs,
        normalize=not args.no_normalize,
    )


if __name__ == "__main__":
    main()
