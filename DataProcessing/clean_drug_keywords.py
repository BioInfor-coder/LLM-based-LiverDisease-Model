# -*- coding: utf-8 -*-


# 标准库
import argparse
import os
import re
from typing import Dict, List, Tuple

DRUG_DICT: Dict[str, List[str]] = {
    # 1. 糖皮质激素（AIH 一线）
    "糖皮质激素": [
        "泼尼松", "强的松", "泼尼松龙", "强的松龙", "醋酸泼尼松龙",
        "甲泼尼龙", "美卓乐", "甲基强的松龙"
    ],

    # 2. 免疫抑制剂（AIH 联合）
    "免疫抑制剂": [
        "硫唑嘌呤", "依木兰", "环孢素", "新山地明", "甲氨蝶呤",
        "吗替麦考酚酯", "骁悉", "他克莫司", "普乐可复"
    ],

    # 3. 熊去氧胆酸类（PBC/PSC 一线）
    "熊去氧胆酸": [
        "熊去氧胆酸", "优思弗", "滔罗特", "牛磺熊去氧胆酸",
        "熊去氧胆酸软胶囊", "熊去氧胆酸胶囊",
        "UDCA", "Ursofalk", "Ursodeoxycholic"
    ],

    # 4. 奥贝胆酸（PBC 二线）
    "奥贝胆酸": ["奥贝胆酸", "OCA", "Obeticholic"],

    # 5. 甘草酸制剂（保肝降酶）
    "甘草酸": [
        "甘草酸二铵", "天晴甘平", "甘草酸二铵肠溶胶囊", "甘草酸二铵肠溶片",
        "复方甘草酸苷", "复方甘草酸苷注射液", "异甘草酸镁",
        "异甘草酸镁注射液"
    ],

    # 6. 水飞蓟素/水飞蓟宾（抗氧化）
    "水飞蓟": [
        "水飞蓟宾", "水林佳", "水飞蓟宾葡甲胺片", "水飞蓟宾葡甲胺",
        "水飞蓟宾胶囊", "水飞蓟素", "利加隆", "水飞蓟素胶囊"
    ],

    # 7. 双环醇/联苯双酯（降酶）
    "降酶药": [
        "双环醇", "百赛诺", "双环醇片",
        "联苯双酯", "联苯双酯片"
    ],

    # 8. 磷脂酰胆碱/谷胱甘肽（膜修复/抗氧化）
    "膜修复抗氧化": [
        "多烯磷脂酰胆碱", "易善复",
        "多烯磷脂酰胆碱注射液", "多烯磷脂酰胆碱胶囊",
        "还原型谷胱甘肽", "谷胱甘肽", "注射用谷胱甘肽", "绿汀诺"
    ],

    # 9. 腺苷蛋氨酸/门冬氨酸鸟氨酸（退黄利胆）
    "退黄利胆": [
        "腺苷蛋氨酸", "丁二磺酸腺苷蛋氨酸", "丁二磺酸腺苷蛋氨酸肠溶片",
        "思美泰",
        "门冬氨酸鸟氨酸", "门冬氨酸鸟氨酸颗粒", "瑞甘"
    ],

    # 10. 利胆中成药
    "利胆中成药": [
        "茵栀黄", "茵栀黄颗粒", "茵栀黄胶囊",
        "苦黄", "苦黄注射液", "大黄利胆", "大黄利胆胶囊",
        "茴三硫", "亮菌", "亮菌口服溶液"
    ],

    # 11. 肝硬化并发症
    # 删除裸词"白蛋白"，改为明确药用形式，
    # 避免与"白蛋白/球蛋白比值"等检验描述产生误匹配
    "肝硬化并发症": [
        "螺内酯", "螺内酯片", "呋塞米", "呋塞米片", "呋塞米注射液",
        "乳果糖", "乳果糖口服溶液", "杜密克", "利福昔明", "昔服申",
        "人血白蛋白", "白蛋白注射液", "静注人血白蛋白",
        "氯化钾", "氯化钾缓释片"
    ],

    # 12. 抗病毒（HBV）
    "HBV抗病毒": [
        "恩替卡韦", "博路定", "恩替卡韦片", "恩替卡韦胶囊",
        "替诺福韦", "TDF", "TAF", "韦瑞德", "韦立得",
        "富马酸替诺福韦二吡呋酯", "富马酸丙酚替诺福韦",
        "富马酸丙酚替诺福韦片",
        "艾米替诺福韦", "艾米替诺福韦片", "恒沐",
        "恩曲他滨替诺福韦", "恩曲他滨替诺福韦片",
        "拉替拉韦", "拉替拉韦钾", "艾生特",
        "拉米夫定", "干扰素", "聚乙二醇干扰素",
        "聚乙二醇干扰素a-2b", "聚乙二醇干扰素α-2b",
        "派格宾"
    ],

    # 13. 辅助/胃保护/维生素
    "辅助用药": [
        "骨化三醇", "盖三淳", "碳酸钙D3", "钙尔奇", "琥珀酸亚铁", "速力菲",
        "奥美拉唑", "奥美拉唑镁肠溶片", "奥美拉唑肠溶胶囊", "洛赛克",
        "泮托拉唑", "泮立苏",
        "艾司奥美拉唑", "艾司奥美拉唑镁肠溶片", "帮卡欣",
        "维生素K1", "维生素K"
    ],

    # 14. 抗纤维化/中成药
    "抗纤维化中成药": [
        "强肝胶囊", "强肝片", "强肝丸", "五酯滴丸",
        "去甲斑蝥素", "去甲斑蝥素片",
        "参芪肝康", "参芪肝康片", "参芪肝康胶囊",
        "肝达康", "肝达康颗粒",
        "九味肝泰", "九味肝泰胶囊",
        "复方益肝灵", "复方益肝灵胶囊",
        "复方鳖甲软肝", "复方鳖甲软肝片", "鳖甲煎丸", "安络化纤丸",
        "当飞利肝宁", "当飞利肝宁片", "当飞利肝宁胶囊",
        "扶正化瘀片", "扶正化瘀",
        "肝苏胶囊", "肝苏",
        "茵芪肝复颗粒", "茵芪肝复",
        "五灵胶囊",
        "降脂灵片", "降脂灵",
        "复方丹参滴丸",
        "生血宝", "生血宝合剂", "养血饮",
        "速效救心丸", "牛黄上清丸", "蛹油α-亚麻酸乙酯", "黄葵胶囊",
        "大柴胡颗粒", "复方消化酶", "荆花胃康", "康复新液"
    ],

    # 15. 造影剂/辅助注射液
    "造影剂及辅助": [
        "碘普罗胺", "碘佛醇", "碘佛醇注射液",
        "碘美普尔", "碘美普尔注射液", "典迈伦",
        "钆贝葡胺", "钆贝葡胺注射液", "莫迪司",
        "钆特酸葡胺",
        "钆喷酸葡胺", "钆喷酸葡胺注射液",
        "钆塞酸二钠", "钆塞酸二钠注射液", "普美显",
        "盐酸利多卡因胶浆", "凝血酶冻干粉", "盐酸达克罗宁胶浆", "达己苏",
        "氢溴酸山莨菪碱", "间苯三酚",
        "聚乙二醇电解质散II", "和爽", "甘油灌肠剂"
    ],

    # 16. 中药饮片（出现过的）
    "中药饮片": [
        "焦槟榔", "焦山楂", "干姜", "附片", "甘草片", "生白术", "茯苓",
        "北柴胡", "白芍", "当归", "桂枝", "麸炒白术", "防己", "黄芪",
        "薏苡仁", "北败酱草", "荷叶", "黄芩片", "绞股蓝", "麦芽", "山楂",
        "阿胶珠", "白及", "川牛膝", "醋香附", "刀豆",
        "姜厚朴", "熟地黄", "太子参", "仙鹤草", "泽兰"
    ],

    # 17. 乙肝免疫球蛋白（被动免疫）
    "乙肝免疫球蛋白": [
        "乙型肝炎人免疫球蛋白", "乙肝免疫球蛋白", "HBIG"
    ],

    # 18. 肝癌/肿瘤靶向药
    "肝癌靶向药": [
        "仑伐替尼", "甲磺酸仑伐替尼", "甲磺酸仑伐替尼胶囊", "乐卫玛",
        "瑞戈非尼", "瑞戈非尼片", "拜万戈",
        "多纳非尼", "甲苯磺酸多纳非尼", "甲苯磺酸多纳非尼片", "泽普生",
        "索拉非尼", "多吉美",
        "阿帕替尼", "艾坦"
    ],

    # 19. 镇静催眠
    "镇静催眠": [
        "佐匹克隆", "佐匹克隆片", "金盟",
        "右佐匹克隆", "艾司唑仑", "地西泮"
    ],

    # 20. 抗组胺
    "抗组胺": [
        "西替利嗪", "盐酸西替利嗪", "盐酸西替利嗪片", "敏达",
        "氯雷他定", "依巴斯汀"
    ],

    # 21. 胸腺肽类免疫调节
    "免疫调节": [
        "胸腺法新", "注射用胸腺法新", "胸腺肽",
        "胸腺五肽", "脾氨肽"
    ],

    # 22. 肠道菌群/消化辅助
    "消化辅助及益生菌": [
        "双歧杆菌三联活菌", "双歧杆菌三联活菌肠溶胶囊", "贝飞达",
        "米曲菌胰酶", "米曲菌胰酶片", "慷彼申",
        "乌灵胶囊",
        "甲钴胺", "甲钴胺片", "弥可保"
    ],

    # 23. 造血/补血
    "造血补血": [
        "利可君", "利可君片",
        "多糖铁复合物", "多糖铁复合物胶囊", "红源达"
    ],

    # 24. AIH 临床描述及扩展用药（补充"高泄露风险词"表中遗漏项）
    # 含疾病诊断/病程描述短语，以及原属类别名但未收入别名列表的
    # "糖皮质激素""免疫抑制剂"等词
    "AIH临床描述及扩展用药": [
        "自身免疫性肝炎", "免疫介导性肝炎", "免疫介导性肝病", "自身免疫相关肝病",
        "激素反应性肝炎", "使用激素治疗的肝炎", "免疫抑制治疗的肝病",
        "正在接受免疫抑制治疗",
        "糖皮质激素", "皮质类固醇", "激素治疗", "免疫抑制剂"
    ],

    # 25. PBC 临床描述及扩展用药
    "PBC临床描述及扩展用药": [
        "原发性胆汁性胆管炎", "原发性胆汁性肝硬化", "胆汁酸治疗的胆汁淤积",
        "胆汁酸治疗",
        # 以下为含"UDCA/熊去氧胆酸"的完整短语，长度大于单独别名，
        # 排序时会优先整体匹配，避免只删掉药名残留半句话
        "UDCA治疗的胆汁淤积性肝病", "长期UDCA治疗", "UDCA治疗有效",
        "使用熊去氧胆酸治疗", "熊去氧胆酸（ursodiol）"
    ],

    # 26. HBV 临床描述及扩展用药
    "HBV临床描述及扩展用药": [
        "慢性乙型肝炎", "乙型肝炎感染", "HBV感染", "乙型肝炎病毒感染",
        "慢性HBV感染", "乙肝携带状态", "已知乙肝携带者", "乙肝感染史",
        "HBV抗病毒治疗", "正在接受乙肝抗病毒治疗",
        "阿德福韦", "抗病毒药物", "核苷类似物", "核苷酸类似物", "抗病毒治疗"
    ],

    # 27. DILI 临床描述及扩展用药
    "DILI临床描述及扩展用药": [
        "药物性肝损伤", "药物性肝炎", "药物相关肝损伤", "药物诱导性肝损伤",
        "药物相关性肝损伤", "药物导致的肝损伤", "继发于药物的肝损伤",
        "肝毒性肝损伤", "疑似药物性损伤", "疑似肝毒性", "毒性肝损伤",
        "停药后好转", "停用药物后改善", "对乙酰氨基酚过量", "肝毒性药物",
        "致病药物", "停药",
        "甘草酸类制剂", "降转氨酶治疗", "SAMe", "退黄治疗", "胆汁淤积治疗",
        "N-乙酰半胺酸", "对乙酰氨基酚解毒治疗", "对乙酰氨基酚中毒解毒",
        # 以下为含短别名的完整短语，避免只删部分残留括号/前后缀
        "谷胱甘肽（GSH）", "水飞蓟素制剂", "S-腺苷蛋氨酸"
    ]
}

# 删除单独的"停药"，改为更严格的短语，
# 避免在非 DILI 场景中（如"监测药物副作用"等）产生误触发
DILI_KEYWORDS: List[str] = [
    "立即停用所有可疑药物",
    "停用可疑药物",
    "停用疑似",
    "停用所有可疑",
    "N-乙酰半胱氨酸",
    "NAC"
]

# 汇总 DRUG_DICT 全部别名 + DILI_KEYWORDS，去重后按字符串长度降序排序，
# 保证正则引擎优先尝试匹配更长的别名（例如优先匹配"人血白蛋白"而不是
# 被更短的子串抢先命中后留下残缺片段）。
_ALL_KEYWORDS: List[str] = []
for _aliases in DRUG_DICT.values():
    _ALL_KEYWORDS.extend(_aliases)
_ALL_KEYWORDS.extend(DILI_KEYWORDS)

ALL_KEYWORDS_SORTED: List[str] = sorted(set(_ALL_KEYWORDS), key=len, reverse=True)

CLEAN_PATTERN: "re.Pattern" = re.compile(
    "|".join(map(re.escape, ALL_KEYWORDS_SORTED)), flags=re.I
)


def clean_line(line: str, collapse_whitespace: bool = True) -> Tuple[str, List[str]]:
    """
    清洗单行文本，将 DRUG_DICT / DILI_KEYWORDS 中出现过的全部关键词删除

    参数:
        line: 待清洗的原始文本行（不含末尾换行符）
        collapse_whitespace: 是否将删词后产生的连续空白归并为单个空格，
            默认True

    返回值:
        cleaned_line: 清洗后的文本行
        hit_keywords: 本行命中并被删除的关键词列表（按出现顺序，可能含重复）
    """
    if line is None:
        return "", []

    # 按出现顺序记录命中词，便于核查清洗效果
    hit_keywords = CLEAN_PATTERN.findall(line)

    cleaned_line = CLEAN_PATTERN.sub("", line)

    if collapse_whitespace:
        # 删词后可能残留连续空格/制表符，归并为单个空格；
        # 保留原有换行结构由调用方逐行处理，此处不处理换行符
        cleaned_line = re.sub(r"[ \t]{2,}", " ", cleaned_line).strip()

    return cleaned_line, hit_keywords


def clean_drug_keywords_from_txt(
    input_path: str,
    output_path: str,
    encoding: str = "utf-8",
    collapse_whitespace: bool = True
) -> None:
    """
    逐行读取 txt 文件，清洗掉词典中出现过的全部药品关键词，并写出结果

    参数:
        input_path: 输入 txt 文件路径
        output_path: 输出 txt 文件路径（清洗后文本，逐行对应输入）
        encoding: 文件编码，默认 utf-8
        collapse_whitespace: 是否归并删词后产生的连续空白，默认True

    异常:
        FileNotFoundError: 当输入文件不存在时

    说明:
        输出文件与输入文件行数一一对应；同时在控制台打印每类关键词
        的清洗命中次数汇总，便于核查是否有遗漏或误删。
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    total_lines = 0
    total_hits = 0
    # 统计每个关键词被清洗掉的总次数，用于核查效果
    hit_counter: Dict[str, int] = {}

    with open(input_path, "r", encoding=encoding) as fin, \
            open(output_path, "w", encoding=encoding) as fout:
        for raw_line in fin:
            total_lines += 1
            # 去掉行尾换行符，清洗后统一重新写入 "\n"
            line_content = raw_line.rstrip("\n").rstrip("\r")

            cleaned_line, hit_keywords = clean_line(
                line_content, collapse_whitespace=collapse_whitespace
            )

            for kw in hit_keywords:
                hit_counter[kw] = hit_counter.get(kw, 0) + 1
            total_hits += len(hit_keywords)

            fout.write(cleaned_line + "\n")

    # 打印清洗结果汇总
    print("===== 文本清洗完成 =====")
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"处理行数: {total_lines}")
    print(f"共清洗掉关键词出现次数: {total_hits}")

    if hit_counter:
        print("----- 各关键词命中次数（降序前20）-----")
        sorted_hits = sorted(hit_counter.items(), key=lambda x: x[1], reverse=True)
        for kw, cnt in sorted_hits[:20]:
            print(f"  {kw}: {cnt}")
    else:
        print("未命中任何词典关键词，输出文件与输入内容基本一致。")


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数

    返回值:
        包含 input、output、encoding、collapse_whitespace 字段的命名空间

    示例:
        python clean_drug_keywords.py -i raw.txt -o cleaned.txt
        python clean_drug_keywords.py -i raw.txt -o cleaned.txt --encoding gbk
        python clean_drug_keywords.py -i raw.txt -o cleaned.txt --keep-whitespace
    """
    parser = argparse.ArgumentParser(
        description="逐行清洗 txt 文本中出现的药品关键词（DRUG_DICT/DILI_KEYWORDS）"
    )
    parser.add_argument(
        "-i", "--input", required=True, type=str,
        help="输入 txt 文件路径"
    )
    parser.add_argument(
        "-o", "--output", required=True, type=str,
        help="输出 txt 文件路径（清洗后文本，逐行对应输入）"
    )
    parser.add_argument(
        "--encoding", default="utf-8", type=str,
        help="文件编码，默认 utf-8；如遇乱码可改为 gbk 等"
    )
    parser.add_argument(
        "--keep-whitespace", action="store_true",
        help="关闭删词后的空白归并（默认会将连续空白归并为单个空格）"
    )
    return parser.parse_args()


def main() -> None:
    """
    程序入口：解析命令行参数，逐行清洗药品关键词，写出到指定输出文件
    """
    args = parse_args()
    clean_drug_keywords_from_txt(
        input_path=args.input,
        output_path=args.output,
        encoding=args.encoding,
        collapse_whitespace=not args.keep_whitespace
    )


if __name__ == "__main__":
    main()