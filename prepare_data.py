"""
数据预备：从 datedsv4f 提取"编程 / 数学"两个领域的最优样本，
统一为对话式文本（含宿主模型可读的提问+回答），供 MoE 域隔离训练/测试使用。

每个领域输出两个文件：
  data/prog.train.txt  /  data/prog.test.txt
  data/math.train.txt  /  data/math.test.txt
每行一条样本： "用户:...\n\n回答正文"，连续多行文本在生成时再按样本拆分。
为路径健壮起见，这里统一存成 JSON 字段，后续由训练脚本读取。
"""
import json
import os
import random
import zipfile

DATASV4F = r"D:\dev\aisa-llm\datedsv4f"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

random.seed(7)


def ensure_out():
    os.makedirs(OUT, exist_ok=True)
    return OUT


def to_dialogue(inst, response):
    """编程指令数据 -> 对话文本 (instruction; response)"""
    return {"text": f"用户:{inst}\n\n{response}"}


def read_lines(p):
    with open(p, encoding="utf-8") as f:
        return [l for l in f if l.strip()]


def load_programming(seed=7):
    """读取 fst jsonl -> 结构化样本列表 (instruction, thought, response). 拣质量高/适中长度的。"""
    rnd = random.Random(seed)
    samples = []
    for name in ("fst_train_data_L1.jsonl", "fst_train_data_L2.jsonl"):
        p = os.path.join(DATASV4F, name)
        if not os.path.exists(p):
            continue
        for line in read_lines(p):
            try:
                o = json.loads(line)
            except Exception:
                continue
            inst = o.get("instruction", "")
            resp = o.get("response", "")
            thought = o.get("thought", "")
            if not inst or not resp:
                continue
            samples.append({"inst": inst, "resp": resp, "thought": thought})
    return samples


def load_math_zip(zf):
    """从 zip 读数学逻辑语料 -> text 条目标签。"""
    samples = []
    inner = "TGAI 训练语料V2.6/TGAI数学逻辑语料COT.jsonl"
    with zipfile.ZipFile(zf) as z:
        for line in z.read(inner).decode("utf-8", "ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            t = o.get("text", "")
            if t and len(t) > 40:
                samples.append(t)
    return samples


def split(samples, test_frac=0.15, seed=7):
    rnd = random.Random(seed)
    al = list(samples)
    rnd.shuffle(al)
    n = max(1, int(len(al) * test_frac))
    return al[n:], al[:n]


def main():
    out = ensure_out()
    summary = {}

    # 编程：优先 thought (L2 质量更高) 保留
    prog = load_programming()
    # 简单质量过滤：长度适中（instruction 3~120 token 的编辑强度样本）
    prog_ok = [s for s in prog if 10 <= len(s["inst"]) <= 160]
    # 转对话文本，带 thought 作为回答前缀（体现推理）
    def prog_text(s):
        t = s.get("thought", "").strip()
        body = f"【思考】{t}\n{s['resp']}" if t else s["resp"]
        return to_dialogue(s["inst"], body)

    prog_dial = [prog_text(s) for s in prog_ok]
    ptr, pte = split(prog_dial)

    # 数学：zip 内 text
    zf = os.path.join(DATASV4F, "TGAI 训练语料V2.6.zip")
    math = load_math_zip(zf) if os.path.exists(zf) else []
    math_dial = [{"text": t} for t in math]
    mtr, mte = split(math_dial)

    files = {
        "prog.train.jsonl": ptr,
        "prog.test.jsonl": pte,
        "math.train.jsonl": mtr,
        "math.test.jsonl": mte,
    }
    for fn, arr in files.items():
        with open(os.path.join(out, fn), "w", encoding="utf-8") as f:
            for o in arr:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")

    summary = {
        "programming_src": len(prog),
        "programming_filtered": len(prog_ok),
        "programming_train": len(ptr),
        "programming_test": len(pte),
        "math_src": len(math),
        "math_train": len(mtr),
        "math_test": len(mte),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()