import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# --------- Step 1: 读取并组合 TEXT + QAs ---------
def parse_and_merge_blocks(filepath):
    blocks = []
    current_text = ""
    qa_list = []
    current_question, current_answer = None, None

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith("[TEXT]"):
            if current_text and qa_list:
                blocks.append({
                    "text": current_text.strip(),
                    "qas": qa_list
                })
            current_text = line.replace("[TEXT]", "").strip()
            qa_list = []
        elif line.startswith("[QUESTION]"):
            current_question = line.replace("[QUESTION]", "").strip()
        elif line.startswith("[ANSWER]"):
            current_answer = line.replace("[ANSWER]", "").strip()
            if current_question and current_answer:
                qa_list.append({
                    "question": current_question,
                    "answer": current_answer
                })
                current_question, current_answer = None, None

    # 最后一组
    if current_text and qa_list:
        blocks.append({
            "text": current_text.strip(),
            "qas": qa_list
        })

    # 为每个文档块分配 ID
    for i, block in enumerate(blocks):
        block["id"] = i

    return blocks


# --------- Step 2: 构建向量索引，合并 text + qa 内容 ---------
def build_combined_index(blocks, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
    model = SentenceTransformer(model_name)

    combined_texts = []
    for block in blocks:
        combined = block['text']
        for qa in block['qas']:
            combined += f"。问题：{qa['question']} 答案：{qa['answer']}"
        combined_texts.append(combined)

    embeddings = model.encode(combined_texts, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, model


# --------- Step 3: 检索函数 ---------
def retrieve_combined_blocks(query, model, index, blocks, top_k=5):
    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        block = blocks[idx]
        similarity = 1 / (1 + distances[0][i])
        results.append((similarity, block["id"], block['text'], block['qas']))

    return results


# --------- Step 3.5: 通过 ID 查找指定文档 ---------
def retrieve_block_by_id(block_id, blocks):
    if 0 <= block_id < len(blocks):
        return blocks[block_id]
    else:
        return None


# --------- Step 4: 主程序交互 ---------
if __name__ == "__main__":
    blocks = parse_and_merge_blocks(r"C:\Users\wdg\Desktop\qa_data(1).txt")  # 修改路径

    print(f"✅ 共载入 {len(blocks)} 个文档块。正在构建全文+问答组合索引...")

    index, model = build_combined_index(blocks)

    print("\n🧠 输入你的问题（输入 q 退出），输入 id:xxx 来通过 ID 查找文档：")
    while True:
        user_input = input(">> ")
        if user_input.lower() == 'q':
            break
        elif user_input.startswith("id:"):
            try:
                block_id = int(user_input.split(":")[1])
                block = retrieve_block_by_id(block_id, blocks)
                if block:
                    print(f"\n📘 文档 ID: {block_id} 的内容：")
                    print(f"📘 文本内容：\n{block['text']}")
                    print(f"🗂️ 相关问答：")
                    for qa in block['qas']:
                        print(f"  Q: {qa['question']}")
                        print(f"  A: {qa['answer']}")
                    print("-" * 100 + '\n')
                else:
                    print("❌ 无效的文档 ID，请输入有效的 ID。")
            except ValueError:
                print("❌ 输入的 ID 格式不正确，请输入 id:数字。")
        else:
            results = retrieve_combined_blocks(user_input, model, index, blocks, top_k=5)

            print("\n🔍 匹配度最高的前 5 个文档块：\n")
            for i, (score, block_id, text, qas) in enumerate(results, 1):
                print(f"{i}. 匹配度：{score:.4f}，文档 ID: {block_id}")
                print(f"📘 文本内容：\n{text}")
                print(f"🗂️ 相关问答：")
                for qa in qas:
                    print(f"  Q: {qa['question']}")
                    print(f"  A: {qa['answer']}")
                print("-" * 100 + '\n')
