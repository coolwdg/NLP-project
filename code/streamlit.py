import streamlit as st
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import hashlib

# --------- 解析并组合 TEXT + QAs ---------
def parse_and_merge_blocks(file_obj):
    blocks = []
    current_text = ""
    qa_list = []
    current_question, current_answer = None, None

    lines = file_obj.read().decode("utf-8").splitlines()

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

    if current_text and qa_list:
        blocks.append({
            "text": current_text.strip(),
            "qas": qa_list
        })

    return blocks

# --------- 计算文件hash用于判断是否是新文件 ---------
def compute_file_hash(file_obj):
    file_obj.seek(0)
    file_bytes = file_obj.read()
    file_obj.seek(0)
    return hashlib.md5(file_bytes).hexdigest()

# --------- 构建向量索引 ---------
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

# --------- 相似度检索 ---------
def retrieve_combined_blocks(query, model, index, blocks, top_k=5):
    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        block = blocks[idx]
        similarity = 1 / (1 + distances[0][i])
        results.append((similarity, block['text'], block['qas']))

    return results

# --------- 调用大模型 ---------
def ask_doubao(context, user_question):
    client = OpenAI(
        base_url="",
        api_key=""
    )

    completion = client.chat.completions.create(
        model="doubao-pro-256k-241115",
        messages=[
            {"role": "system", "content": "你是人工智能助手，只能基于提供的上下文回答问题。如果上下文中没有相关信息，请直接回答：'我无法回答该问题，因为提供的内容中没有相关信息。'"},
            {"role": "user", "content": f"已知内容如下：\n{context}\n\n请基于上述内容回答：{user_question}"},
        ],
    )
    return completion.choices[0].message.content

# --------- 初始化 Streamlit 状态 ---------
st.set_page_config(page_title="问答助手", layout="wide")
st.title("📘 文本问答智能助手")

if 'blocks' not in st.session_state:
    st.session_state.blocks = None
if 'index' not in st.session_state:
    st.session_state.index = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'file_hash' not in st.session_state:
    st.session_state.file_hash = None

# --------- 上传文件并缓存处理结果 ---------
uploaded_file = st.file_uploader("请上传包含 [TEXT]/[QUESTION]/[ANSWER] 标记的文本文件", type=["txt"])

if uploaded_file:
    current_hash = compute_file_hash(uploaded_file)

    if st.session_state.file_hash != current_hash:
        with st.spinner("⏳ 正在解析文件并构建索引..."):
            blocks = parse_and_merge_blocks(uploaded_file)
            index, model = build_combined_index(blocks)

            st.session_state.blocks = blocks
            st.session_state.index = index
            st.session_state.model = model
            st.session_state.file_hash = current_hash

        st.success(f"✅ 成功载入 {len(blocks)} 个文档块。可以开始提问了！")
    else:
        st.info("📁 文件未更改，使用已缓存的模型和索引。")

# --------- 提问 & 显示结果 ---------
if st.session_state.blocks and st.session_state.index and st.session_state.model:
    user_question = st.text_input("💬 输入你的问题：")

    if user_question:
        with st.spinner("🔍 正在检索相关内容..."):
            results = retrieve_combined_blocks(
                user_question,
                st.session_state.model,
                st.session_state.index,
                st.session_state.blocks,
                top_k=5
            )

        # 拼接上下文内容
        all_context_parts = []
        st.subheader("📚 匹配的文本段落（Top 5）")
        for i, (score, text, qas) in enumerate(results[:5], 1):
            st.markdown(f"**{i}. 匹配度：{score:.4f}**")
            st.markdown(f"📘 {text}")
            context_part = f"段落 {i}：{text}"
            for qa in qas:
                context_part += f"\nQ: {qa['question']}\nA: {qa['answer']}"
            all_context_parts.append(context_part)

        full_context = "\n\n".join(all_context_parts)

        with st.spinner("🤖 正在向大模型请求答案..."):
            response = ask_doubao(full_context, user_question)

        st.subheader("🧠 大模型回答")
        st.markdown(response)
