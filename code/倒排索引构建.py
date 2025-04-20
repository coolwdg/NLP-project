import math
from collections import defaultdict
import jieba
import os
from openai import OpenAI


class Preprocessor:
    def preprocess(self, text):
        words = jieba.lcut_for_search(text)
        return [word.strip() for word in words if word.strip()]


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(list)
        self.doc_id_counter = 0
        self.doc_lengths = defaultdict(int)
        self.doc_count = 0

    def add_document(self, content):
        doc_id = self.doc_id_counter
        self.doc_id_counter += 1
        terms = Preprocessor().preprocess(content)
        term_positions = defaultdict(list)
        for position, term in enumerate(terms):
            term_positions[term].append(position)

        for term, positions in term_positions.items():
            self.index[term].append({'doc_id': doc_id, 'positions': positions})

        self.doc_lengths[doc_id] = len(terms)
        self.doc_count += 1

    def build_index(self, documents):
        for doc in documents:
            self.add_document(doc)

    def query(self, query_terms):
        results = None
        for term in query_terms:
            if term in self.index:
                current_docs = {entry['doc_id'] for entry in self.index[term]}
                if results is None:
                    results = current_docs
                else:
                    results &= current_docs
            else:
                return []
        return sorted(results) if results else []

    def rank(self, query):
        query_terms = Preprocessor().preprocess(query)
        relevant_docs = self.query(query_terms)
        scores = {}
        for doc_id in relevant_docs:
            score = 0
            doc_length = self.doc_lengths.get(doc_id, 1)
            for term in query_terms:
                if term in query_terms:
                    posting = [entry for entry in self.index[term] if entry['doc_id'] == doc_id]
                    #print(posting)
                    if posting:
                        tf = len(posting[0]['positions'])
                        idf = math.log(self.doc_count / (1+len(self.index[term])))
                        score += tf * idf
                        #print(tf, idf, score, self.doc_count)
            scores[doc_id] = score/doc_length
        return sorted(scores.items(), reverse=True, key=lambda x: x[1])


test_documents = []
top_n = 1

with open('processed_data.txt', 'r', encoding='utf-8') as f:
    data = f.readlines()
    print(data)
    for line in data:
        test_documents.append(line)
#print(test_documents)

que_txt = open('./test_word.txt', 'r', encoding='utf-8')
que_lis = que_txt.readlines()
index = InvertedIndex()
index.build_index(test_documents)
total = 0
correct = 0
for que in que_lis:
    total += 1
    print('Q:', que, '\n')
    question = que.strip('\n')
    query = question
    res = index.rank(query)

    final_res = res[0:top_n]
    final_text = []
    for single in final_res:
        idx = single[0]
        print(test_documents[idx], 'idx: ', idx)
        if (idx + 1) == total:
            correct += 1
        final_text.append(test_documents[idx])
print(correct/float(total))
    #client = OpenAI(
        # 从环境变量中读取您的方舟API Key
    #    api_key=os.environ.get("ARK_API_KEY"),
    #    base_url="https://ark.cn-beijing.volces.com/api/v3",
    #    )
    #completion = client.chat.completions.create(
        # 将推理接入点 <Model>替换为 Model ID
    #    model="doubao-1-5-pro-32k-250115",
    #    messages=[
    #        {"role": "system", "content": "根据以下知识回答用户的问题。问题与知识无关时要拒绝回答。如果用户只提供关键词，"
    #                                      "尝试猜测用户要询问的问题。知识如下: {}".format(final_text)},
    #        {"role": "user", "content": "{}".format(question)}
    #    ]
    #)
    #print(completion.choices[0].message)
