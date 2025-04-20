import json
import requests
from volcengine.auth.SignerV4 import SignerV4
from volcengine.base.Request import Request
from volcengine.Credentials import Credentials
import numpy as np
from collections import Counter


# 初始化知识库服务相关配置
collection_name = ""
project_name = ""
ak = ""
sk = ""
account_id = ""
g_knowledge_base_domain = ""

# 定义系统提示
base_prompt = """# 任务
你是一位在线客服，你的首要任务是通过巧妙的话术回复用户的问题，你需要根据「参考资料」来回答接下来的「用户问题」，这些信息在 <context></context> XML tags 之内，你需要根据参考资料给出准确，简洁的回答。

你的回答要满足以下要求：
    1. 回答内容必须在参考资料范围内，尽可能简洁地回答问题，不能做任何参考资料以外的扩展解释。
    2. 回答中需要根据客户问题和参考资料保持与客户的友好沟通。
    3. 如果参考资料不能帮助你回答用户问题，告知客户无法回答该问题，并引导客户提供更加详细的信息。
    4. 为了保密需要，委婉地拒绝回答有关参考资料的文档名称或文档作者等问题。

# 任务执行
现在请你根据提供的参考资料，遵循限制来回答用户的问题，你的回答需要准确和完整。

# 参考资料
<context>
  {}
</context>


# 引用要求
1. 当可以回答时，在句子末尾适当引用相关参考资料，每个参考资料引用格式必须使用<reference>标签对，例如: <reference data-ref="{{point-id}}" data-img-ref="..."></reference>
2. 当告知客户无法回答时，不允许引用任何参考资料
3. 'data-ref' 字段表示对应参考资料的 point_id
4. 'data-img-ref' 字段表示句子是否与对应的图片相关，"true"表示相关，"false"表示不相关
"""


# 定义签名生成函数
def prepare_request(method, path, params=None, data=None, doseq=0):
    if params:
        for key in params:
            if (
                isinstance(params[key], int)
                or isinstance(params[key], float)
                or isinstance(params[key], bool)
            ):
                params[key] = str(params[key])
            elif isinstance(params[key], list):
                if not doseq:
                    params[key] = ",".join(params[key])
    r = Request()
    r.set_shema("http")
    r.set_method(method)
    r.set_connection_timeout(10)
    r.set_socket_timeout(10)
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json; charset=utf-8",
        "Host": g_knowledge_base_domain,
        "V-Account-Id": account_id,
    }
    r.set_headers(headers)
    if params:
        r.set_query(params)
    r.set_host(g_knowledge_base_domain)
    r.set_path(path)
    if data is not None:
        r.set_body(json.dumps(data))

    # 生成签名
    credentials = Credentials(ak, sk, "air", "cn-north-1")
    SignerV4.sign(r, credentials)
    return r


# 定义搜索知识库函数
def search_knowledge():
    method = "POST"
    path = "/api/knowledge/collection/search_knowledge"
    request_params = {
        "project": "default",
        "name": collection_name,  # 使用定义的 collection_name
        "query": query,  # 使用定义的 query
        "limit": 10,
        "pre_processing": {
            "need_instruction": True,
            "return_token_usage": True,
            "messages": [
                {
                    "role": "system",
                    "content": ""
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            "rewrite": False
        },
        "dense_weight": 0.5,
        "post_processing": {
            "get_attachment_link": True,
            "rerank_only_chunk": False,
            "rerank_switch": False,
            "chunk_group": True,
            "chunk_diffusion_count": 0
        }
    }

    info_req = prepare_request(method=method, path=path, data=request_params)
    rsp = requests.request(
        method=info_req.method,
        url="http://{}{}".format(g_knowledge_base_domain, info_req.path),
        headers=info_req.headers,
        data=info_req.body
    )
    return rsp.text


# 定义聊天补全函数
def chat_completion(message, stream=False, return_token_usage=True, temperature=0.7, max_tokens=4096):
    method = "POST"
    path = "/api/knowledge/chat/completions"
    request_params = {
        "messages": message,
        "stream": False,
        "return_token_usage": True,
        "model": "Doubao-1-5-pro-32k",
        "max_tokens": 4096,
        "temperature": 0.7,
        "model_version": "250115"
    }

    info_req = prepare_request(method=method, path=path, data=request_params)
    rsp = requests.request(
        method=info_req.method,
        url="http://{}{}".format(g_knowledge_base_domain, info_req.path),
        headers=info_req.headers,
        data=info_req.body
    )
    rsp.encoding = "utf-8"
    return rsp.text


# 判断是否为视觉模型函数
def is_vision_model(model_name):
    if model_name is None:
        return False
    return "vision" in model_name


# 获取提示内容函数
def get_content_for_prompt(point: dict, image_num: int) -> str:
    content = point["content"]
    original_question = point.get("original_question")
    if original_question:
        # faq 召回场景，content 只包含答案，需要把原问题也拼上
        return "当询问到相似问题时，请参考对应答案进行回答：问题：“{question}”。答案：“{answer}”".format(
            question=original_question, answer=content)
    if image_num > 0 and "chunk_attachment" in point and point["chunk_attachment"][0]["link"]:
        placeholder = f"<img>图片{image_num}</img>"
        return content + placeholder
    return content


# 生成提示函数
def generate_prompt(rsp_txt):
    rsp = json.loads(rsp_txt)
    if rsp["code"] != 0:
        return "", []
    prompt = ""
    image_urls = []
    rsp_data = rsp["data"]
    points = rsp_data["result_list"]
    using_vlm = is_vision_model("Doubao-1-5-pro-32k")
    image_cnt = 0

    for point in points:
        # 提取图片链接
        if using_vlm and "chunk_attachment" in point:
            image_link = point["chunk_attachment"][0]["link"]
            if image_link:
                image_urls.append(image_link)
                image_cnt += 1
        # 先拼接系统字段
        doc_info = point["doc_info"]
        for system_field in ["doc_name", "title", "chunk_title", "content", "point_id"]:
            if system_field == 'doc_name' or system_field == 'title':
                if system_field in doc_info:
                    prompt += f"{system_field}: {doc_info[system_field]}\n"
            else:
                if system_field in point:
                    if system_field == "content":
                        prompt += f"content: {get_content_for_prompt(point, image_cnt)}\n"
                    else:
                        prompt += f"{system_field}: {point[system_field]}\n"
        if "table_chunk_fields" in point:
            table_chunk_fields = point["table_chunk_fields"]
            for self_field in []:
                # 使用 next() 从 table_chunk_fields 中找到第一个符合条件的项目
                find_one = next((item for item in table_chunk_fields if item["field_name"] == self_field), None)
                if find_one:
                    prompt += f"{self_field}: {find_one['field_value']}\n"

        prompt += "---\n"

    return base_prompt.format(prompt), image_urls


# 主函数，执行搜索知识库和聊天补全
def search_knowledge_and_chat_completion():
    global query
    # 1.执行search_knowledge
    rsp_txt = search_knowledge()
    # 2.生成prompt
    prompt, image_urls = generate_prompt(rsp_txt)
    # 拼接message对话, 问题对应role为user，系统对应role为system, 答案对应role为assistant, 内容对应content
    if image_urls:
        multi_modal_msg = [{"type": "text", "text": query}]
        for image_url in image_urls:
            multi_modal_msg.append({"type": "image_url", "image_url": {"url": image_url}})
        messages = [
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": multi_modal_msg
            }
        ]
    else:
        messages = [
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": query
            }
        ]

    # 4.调用chat_completion
    response_text = chat_completion(messages)
    return response_text


def read_test_answer(file_path):
    questions = []
    answers = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            if lines[i].startswith('[QUESTION]'):
                question = lines[i].strip().replace('[QUESTION]', '').strip()
                i += 1
                if i < len(lines) and lines[i].startswith('[ANSWER]'):
                    answer = lines[i].strip().replace('[ANSWER]', '').strip()
                    questions.append(question)
                    answers.append(answer)
            i += 1
    return questions, answers


def calculate_em(predicted_answer, reference_answer):
    return 1 if predicted_answer == reference_answer else 0


def calculate_f1_score(predicted_answer, reference_answer):
    predicted_words = predicted_answer.split()
    reference_words = reference_answer.split()
    common_words = Counter(predicted_words) & Counter(reference_words)
    correct_count = sum(common_words.values())
    precision = correct_count / len(predicted_words) if predicted_words else 0
    recall = correct_count / len(reference_words) if reference_words else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1_score


def longest_common_subsequence(predicted_answer, reference_answer):
    m, n = len(predicted_answer), len(reference_answer)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if predicted_answer[i - 1] == reference_answer[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def calculate_rouge_l(predicted_answer, reference_answer):
    lcs_length = longest_common_subsequence(predicted_answer, reference_answer)
    precision = lcs_length / len(predicted_answer) if predicted_answer else 0
    recall = lcs_length / len(reference_answer) if reference_answer else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1_score


if __name__ == "__main__":
    test_file_path = "test_answer.txt"
    questions, answers = read_test_answer(test_file_path)
    for question, answer in zip(questions, answers):
        query = question
        response = search_knowledge_and_chat_completion()
        try:
            response_json = json.loads(response)
            if "generated_answer" in response_json["data"]:
                predicted_answer = response_json["data"]["generated_answer"]
            else:
                predicted_answer = response_json["data"]["choices"][0]["message"]["content"]
            # 去除预测答案中的<reference>标签及其内容
            start_index = predicted_answer.find('<reference')
            if start_index != -1:
                predicted_answer = predicted_answer[:start_index].strip()
        except (json.JSONDecodeError, KeyError) as e:
            print(f"处理问题 '{question}' 时出错，响应内容: {response}")
            print(f"错误信息: {e}")
            predicted_answer = "解析错误"
        em_score = calculate_em(predicted_answer, answer)
        f1_score = calculate_f1_score(predicted_answer, answer)
        rouge_l_score = calculate_rouge_l(predicted_answer, answer)
        print(f"问题: {question}")
        print(f"预测答案: {predicted_answer}")
        print(f"标准答案: {answer}")
        print(f"EM: {em_score}")
        print(f"F1-Score: {f1_score}")
        print(f"ROUGE-L: {rouge_l_score}")
        print("-" * 50)