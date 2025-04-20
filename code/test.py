import json
import requests
from volcengine.auth.SignerV4 import SignerV4
from volcengine.base.Request import Request
from volcengine.Credentials import Credentials

# 初始化知识库服务相关配置
collection_name = ""
project_name = ""
ak = ""
sk = ""
account_id = ""
g_knowledge_base_domain = ""

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
def search_knowledge(query):
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
    # print("search res = {}".format(rsp.text))
    return rsp.text

# 主函数，执行搜索知识库并返回前三个切片ID
def search_knowledge_and_get_id(query):
    rsp_txt = search_knowledge(query)
    rsp = json.loads(rsp_txt)
    retrieved_ids = []
    if rsp["code"] == 0 and rsp["data"]["result_list"]:
        for result in rsp["data"]["result_list"][:3]:
            retrieved_ids.append(result["id"])
    return retrieved_ids

if __name__ == "__main__":
    correct_ids = []
    guessed_ids = []
    correct_count_top1 = 0  # 初始化 top1 正确次数
    correct_count_top3 = 0  # 初始化 top3 正确次数

    # 读取answer.txt中的正确ID
    with open('answer.txt', 'r', encoding='ascii') as file:
        correct_ids = [line.strip() for line in file.readlines()]

    # 读取question.txt中的问题
    with open('keyword.txt', 'r', encoding='utf-8') as file:
        questions = [line.strip() for line in file.readlines()]

    for query, correct_id in zip(questions, correct_ids):
        guessed_id_list = search_knowledge_and_get_id(query)
        guessed_ids.append(guessed_id_list)
        print(f"检索问题 '{query}' 对应的ID: {guessed_id_list}")

        # 计算 top1 准确率
        if guessed_id_list and guessed_id_list[0] == correct_id:
            correct_count_top1 += 1

        # 计算 top3 准确率
        is_correct_top3 = any(correct_id in guessed_id_list for guessed_id_list in guessed_id_list)
        if is_correct_top3:
            correct_count_top3 += 1

    accuracy_top1 = correct_count_top1 / len(correct_ids) if correct_ids else 0
    accuracy_top3 = correct_count_top3 / len(correct_ids) if correct_ids else 0

    print(f"top1 准确率: {accuracy_top1 * 100:.2f}%")
    print(f"top3 准确率: {accuracy_top3 * 100:.2f}%")