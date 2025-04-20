import json
from volcengine.viking_knowledgebase import VikingKnowledgeBaseService
from volcengine.viking_knowledgebase.common import FieldType, IndexType, EmbddingModelType
from volcengine.auth.SignerV4 import SignerV4
from volcengine.base.Request import Request
from volcengine.Credentials import Credentials

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
    r.set_shema("https")
    r.set_method(method)
    r.set_connection_timeout(30)
    r.set_socket_timeout(30)
    mheaders = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    r.set_headers(mheaders)
    if params:
        r.set_query(params)
    r.set_path(path)
    if data is not None:
        r.set_body(json.dumps(data))  # 将数据转换为 JSON 字符串

    # 生成签名
    credentials = Credentials(ak, sk, "air", "cn-north-1")
    SignerV4.sign(r, credentials)
    return r

# 初始化知识库服务
viking_knowledgebase_service = VikingKnowledgeBaseService(host="api-knowledgebase.mlp.cn-beijing.volces.com", scheme="https", connection_timeout=30, socket_timeout=30)
ak = ""
sk = ""
viking_knowledgebase_service.set_ak(ak)
viking_knowledgebase_service.set_sk(sk)

collection_name = ""
description = ""

try:
    # 尝试删除已存在的集合
    viking_knowledgebase_service.drop_collection(collection_name)
except Exception as e:
    print(f"删除集合时出现异常: {e}")
    pass  # 忽略异常

# 自定义index配置、preprocess文档配置构建知识库
index = {
    "index_type": IndexType.HNSW_HYBRID.value,  # 将 IndexType 转换为字符串值
    "index_config": {
        "fields": [{
            "field_name": "chunk_len",
            "field_type": FieldType.Int64.value,  # 将 FieldType 转换为字符串值
            "default_val": 0
        }],
        "cpu_quota": 2,
        "embedding_model": EmbddingModelType.EmbeddingModelDoubaoLargeAndM3.value  # 将 EmbddingModelType 转换为字符串值
    }
}

# 修正 chunk_length 到有效范围，并使用自定义切片策略
preprocessing = {
    "chunking_strategy": "custom",  # 使用自定义切片策略
    "chunk_identifier": "[TEXT]",  # 以 [TEXT] 为分隔符
    "chunk_length": 2500,  # 这里仍然超出范围，建议修改为 100 到 500 之间的值
    "merge_small_chunks": False  # 开启合并短文本片
}

try:
    # 创建集合的请求方式
    method = "POST"
    # 创建集合的请求路径
    path = "/api/knowledge/collection/create"
    # 创建集合所需要的一些参数
    request_params = {
        "collection_name": collection_name,
        "description": description,
        "index": index,
        "preprocessing": preprocessing
    }
    info_req = prepare_request(method=method, path=path, data=request_params)
    my_collection = viking_knowledgebase_service.create_collection(collection_name=collection_name, description=description, index=index, preprocessing=preprocessing)
    print(f"成功创建集合: {collection_name}")
    # 只有在创建集合成功后才进行获取集合信息和上传文档的操作
    try:
        # 获取集合信息的请求方式
        method = "GET"
        # 获取集合信息的请求路径
        path = f"/api/knowledge/collection/{collection_name}"
        info_req = prepare_request(method=method, path=path)
        my_collection = viking_knowledgebase_service.get_collection(collection_name=collection_name)
        print(f"成功获取集合 {collection_name} 的详细信息")

        # 由url上传doc
        url = ""  # 注意转义字符
        # 上传文档的请求方式
        method = "POST"
        # 上传文档的请求路径
        path = f"/api/knowledge/collection/{collection_name}/doc/add"
        request_params = {
            "add_type": "url",
            "doc_id": "doc_111",
            "doc_name": "qa_data.txt",
            "doc_type": "txt",
            "url": url
        }
        info_req = prepare_request(method=method, path=path, data=request_params)
        my_collection.add_doc(add_type="url", doc_id="doc_111", doc_name="qa_data.txt", doc_type="txt", url=url)
        print(f"成功通过 URL {url} 上传文档")
    except Exception as e:
        print(f"获取集合信息或上传文档时出现异常: {e}")
except Exception as e:
    print(f"创建集合时出现异常: {e}")