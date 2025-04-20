api.py  该文件直接调用大模型，完整的多轮对话代码演示，可直接运行。
evaluate.py 用于评估关键词或者问题的准确率（不加入大模型评估结果，且EM，F1指标还在）
model_evaluate.py相较于上述代码加入了大模型评估
evaluation_res保存了评估结果
fanli.txt包括了设计的反例和源TEXT
fanli_res.txt包括了反例测试最后的结果
fanliceshi.py用于反例测试
keyword.txt是随机选取的关键词 question.txt是随机选取的问题
output.jsonl是我按照标准数据集格式生成的样例，可用作微调
read.py是一个小脚本，用于读取文档并切分，同时随机选取一些TEXT，按照需要格式输出（比如keyword.txt）
sdk.py是py创建知识库的代码实现
test.py是用于测试文档检索
test_answer是根据答案设计的模板