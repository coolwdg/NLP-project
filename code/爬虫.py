import requests
from bs4 import BeautifulSoup
from openai import OpenAI

# 初始化豆包大模型客户端
client = OpenAI(
    base_url="",
    api_key=""
)

# 抓取页面正文内容
def fetch_ai_phd_notice_text(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    response.encoding = 'utf-8'
    soup = BeautifulSoup(response.text, 'html.parser')

    content_div = soup.find('div', class_='tj-intro')
    if not content_div:
        print("❌ 没有找到正文内容区域")
        return ""

    paragraphs = content_div.find_all('p')
    full_text = ""
    for p in paragraphs:
        text = p.get_text(strip=True)
        if text:
            full_text += text + "\n"
    return full_text.strip()

# 生成多个问答对
def generate_qa_10(text):
    prompt = (
        f"请根据以下内容生成多个高质量的问题及其对应的准确答案。多个问题使用相同的[TEXT]可以将它们放在同一组[TEXT]下，每个[TEXT]标签对应一个段落，格式如下：\n"
        f"- 输出格式必须严格如下：\n"
        f"[TEXT] 段落内容\n"
        f"[QUESTION] 问题1\n"
        f"[ANSWER] 答案1\n"
        f"[QUESTION] 问题2\n"
        f"[ANSWER] 答案2\n"
        f"\n"
        f"[TEXT] 第二段内容\n"
        f"[QUESTION] 问题3\n"
        f"[ANSWER] 答案3\n"
        f"[QUESTION] 问题4\n"
        f"[ANSWER] 答案4\n"
        f"\n"
        f"注意：问答之间不要空行，段落之间必须有一个空行。\n"
        f"内容如下：\n{text}\n"
        f"生成的问答对应该覆盖不同的信息点，并且相同的[TEXT]下可以有多个问答。"
    )

    try:
        completion = client.chat.completions.create(
            model="doubao-pro-256k-241115",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ 生成问答失败：{e}")
        return None

# 主程序
if __name__ == "__main__":
    url = "https://ai.bnu.edu.cn/ggjxz/tzgg/4b02d0641bfb49a7b1c4468b6c128d92.htm"
    text = fetch_ai_phd_notice_text(url)

    if text:
        print("📄 抓取到的正文内容：\n")
        print(text)
        print("\n🤖 正在生成问答对...\n")

        qa_output = generate_qa_10(text)
        if qa_output:
            print("🤖 生成的问答对：\n")
            print(qa_output)  # 打印生成的问答对

            # 保存问答对到文件
            with open("bnu_qa_dataset.txt", "w", encoding="utf-8") as f:
                f.write(qa_output)

            print("📁 数据集已保存为 bnu_qa_dataset.txt")
