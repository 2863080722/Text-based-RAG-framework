import json
import pdfplumber
import torch
import transformers
import jieba
import sklearn
import numpy
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from rank_bm25 import BM25Okapi
from zhipuai import ZhipuAI
from scipy.stats import rankdata

print(torch.cuda.is_available())

# ----------------------------------------------------------------------------------------------------------------------
# 定义访问ChatGLM的函数
def ask_alm(content):
    # 填写自己的APIKey
    client = ZhipuAI(api_key="c3122cd0e5c11b9312ef7c6559826bc2.f3eufPOzX6u7U4jO")
    response = client.chat.completions.create(
        # 填写需要调用的模型名称
        model="glm-4",
        messages=[
            {"role": "user", "content": content},
        ],
    )
    return response.choices[0].message.content


# ----------------------------------------------------------------------------------------------------------------------
# 读取问题的JSON文件
# 共有301个问题
print("读取文件中.....")
questions = json.load(open("questions.json", encoding='utf-8'))

# 读取数据集的PDF文件
# 共有354页
pdf = pdfplumber.open("初赛训练数据集.pdf")
pdf_content = []
for page_idx in range(len(pdf.pages)):
    # 从0开始
    pdf_content.append({
        'page': 'page_' + str(page_idx + 1),
        # 第一页为1，也就是在pdf的content中，第0项对应第1页
        'content': pdf.pages[page_idx].extract_text()
    })

# 提取问题及PDF文档中的句子
question_sentences = [x['question'] for x in questions]
pdf_content_sentences = [x['content'] for x in pdf_content]
print("读取文件完成")

# ----------------------------------------------------------------------------------------------------------------------
# 加载重排序模型
print("加载重排序模型中.....")
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large')
rerank_model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large')
rerank_model.cuda()
rerank_model.eval()
print("加载重排序模型完成")

# ----------------------------------------------------------------------------------------------------------------------
# 进行BM25检索
print("BM25检索中.....")
pdf_content_words = [jieba.lcut(x['content']) for x in pdf_content]
bm25 = BM25Okapi(pdf_content_words)

# 进行BGE检索
model = SentenceTransformer('BAAI/bge-base-zh-v1.5')
question_embeddings = model.encode(question_sentences, normalize_embeddings=True)
pdf_embeddings = model.encode(pdf_content_sentences, normalize_embeddings=True)

# ----------------------------------------------------------------------------------------------------------------------
# 遍历每一个问题
for query_idx, feat in enumerate(question_embeddings):
    score1 = feat @ pdf_embeddings.T
    score2 = bm25.get_scores(jieba.lcut(questions[query_idx]["question"]))
    score = rankdata(score1) + rankdata(score2)
    max_score_page_idxs = score.argsort()[-3:]
    # 采用正向传播模型对TOP3进行重排序
    print("第", query_idx + 1, "问题重排序中.....")
    pairs = []

    # pairs中，每个问题对应三页在BM25检索下最相关的页面
    # 构建文本对，将用户提问与检索得到的文本拼接为文本对,并进行编码
    for idx in max_score_page_idxs:
        pairs.append([questions[query_idx]["question"], pdf_content[idx]['content']])

    # 通过tokenizer函数将pairs中的各项的长度等要素归一化，并转化为pytorch张量
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)

    # 采用正向传播对文本对进行检索排序
    # 将文本对进行正向传播，得到匹配度的打分
    with torch.no_grad():
        # 将所有值转移到GPU上，加速计算
        inputs = {key: inputs[key].cuda() for key in inputs.keys()}
        # 通过rerank_model模型对文本进行正向传播，得到匹配打分logits
        scores = rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()

    # 得到重排序的结果，通过scores.cpu().numpy().argmax()找到最大项在TOP3中选取匹配度最高的文本
    max_score_page_idx = max_score_page_idxs[scores.cpu().numpy().argmax()]
    questions[query_idx]['reference'] = 'page_' + str(max_score_page_idx + 1)
    print("第", query_idx + 1, "问题检索完成")

    # ------------------------------------------------------------------------------------------------------------------
    # 对每个问题生成prompt
    # {0}，{1}为占位符，通过format()将其替换
    prompt = '''
    你是一个汽车专家，帮我结合给定的资料，回答一个问题。如果问题无法从资料中获得，请输出结合给定的资料，无法回答问题。
    资料：{0}
    问题：{1} 
    '''.format(
        pdf_content[max_score_page_idx]['content'],
        questions[query_idx]["question"]
    )

    print("调取答案中.....")
    answer = ask_alm(prompt)
    questions[query_idx]['answer'] = answer
    print("导入答案完成")

# ----------------------------------------------------------------------------------------------------------------------
# 生成提交结果
with open('submit.json', 'w', encoding='utf8') as up:
        json.dump(questions, up, ensure_ascii=False, indent=4)
        # open('submit.json', 'w', encoding='utf8') 打开一个名为 submit.json 的文件用于写入（'w' 模式）。
        # 如果文件不存在，它会被创建；如果文件已存在，它会被覆盖。encoding='utf8' 指定文件使用UTF-8编码。
        # json.dump() 函数用于将Python对象(这里是 questions)转换为JSON格式，并写入到指定的文件对象
print("建立submit.json成功")

for i in range(len(questions)):
    print(questions[i])

# written by ZI HANG_LIU























