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
print(torch.cuda.is_available())

# ----------------------------------------------------------------------------------------------------------------------
# 读取问题的JSON文件
# 共有301个问题
print("读取文件中.....")
questions = json.load(open("questions.json", encoding='utf-8'))
print(len(questions))

# 读取数据集的PDF文件
# 共有354页
pdf = pdfplumber.open("初赛训练数据集.pdf")
print(len(pdf.pages))
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
print("加载重排序模型完成")

# 先进行BM25检索
print("BM25检索中.....")
pdf_content_words = [jieba.lcut(x['content']) for x in pdf_content]
bm25 = BM25Okapi(pdf_content_words)

for query_idx in range(len(questions)):
    doc_scores = bm25.get_scores(jieba.lcut(questions[query_idx]["question"]))
    # 取TOP3的页码
    max_score_page_idxs = doc_scores.argsort()[-3:]

    # 对TOP3进行重排序
    # 构建文本对，将用户提问与检索得到的文本拼接为文本对,并进行编码
    # pairs中，每个问题对应三页在BM25检索下最相关的页面
    pairs = []
    for idx in max_score_page_idxs:
        pairs.append([questions[query_idx]["question"], pdf_content[idx]['content']])
    print(pairs)

    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)

    # 将文本对进行正向传播，得到匹配度的打分
    with torch.no_grad():
        inputs = {key: inputs[key].cuda() for key in inputs.keys()}
        scores = rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()

    # 得到重排序的结果，在TOP3中再选取匹配度最高的文本
    max_score_page_idx = max_score_page_idxs[scores.cpu().numpy().argmax()]
    questions[query_idx]['reference'] = 'page_' + str(max_score_page_idx + 1)

# ----------------------------------------------------------------------------------------------------------------------
# 生成结果
print("生成结果文件中.....")
with open('submit.json', 'w', encoding='utf8') as up:
    json.dump(questions, up, ensure_ascii=False, indent=4)
print("生成结果文件完成")

























