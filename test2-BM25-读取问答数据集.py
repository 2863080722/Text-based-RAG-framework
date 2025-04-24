import json
import pdfplumber
import torch
import transformers
import jieba
import sklearn
import numpy
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('moka-ai/m3e-base')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from rank_bm25 import BM25Okapi

# ----------------------------------------------------------------------------------------------------------------------
# 读取问题的JSON文件
# 共有301个问题
questions=json.load(open("questions.json",encoding='utf-8'))
print(len(questions))

# 读取数据集的PDF文件
# 共有354页
pdf = pdfplumber.open("初赛训练数据集.pdf")
print(len(pdf.pages))
pdf_content = []
for page_idx in range(len(pdf.pages)):
    # 从0开始
    pdf_content.append({
        'page': 'page_'+str(page_idx+1),
        # 第一页为1，也就是在pdf的content中，第0项对应第1页
        'content': pdf.pages[page_idx].extract_text()
    })

# ----------------------------------------------------------------------------------------------------------------------
# 对文本进行切分-采用单词切分
# 通过BM25Okapi将切分后单词导入bm25中
pdf_content_words = [jieba.lcut(x['content']) for x in pdf_content]
bm25 = BM25Okapi(pdf_content_words)

# ----------------------------------------------------------------------------------------------------------------------
# 检索进行排序
# # bm25.get_scores()计算第query_idx个问题与每一页的BM25值
for query_idx in range(len(questions)):
    # bm25.get_scores返回第query_idx个问题与每一页的bm25值
    doc_scores = bm25.get_scores(jieba.lcut(questions[query_idx]['question']))
    max_score_page_idx = doc_scores.argsort()[-1] + 1
    questions[query_idx]['reference'] = 'page_' + str(max_score_page_idx)

# ----------------------------------------------------------------------------------------------------------------------
# 生成提交结果
with open('submit.json', 'w', encoding='utf8') as up:
        json.dump(questions, up, ensure_ascii=False, indent=4)
        # open('submit.json', 'w', encoding='utf8') 打开一个名为 submit.json 的文件用于写入（'w' 模式）。
        # 如果文件不存在，它会被创建；如果文件已存在，它会被覆盖。encoding='utf8' 指定文件使用UTF-8编码。
        # json.dump() 函数用于将Python对象（这里是 questions）转换为JSON格式，并写入到指定的文件对象
print("建立submit.json成功")




















