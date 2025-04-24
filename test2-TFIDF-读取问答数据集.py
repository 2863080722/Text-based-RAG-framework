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
# 创建x['question']获取JSON文件中question对应的内容，赋予question_words
# jieba.lcut()函数用于将这段文本进行分词，即将一个句子分割成多个词语，并返回一个包含所有词语的列表。
# join()函数用于将列表中的元素连接成一个字符串，元素之间用指定的分隔符分隔。
question_words = [' '.join(jieba.lcut(x['question'])) for x in questions]
pdf_content_words = [' '.join(jieba.lcut(x['content'])) for x in pdf_content]

# fit方法会分析传入的文本数据，构建一个词汇表(包含所有单词），并为词汇表中的每个词计算IDF值。
tfidf = TfidfVectorizer()
tfidf.fit(question_words + pdf_content_words)


# ----------------------------------------------------------------------------------------------------------------------
# 提取TFIDF
# transform方法将question_words和pdf_content_words中的文本转换为TF-IDF特征矩阵。
# transform方法会利用fit方法学习到的词汇表和IDF值，将新的文本数据转换为特征矩阵。question_feat和pdf_feat分别包含了问题和PDF内容的TF-IDF特征矩阵。
# question_feat中，包含301行，对应301个问题，有好多列，每一列对应一个单词，每个值代表每个单词的TFIDF值，也就是该词的重要性
# pdf_content_feat中，包含354行，对应354页，有好多列，每一列对应一个单词，每个值代表每个单词的TFIDF值，也就是该词的重要性
question_feat = tfidf.transform(question_words)
pdf_content_feat = tfidf.transform(pdf_content_words)


# 进行归一化
# 将所有数据放在统一的尺度上
question_feat = normalize(question_feat)
pdf_content_feat = normalize(pdf_content_feat)


# ----------------------------------------------------------------------------------------------------------------------
# 检索进行排序
# 计算第query_idx个问题与每一页的相关性
for query_idx, feat in enumerate(question_feat):
    # enumerate函数会返回第query_idx行的所有元素，这些元素包含在feat中
    # for循环中，query_idx代表问题序号，feat代表该问题中每一个词对应的TFIDF值
    # 使用feat @ pdf_content_feat.T计算第 query_idx 个问题的特征向量与PDF内容中每个页面的特征向量之间的点积。点积是衡量两个向量相似度的一种方法
    score = feat @ pdf_content_feat.T

    # score.toarray()[0]将score从稀疏矩阵格式转换为NumPy数组，
    score = score.toarray()[0]

    # 通过score.argsort()获取按得分排序的页面索引。argsort()函数返回的是数组值从小到大的索引值。
    # [-1]取最后一位，得到最相关的页数
    max_score_page_idx = score.argsort()[-1] + 1
    # 赋予reference最相关的页数
    questions[query_idx]['reference'] = 'page_' + str(max_score_page_idx)


# ----------------------------------------------------------------------------------------------------------------------
# 生成提交结果
with open('submit.json', 'w', encoding='utf8') as up:
        json.dump(questions, up, ensure_ascii=False, indent=4)
        # open('submit.json', 'w', encoding='utf8') 打开一个名为 submit.json 的文件用于写入（'w' 模式）。
        # 如果文件不存在，它会被创建；如果文件已存在，它会被覆盖。encoding='utf8' 指定文件使用UTF-8编码。
        # json.dump() 函数用于将Python对象（这里是 questions）转换为JSON格式，并写入到指定的文件对象
print("建立submit.json成功")





















