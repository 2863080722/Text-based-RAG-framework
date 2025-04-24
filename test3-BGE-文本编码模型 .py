import json
import pdfplumber
import torch
import transformers
import jieba
import sklearn
import numpy
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# ----------------------------------------------------------------------------------------------------------------------
# 读取问题的JSON文件
# 共有301个问题
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


# ----------------------------------------------------------------------------------------------------------------------
# 对文本进行切分-采用token分割器
# 从0开始，递进到len(text),以chunk_size为步长
# 该函数的返回值为第i段的文本
def split_text_fixed_size(text, chunk_size):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


# 从0遍历到最后一页
for page_idx in range(len(pdf.pages)):
    # 读取第page_idx页的text
    text = pdf.pages[page_idx].extract_text()
    # 对第page_idx页的text进行切分，以40为chunk_size
    for chunk_text in split_text_fixed_size(text, 40):
        # 得到每一句切分后的小块以及对应的页码，如有需要还可以得到每个小块对应的序号
        pdf_content.append({
            'page': 'page_' + str(page_idx + 1),
            'content': chunk_text
        })

# ----------------------------------------------------------------------------------------------------------------------
# 对文本进行编码，采用BGE文本编码模型
# 得到每一个问题与每一页的embeddings值
# normalize_embeddings=True表示默认归一化
model = SentenceTransformer('BAAI/bge-base-zh-v1.5')
question_embeddings = model.encode(question_sentences, normalize_embeddings=True, show_progress_bar=True)
pdf_embeddings = model.encode(pdf_content_sentences, normalize_embeddings=True, show_progress_bar=True)

# ----------------------------------------------------------------------------------------------------------------------
# 检索进行排序
# 计算每个问题与每个页面的相似度
for query_idx, feat in enumerate(question_embeddings):
    score = feat @ pdf_embeddings.T
    max_score_page_idx = score.argsort()[-1] + 1
    questions[query_idx]['reference'] = pdf_content[max_score_page_idx]['page']

# ----------------------------------------------------------------------------------------------------------------------
# 生成结果
for i in range(len(questions)):
    print(questions[i])
