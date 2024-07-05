import json
from tqdm import tqdm
import random
# from config import load_config
import os
import pickle
import sys
sys.path.append('..')

with open('./openai_key.txt','r',encoding='utf-8') as k:
    KEY = k.readline().strip()
os.environ['OPENAI_API_KEY'] = KEY

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from interface import load_retriever, load_model, load_chain
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from datasets import Dataset

data_path = '../tran_dataset_single_0/tran_dataset_single_0.json'
database_path = './database.json'
others_path = './others.json'
ood_test = False # Out-of-dataset test 

'''
Stage 1: build 100k RAG database(run only once)
'''
# with open(data_path,'r',encoding='utf-8') as fr:
#     data = json.load(fr)
#     idx_range = range(len(data))
#     for _ in range(3):
#         random.shuffle(data)
#     idxs = random.sample(idx_range,100000)
#     database = []
#     others = []
#     for i in range(len(data)):
#         if i in idxs:
#             database.append(data[i])
#         else: 
#             others.append(data[i])
#     with open(database_path,'w',encoding='utf-8') as fw1:
#         json.dump(database, fw1)
#     with open(others_path,'w',encoding='utf-8') as fw2:
#         json.dump(others, fw2)


'''
Stage 2: vectorize the database(run only once)
'''
# # most of the code is derived from rag/create_db.json

# with open(database_path, 'r', encoding='utf-8') as f:
#     json_data = json.load(f)

# # 取前test_count个元素, 用于测试
# # test_count小于等于0时, 取全部元素

# emb_strategy = load_config('rag', 'emb_strategy')
# assert emb_strategy['source_caipu'] or emb_strategy['HyQE'], "source_caipu and HyQE cannot be both False"

# # 创建待编码文档集
# split_docs = []
# for i in range(len(json_data)):
#     question = json_data[i]['conversation'][0]['input']
#     # 如果input只有菜名，则加上“的做法”
#     if "做" not in question:
#         question += "的做法"
#     answer = json_data[i]['conversation'][0]['output']
#     # 加入原始菜谱
#     if emb_strategy['source_caipu']:
#         split_docs.append(Document(page_content=question + "\n" + answer))
#     # 假设问题为“菜谱名+怎么做”
#     # 加入假设问题，原始菜谱存放入metadata
#     if emb_strategy['HyQE']:
#         split_docs.append(Document(page_content=question, metadata={"caipu": question + "\n" + answer}))

# # 加载编码模型
# bce_emb_config = load_config('rag', 'bce_emb_config')
# embeddings = HuggingFaceEmbeddings(**bce_emb_config)

# # 构建BM25检索器
# bm25_config = load_config('rag', 'bm25_config')
# bm25retriever = BM25Retriever.from_documents(documents=split_docs)
# bm25retriever.k = bm25_config['search_kwargs']['k']

# # BM25Retriever序列化到磁盘
# if not os.path.exists(bm25_config['dir_path']):
#     os.mkdir(bm25_config['dir_path'])
# pickle.dump(bm25retriever, open(bm25_config['save_path'], 'wb'))

# # 构建向量数据库
# rag_model_type = load_config('rag', 'rag_model_type')
# if rag_model_type == "chroma":
#     vectordb = Chroma.from_documents(documents=split_docs, embedding=embeddings,
#                                      persist_directory=load_config('rag', 'chroma_config')['save_path'])
#     # 持久化到磁盘
#     vectordb.persist()
# else:
#     faiss_index = FAISS.from_documents(documents=split_docs, embedding=embeddings,
#                                        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE)
#     # 保存索引到磁盘
#     faiss_index.save_local(load_config('rag', 'faiss_config')['save_path'])


'''
Stage 3: build test dataset
'''
ds = []
with open(database_path,'r',encoding='utf-8') as database:
    database_data = json.load(database)
    if not ood_test:
        ds.extend(random.sample(database_data, 100))
    else:
        ds.extend(random.sample(database_data,50))
        with open(others_path,'r',encoding='utf-8') as others:
            other_data = json.load(others)
            ds.extend(random.sample(other_data, 50))
    random.shuffle(ds)

'''
Stage 4: evaluate
'''
_ , _ , llm = load_model()

chain = load_chain(llm)
retriever = load_retriever()

# qa_chain({"query": query})['result']

questions = [data['conversation'][0]['input'] for data in ds]
ground_truths = [data['conversation'][0]['output'] for data in ds]
contexts = []
for i, query in enumerate(questions):
    docs = retriever.get_relevant_documents(query)
    if len(docs) == 0:
        contexts.append([])
    else:
        contexts.append([d.page_content for d in docs])
    print(f'Context #{i + 1} retrieved')
answers = []
for i, query in enumerate(questions):
    answers.append(chain({'query':query})['result'])
    print(f'Query #{i + 1} finished')

eval_data = {
    'question': questions,
    'answer': answers,
    'contexts': contexts,
    'ground_truth': ground_truths
}
with open('./eval_data.pkl','wb') as f:
    pickle.dump(eval_data, f)

# dataset = Dataset.from_dict(eval_data)
# result = evaluate(
#     dataset = dataset, 
#     metrics=[
#         context_precision,
#         context_recall,
#         faithfulness,
#         answer_relevancy,
#     ],
# )
 
# df = result.to_pandas()
# df.to_csv('./result.csv')
