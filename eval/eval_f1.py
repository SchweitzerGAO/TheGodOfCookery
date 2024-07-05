from collections import Counter
import re
import json
# from tqdm import tqdm
from interface import load_retriever, load_model, load_chain
import random

question_template = [
    '{cuisine}怎么做？',
    '我想做{cuisine}',
    '告诉我{cuisine}怎么做',
    '你知道{cuisine}怎么做吗',
    '{cuisine}的菜谱是什么',
    '{cuisine}的做法'
]

def de_punct(output: str):
    rule = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fa5]")
    output = rule.sub('', output)
    return output


def f1_score(output, gt):
    output = de_punct(output)
    gt = de_punct(gt)
    common = Counter(output) & Counter(gt)

    # Same words
    num_same = sum(common.values())
    if num_same == 0:
        return 0

    # precision
    precision = 1.0 * num_same / len(output)

    # recall
    recall = 1.0 * num_same / len(gt)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def evaluate_retriever():
    # 正式评估请手动替换下一行的路径
    log_file = open('./log_new.txt','w',encoding='utf-8')
    retriever = load_retriever()
    trials = 5
    p_ave = 0
    r_ave = 0
    f1_ave = 0
    for t in range(trials):
        # print()
        # print('################################')
        # print(f'Trial #{t + 1}')
        log_file.write('\n')
        log_file.write('################################\n')
        log_file.write(f'Trial #{t + 1}\n')
        with open('./eval_dataset_test_new.json', 'r', encoding='utf-8') as f:
            data = random.sample(json.load(f),100)
        random.shuffle(data)
        f1_sum = 0
        p_sum = 0
        r_sum = 0
        idx = random.randint(0, 99)
        for i, d in enumerate(data):
            # log_file.write(f'Query #{i + 1}\n')
            # print(f'Query #{i + 1}')
            query = random.choice(question_template).replace('{cuisine}',d["conversation"][0]['input'][:-3])
            docs = retriever.get_relevant_documents(query)
            if len(docs) == 0:
                output = ""
            else:
                output = docs[0].page_content
            gt = d["conversation"][0]['input'] + '\n' + d["conversation"][0]['output']
            p, r, f1 = f1_score(output, gt)
            if i == idx:
                # print()
                # print(f'Query #{i + 1}')
                # print('----------------------------')
                # print("查询：", query)
                # print("查询结果：", output)
                # print("答案：", gt)
                # print('----------------------------')
                # print()
                log_file.write('\n')
                log_file.write(f'Query #{i + 1}\n')
                log_file.write('----------------------------\n')
                log_file.write(f"查询：{query}\n")
                log_file.write(f"查询结果：{output}\n")
                log_file.write(f"答案：{gt}\n")
                log_file.write('----------------------------\n')
                log_file.write('\n')

            f1_sum += f1
            p_sum += p
            r_sum += r
        p_ave += (p_sum / len(data))
        r_ave += (r_sum / len(data))
        f1_ave += (f1_sum / len(data))
        # print(f'The number of data: {len(data)}')
        # # print(f'F1 score sum: {f1_sum}')
        # print(f'Precision average: {p_sum / len(data)}')
        # print(f'Recall average: {r_sum / len(data)}')
        # print(f'F1 average: {f1_sum / len(data)}')
        # print('################################')
        # print()
        log_file.write(f'The number of data: {len(data)}\n')
        log_file.write(f'Precision average: {p_sum / len(data)}\n')
        log_file.write(f'Recall average: {r_sum / len(data)}\n')
        log_file.write(f'F1 average: {f1_sum / len(data)}\n')
        log_file.write('################################\n')
        log_file.write('\n')
    log_file.write('################################\n')
    log_file.write(f'The number of trials: {trials}\n')
    log_file.write(f'Precision average: {p_ave / trials}\n')
    log_file.write(f'Recall average: {r_ave / trials}\n')
    log_file.write(f'F1 average: {f1_ave / trials}\n')
    log_file.write('################################\n')
    log_file.write('\n')
    log_file.close()

def evaluate_model():
    # 正式评估请手动替换下一行的路径
    with open('./eval_dataset_test.json', 'r', encoding='utf-8') as f:
        # 评测模型耗时较长，可以只评测部分数据
        data = random.sample(json.load(f),100)
    random.shuffle(data)
    f1_sum = 0
    p_sum = 0    
    r_sum = 0
    model, tokenizer, llm = load_model()
    qa_chain = load_chain(llm)
    for i, d in enumerate(data):
        print(f'Query #{i}')
        print('----------------------------')
        query = d["conversation"][0]['input']
        output = qa_chain({"query": query})['result']
        gt = d["conversation"][0]['input'] + '\n' + d["conversation"][0]['output']
        p, r, f1 = f1_score(output, gt)
        print("用户输入：", query)
        print("回答：", output)
        print("答案：", gt)
        print('----------------------------')
        print()
        f1_sum += f1
        p_sum += p
        r_sum += r
    print()
    print('################################')
    print(f'The number of data: {len(data)}')
    # print(f'F1 score sum: {f1_sum}')
    print(f'Precision average: {p_sum / len(data)}')
    print(f'Recall average: {r_sum / len(data)}')
    print(f'F1 average: {f1_sum / len(data)}')
    print('################################')
    print()



if __name__ == '__main__':
    evaluate_retriever()
    # evaluate_model()
