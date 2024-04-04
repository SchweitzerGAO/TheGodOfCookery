import json

'''
Stage 1: extract the questions(run only once)
'''
# with open('./eval_dataset_test.json','r',encoding='utf-8') as f:
#     questions = [d['conversation'][0]['input'] for d in json.load(f)]

'''
Stage 2: write the original questions(run only once)
'''
# with open('./questions.txt','w',encoding='utf-8') as f:
#     for q in questions:
#         f.write(f'{q}\n')

'''
Stage 3: rewrite the questions
'''
# Leverage KimiChat, details at https://kimi.moonshot.cn/share/co6kshgnsmmh5fcns79g

'''
Stage 4: alter the questions
'''
with open('./eval_dataset_test.json','r',encoding='utf-8') as f:
    data = json.load(f)
with open('./questions_new.txt','r',encoding='utf-8') as f:
    questions = f.readlines()

# ori_questions = [d['conversation'][0]['input'] for d in data]

# for q in ori_questions:
#     if '\n' in q:
#         print(q)

for d, q in zip(data, questions):
    d['conversation'][0]['input'] = q
with open('./eval_dataset_test_new.json','w',encoding='utf-8') as f:
    json.dump(data, f,ensure_ascii=False, indent=4)