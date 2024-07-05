# RAG retriever 评估

## 1. 评估目的

- 评估 RAG retriever 的检索结果质量
- 评估附加 RAG 的 LLM 的回答质量

## 2. 评估指标

## 2.1 F1-score

这是机器学习中常见的评估标准，可以用来评估本项目场景下retriever检索结果以及大模型生成结果的质量。

## 2.2 ragas

这是一个通用的RAG模型评测框架， 该框架可以使用多种不同的metric评估附加RAG的LLM生成的答案，适用范围较为广泛。其中常用的metric包括：

- 忠实度(faithfulness)

将答案拆分成多个statement，利用大模型评估检索出的上下文是否可以推断出statement，忠实度分数 =  可以推断出的statement个数 / 总statement个数

- 

## 3. 评估方法

### 3.1 F1-score

首先给出F1-score的定义：
$F1=\dfrac{2PR}{P+R}$

其中$P=\dfrac{TP}{TP+FP}$是precision(准确率)， $R=\dfrac{TP}{TP+FN}$是recall(召回率)

在本项目中，$TP+FP$是RAG的检索结果或LLM产生的回答，$TP+FN$是测评数据集中的正确答案

### 3.2 ragas

## 4. 如何开始评估

### 4.1 使用 F1-score 评估

1. 构建评估数据集

只需运行`build_eval_dataset.py`。脚本会将`data_source.txt`中的数据进行清洗并格式化为JSON格式，示例如下

```json
 {
    "conversation": [
      {
        "system": "你是一个专业的厨师，你会做很多菜。用户报上自己所需的菜名后，你可以把做菜所需要的原料，以及做菜的方法告诉用户",
        "input": "红烧滩羊肉的做法",
        "output": "您需要准备以下食材:\n['1kg羊肉', '5片姜', '3瓣蒜', '适量花椒', '3勺老抽', '3片香叶', '2个八角', '1个干辣椒', '1块桂皮', '2勺料酒', '适量盐', '3根香菜', '2个小洋葱', '适量乱泉水']\n按以下方法制作:\n['滩羊肉在姜水里焯3分钟，姜水里加点料酒去腥', '再将羊肉反过来焯水2分钟', '把羊肉切2/3手掌大小，羊肉煮熟后会缩水，所以可以稍微大一点的', '热油里放入花椒、桂皮、香叶、生姜爆炒30秒，放入切好的羊肉翻炒3分钟左右，把羊肉里面的油煸炒出来，再加入老抽上色，把所有羊肉都上色后加入乱泉水，以漠过羊肉上面为准', '锅里水烧开后，换成砂锅中火慢炖40分钟', '分次吃', '超级软糯', '加点萝卜进去炖起来，解油腻']"
      }
    ]
  }
```

每个`conversation`下仅有一轮对话，每轮对话包含以下字段

- `system`：自我认知prompt，在所有测试数据中均为

```py
"""
你是一个专业的厨师，你会做很多菜。用户报上自己所需的菜名后，你可以把做菜所需要的原料，以及做菜的方法告
"""
```

- `input` 用户输入

- `output` 正确答案

完整测试数据在`eval_dataset.json`文件中。

2. 进行评估

运行`eval_f1.py`，就可以评估retriever检索结果的F1-score以及LLM回答的F1-score

**进行完整评估时，注意修改35、58行的数据集路径为`./eval_dataset.json`**

### 4.2 使用 ragas 评估

## 5. 评估结果

### 5.1 F1-score

我们目前从`eval_dataset.json`中随机抽取了1000条进行retriever评估，随机抽取了10条数据进行LLM评估

Retriever检索 F1-score：  

```bash
F1 score sum: 993.7684651584043
The number of data: 1000
F1 average: 0.9937684651584043
```

LLM回答 F1-score：  

```bash
F1 score sum: 5.221176107197454  
The number of data: 10  
F1 average: 0.5221176107197454
```

### 5.2 ragas
