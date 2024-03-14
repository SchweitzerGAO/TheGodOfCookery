<div align="center">
  <img src="https://github.com/SmartFlowAI/TheGodOfCookery/blob/main/images/cooker.png" width="1092"/>
  <br /><br />

![license](https://img.shields.io/github/license/SmartFlowAI/TheGodOfCookery.svg)  [![issue resolution](https://img.shields.io/github/issues-closed-raw/SmartFlowAI/TheGodOfCookery)](https://github.com/SmartFlowAI/TheGodOfCookery/issues)   [![open issues](https://img.shields.io/github/issues-raw/SmartFlowAI/TheGodOfCookery)](https://github.com/SmartFlowAI/TheGodOfCookery/issues)

[中文 ](https://github.com/SmartFlowAI/TheGodOfCookery/blob/main/README.md)| [English](https://github.com/SmartFlowAI/TheGodOfCookery/blob/main/README_EN.md)

🔍 探索我们的模型：
[![Static Badge](https://img.shields.io/badge/-gery?style=social&label=🤖%20ModelScope)](https://www.modelscope.cn/models/zhanghuiATchina/zhangxiaobai_shishen2_full/summary)

</div>
</p>

## 简介

​		本项目名称为“食神”（ The God Of Cookery ），灵感来自喜剧大师周星驰主演的著名电影《食神》，旨在通过人工智能技术为用户提供烹饪咨询和食谱推荐，帮助用户更好地学习和实践烹饪技巧，降低烹饪门槛，实现《食神》电影中所讲的“只要用心，人人皆能做食神”。

​		本APP的基本思想，是基于InternLM的对话模型，采用 XiaChuFang Recipe Corpus 提供的1,520,327种中国食谱进行微调，生成食谱模型。 模型存放在[ModelScope](https://www.modelscope.cn/models/zhanghuiATchina/zhangxiaobai_shishen2_full/summary)上，应用部署在[OpenXlab](https://openxlab.org.cn/apps/detail/zhanghui-china/shishen2024)上。为此感谢魔搭社区提供免费的模型存放空间，感谢OpenXLab提供应用部署环境及GPU资源。

​		本APP提供的回答仅供参考，不作为正式菜谱的真实制作步骤。由于大模型的“幻觉”特性，很可能有些食谱会给用户带来心理或生理上的不利影响，切勿上纲上线。

### 整体技术架构说明

​		项目主要依赖上海人工智能实验室开源模型internlm-chat-7b（包含1代和2代），在XiaChuFang Recipe Corpus 提供的1,520,327种中国食谱数据集上借助Xtuner进行LoRA微调，形成shishen2_full模型，并将微调后模型与向量数据库整合入langchain，实现RAG检索增强的效果，并可进行多模态（语音、文字、图片）问答对话，前端基于streamlit实现与用户的交互。

![](README.assets/ddeeaf42-097f-4e70-b265-314fa7309685.png)

### 应用整体流程说明

​		用户发出请求后，应用加载模型（语音模型，文生图模型，微调后的对话模型），并处理用户的文字输入或者语音输入，如果未打开RAG开关，则直接调用微调后的对话模型生成回复，对结果进行格式化输出，并调用stable diffusion模型生成图片，最后将相应结果返回用户；如果打开RAG开关，则利用langchain检索向量数据库，并将检索结果输入微调后的对话模型生成回复，对结果进行格式化输出，并调用stable diffusion模型生成图片，最后将相应结果返回用户。

![](README.assets/output-1710423526789.png)

## 更新说明

- [2024.3.9] 基于团队成员 @乐正萌 的RAG模块(faiss)，整合 text2image分支，发布二阶段第4个基于openxlab A100的应用 [openxlab A100 app](https://openxlab.org.cn/apps/detail/zhanghui-china/shishen2024) 和 openxlab A10的应用 [openxlab A10 app](https://openxlab.org.cn/apps/detail/zhanghui-china/shishen2024_1.8b)  
- [2024.3.4] 增加英文readme
- [2024.3.3] 基于团队成员 @solo fish 的 paraformer语音输入模块，整合 text2image分支，发布二阶段第3个基于openxlab A100的应用 [openxlab app](https://openxlab.org.cn/apps/detail/zhanghui-china/nlp_shishen3)
- [2024.2.24] 基于团队成员 @Charles 的RAG模块(Chroma)，整合 text2image分支，发布二阶段第2个基于openxlab A100的应用 [openxlab app](https://openxlab.org.cn/apps/detail/zhanghui-china/nlp_shishen3)
- [2024.2.22] 基于团队成员 @房生亮 的文生图模块 以及 @solo fish 的 whisper语音输入模块，整合 text2image分支，发布二阶段第1个基于openxlab A100的应用 [openxlab app](https://openxlab.org.cn/apps/detail/zhanghui-china/nlp_shishen3)
- [2024.1.30] 基于二代150万菜谱微调的模型和APP发布。（使用InternStudio+A100 1/4X2 40G显存微调，1.25 15:46-1.30 12:25，微调历时4天20小时39分钟）
- [2024.1.28] 基于一代150万菜谱微调的模型和APP发布。（使用WSL+Ubuntu22.04+RTX4090 24G显存微调，1.26 18:40-1.28 13:46历时1天19小时6分钟）。

## 使用指南

### 数据集

[150万下厨房微调数据集:提取密码8489](https://pan.baidu.com/s/1TyqDWRI5jOs621VXr-uMoQ)

### 安装

1. 准备 Python 虚拟环境：

   ```bash
   conda create -n cook python=3.10 -y
   conda activate cook
   ```

2. 克隆该仓库：

   ```shell
   git clone https://github.com/SmartFlowAI/TheGodOfCookery.git
   cd ./TheGodOfCookery
   ```

3. 安装Pytorch和依赖库：

   ```shell
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   pip install -r requirements.txt
   ```
这里cuda的版本根据用户自己的cuda版本确定。一般为 11.8或12.1

### 训练

​		一阶段一代模型 使用 xtuner 0.1.9 训练，在 internlm-chat-7b 上进行微调 <br />
​		一阶段二代模型 使用 xtuner 0.1.13 训练，在 internlm2-chat-7b 上进行微调 <br />
    二阶段二代模型 使用 xtuner 0.1.15.dev0 训练，在 internlm2-chat-1.8b 上进行微调 <br />

1. 微调方法如下

   ```shell
   xtuner train ${YOUR_CONFIG} --deepspeed deepspeed_zero2
   ```

   - `--deepspeed` 表示使用 [DeepSpeed](https://github.com/microsoft/DeepSpeed) 🚀 来优化训练过程。XTuner 内置了多种策略，包括 ZeRO-1、ZeRO-2、ZeRO-3 等。如果用户期望关闭此功能，请直接移除此参数。

2. 将保存的 `.pth` 模型（如果使用的DeepSpeed，则将会是一个文件夹）转换为 LoRA 模型：

   ```shell
   export MKL_SERVICE_FORCE_INTEL=1
   xtuner convert pth_to_hf ${YOUR_CONFIG} ${PTH} ${LoRA_PATH}
   ```

   3.将LoRA模型合并入 HuggingFace 模型：
   
   ```
   xtuner convert merge ${Base_PATH} ${LoRA_PATH} ${SAVE_PATH}
   ```

### 对话

```shell
xtuner chat ${SAVE_PATH} [optional arguments]
```

参数：

- `--prompt-template`: 一代模型使用 internlm_chat，二代使用  internlm2_chat。
- `--system`: 指定对话的系统字段。
- `--bits {4,8,None}`: 指定 LLM 的比特数。默认为 fp16。
- `--no-streamer`: 是否移除 streamer。
- `--top`: 对于二代模型，建议为0.8。
- `--temperature`: 对于二代模型，建议为0.8。
- `--repetition-penalty`: 对于二代7b模型，建议为1.002，对于二代1.8b模型，建议为1.17，对于一代模型可不填。
- 更多信息，请执行 `xtuner chat -h` 查看。

### 演示

一阶段：

Demo 访问地址：https://openxlab.org.cn/apps/detail/zhanghui-china/nlp_shishen

![](README.assets/answer001.png)

![](README.assets/answer002.png)

二阶段：

Demo 访问地址：<https://openxlab.org.cn/apps/detail/zhanghui-china/shishen2024>

![1710422208862](README.assets/1710422208862.png)

![1710422224731](README.assets/1710422224731.png)

### 模型地址

​[modelscope一代7b模型](https://www.modelscope.cn/models/zhanghuiATchina/zhangxiaobai_shishen_full/summary)    <br />
[modelscope二代7b模型](https://www.modelscope.cn/models/zhanghuiATchina/zhangxiaobai_shishen2_full/summary)    <br />
[modelscope二代1.8b模型](https://www.modelscope.cn/models/zhanghuiATchina/zhangxiaobai_shishen2_1_8b/summary)    <br />
[openxlab一代7b模型](https://openxlab.org.cn/models/detail/zhanghui-china/zhangxiaobai_shishen_full)    <br />
[openxlab二代7b模型](https://openxlab.org.cn/models/detail/zhanghui-china/zhangxiaobai_shishen2_full)    <br />

```shell
import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM
from tools.transformers.interface import GenerationConfig, generate_interactive

model_name_or_path = "zhanghuiATchina/zhangxiaobai_shishen_full" #对于二代模型改为 zhanghuiATchina/zhangxiaobai_shishen2_full

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
model = model.eval()

messages = []
generation_config = GenerationConfig(max_length=max_length, top_p=0.8, temperature=0.8, repetition_penalty=1.002)

response, history = model.chat(tokenizer, "你好", history=[])
print(response)
response, history = model.chat(tokenizer, "酸菜鱼怎么做", history=history)
print(response)
```


### 一阶段实践文档

[一代实践](https://zhuanlan.zhihu.com/p/678019309)  <br />
[二代实践](https://zhuanlan.zhihu.com/p/678376843)  <br />

[实践视频](https://www.bilibili.com/video/BV1Ut421W7Qg)  <br />

## 项目代码结构（二阶段）

   ```shell
项目目录
|---config   # 配置文件目录（主要贡献者 @房宇亮）
|     |---__init__.py                                      #初始化脚本
|     |---config.py                                        #配置脚本
|
|---gen_image    # 文生图目录（主要贡献者 @房宇亮）
|     |---__init__.py                                      #初始化脚本
|     |---sd_gen_image.py                                  #使用Stabble Disffion的文生图模块
|     |---zhipu_ai_image.py                                #使用智谱AI的文生图模块
|
|---images  # 的图片目录，生成的图片临时也放在这里，今后会考虑迁移到其他目录
|     |---robot.png                                        #对话机器人图标 
|     |---user.png                                         #对话用户图标 
|     |---shishen.png                                      #项目图标 （主要贡献者 @刘光磊）
|
|---rag   # 二代RAG代码目录（主要贡献者 @乐正萌）
|     |---source_data                                      #原始数据集目录
|     |     |- text.txt                                    #原始菜谱数据集
|     |---data                                             #处理后的数据集目录
|     |     |- caipu.txt                                   #处理后的菜谱数据集
|     |---chroma_db                                        #chroma数据库目录
|     |     |- chroma.sqlite3                              #chroma库文件
|     |---faiss_index                                      #FAISS数据库目录
|     |     |- index.faiss   
|     |     |- index.pkl
|     |---retrieve                                         #retrieve目录
|     |     |- bm25retriever.pkl
|     |---CookMasterLLM.py
|     |---convert_txt.py
|     |---create_db.py
|     |---interface.py
|     |---rag_test.py
|     |---run_local.py
|
|---rag_chroma   # 二代RAG代码目录（主要贡献者 @Charles）
|     |---database                                         #chroma数据库目录
|     |     |- chroma.sqlite3                              #chroma库文件
|     |---LLM.py
|     |---create_db.py
|     |---interface.py
|
|---src    # 食材图标目录
|     |---*.png                                            #各类食材图标
|
|---tools    # 工具文件目录
|
|---whisper_app    # 语音识别目录（主要贡献者 @solo fish）
|     |---__init__.py                                      #初始化脚本
|     |---whisper.py                                       #语音识别处理脚本
|
|---speech    # paraformer语音识别目录（主要贡献者 @solo fish）
|     |---__init__.py                                      #初始化脚本
|     |---utils.py                                         #语音识别处理脚本
|
|---requirements.txt                                       #系统依赖包（请使用pip install -r requirements.txt安装）
|---convert_t2s.py                                         #繁体字转简体字工具（主要贡献者 @彬彬）
|---parse_cur_response.py                                  #输出格式化处理工具 （主要贡献者 @彬彬）
|---README.md                                              #本文档
|---cli_demo.py                                            #模型测试脚本
|---download.py                                            #模型下载脚本
|---start.py                                               #Web Demo启动脚本
|---app.py                                                 #Web Demo主脚本

   ```

## 项目参与人员（排名不分先后）

|                        用户名                         |                      组织                      |                            贡献                            |                             备注                             |
| :---------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------------------: | :----------------------------------------------------------: |
| [张小白](https://www.zhihu.com/people/zhanghui_china) |     南京大学本科毕业，现为某公司数据工程师     |                    项目策划、测试和打杂                    | 华为云HCDE（原华为云MVP），2020年华为云社区十佳博主，2022年昇腾社区优秀开发者，2022年华为云社区年度优秀版主，MindSpore布道师，DataWhale优秀学习者 |
|      [sole fish](https://github.com/YanxingLiu)       |          中国科学院大学在读博士研究生          |                        语音输入模块                        |                                                              |
|      [Charles](https://github.com/SchweitzerGAO)      |           同济大学本科毕业生，考研中           |                 一代RAG模块（基于Chroma）                  |                                                              |
|       [乐正萌](https://github.com/YueZhengMeng)       |         上海海洋大学本科毕业生，考研中         |              二代RAG模块（基于faiss&Chroma）               |                                                              |
|        [彬彬](https://github.com/Everfighting)        | 华东师范大学本科毕业、现为某公司算法开发工程师 |                         格式化输出                         |                                                              |
|        [房宇亮](https://github.com/leonfrank)         |     南京大学本科毕业，现为某公司算法工程师     |                    文生图模块、配置工具                    |                                                              |
|        [刘光磊](https://github.com/Mrguanglei)        |                       -                        |                     图标设计，前端优化                     |                                                              |
|         [喵喵咪](https://github.com/miyc1996)         | 北京航空航天大学硕士毕业，现为上海某国企工程师 |             数据集准备，后续本地小模型部署测试             |                                                              |
|                        王巍龙                         |                       -                        |                        数据集，微调                        |                                                              |
|          [轩辕](https://github.com/zzd2001)           |                南京大学在读硕士                |                   文档准备，数据集，微调                   |                                                              |
|  [浦语小助手](https://github.com/InternLM/InternLM)   |                       -                        | 提供书生浦语大模型、工具链、训练环境、人才培养等全方面支持 |                                                              |
|                                                       |                                                |                                                            |                                                              |




## 开源许可证

本项目采用 [Apache License 2.0 开源许可证](LICENSE.txt)。

## 特别鸣谢

<center>感谢上海人工智能实验室组织的 书生·浦语实战营 学习活动~~~<br />
<img src="https://github.com/SmartFlowAI/TheGodOfCookery/blob/main/images/shanghaiailab.png" width="150"/>
  <br /></center>



<center>感谢 OpenXLab 对项目部署的算力支持~  <br />
<img src="https://github.com/SmartFlowAI/TheGodOfCookery/blob/main/images/openxlab.png" width="150"/></center>

<center>感谢 浦语小助手 对项目的支持~<br />
<img src="https://github.com/SmartFlowAI/TheGodOfCookery/blob/main/images/internlm.jpg" width="150"/></center>




## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=SmartFlowAI/TheGodOfCookery&type=Date)](https://star-history.com/#SmartFlowAI/TheGodOfCookery&Date)
