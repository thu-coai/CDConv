# CDConv
Data and codes for EMNLP 2022 paper "[CDConv: A Benchmark for Contradiction Detection in Chinese Conversations](https://arxiv.org/abs/2210.08511)"

## Data

`cdconv.txt`中的每一行为一段对话session，各字段含义如下：

- `u1`, `b1`, `u2`, `b2`表示user和bot之间的对话（各两句，交替发言）
- `file`表示标注批次，共包含5个标注批次
- `model`表示bot所采用的模型（eva或plato）。其中eva为EVA 2.0模型（编码器-解码器模型，各24层、共2.8B参数，项目地址：https://github.com/thu-coai/EVA/），plato为32层版本的模型（共1.6B参数）
- `method`表示u2的构造方法，具体含义如下：
  - `短句`：u2为无信息量的短句
  - `设问-bot`：u2对b1中的实体信息提问
  - `设问-user(-v2)`：u2对u1中的实体信息提问
  - `同义-回译`：将u1翻译成英文、再回译成中文
  - `同义-同义词`：替换u1中的词为同义词
  - `反义-反义词`：替换u1中的词为反义词
  - `反义-否定词`：在u1中插入否定词
- `label`表示矛盾类型标注（0：无矛盾，1：b2句内矛盾，2：b2角色混淆，3：b2与对话历史矛盾）
  - `persona`表示从人设角度，对对话历史矛盾进行了矛盾内容的标注（1：人物属性，2：人物观点和偏好，3：人物经历，0：其他）

数据集的统计指标如下：

|                                                    | EVA       | PLATO     | Total       |
| -------------------------------------------------- | --------- | --------- | ----------- |
| # Conversations                                    | 5,458     | 6,202     | 11,660      |
| # Positive                                         | 3,233     | 4,076     | 7,309       |
| # Negative                                         | 2,225     | 2,126     | 4,351       |
| **Trigger Methods (Positive / Negative Samples)**  |           |           |             |
| # Short                                            | 429 / 91  | 692 / 304 | 1,121 / 395 |
| # Inquiring (Bot)                                  | 764 / 577 | 845 / 406 | 1,609 / 983 |
| # Inquiring (User)                                 | 127 / 116 | 131 / 106 | 258 / 222   |
| # Inquiring (User-M)                               | 251 / 552 | 477 / 541 | 728 / 1,093 |
| # Paraphrasing                                     | 962 / 448 | 846 / 389 | 1,808 / 837 |
| # Synonym                                          | 288 / 145 | 376 / 147 | 664 / 292   |
| # Antonym                                          | 185 / 143 | 319 / 103 | 504 / 246   |
| # Negative                                         | 227 / 153 | 390 / 130 | 617 / 283   |
| **Contradiction Categories (of Negative Samples)** |           |           |             |
| Intra-sentence                                     | 17.3%     | 6.8%      | 12.2%       |
| Role                                               | 5.8%      | 29.9%     | 17.6%       |
| History                                            | 76.9%     | 63.3%     | 70.2%       |
| **Persona Labels (of History Contradiction)**      |           |           |             |
| Attributes                                         | 48.8%     | 46.2%     | 47.7%       |
| Opinions                                           | 22.2%     | 20.7%     | 21.5%       |
| Experiences                                        | 26.3%     | 31.5%     | 28.6%       |
| Unrelated                                          | 2.7%      | 1.6%      | 2.2%        |

## Codes

参见`codes`文件夹
