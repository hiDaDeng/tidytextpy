# TidyTextPy

tidytext是R语言的文本分析包，一般数据会整理为dataframe，每行都是由docid-word-freq组成。有一本R语言的文本挖掘书《Text mining with R》，知识体系挺完整的，该书主力分析工具是R语言的tidytext包。



最早 https://github.com/machow/tidytext-py 项目初步实现了R语言中的unntest_tokens和bind_tf_idf，但未实现get_sentiments和get_stopwords，本项目主要是基于https://github.com/machow/tidytext-py，将其完善。



本项目可能图片看不到，大家可以点击链接:https://pan.baidu.com/s/1Lf8fd7Ra4A3GwoViLyZQnA  密码:wucj 下载本文代码和数据。

# 安装

```
pip install tidytextpy
```



# 实验数据

这里使用中文科幻小说《三体》为例子，含注释共213章，使用正则表达式构建三体小说数据集，该数据集涵
- chapterid 第几章
- title 章(节)标题
- text 每章节的文本内容(分词后以空格间隔的文本，形态类似英文)


```python
import pandas as pd
import jieba
import re
pd.set_option('display.max_rows', 6)

raw_texts = open('三体.txt', encoding='utf-8').read()
texts = re.split('第\d+章', raw_texts)
texts = [text for text in texts if text]
#中文多了下面一行代码（构造用空格间隔的字符串）
texts = [' '.join(jieba.lcut(text)) for text in texts if text]
titles = re.findall('第\d+章 (.*?)\n', raw_texts)

data = {'chapterid': list(range(1, len(titles)+1)),
        'title': titles,
        'text': texts}
df = pd.DataFrame(data)
df
```

    Building prefix dict from the default dictionary ...
    Loading model from cache /var/folders/sc/3mnt5tgs419_hk7s16gq61p80000gn/T/jieba.cache
    Loading model cost 0.592 seconds.
    Prefix dict has been built successfully.





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>chapterid</th>
      <th>title</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>科学边界(1)</td>
      <td>科学 边界 ( 1 ) \n \n         恋上你 看书 网   630book...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>科学边界(2)</td>
      <td>科学 边界 ( 2 ) \n \n         恋上你 看书 网   630book...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>台球</td>
      <td>台球 \n \n         恋上你 看书 网   630bookla   ， 最快...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>210</th>
      <td>211</td>
      <td>【时间之外，我们的宇宙】(2)</td>
      <td>【 时间 之外 ， 我们 的 宇宙 】 ( 2 ) \n \n         恋上你 ...</td>
    </tr>
    <tr>
      <th>211</th>
      <td>212</td>
      <td>【时间之外，我们的宇宙】(3)</td>
      <td>【 时间 之外 ， 我们 的 宇宙 】 ( 3 ) \n \n         恋上你 ...</td>
    </tr>
    <tr>
      <th>212</th>
      <td>213</td>
      <td>注释</td>
      <td>注释 \n \n         恋上你 看书 网   630bookla   ， 最快...</td>
    </tr>
  </tbody>
</table>
<p>213 rows × 3 columns</p>
</div>



# tidytextpy库
- get_stopwords 停用词表
- get_sentiments 情感词典
- unnest_tokens 分词函数
- bind_tf_idf  计算tf-idf




# 停用词表
get_stopwords(language)  获取对应语言的停用词表，目前仅支持chinese和english两种语言


```python
from tidytextpy import get_stopwords

cn_stps = get_stopwords('chinese')
#前20个中文的停用词
cn_stps[:20]
```




    ['、',
     '。',
     '〈',
     '〉',
     '《',
     '》',
     '一',
     '一些',
     '一何',
     '一切',
     '一则',
     '一方面',
     '一旦',
     '一来',
     '一样',
     '一般',
     '一转眼',
     '七',
     '万一',
     '三']




```python
en_stps = get_stopwords()
#前20个英文文的停用词
en_stps[:20]
```




    ['i',
     'me',
     'my',
     'myself',
     'we',
     'our',
     'ours',
     'ourselves',
     'you',
     'your',
     'yours',
     'yourself',
     'yourselves',
     'he',
     'him',
     'his',
     'himself',
     'she',
     'her',
     'hers']



# 情感词典
**get_sentiments('词典名')** 调用词典，返回词典的dataframe数据。

- [afinn](http://www2.imm.dtu.dk/pubdb/pubs/6010-full.html) sentiment取值-5到5
- [bing](https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html) sentiment取值为positive或negative
- [nrc](http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm) sentiment取值为positive或negative，及细粒度的情绪分类信息
- [dutir](https://github.com/ZaneMuir/DLUT-Emotionontology) sentiment为中文七种情绪类别（细粒度情绪分类信息）
- hownet sentiment为positive或negative

其中hownet和dutir为**中文情感词典**


```python
from tidytextpy import get_sentiments

#大连理工大学情感本体库，共七种情绪（sentiment）
get_sentiments('dutir')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sentiment</th>
      <th>word</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>惊</td>
      <td>冷不防</td>
    </tr>
    <tr>
      <th>1</th>
      <td>惊</td>
      <td>惊动</td>
    </tr>
    <tr>
      <th>2</th>
      <td>惊</td>
      <td>珍闻</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>27411</th>
      <td>惧</td>
      <td>匆猝</td>
    </tr>
    <tr>
      <th>27412</th>
      <td>惧</td>
      <td>忧心仲忡</td>
    </tr>
    <tr>
      <th>27413</th>
      <td>惧</td>
      <td>面面厮觑</td>
    </tr>
  </tbody>
</table>
<p>27414 rows × 2 columns</p>
</div>




```python
get_sentiments('nrc')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>word</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>abacus</td>
      <td>trust</td>
    </tr>
    <tr>
      <th>1</th>
      <td>abandon</td>
      <td>fear</td>
    </tr>
    <tr>
      <th>2</th>
      <td>abandon</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>13898</th>
      <td>zest</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>13899</th>
      <td>zest</td>
      <td>trust</td>
    </tr>
    <tr>
      <th>13900</th>
      <td>zip</td>
      <td>negative</td>
    </tr>
  </tbody>
</table>
<p>13901 rows × 2 columns</p>
</div>



# 分词
``unnest_tokens(__data, output, input)``
- ``__data`` 待处理的dataframe数据
- output 新生成的dataframe中，用于存储分词结果的字段名
- input 待分词数据的字段名(待处理的dataframe数据)



```python
from tidytextpy import unnest_tokens

tokens = unnest_tokens(df, output='word', input='text')
tokens
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>chapterid</th>
      <th>title</th>
      <th>word</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>科学边界(1)</td>
      <td>科学</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>科学边界(1)</td>
      <td>边界</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>科学边界(1)</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>212</th>
      <td>213</td>
      <td>注释</td>
      <td>想到</td>
    </tr>
    <tr>
      <th>212</th>
      <td>213</td>
      <td>注释</td>
      <td>暗物质</td>
    </tr>
    <tr>
      <th>212</th>
      <td>213</td>
      <td>注释</td>
      <td>。</td>
    </tr>
  </tbody>
</table>
<p>556595 rows × 3 columns</p>
</div>



## 各章节用词量
从这里开始会用到plydata的管道符>> 和相关的常用函数，建议大家遇到不懂的地方查阅plydata文档


```python
from plydata import count, group_by, ungroup


wordfreq = (df 
            >> unnest_tokens(output='word', input='text') #分词
            >> group_by('chapterid')  #按章节分组
            >> count() #对每章用词量进行统计
            >> ungroup() #去除分组
           )

wordfreq
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>chapterid</th>
      <th>n</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2549</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2666</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1726</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>210</th>
      <td>211</td>
      <td>2505</td>
    </tr>
    <tr>
      <th>211</th>
      <td>212</td>
      <td>2646</td>
    </tr>
    <tr>
      <th>212</th>
      <td>213</td>
      <td>2477</td>
    </tr>
  </tbody>
</table>
<p>213 rows × 2 columns</p>
</div>



## 章节用词量可视化
使用plotnine进行可视化


```python
from plotnine import ggplot, aes, theme, geom_line, labs, theme, element_text
from plotnine.options import figure_size

(ggplot(wordfreq, aes(x='chapterid', y='n'))+
 geom_line()+
 labs(title='三体章节用词量折线图',
      x='章节', 
      y='用词量')+
 theme(figure_size=(12, 8),
       title=element_text(family='Kai', size=15), 
       axis_text_x=element_text(family='Kai'),
       axis_text_y=element_text(family='Kai'))
)
```


![png](output_14_0.png)





    <ggplot: (338899281)>



# 情感分析
重要的事情多重复一遍o(*￣︶￣*)o

**get_sentiments('词典名')** 调用词典，返回词典的dataframe数据。

- [afinn](http://www2.imm.dtu.dk/pubdb/pubs/6010-full.html) sentiment取值-5到5
- [bing](https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html) sentiment取值为positive或negative
- [nrc](http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm) sentiment取值为positive或negative，及细粒度的情绪分类信息
- [dutir](https://github.com/ZaneMuir/DLUT-Emotionontology) sentiment为中文七种情绪类别（细粒度情绪分类信息）
- hownet sentiment为positive或negative

其中hownet和dutir为**中文情感词典**


## 情感计算
这里会用到plydata的很多知识点，大家可以查看https://plydata.readthedocs.io/en/latest/index.html 相关函数的文档。

![](img/inner-join.gif)


```python
from plydata import inner_join, count, define, call
from plydata.tidy import spread

chapter_sentiment_score = (
    df #分词
    >> unnest_tokens(output='word', input='text') 
    >> inner_join(get_sentiments('hownet')) #让分词结果与hownet词表交集，给每个词分配sentiment
    >> count('chapterid', 'sentiment')#统计每章中每类sentiment的个数
    >> spread('sentiment', 'n') #将sentiment中的positive和negative转化为两列
    >> call('.fillna', 0) #将缺失值替换为0
    >> define(score = '(positive-negative)/(positive+negative)') #计算每一章的情感分score
)

chapter_sentiment_score
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>chapterid</th>
      <th>negative</th>
      <th>positive</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>93.0</td>
      <td>56.0</td>
      <td>-0.248322</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>98.0</td>
      <td>83.0</td>
      <td>-0.082873</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>54.0</td>
      <td>37.0</td>
      <td>-0.186813</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>210</th>
      <td>211</td>
      <td>56.0</td>
      <td>73.0</td>
      <td>0.131783</td>
    </tr>
    <tr>
      <th>211</th>
      <td>212</td>
      <td>71.0</td>
      <td>67.0</td>
      <td>-0.028986</td>
    </tr>
    <tr>
      <th>212</th>
      <td>213</td>
      <td>75.0</td>
      <td>74.0</td>
      <td>-0.006711</td>
    </tr>
  </tbody>
</table>
<p>213 rows × 4 columns</p>
</div>



## 三体小说情感走势
我记得看完《三体》后，很悲观，觉得人类似乎永远逃不过宇宙的时空规律，心情十分压抑。如果对照小说进行章节的情感分析，应该整体情感分的走势大多在0以下。


```python
from plotnine import ggplot, aes, geom_line, element_text, labs, theme

(ggplot(chapter_sentiment_score, aes('chapterid', 'score'))+
 geom_line()+
 labs(x='章节', y='情感值score', title='《三体》小说情感走势图')+
 theme(title=element_text(family='Kai'))
)
```


![png](output_18_0.png)





    <ggplot: (364328989)>



# tf-idf
相比之前的代码，bind_tf_idf运行起来很慢很慢。所以这里用别的数据做实验

## tf-idf实验数据


```python
import pandas as pd
pd.set_option('display.max_rows', 6)

zen = """
The Zen of Python, by Tim Peters

Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!
"""

zen_split = zen.splitlines()

df = pd.DataFrame({'docid': list(range(len(zen_split))),
                  'text': zen_split})

df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>docid</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td></td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>The Zen of Python, by Tim Peters</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td></td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>19</td>
      <td>If the implementation is hard to explain, it's...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>20</td>
      <td>If the implementation is easy to explain, it m...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>21</td>
      <td>Namespaces are one honking great idea -- let's...</td>
    </tr>
  </tbody>
</table>
<p>22 rows × 2 columns</p>
</div>



## bind_tf_idf
tf表示词频，idf表示词语在文本中的稀缺性，两者的结合体现了一个词的信息量。找出小说中tf-idf最大的词。

``bind_tf_idf(_data, term, document, n)``

- ``_data`` 传入的df
- term df中词语对应的字段名
- document df中文档id的字段名
- n df中词频数对应的字段名


```python
from tidytextpy import bind_tf_idf
from plydata import count, group_by, ungroup

tfidfs = (df
          >> unnest_tokens(output='word', input='text')
          >> count('docid', 'word')
          >> bind_tf_idf(term='word', document='docid', n='n')
         )

tfidfs
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>docid</th>
      <th>word</th>
      <th>n</th>
      <th>tf</th>
      <th>idf</th>
      <th>tf_idf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>the</td>
      <td>1</td>
      <td>0.142857</td>
      <td>1.386294</td>
      <td>0.198042</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>zen</td>
      <td>1</td>
      <td>0.142857</td>
      <td>2.995732</td>
      <td>0.427962</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>of</td>
      <td>1</td>
      <td>0.142857</td>
      <td>1.897120</td>
      <td>0.271017</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>137</th>
      <td>21</td>
      <td>more</td>
      <td>1</td>
      <td>0.090909</td>
      <td>2.995732</td>
      <td>0.272339</td>
    </tr>
    <tr>
      <th>138</th>
      <td>21</td>
      <td>of</td>
      <td>1</td>
      <td>0.090909</td>
      <td>1.897120</td>
      <td>0.172465</td>
    </tr>
    <tr>
      <th>139</th>
      <td>21</td>
      <td>those</td>
      <td>1</td>
      <td>0.090909</td>
      <td>2.995732</td>
      <td>0.272339</td>
    </tr>
  </tbody>
</table>
<p>140 rows × 6 columns</p>
</div>





# 如果

如果您是经管人文社科专业背景，编程小白，面临海量文本数据采集和处理分析艰巨任务，个人建议学习[《python网络爬虫与文本数据分析》](https://ke.qq.com/course/482241?tuin=163164df)视频课。作为文科生，一样也是从两眼一抹黑开始，这门课程是用五年时间凝缩出来的。自认为讲的很通俗易懂o(*￣︶￣*)o，

- python入门
- 网络爬虫
- 数据读取
- 文本分析入门
- 机器学习与文本分析
- 文本分析在经管研究中的应用

感兴趣的童鞋不妨 戳一下[《python网络爬虫与文本数据分析》](https://ke.qq.com/course/482241?tuin=163164df)进来看看~



# 更多

- [B站:大邓和他的python](https://space.bilibili.com/122592901/channel/detail?cid=66008)
- **公众号：大邓和他的python**
- [知乎专栏：数据科学家](https://zhuanlan.zhihu.com/dadeng)


<br>

![](img/大邓和他的Python.png)

      
