from setuptools import setup
import setuptools

setup(
    name='tidytextpy',     # 包名字
    version='0.0.1',   # 包版本
    description='将R语言tidytext包移植到Python，可简单调用unnest_tokens、get_sentiments、get_stopwords、bind_tf_idf等函数。',   # 简单描述
    author='大邓',  # 作者
    author_email='thunderhit@qq.com',  # 邮箱
    url='https://github.com/thunderhit/tidytextpy',      # 包的主页
    packages=setuptools.find_packages(),
    package_data = {'': ['dictionary/*.txt','dictionary/*.csv']},  #所有目录下的csv、txt词典文件
    python_requires='>=3.5',
    license="MIT",
    keywords=['tidytext', 'text analysis', 'sentiment', 'sentiment analysis', 'natural language processing', 'R', 'python'],
    long_description=open('README.md').read(), # 读取的Readme文档内容
    long_description_content_type="text/markdown")  # 指定包文档格式为markdown
