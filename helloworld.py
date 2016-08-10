#  -*-encoding: utf-8 -*-
import email
from email.Iterators import typed_subpart_iterator
import sys
import re
import imp
#  import jieba
import os
from bs4 import BeautifulSoup
import jieba.analyse
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

imp.reload(sys)
sys.setdefaultencoding('utf-8')


def get_charset(message, default="ascii"):
    """Get the message charset"""

    if message.get_content_charset():
        # print message.get_content_charset()
        return message.get_content_charset()

    if message.get_charset():
        # print message.get_charset()
        return message.get_charset()

    return default


def get_body(message):
    """Get the body of the email message"""

    if message.is_multipart():
        # get the html text version only
        text_parts = [part
                      for part in typed_subpart_iterator(message,
                                                         'text',
                                                         'html')]
        body = []
        for part in text_parts:
            charset = get_charset(part, get_charset(message))
            body.append(unicode(part.get_payload(decode=True),
                                charset,
                                "replace"))

        return u"\n".join(body).strip()
    else:
        # if it is not multipart, the payload will be a string
        # representing the message body
        body = unicode(message.get_payload(decode=True),
                       get_charset(message),
                       "replace")
        return body.strip()


def filter_tags(htmlstr):
    # 先过滤CDATA
    re_cdata = re.compile('//<!\[CDATA\[[^>]*//\]\]>', re.I)  # 匹配CDATA
    # Script
    re_script = re.compile('<\s*script[^>]*>[^<]*<\s*/\s*script\s*>', re.I)
    # style
    re_style = re.compile('<\s*style[^>]*>[^<]*<\s*/\s*style\s*>', re.I)
    re_br = re.compile('<br\s*?/?>')  # 处理换行
    re_h = re.compile('</?!?\w+[^>]*>')  # HTML标签
    re_comment = re.compile('<!--[^>]*-->')  # HTML注释
    re_blankline = re.compile('\n+')
    s = re_cdata.sub('', htmlstr)  # 去掉CDATA
    s = re_script.sub('', s)  # 去掉SCRIPT
    s = re_style.sub('', s)  # 去掉style
    s = re_br.sub('', s)  # 将br转换为换行
    s = re_h.sub('', s)  # 去掉HTML 标签
    s = re_comment.sub('', s)  # 去掉HTML注释
    s = re_blankline.sub('', s)  # 去掉多余的空行
    s = replaceCharEntity(s)  # 替换实体
    return s


def replaceCharEntity(htmlstr):
    CHAR_ENTITIES = {'nbsp': '', '160': '',
                     'lt': '<', '60': '<',
                     'gt': '>', '62': '>',
                     'amp': '&', '38': '&',
                     'quot': '"', '34': '"'}
    re_charEntity = re.compile(r'&#?(?P<name>\w+);')
    sz = re_charEntity.search(htmlstr)
    while sz:
        # entity = sz.group()  # entity全称，如&gt;
        key = sz.group('name')  # 去除&;后entity,如&gt;为gt
        try:
            htmlstr = re_charEntity.sub(CHAR_ENTITIES[key], htmlstr, 0)
            sz = re_charEntity.search(htmlstr)
        except KeyError:
            #  以空串代替
            htmlstr = re_charEntity.sub('', htmlstr, 0)
            sz = re_charEntity.search(htmlstr)
    return htmlstr


def repalce(s, re_exp, repl_string):
    return re_exp.sub(repl_string, s)


def isWord(word):
    if(word != ' ' and word != '\t' and word != '(' and
       word != ')' and word != '：' and word != '，' and
       word != '。' and word != '！' and word != '.' and
       word != '《' and word != '》' and word != '/'):
        return True
    else:
        return False

fp = open("zsyh7.eml")
msg = email.message_from_file(fp)  # 直接文件创建message对象
# print msg
words = get_body(msg)
print words
# soup = BeautifulSoup(words, 'html.parser')
# print soup
# for string in soup.stripped_strings:
#    print(string)
# vectorizer = CountVectorizer(min_df=1)
# x = vectorizer.fit_transform((soup))
# print x
# filter_words = filter_tags(words)
# content = open('stop_words.txt','rb').read()
print os.getcwd()
jieba.analyse.set_stop_words("c:\mail\python\stop_words.txt")
tags = jieba.analyse.extract_tags(words, topK=20)

# seg_list = jieba.lcut(words)
# print (repr(seg_list))
print ','.join(tags)
'''vectorizer = CountVectorizer(min_df=1)
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(seg_list))
wordlist = vectorizer.get_feature_names()
weightlist = tfidf.toarray()
# print weightlist
# for i in range(len(weightlist)):
#    for j in range(len(wordlist)):
#        print wordlist[j], weightlist[i][j]
res = {}
for word in seg_list:
    if(isWord(word)):
        if(word not in res):
            res[word] = 1
        else:
            res[word] += 1
    else:
        continue

for a in res:
    print a, res[a]'''
