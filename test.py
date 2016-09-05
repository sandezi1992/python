#!/usr/bin/python
# -*-coding:utf-8-*-
# 提取出正负文档中的所有词汇
import os
import jieba.analyse
import sys
import email
import email.header
import chardet
from email.Iterators import typed_subpart_iterator
from sklearn import feature_extraction #  noqa
from sklearn import metrics
from sklearn import svm
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
# from sklearn.feature_selection import chi
from bs4 import BeautifulSoup
# from pandas import Series, DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.cross_validation import cross_val_score, KFold , StratifiedKFold
from sklearn.feature_selection import SelectKBest, chi2
from scipy.stats import sem
import numpy as np
from operator import itemgetter, attrgetter

reload(sys)
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


# 中文目录处理
def handleChinese(ustring):
    codeDetect = chardet.detect(ustring)["encoding"]
    ustring = unicode(ustring, codeDetect)
    ustring.encode("utf-8")
    return ustring


posPath = handleChinese('C:\mail\\trainMail\pos')
negPath = handleChinese('C:\mail\\trainMail\\neg')


def loadDataWithSubject(path):
    postingList = []
    name = []
    classVec = []
    for parent, dirname, filenames in os.walk(handleChinese(path)):
        for filename in filenames:
            fp = open(os.path.join(parent, filename))
            msg = email.message_from_file(fp)
            subject = msg.get('subject')
            subjectTmp = email.Header.decode_header(subject)[0][0]
            fromTmp1 = email.Header.decode_header(email.utils.parseaddr(msg.get('from'))[0])[0][0]
            fromTmp2 = email.utils.parseaddr(msg.get('from'))[1]
            words = subjectTmp + ' ' + fromTmp1 + ' ' + fromTmp2
            seg_list = jieba.lcut(words)
            res = ""
            for tmp in seg_list:
                if len(tmp) >= 2:
                    res += tmp + " "
            postingList.append(res)
            name.append(filename)
            if parent == posPath:
                classVec.append(1)
            elif parent == negPath:
                classVec.append(0)
    classVecFile = open("classVecFileWithSubject.txt", 'w')
    y = [str(line) + '\n' for line in classVec]
    for tmp in y:
        classVecFile.write(tmp)
    return postingList, classVec


def createWordBagWithSubject(path):
    totalWordBag = []
    subjectArray = []
    fromArray = []
    totalFileNum = 0
    for parent, dirname, filenames in os.walk(handleChinese(path)):
        for filename in filenames:
            totalFileNum += 1
            fp = open(os.path.join(parent, filename))
            msg = email.message_from_file(fp)
            subject = msg.get('subject')
            subjectTmp = email.Header.decode_header(subject)[0][0]
            fromTmp1 = email.Header.decode_header(email.utils.parseaddr(msg.get('from'))[0])[0][0]
            fromTmp2 = email.utils.parseaddr(msg.get('from'))[1]
            words = subjectTmp + ' ' + fromTmp1 + ' ' + fromTmp2
            # print filename
            # print words
            seg_list = set(jieba.lcut(words))
            for tmp in seg_list:
                if tmp not in totalWordBag and len(tmp) >= 2:
                    totalWordBag.append(tmp)
            subjectArray.append(subjectTmp)
            fromArray.append(fromTmp1)
    totalWordBagFile = open("totalWordBagSubject.txt", 'w')
    wordBag = [line + '\n' for line in totalWordBag]
    for x in wordBag:
        totalWordBagFile.write(x)
    totalWordBagFile.close()
    return totalWordBag, totalFileNum


if __name__ == '__main__':
    trainPath = 'C:\mail\\trainMail'
    totalWordBag, totalFilenNum = createWordBagWithSubject(trainPath)
    listPosts, listClasses = loadDataWithSubject(trainPath)
    print "loadDataSet finished"
    dataSet = []
    dataSetTfIdf = []
    stop_words = []
    for line in open("stop_words.txt"):
        line = line.strip('\n')
        stop_words.append(line.decode('utf-8'))
    x_train, x_test, y_train, y_test = train_test_split(listPosts, listClasses,
         test_size=0.3, stratify=listClasses, random_state=32)
    vectorizer = CountVectorizer(stop_words=stop_words)
    count = vectorizer.fit_transform(x_train)
    word = vectorizer.get_feature_names()  # 词的合集
    tword = tuple(word)
    svmClf = svm.SVC(kernel='linear')
    # berClf = BernoulliNB()
    svmClf.fit(count, y_train)
    # berClf.fit(count, y_train)
    ax = svmClf.coef_.transpose()
    # ax, ay = berClf.feature_log_prob_
    tax = tuple(ax)
    # tay = tuple(ay)
    axword = zip(tword, tax)
    # ayword = zip(tword, tay)

    res1 = sorted(axword, key=itemgetter(1), reverse=True)
    # res2 = sorted(ayword, key=itemgetter(1), reverse=True)
    # res1txt = open("berClfNeg.txt", 'w')
    res1txt = open("svmClfCoef.txt", 'w')
    # res2txt = open("berClfPos.txt", 'w')
    for x in res1:
        res1txt.write(x[0] + ':' + str(x[1]) + '\n')
    res1txt.close()
    '''for x in res2:
        res2txt.write(x[0] + ":" + str(x[1]) + '\n')
    res2txt.close()'''
