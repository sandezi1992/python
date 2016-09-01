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
from sklearn import feature_extraction
from sklearn import metrics
from sklearn import svm
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.cross_validation import cross_val_score, KFold , StratifiedKFold
from sklearn.feature_selection import SelectKBest, chi2
from scipy.stats import sem
import numpy as np

reload(sys)
sys.setdefaultencoding('utf-8')
stop_words = []
for line in open("stop_words.txt"):
    line = line.strip('\n')
    stop_words.append(line.decode('utf-8'))


nbc_1 = Pipeline([
    ('vect', CountVectorizer(stop_words=stop_words)),
    ('clf', MultinomialNB()),
])

nbc_2 = Pipeline([
    ('vect', TfidfVectorizer(stop_words=stop_words)),
    ('clf', MultinomialNB()),
])
#  BernoulliNB
nbc_3 = Pipeline([
    ('vect', CountVectorizer(stop_words=stop_words)),
    ('clf', BernoulliNB()),
])


nbc_4 = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', BernoulliNB()),
])
# svm
nbc_5 = Pipeline([
    ('vect', CountVectorizer(stop_words=stop_words)),
    ('clf', svm.SVC(kernel='linear')),
])


nbc_6 = Pipeline([
    ('vect', TfidfVectorizer(stop_words=stop_words)),
    ('clf', svm.SVC(kernel='linear')),
])


nbcs = [nbc_1, nbc_2, nbc_3, nbc_4, nbc_5, nbc_6]


def evaluate_cross_validation(clf, X, y, K):
    cv = StratifiedKFold(y, K, shuffle=True, random_state=0)
    scores = cross_val_score(clf, X, y, cv=cv)
    # print scores
    print ("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores),
                                                      sem(scores))
    return np.mean(scores)


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

posPath = handleChinese('C:\mail\\testMail\pos')
negPath = handleChinese('C:\mail\\testMail\\neg')


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
                if len(tmp) >= 2 and tmp not in stop_words:
                    res += tmp + " "
            postingList.append(res)
            name.append(filename)
            if parent == posPath:
                classVec.append(1)
            elif parent == negPath:
                classVec.append(0)
            else:
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
                if tmp not in totalWordBag and len(tmp) >= 2 and tmp not in stop_words:
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
    path = 'C:\mail\\testMail'
    prePath = 'C:\mail\python\pre'
    totalWordBag, totalFilenNum = createWordBagWithSubject(path)
    listPosts, listClasses = loadDataWithSubject(path)
    print "loadDataSet finished"
    x_train, x_test, y_train, y_test = train_test_split(listPosts, listClasses,
         test_size=0.3, stratify=listClasses, random_state=32)
    for nbc in nbcs:
        scores = evaluate_cross_validation(nbc, x_train, y_train, 10)
    print "Predicted finished"