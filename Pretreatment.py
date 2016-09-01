<<<<<<< HEAD
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

reload(sys)
sys.setdefaultencoding('utf-8')
#  MultinomialNB
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

shoppingPath = handleChinese('C:\mail\python\邮件样例\购物')
registerPath = handleChinese('C:\mail\python\邮件样例\注册')
unsubscribePath = handleChinese('C:\mail\python\邮件样例\退订')
airPath = handleChinese('C:\mail\python\邮件样例\旅行')
advPath = handleChinese('C:\mail\python\邮件样例\广告邮件')
subscribePath = handleChinese('C:\mail\python\邮件样例\订阅')
financePath = handleChinese('C:\mail\python\邮件样例\财务')
posPath = handleChinese('C:\mail\python\\testMail\pos')
negPath = handleChinese('C:\mail\python\\testMail\\neg')


def loadDataSet(path):
    postingList = []
    name = []
    classVec = []
    for parent, dirname, filenames in os.walk(handleChinese(path)):
        for filename in filenames:
            fp = open(os.path.join(parent, filename))
            msg = email.message_from_file(fp)
            text = get_body(msg)
            soup = BeautifulSoup(text, 'html.parser')
            soupWordsTmp = [word for word in soup.stripped_strings]
            soupWords = "".join(soupWordsTmp)
            seg_list = jieba.lcut(soupWords)
            res = []
            for tmp in seg_list:
                if len(tmp) >= 2:
                    res.append(tmp)
            postingList.append(res)
            name.append(filename)
            if parent == shoppingPath:
                classVec.append(1)
            elif parent == registerPath:
                classVec.append(2)
            elif parent == unsubscribePath:
                classVec.append(3)
            elif parent == airPath:
                classVec.append(4)
            elif parent == advPath:
                classVec.append(5)
            elif parent == subscribePath:
                classVec.append(6)
            elif parent == financePath:
                classVec.append(7)
    classVecFile = open("classVecFile.txt", 'w')
    y = [str(line) + '\n' for line in classVec]
    for tmp in y:
        classVecFile.write(tmp)
    return postingList, classVec


# 创建词袋
def createWordBagWithText(path):
    totalWordBag = []
    totalFilenNum = 0
    for parent, dirname, filenames in os.walk(handleChinese(path)):
        for filename in filenames:
            totalFilenNum += 1
            fp = open(os.path.join(parent, filename))
            msg = email.message_from_file(fp)
            text = get_body(msg)
            soup = BeautifulSoup(text, 'html.parser')
            soupWordsTmp = [word for word in soup.stripped_strings]
            soupWords = "".join(soupWordsTmp)
            seg_list = jieba.lcut(soupWords)
            res = set(seg_list)
            for tmp in res:
                if tmp not in totalWordBag and len(tmp) >= 2 and tmp not in stop_words:
                    totalWordBag.append(tmp)
    totalWordBagFile = open("totalWordBag.txt", 'w')
    wordBag = [line + '\n' for line in totalWordBag]
    for x in wordBag:
        totalWordBagFile.write(x)
    totalWordBagFile.close()
    return totalWordBag, totalFilenNum


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


def setofWords2Vec(wordBag, article):
    returnVec = [0] * len(wordBag)
    for word in article:
        if word in wordBag:
            returnVec[wordBag.index(word)] = 1
        else:
            print "the word: %s is not in my Vocabulary!" % word
    return returnVec


if __name__ == '__main__':
    path = 'C:\mail\python\\testMail'
    prePath = 'C:\mail\python\pre'
    totalWordBag, totalFilenNum = createWordBagWithSubject(path)
    listPosts, listClasses = loadDataWithSubject(path)
    # totalWordBag, totalFilenNum = createWordBagWithText(path)
    # listPosts, listClasses = loadDataSet(path)
    print "loadDataSet finished"
    dataSet = []
    dataSetTfIdf = []
    # for index in range(len(listClasses)):
        # print listPosts[index]
        # print listClasses[index]
    # for postinDoc in listPosts:
        # dataSet.append(setofWords2Vec(totalWordBag, postinDoc))
    '''x = TfidfVectorizer().fit_transform(listPosts)
    x_new = SelectKBest(chi2, k=100).fit_transform(x, listClasses)
    print x_new'''
    x_train, x_test, y_train, y_test = train_test_split(listPosts, listClasses,
         test_size=0.3, stratify=listClasses, random_state=32)
    for nbc in nbcs:
        scores = evaluate_cross_validation(nbc, x_train, y_train, 10)
    '''nbc_1.fit(x_train, y_train)
    listPre, listCla = loadDataWithSubject(prePath)
    pren = nbc_1.predict(listPre)
    preNev = zip(listPre, pren)
    count = 0
    for i, j in preNev:
        if j == 0:
            count += 1
            print i
    print "count:", count
    print "Accuracy on testing set:"
    print nbc_1.score(x_test, y_test)
    y_predict = nbc_1.predict(x_test)
    preZip = zip(x_test, y_predict, y_test)
    for i, j, k in preZip:
        if j != k:
            print i
    print "Classification Report:"
    print metrics.classification_report(y_test, y_predict)
    print "Confusion Matrix:"
    print metrics.confusion_matrix(y_test, y_predict)'''
    '''tfidf = transformer.fit_transform(vectorizer.fit_transform(x_train))
    word = vectorizer.get_feature_names()
    weight = tfidf.toarray()
    print len(x_train)
    print len(y_train)
    print len(x_test)
    print len(y_test)
    clfMul = MultinomialNB()
    clfBer = BernoulliNB()
    clfSvm = svm.SVC()
    clfMul.fit(tfidf, y_train)
    # clfSvm.fit(x_train, y_train)
    # clfBer.fit(x_train, y_train)
    # predictedBer = clfBer.predict(x_test)
    # predictedSvm = clfSvm.predict(x_test)
    tfidfTest = transformer.fit_transform(vectorizer.fit_transform(x_test))
    predictedMul = clfMul.predict(tfidfTest)
    print "matrix Ber:"
    # print metrics.confusion_matrix(y_test, predictedBer)
    print "matrix Svm:"
    # print metrics.confusion_matrix(y_test, predictedSvm)
    print "matrix Mul:"
    print metrics.confusion_matrix(y_test, predictedMul)
    # precisionBer = metrics.precision_score(y_test, predictedBer)
    # recallBer = metrics.recall_score(y_test, predictedBer)
    # precisionSvm = metrics.precision_score(y_test, predictedSvm)
    # recallSvm = metrics.recall_score(y_test, predictedSvm)
    precisionMul = metrics.precision_score(y_test, predictedMul)
    recallMul = metrics.recall_score(y_test, predictedMul)
    # print 'BernoulliNB precison: %.3f' % precisionBer
    # print 'BernoulliNB recall: %.3f' % recallBer
    # print 'svm precison: %.3f' % precisionSvm
    # print 'svm recall: %.3f' % recallSvm
    # print 'Mul precison: %.3f' % precisionMul
    # print 'Mul recall: %.3f' % recallMul
    print "predicted finished"'''


'''def createVSM(totalWordBag, path):
    data = []
    data1 = ""
    data2 = ""
    for parent, dirname, filenames in os.walk(handleChinese(path)):
        for filename in filenames:
            fp = open(os.path.join(parent, filename))
            msg = email.message_from_file(fp)
            text = get_body(msg)
            soup = BeautifulSoup(text, 'html.parser')
            soupWordsTmp = [word for word in soup.stripped_strings]
            soupWords = "".join(soupWordsTmp)
            seg_list = jieba.lcut(soupWords)
            if parent == posPath:
                data1 = " ".join(seg_list)
            else:
                data2 = " ".join(seg_list)
    data1File = open("data1File.txt", 'w')
    data1File.write(data1)
    data2File = open("data2File.txt", 'w')
    data2File.write(data2)
    data.append(data1)
    data.append(data2)
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(data))
    word = vectorizer.get_feature_names()
    weight = tfidf.toarray()
    print data1
    for i in range(len(weight)):
        print "-----这里输出第", i, "类文本的词语的tf-idf权重-----"
        weight[i].sort()
        for j in range(len(word)):
            print word[j], weight[i][j]
        print '第', i, '类文本中总共单词长度', len(word)
    # df = DataFrame(data)  # index=indexArray, columns=totalWordBag)
    # dfT = df.T
    # df.to_csv('C:\mail\python\DataFrame.csv')





# dirs = os.listdir(handleChinese(path))
stopWordDic = open("stop_words.txt", 'rb')
stopWordList = stopWordDic.read().splitlines()
totalWordDic = open("totalWordBag.txt", 'rb')
totalWordBag = totalWordDic.read().splitlines()
'''
=======
#!/usr/bin/python
# coding:utf-8

import os
import jieba
import sys
import email
from email.Iterators import typed_subpart_iterator

reload(sys)
sys.setdefaultencoding('utf8')


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

index = 1
wordBag = {}
print "开始创建词袋"
dirs = os.listdir("finance")
print dirs
for dir in dirs:

    pathName = os.path.join("c:/mail/python/finance/", dir)
    # print pathName
    print '\n'
    fp = open(pathName)
    msg = email.message_from_file(fp)
    words = get_body(msg)
   # print index, words'''
>>>>>>> 85922265612230c1672ed1110a41558e888c188d
