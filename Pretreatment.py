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
from sklearn import metrics, cross_validation
from sklearn.metrics import auc, roc_auc_score, roc_curve, precision_recall_curve
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
from sklearn.grid_search import GridSearchCV
from scipy import interp
import numpy as np
import matplotlib.pyplot as plt

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
stop_words = []
for line in open("stop_words.txt"):
    line = line.strip('\n')
    stop_words.append(line.decode('utf-8'))

posPath = handleChinese('C:\mail\\trainMail\pos')
negPath = handleChinese('C:\mail\\trainMail\\neg')


nbc_1 = Pipeline([
    ('vect', CountVectorizer(stop_words=stop_words)),
    ('select', SelectKBest(chi2, k=150)),
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
    ('select', SelectKBest(chi2)),
    ('clf', svm.SVC(kernel='linear')),
])


parameters = {
    'select__k': (10, 100, 300, 500, 1000, 2000, 3500),
}

# nbcs = [nbc_1, nbc_2, nbc_3, nbc_4, nbc_5, nbc_6]
nbcs = [nbc_6]


def evaluate_cross_validation(clf, X, y, K):
    cv = StratifiedKFold(y, K, shuffle=True, random_state=0)
    # cv = KFold(len(y), K, shuffle=True, random_state=0)
    grid_search = GridSearchCV(clf, parameters, n_jobs=-1, verbose=1, cv=cv)
    grid_search.fit(X, y)
    print "grid search", grid_search.best_score_
    scores = cross_val_score(clf, X, y, cv=cv)
    # print scores
    print ("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores),
                                                      sem(scores))
    return grid_search, np.mean(scores)


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


def plot_ROC_curve(classifier, X, y, pos_label=1, n_folds=10):
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    for i, (train, test) in enumerate(StratifiedKFold(y, n_folds=n_folds)):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area under the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1], pos_label=1)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')
    mean_tpr /= n_folds
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
                    label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def exactFeature(listPosts, lis):
    Xfit = CountVectorizer(stop_words=stop_words).fit(listPosts)
    X = Xfit.transform(listPosts)
    select = SelectKBest(chi2, k=500)
    select.fit_transform(X, listClasses)
    features = []
    for idx, val in enumerate(select.get_support()):
        if val == True:
            features.append(Xfit.get_feature_names()[idx])
    featureTxt = open("featureTxt.txt", 'w')
    wordBag = [line + '\n' for line in features]
    for x in wordBag:
        featureTxt.write(x)
    featureTxt.close()
    return features


if __name__ == '__main__':
    trainPath = 'C:\mail\\trainMail'
    testPath = 'C:\mail\\testMail'
    listPosts, listClasses = loadDataWithSubject(trainPath)
    print "loadDataSet finished"
    features = exactFeature(listPosts, listClasses)  # 后续只考虑这些特征，不是全部维度的
    Xfit = CountVectorizer(stop_words=stop_words).fit(listPosts)
    X = Xfit.transform(listPosts)
    y = np.array(listClasses)
    # plot_ROC_curve(svm.SVC(kernel='linear', probability=True), X, y, 10)
    # plot_ROC_curve(BernoulliNB(), X, y, 10)
    x_train, x_test, y_train, y_test = train_test_split(listPosts, listClasses,
        test_size=0.3, stratify=listClasses, random_state=32)
    for nbc in nbcs:
        grid_search, scores = evaluate_cross_validation(nbc, x_train, y_train, 10)
    predictListPosts, predictListClasses = loadDataWithSubject(testPath)
    nbc_1.fit(x_train, y_train)
    y_pred = nbc_1.predict(predictListPosts)
    count = 0
    for i in y_pred:
        if i == 1:
            count += 1
    print count
    preZip = zip(predictListPosts, y_pred)
    print "Predicted finished"
