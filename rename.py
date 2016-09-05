#!/usr/bin/python
# -*-coding:utf-8-*-
import os
import email
import chardet
import re
import email.header


def handleChinese(ustring):
    codeDetect = chardet.detect(ustring)["encoding"]
    ustring = unicode(ustring, codeDetect)
    ustring.encode("utf-8")
    return ustring

path = 'c:\mail\\rename'
for parent, dirname, filenames in os.walk(handleChinese(path)):
    for filename in filenames:
        fp = open(os.path.join(parent, filename))
        msg = email.message_from_file(fp)
        subject = msg.get('subject')
        subject1 = email.Header.decode_header(subject)[0][0]
        subjectTmp = re.sub(r"\.", "", subject1)
        res = re.sub(":", "", subjectTmp)
        res1 = re.sub("：", "", res)
        fp.close()
        tet = res1.decode('utf-8') + '.eml'
        if(os.path.exists(os.path.join(parent, tet))):
            continue
        else:
            os.rename(os.path.join(parent, filename), os.path.join(parent, tet))
