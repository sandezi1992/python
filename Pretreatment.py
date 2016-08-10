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
