# -*- coding: utf-8 -*-
import itchat, time
from itchat.content import *


def quqian(txt):
    slen=len(txt)
    return txt[:slen+1]


@itchat.msg_register([TEXT, MAP, CARD, NOTE, SHARING])
def text_reply(msg):
    ans2=quqian(msg['Text'])
    itchat.send('%s' % (ans2), msg['FromUserName'])


@itchat.msg_register(TEXT, isGroupChat=True)
def text_reply(msg):
    if msg['isAt']:
        itchat.send(u'@%s\u2005I received: %s' % (msg['ActualNickName'], msg['Content']), msg['FromUserName'])

itchat.auto_login(True)
itchat.run()
