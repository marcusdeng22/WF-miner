import ctypes
import time

import win32api

from ctypes import windll, Structure, c_long, byref

class POINT(Structure):
    _fields_ = [("x", c_long), ("y", c_long)]


def getMousePos():
    pt = POINT()
    windll.user32.GetCursorPos(byref(pt))
    return pt.x, pt.y
##    return { "x": pt.x, "y": pt.y}

left = win32api.GetKeyState(0x01)

##SendInput = ctypes.windll.user32.SendInput
##
##def PressKey(hexKeyCode):
##    extra = ctypes.c_ulong(0)
##    ii_ = Input_I()
##    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
##    x = Input( ctypes.c_ulong(1), ii_ )
##    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
##
##def ReleaseKey(hexKeyCode):
##    extra = ctypes.c_ulong(0)
##    ii_ = Input_I()
##    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
##    x = Input( ctypes.c_ulong(1), ii_ )
##    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


while True:
##    a = win32api.GetKeyState(0x01)
##    if a != left:
##        left = a
##        print(a)
##        if a < 0:
##            print("pressed")
##        else:
##            print("released")

    #trigger a mouse release if down for 3 sec
    a = win32api.GetKeyState(0x01)
    if a != left:
        left = a
        print(a)
        if a < 0:
            print("pressed")
            triggerTime = time.time()
        else:
            print("released")
    elif a == left:
        if a < 0 and time.time() - triggerTime > 3:
            posX, posY = getMousePos()
            print(posX, posY)
##            win32api.mouse_event(0, posX, posY)
            ctypes.windll.user32.mouse_event(0x0004, posX, posY, 0, 0)
    time.sleep(0.001)
