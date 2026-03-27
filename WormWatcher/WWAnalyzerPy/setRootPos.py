"""
    Set the position and size  of the root Tkinter figure, similar to setting the 
    'Position' of a Matlab figure/gui

    Adapted from:
    https://stackoverflow.com/questions/14910858/how-to-specify-where-a-tkinter-window-opens

    Leave x and y as NONE to assign the figure to the middle of the screen

"""


import tkinter as tk

def setRootPos(root,w,h,x = None, y = None):

    # get screen width and height
    ws = root.winfo_screenwidth() # width of the screen
    hs = root.winfo_screenheight() # height of the screen

    # calculate x and y coordinates for the Tk root window
    if x == None or y == None:
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)

    # set the dimensions of the screen 
    # and where it is placed
    root.geometry('%dx%d+%d+%d' % (w, h, x, y))