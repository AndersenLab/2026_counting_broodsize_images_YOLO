"""

Mainly by Chris Teng
Hongjing Sun & 
Christian de AbreuTh
Created in Nov 2019

"""
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import font
from PIL import Image, ImageTk, ImageChops, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import  FigureCanvasTkAgg
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from matplotlib.widgets import RectangleSelector
from ErrorNotifier import ErrorNotifier

from ReformatParentDirectory import verifyValidParentDirectory
from csvio import *
from verifyDirlist import verifyDirlist
import os
import numpy as np
import json
import glob
import random
import threading
import queue
import time
import ctypes
from datetime import datetime
import sys

def from_rgb(rgb):
    """translates an rgb tuple of int to a tkinter friendly color code
    """
    return "#%02x%02x%02x" % rgb   


#############################
#       Images Class        #
#############################
class images():
    def __init__(self, main, width, height, gui, enable_groups=False,):
        print("Created images")
        #Getting screen size and adjusting image size to that

        self.screen_width = width
        self.screen_height = height
        self.imagegui = gui
        self.canvas = tk.Canvas(main, width=int(self.screen_width/2.5), height=int(self.screen_height/2), 
                                bg="black", highlightthickness = 0, relief="ridge")
        self.canvas.grid(row=1, column=1, columnspan = 3, pady = 5)
        self.button_swap = ttk.Button(main,text="Swap frame", command = self.pushbutton_swap_callback)
        self.button_swap.grid(row=0, column=3, sticky = 'w')
        self.fr_val = 0                     # Initial frame
        self.fr_jump = 12
        self.bttn_clicks = 0
        self.total_frames = 0
        self.total_sessions = 0
        self.img_dict = {}
        self.curr_imgList = []
        self.row_col_dict = {}
        self.load_row_col_dict()

        # Zoomer type one
        self.zoomcycle = 0
        self.zimg_id = None
        self.zoom_index = 0 
        self.zoomed = False
        self.cm = plt.get_cmap("jet", lut=12)

        # Zoomer type two
        #self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        #self.canvas.bind("<B1-Motion>", self.on_move_press)
        #self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.rect = None
        self.start_x = None
        self.start_y = None
        self.mouse_x = None
        self.mouse_y = None

        #Roi group viewing/editing
        self.canvas.bind("<ButtonPress-1>", self.circle_clicked)
        
        # Current variables for sharing
        self.curr_session = ""
        self.curr_value = ""
        self.curr_group = ""
        self.circle_centers = []
        self.editing = False
        self.enable_groups = enable_groups

    # Called when double click in session box
    def ImportImage(self,event):
        global session_selected
        global activity_plot

        if(self.imagegui.applying):
            return

        widget = event.widget
        selection = widget.curselection()
        if(self.imagegui.box_session.itemcget(selection[0],"foreground")=="red"):
            tk.messagebox.showwarning("Error!","ROI Files or Images not Found! Please check directory!")
            return
        if(len(selection)==0):
            return
        elif(selection[0]>(len(self.img_dict)-1)):
            self.imagegui.box_session.selection_clear(selection[0])
            return
            

        self.editing = False

        print("Image Selected!")
        self.bttn_clicks = 0       # Renew the bottun-click number when new session chosen
        self.fr_val = 0

        session = widget.get(selection[0]).split(" ")                        # The session selected
        session.pop(2)
        days = session.pop(2)
        days = days.replace(")","")
        days = float(days)
        days = int(days)
        self.day = days
        value = " ".join(session)
        session_path = os.path.join(self.imagegui.plate_path, value)
        self.curr_value = value
        self.curr_session = session_path
        self.curr_frame = 0
        self.curr_index = 2
        
        self.curr_imgList = self.img_dict[value]
        imgpaths = verifyDirlist(session_path, False)[0]
        self.imagegui.tot_frame.set(str(len(imgpaths)-2))
        self.imagegui.input_frame.set(str(self.curr_frame))
        self.fr_jump = len(imgpaths)-2
        self.imagegui.input_frame.set(self.fr_val)
        self.imagegui.input_fr_jump.set(self.fr_jump)

        if(self.total_frames != 0):
            avg = self.total_frames//self.total_sessions
            if(int(self.fr_jump)<avg//2):
                tk.messagebox.showwarning("Low Frames!","A low number of frames on this session has been detected.")

        #Initialize important coordinates
        self.image_width = self.curr_imgList[0].width
        self.image_height = self.curr_imgList[1].height
        self.resize_factor_width = self.image_width/(self.screen_width/2.5)
        self.resize_factor_height = self.resize_factor_width
        self.load_roi_list()
        self.load_roi_dict()

        #Display resized image
        self.imagegui.image2.delete("all")                                    # Delete previous difference image when a new session chosen
        if(self.imagegui.first_loading):
            self.canvas["width"] = self.image_width/self.resize_factor_width
            self.canvas["height"] = self.image_height/self.resize_factor_height
            self.aspect_ratio = (self.image_width/self.resize_factor_width)/(self.image_height/self.resize_factor_height)
            self.imagegui.image2["width"] = self.image_width/self.resize_factor_width
            self.imagegui.image2["height"] = self.image_height/self.resize_factor_height
            self.imagegui.image3["width"] = self.image_width/(self.resize_factor_width/2)
            self.imagegui.first_loading=False


        # Display difference and default image
        diff = self.XYdiff() 
        if(self.zoomed):
            self.display_zoom(self.zoom_index)
        else:
            self.display_roi()                                   # Default difference image with 12 frames-difference
        self.imagegui.session_selected = True
        if(self.zoomed):
            self.imagegui.activity_plot.highlight_curves(self.zoom_index)
        else:
            self.imagegui.activity_plot.plot_data()
        self.imagegui.activity_plot.mark_day(days)
        return

    def load_row_col_dict(self): # For well labeling, loading each well's row and col number
        index = 0
        for c in range(6):
            for r in range(4):
                label = "Row: "+str(r)+" Col: "+str(c)
                self.row_col_dict[index] = label
                index+=1

    def load_roi_list(self): # Reads CSV and gets well ROI data
        session = self.curr_session+'\\'
        files = os.listdir(session)
        roi_file = ""
        for f in files:
            if f.endswith(".csv"):
                roi_file = f
    
        self.curr_roi_list = readArrayFromCsv(session+roi_file,"<ROI Coordinates>")

    def load_roi_dict(self): # Reformats roi data to find well circle coordinates
        self.roi_dict = {}
        self.resized_roi_dict = {}
        centers = []
        for i in range(len(self.curr_roi_list[0])):
            x = self.curr_roi_list[0][i]
            y = self.curr_roi_list[1][i]
            r = self.curr_roi_list[3][i]
            curr_region = [x-r,y-r,x+r,y+r]
            curr_resized_region = [(x-r)/self.resize_factor_width,(y-r)/self.resize_factor_width,
                                   (x+r)/self.resize_factor_height,(y+r)/self.resize_factor_height] # hardcoded resized circle size
            self.roi_dict[i] = curr_region
            self.resized_roi_dict[i] = curr_resized_region
            centers.append((x/self.resize_factor_width,y/self.resize_factor_height))

        self.circle_centers = centers

    def start_loading(self): # Starts loading thread
        self.img_dict.clear()
        for i in range(len(self.imagegui.slist)):
            self.imagegui.box_session.itemconfig(i,fg="gray")
              
    def load_images(self,imgList, roi_list):
        img_list = []

        f = Image.open(imgList[0])
        l = Image.open(imgList[-1])
        
        frame1 = Image.new("RGBA", f.size) # Creates two copies of first and last respectively
        frame1.paste(f)
        last_frame = Image.new("RGBA", l.size)
        last_frame.paste(l)

        img_list.append(f)# Loads 4 images per session, first and last frame and first and last with drawn circles
        img_list.append(l)
        img_list.append(f)
        img_list.append(l)

        img_list = self.draw_circles(img_list,roi_list)# Draw circles on the second and last frame

        return img_list
        
    def find_well(self,x,y):# Find well that is clicked given coords 
        index = -1
        for key in self.resized_roi_dict:
                if(x<self.resized_roi_dict[key][2] and x>self.resized_roi_dict[key][0]):
                    if(y<self.resized_roi_dict[key][3] and y>self.resized_roi_dict[key][1]):
                        index = key

        return index
    
    def circle_clicked(self,event): # Update image and template once a user clicks on a well

        if(not self.imagegui.session_selected):
            return

        x = event.x
        y = event.y
        index = self.find_well(x,y)
        if(index<0): # If user didnt click within well
            return
        if(self.editing):
            if(self.imagegui.groups_template[index]==self.curr_group):
                self.imagegui.groups_template[index] = "No group"
            else:
                self.imagegui.groups_template[index] = self.curr_group
            self.display_roi(True)
        else:
            self.zoomer(index)


    def draw_circles(self,img_list,roi_list,fill=False):
        if(len(img_list)==0):
            return
        img1 = Image.new("RGBA", img_list[0].size)
        img1.paste(img_list[0])
        img2 = Image.new("RGBA", img_list[1].size)
        img2.paste(img_list[1])
        draw1 = ImageDraw.Draw(img1)
        draw2 = ImageDraw.Draw(img2)

        for i in range(len(self.imagegui.groups_template)):
            if(self.imagegui.groups_template[i]==""):
                color = (255,255,255)
            else:
                color = self.imagegui.color_dict[self.imagegui.groups_template[i]]
            x = roi_list[0][i]
            y = roi_list[1][i]
            r = roi_list[3][i]
            circle = (x-r,y-r,x+r,y+r)
            if(fill):
                draw1.ellipse(circle,outline=color,fill=color,width = 15)
                draw2.ellipse(circle,outline=color,fill=color,width = 15)
            else:
                draw1.ellipse(circle,outline=color,fill=None,width = 15)
                draw2.ellipse(circle,outline=color,fill=None,width = 15)
        
        img_list[2] = img1
        img_list[3] = img2
        return img_list
        

    def load_group(self,group):
        self.curr_group = group

    def display_zoom(self,well_num):
        if(self.editing):
            return
        coords = self.roi_dict[well_num]
        height = coords[3]-coords[1]
        width = coords[2]-coords[0]
        adj_width = height*self.aspect_ratio
        diff = adj_width-width
        adj_coords = [coords[0]-diff/2,coords[2]+diff/2]
        
        cropped = self.curr_imgList[self.curr_index].crop((adj_coords[0],coords[1],adj_coords[1],coords[3]))
        cropped = cropped.resize((int(self.image_width/self.resize_factor_width), 
                                  int(self.image_height/self.resize_factor_height)),Image.ANTIALIAS)
        draw = ImageDraw.Draw(cropped)
        draw.rectangle((0,0,125,50),fill="black")
        font = ImageFont.truetype("arial.ttf",30)
        draw.text((3,12),"Well "+str(well_num+1),font=font)

        cropped_diff = self.difference.crop((adj_coords[0],coords[1],adj_coords[1],coords[3]))
        cropped_diff = cropped_diff.resize((int(self.image_width/self.resize_factor_width), 
                                            int(self.image_height/self.resize_factor_height)),Image.ANTIALIAS)
        self.display = ImageTk.PhotoImage(cropped)
        self.image_on_canvas = self.canvas.create_image(0, 0, image=self.display, anchor= 'nw')
        self.plot_image(cropped_diff)

    def display_roi(self,fill=False): # Draws circles and displays image
        if(len(self.curr_imgList)==0):
            return
        self.canvas.delete("all")
        if(self.zoomed):
            self.plot_image(self.resized_diff)
            self.zoomed = False
            self.imagegui.activity_plot.highlight_curves(-1)
            self.imagegui.activity_plot.mark_day(self.day)
        self.draw_circles(self.curr_imgList,self.curr_roi_list,fill)
        display = self.curr_imgList[self.curr_index].resize((int(self.image_width/self.resize_factor_width), 
                                                             int(self.image_height/self.resize_factor_height)),Image.ANTIALIAS)
        self.img = ImageTk.PhotoImage(display)
        self.image_on_canvas = self.canvas.create_image(0, 0, image=self.img, anchor= 'nw')

    def pushbutton_swap_callback(self):

        if(not self.imagegui.session_selected):
            return

        if(self.curr_frame!=self.fr_jump):
            self.curr_frame = self.fr_jump
            self.curr_index = 3                      # HARDCODED SUBJECT TO CHANGE
        else:
            self.curr_frame = self.fr_val
            self.curr_index = 2 
            
        self.imagegui.input_frame.set(str(self.curr_frame))
        if(self.zoomed):
            self.display_zoom(self.zoom_index)
        else:
            self.display_roi()
        
        return

    def plot_image(self, im):

        np_im = np.array(im)
        w = im.width
        h = im.height

        figure = plt.Figure(figsize=(w/self.imagegui.dpi,h/self.imagegui.dpi),dpi=self.imagegui.dpi)
        colormap = plt.Figure(figsize=(w/self.imagegui.dpi,h/8/self.imagegui.dpi),dpi=self.imagegui.dpi)
        colormap.set_facecolor("lightgrey")
        display = FigureCanvasTkAgg(figure,self.imagegui.image2)
        display.get_tk_widget().grid(row=0, column=0, padx = 0, pady = 0)
        colormap_display = FigureCanvasTkAgg(colormap,self.imagegui.colormap)
        colormap_display.get_tk_widget().grid(row=1, column=0, padx = 0, pady = 0)
        a = figure.add_subplot()
        a.set_axis_off()
        figure.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        a.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        b = colormap.add_subplot()
        b.set_axis_off()


        #divider = make_axes_locatable(a)
        #cax = divider.append_axes("right", size="5%", pad=0.10)

        im_plot = a.imshow(np_im,cmap=plt.get_cmap("hot",lut=8))
        colormap.colorbar(im_plot,orientation="horizontal",fraction=1)#may need to calculate dynamically in the future
        display.draw()

    def XYdiff(self):
        if self.bttn_clicks == 1 or self.bttn_clicks == 0: # Generate the difference image only for the first click
            first = self.curr_imgList[0]
            last = self.curr_imgList[1]

            self.difference = ImageChops.difference(first, last)
            self.resized_diff = self.difference.resize((int(self.image_width/self.resize_factor_width), 
                                                        int(self.image_height/self.resize_factor_height)),Image.ANTIALIAS)
            
            self.plot_image(self.resized_diff)
            
        
        #self.difference = Image.fromarray(self.np_im)
        
        #self.diff = self.difference.resize((int(self.image_width/6), int(self.image_height/6)),Image.ANTIALIAS)

        #self.imgDiff = ImageTk.PhotoImage(self.diff)
        #self.imgDiff_on_canvas = diff_frame.create_image(0, 0, image=self.imgDiff, anchor= 'nw')
        return 

    def zoomer(self,index): # Find well and display zoomed well
        global session_selected, activity_plot

        self.zoom_index = index

        self.canvas.delete("all")
        self.imagegui.image2.delete("all")
        if (not self.zoomed):
            self.display_zoom(self.zoom_index)
            self.zoomed = True
            self.imagegui.activity_plot.highlight_curves(self.zoom_index)
            self.imagegui.activity_plot.mark_day(self.day)
        elif (self.zoomed):
            self.display_roi()
            self.plot_image(self.resized_diff)
            self.zoomed = False
            self.imagegui.activity_plot.highlight_curves(-1)
            self.imagegui.activity_plot.mark_day(self.day)

    def crop(self,event):
        if self.zimg_id: self.canvas.delete(self.zimg_id)
        if (self.zoomcycle) != 0:
            x,y = event.x, event.y
            if self.zoomcycle == 1:
                tmp = self.image_open.crop((x-55,y-40,x+55,y+40))
                tmp2 = self.imgDiff_op.crop((x-55,y-40,x+55,y+40))
            elif self.zoomcycle == 2:
                tmp = self.image_open.crop((x-40,y-30,x+40,y+30))
            elif self.zoomcycle == 3:
                tmp = self.image_open.crop((x-15,y-10,x+15,y+10))
            elif self.zoomcycle == 4:
                tmp = self.image_open.crop((x-6,y-4,x+6,y+4))
            size = 300,200
            self.zimg = ImageTk.PhotoImage(tmp.resize(size))
            self.zimg2 = ImageTk.PhotoImage(tmp2.resize(size))
            self.zimg_id = self.canvas.create_image(event.x,event.y,image=self.zimg)
            self.zimg_id2 = image2.create_image(event.x,event.y,image=self.zimg2)

    def on_button_press(self, event):
        # save mouse drag start position
        self.start_x = event.x
        self.start_y = event.y

        # create rectangle 
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x+0.1, self.start_y+0.1, fill="")
        
    def on_move_press(self, event):
        None
        #curX, curY = (event.x, event.y)

        # expand rectangle as dragging the mouse
        #self.canvas.coords(self.rect, self.start_x, self.start_y, curX, self.start_y+0.718*(curX-self.start_x))
        #self.rect_img = self.image_open.crop((self.start_x, self.start_y, curX, self.start_y+0.718*(curX-self.start_x)))
        #self.rect_Diffimg = self.imgDiff_op.crop((self.start_x, self.start_y, curX, self.start_y+0.718*(curX-self.start_x)))

    def on_button_release(self, event):
        None
        #screen_width, screen_height= pyautogui.size()
        #self.canvas.delete(self.rect)
        #size = int(screen_width/3), int(screen_height/3)
        #self.crop_img = ImageTk.PhotoImage(self.rect_img.resize(size))
        #self.crop_Diffimg = ImageTk.PhotoImage(self.rect_Diffimg.resize(size))
        #self.show_img = self.canvas.itemconfig(self.image_on_canvas, image = self.crop_img)
        #self.show_Diffimg = image2.itemconfig(self.imgDiff_on_canvas, image = self.crop_Diffimg)



def toggle_selector(self, event):

    #print(' Key pressed.')
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print(' RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print(' RectangleSelector activated.')
        toggle_selector.RS.set_active(True)
    return

#############################
#       Activity Class      #
#############################
class activities():
    def __init__(self, frame, width, height, dpi, gui):
        
        #Creating canvas for activity curve display
        self.screen_width = width
        self.screen_height = height
        self.dpi = dpi
        self.imagegui = gui
        self.fig = plt.Figure(figsize=(int(self.screen_width/dpi),int(self.screen_height/self.dpi)))
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().pack(side="left",fill="both")
        self.bttn_clicks = 69
        self.ax = self.fig.subplots()
        self.header = ""
        self.good_plot = False
        self.activity_data = None # Stores the plot specific data. 
        self.day = 0

    def import_activity(self,plate_name,header):
        #Flag plate, helps keep track of things
        print("Plate Selected: " + plate_name)
        
        #Entering activity directory
        parent_directory = verifyDirlist(self.imagegui.pparent.get(), 1, "_Analysis", "")[0]
        self.imagegui.activity_folder = verifyDirlist(parent_directory[0], 1, str(plate_name), "")[0]
        self.imagegui.activity_file = verifyDirlist(self.imagegui.activity_folder[0], 0, "csv","")[0][0]

        #Reading Activity File
        try:
            self.activity_data = readArrayFromCsv(self.imagegui.activity_file, header)
            days = readArrayFromCsv(self.imagegui.activity_file, '<Day>')

            #Plotting onto activity canvas number 1
            #activity_plot = self.fig.add_subplot(111)
            self.ax.set_title(header)
            self.ax.set_xlabel('Time (days)')
            self.ax.set_ylabel('Activity (a.u.)')
            self.header = header

            #Time intervals:
            self.time_stamp = [elm[0] for elm in days]
            for roi in range(np.shape(self.activity_data)[1]):
                #ROI Activity
                roi_activity = self.activity_data[:,roi]
                #Converting Everything to floats
                x = list(map(float, self.time_stamp))
                y = list(map(float, roi_activity))
                self.ax.plot(x, y,'gray')

            self.canvas.draw()
            self.good_plot = True
            return True
        except ValueError:
            tk.messagebox.showwarning("CSV not Found!",header+" header not found in CSV!")
            self.good_plot = False
            self.ax.clear()
            self.canvas.draw()
            return False

    def select_activity(self):
        #Draw rectangle and plot new curve in bold (in GUI would be below the initial plot)
        toggle_selector.RS = RectangleSelector(self.ax, self.line_select_callback,
                                                drawtype='box', useblit=True,
                                                button=[1, 3],  # don't use middle button
                                                spancoords='pixels', 
                                                interactive = True)

        self.canvas.mpl_connect('key_press_event', toggle_selector)
        #self.canvas.mpl_connect('button_release_event', find_curve)
        


        return

    def mark_day(self, day):
        if(not self.good_plot):
            return

        self.day = day
        self.ax.axvline(x=day,color='blue',linestyle=':')
        self.canvas.draw()
        return

    def plot_data(self):
        global time_stamp

        if(not self.good_plot):
            return

        self.ax.clear()
        self.ax.set_title(self.header)
        self.ax.set_xlabel('Time (days)')
        self.ax.set_ylabel('Activity (a.u.)')
        for roi in range(np.shape(self.activity_data)[1]):
            #ROI Activity
            roi_activity = self.imagegui.activity[:,roi]
            #Converting Everything to floats
            x = list(map(float, self.time_stamp))
            y = list(map(float, roi_activity))
            self.ax.plot(x,y,"grey")

        self.canvas.draw()

        return

    def highlight_curves(self,index):

        if(not self.good_plot):
            return

        self.ax.clear()
        self.ax.set_title(self.header)
        self.ax.set_xlabel('Time (days)')
        self.ax.set_ylabel('Activity (a.u.)')
        for roi in range(np.shape(self.activity_data)[1]):
            #ROI Activity
            roi_activity = self.imagegui.activity[:,roi]
            #Converting Everything to floats
            x = list(map(float, self.time_stamp))
            y = list(map(float, roi_activity))
            if(index==-1):
                self.ax.plot(x,y,"grey")
            else:
                if(index==roi):
                    self.ax.plot(x,y,'r',linewidth=5)
                else:
                    self.ax.plot(x,y,'grey')

        self.canvas.draw()

        return

    """ Functions to manage highlighting wells """
    def line_select_callback(self, eclick, erelease):

        print(
            "Entered the highlight callback")
        'eclick and erelease are the press and release events'
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        #print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
        self.imagegui.coords = [x1, x2, y1, y2]
        #print(coords)
        self.find_curve()

    def find_curve(self):
        #Activity curves, click coordinates, and timestamps
        #Finding midpoint of rectangular selection

        mid_x = (self.imagegui.coords[0]+self.imagegui.coords[1])/2
        mid_y = (self.imagegui.coords[2]+self.imagegui.coords[3])/2
    
        #Finding time closest to mid_x
        time_idx = min(range(len(self.time_stamp)), key=lambda i: abs(self.time_stamp[i]-mid_x))
    
        #Identifying which curve has closest activity value at that exact time
        min_dif = [] 
        for roi in range(np.shape(self.imagegui.activity)[1]):
            roi_activity = self.imagegui.activity[:,roi]
            #Converting everything to floats
            y = list(map(float, roi_activity))
            min_dif.append(abs(y[time_idx] - mid_y)) 
        selected_curve_idx = np.argmin(min_dif)
        self.imagegui.image1.zoom_index = int(selected_curve_idx)
        self.imagegui.image1.zoomed = True
        self.imagegui.image1.display_zoom(int(selected_curve_idx))
        self.imagegui.activity_plot.highlight_curves(int(selected_curve_idx))
        self.imagegui.activity_plot.mark_day(self.imagegui.activity_plot.day)

        #Plotting that specific curve onto second activity plot pane: TODO (edit this portion)
        #plt.figure(2)
        #plt.plot(time, curves[selected_curve+1], lw=5, c='k', alpha = 0.9)
        return


# ImageGUI2 Class
class ImageGUI2:
    activity_frame = None
    activity_file = None
    activity = None
    coords = None
    times = None
    groups_template = None
    groups = None
    slist = None
    color_dict = None
    json_open = None
    json_path = None
    colors = None
    applying = None
    past_selection = None
    plates_loaded = None
    sessions_loaded = None
    session_selected = None
    activity_plot = None
    first_loading = None
    colors = [(191, 128, 0), (139, 255, 61), (61, 113, 255), (255, 0, 204), (127, 61, 61), 
          (255, 216, 61), (31, 128, 44), (70, 61, 128), (128, 0, 102), (127, 34, 0), (127, 119, 0), 
          (122, 255, 193), (51, 0, 128), (255, 61, 139), (255, 158, 122), (238, 255, 0), (0, 179, 191), 
          (170, 0, 255), (255, 61, 87), (127, 88, 61), (158, 191, 92), (0, 128, 191), (171, 92, 191),
          (127, 10, 10), (33, 23, 2), (15, 252, 15), (40, 150, 247), (236, 190, 247), (64, 8, 8), 
          (252, 202, 3), (17, 64, 17), (184, 215, 242), (236, 5, 252), (214, 99, 99), 
          (125, 100, 3), (140, 255, 140), (11, 53, 115), (189, 81, 182), (76, 65, 19), 
          (32, 168, 105), (88, 119, 166), (82, 23, 70), (252, 86, 25), (255, 238, 168), 
          (21, 33, 29), (41, 46, 61), (31, 2, 21), (252, 188, 164), (169, 179, 36), 
          (51, 255, 214), (0, 15, 115), (252, 23, 145), (102, 94, 91), (205, 255, 3), 
          (174, 252, 237), (11, 18, 66), (127, 54, 73), (107, 61, 30), (154, 168, 125), 
          (91, 128, 120), (103, 103, 214), (227, 159, 177), (184, 106, 17), (72, 112, 26), 
          (27, 107, 128), (110, 21, 189), (252, 8, 40), (176, 140, 99), 
          (104, 219, 247), (109, 88, 128)]
    color_dict = {"Censor":(255,0,0),"No group":(0,0,0)}
    groups = []
    json_open = False
    plates_loaded = False
    sessions_loaded = False
    session_selected = False
    signal_stop = False
    first_loaded = False

    def __init__(self,path):
        self.root = tk.Tk()

        # Get the screen width and height
        user32 = ctypes.windll.user32
        self.screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        self.dpi = 96 #user32.GetDpiForWindow(self.root.winfo_id()) crashes on win 7
        self.screen_width = int(self.screensize[0]*0.95)
        self.screen_height = int(self.screensize[1]*0.9)
        self.gui_size = str(self.screen_width) + "x" + str(self.screen_height) + "+20+10"
        self.root.geometry(self.gui_size)
        #print(gui_size)
        self.root.title("ImageGUI")
        self.root.resizable(False, False)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.bgcolor = (241,241,245)


        """ Setting Style """
        self.s=ttk.Style()
        self.s.configure('M.TFrame', borderwidth=1, background='lightgrey')
        self.s.configure('A.TFrame', borderwidth=1, background='white')

        """ Create all of the main containers """
        self.left_frame = ttk.Frame(self.root, height=self.screen_height, width = int(self.screen_width/4), style = 'M.TFrame')
        self.right_frame = ttk.Frame(self.root, height = self.screen_height)
        self.hybrid_frame  = ttk.Frame(self.right_frame, style="M.TFrame")#, height = screen_height, width = screen_width - 500)   # images and differences
        self.activity_tabs = ttk.Notebook(self.right_frame)
        self.activity_frame1 = ttk.Frame(self.activity_tabs, style="M.TFrame")   # activity graphs
        self.activity_frame2 = ttk.Frame(self.activity_tabs, style="M.TFrame")

        """ Layout all of the main containers """
        self.left_frame.pack(side = 'left', fill='both')
        self.left_frame.columnconfigure(0,weight=1)
        #left_frame.grid_propagate(0)
        self.right_frame.pack(side = 'left', fill = 'both', expand = 'true')
        self.right_frame.columnconfigure(1,weight=4)
        self.hybrid_frame.pack(side = 'top', fill = 'both')
        self.activity_tabs.pack(side = 'top',fill='both')
        self.activity_tabs.add(self.activity_frame1,text="Stimulated Activity")
        self.activity_tabs.add(self.activity_frame2, text="Cumulative Activity")

        """ Create widgets for left frame """
        #Directory
        self.find_dir = ttk.Button(self.left_frame, text = "Set Parent Folder", command=self.pushbutton_path_callback,style='my.TButton')
        self.pparent = tk.StringVar(self.root)
        self.pparent.set(path) #TODO: change this to the previous parent folder
        self.L_label2 = ttk.Label(self.left_frame, textvariable = self.pparent, font='Helvetica 12',wraplength=800, justify=tk.LEFT, background='lightgrey') # Fullpathname of parent folder

        #Folders 
        self.L_label_emp0 = ttk.Label(self.left_frame, text = "   ", background='lightgrey') 
        self.L_label_emp1 = ttk.Label(self.left_frame, text = "   ", background='lightgrey') 
        self.s.configure('my.TButton', font=('Helvetica', 12))

        #Button to change folder 
        self.pushbutton_path = ttk.Button(self.left_frame,text="Change path",command = self.pushbutton_path_callback, style = 'my.TButton')

        #Text input to add group 
        self.input_text = tk.StringVar()   
        self.group_entry = ttk.Entry(self.left_frame, textvariable = self.input_text, justify = tk.CENTER, width = 18) 

        #Button to change/delete/add group
        self.s.configure('C.TButton', font=('Helvetica', 12))
        self.pushbutton_add = ttk.Button(self.left_frame,text="Add group",command = self.pushbutton_add_callback, style = 'C.TButton')
        self.s.configure('A.TButton', font=('Helvetica', 12))
        self.s.configure('B.TButton', font=('Helvetica', 12))
        self.s.configure('D.TButton', font=('Helvetica', 12))
        self.pushbutton_load_template = ttk.Button(self.left_frame,text="Load Template",command = self.pushbutton_load_template_callback, style = 'B.TButton')
        self.pushbutton_apply_template = ttk.Button(self.left_frame,text="Batch Apply",command = self.pushbutton_batch_apply_callback, style = 'D.TButton')
        self.pushbutton_save_as_template = ttk.Button(self.left_frame,text="Save as Template", command = self.pushbutton_save_as_template_callback, style = 'D.TButton')
        self.pushbutton_done_editing = ttk.Button(self.left_frame,text="Done Editing", command = self.pushbutton_done_editing_callback, style = 'D.TButton', state="disabled")

        #List boxes
        self.applying = False
        h = int(self.screen_height/120)
        self.box_plate = tk.Listbox(self.left_frame,exportselection=0, selectmode=tk.SINGLE, width=36, height=h) #Plate Box
        self.plate_label = ttk.Label(self.left_frame, text = "Plate", font='Helvetica 12 bold', background='lightgrey')
        self.scroll_p = ttk.Scrollbar(self.left_frame, orient=tk.VERTICAL) #Plate scrollbar
        self.box_plate.config(yscrollcommand=self.scroll_p.set)
        self.scroll_p.configure(command=self.box_plate.yview)
        self.box_plate.bind("<<ListboxSelect>>", self.select_plate)
        self.box_session = tk.Listbox(self.left_frame,exportselection=0, width=36,height=h) #Session Box
        self.session_label = ttk.Label(self.left_frame, text = "Session", font='Helvetica 12 bold', background='lightgrey')
        self.scroll_s = ttk.Scrollbar(self.left_frame, orient=tk.VERTICAL) #Session Scrollbar
        self.box_session.config(yscrollcommand=self.scroll_s.set)
        self.scroll_s.configure(command=self.box_session.yview)
        self.bolded = font.Font(weight='bold', size=8)
        self.box_group = tk.Listbox(self.left_frame, selectmode='single',exportselection=0, width=36, height=h) #Group Box
        self.group_label = ttk.Label(self.left_frame, text = "Group", font='Helvetica 12 bold', background='lightgrey')
        self.scroll_g = ttk.Scrollbar(self.left_frame, orient=tk.VERTICAL) #Scroll Group
        self.box_group.config(yscrollcommand=self.scroll_g.set, font=self.bolded)
        self.scroll_g.configure(command=self.box_group.yview)
        self.box_group.bind("<<ListboxSelect>>", self.editGroups)

        """ Layout Widgets for left frame """
        #Directory
        self.find_dir.grid(row = 0, column = 1, columnspan=2, sticky = 'n')
        self.L_label2.grid(row = 1, column = 1, columnspan=2, sticky = 'n')

        #Folders
        self.L_label_emp0.grid(row = 0, rowspan = 13, column = 0, sticky = 'w')
        self.L_label_emp1.grid(row = 0, rowspan = 13, column = 3, sticky = 'e')

        #Button to change folder
        #pushbutton_path.grid(row = 2, column = 1, columnspan = 5, sticky = 'n')

        #Text input to add group
        self.group_entry.grid(row = 8, column = 1, padx = 5, pady=10)

        #Button to change/delete/add group
        self.pushbutton_add.grid(row = 8, column = 2)
        self.pushbutton_load_template.grid(row = 9, column =1, columnspan=2, pady=3)
        self.pushbutton_apply_template.grid(row = 10, column = 1, columnspan=2, pady=3)
        self.pushbutton_save_as_template.grid(row = 11, column = 1, columnspan=2, pady=3)
        self.pushbutton_done_editing.grid(row=12,column=1, columnspan=2, pady=3)

        #List boxes
        self.plate_label.grid(row = 2, column = 1, columnspan=2)
        self.scroll_p.grid(row=3,column=2,sticky='nse')
        self.box_plate.grid(row = 3, column = 1, columnspan = 2)

        self.session_label.grid(row = 4, column = 1, columnspan=2)
        self.scroll_s.grid(row=5,column=2,sticky='nse')
        self.box_session.grid(row = 5, column = 1, columnspan = 2)

        self.group_label.grid(row = 6, column = 1, columnspan=2)
        self.scroll_g.grid(row=7, column=2,sticky='nse')
        self.box_group.grid(row = 7, column = 1, columnspan = 2)
        self.box_group.insert(0, "Censor")
        self.box_group.insert(1,"No group")
        self.box_group.itemconfig(0,foreground=from_rgb(self.color_dict["Censor"]))
        self.box_group.itemconfig(1,foreground=from_rgb(self.color_dict["No group"]))


        """ Pane 1: Hybrid View Widgets """
        #Labels and Input Frames
        self.label1 = ttk.Label(self.hybrid_frame, text = "Frame ", font = "Helvetica 12", justify=tk.CENTER, background='lightgrey')
        self.input_frame = tk.StringVar(value=0)
        self.curr_frm = ttk.Label(self.hybrid_frame, textvariable = self.input_frame, justify = tk.CENTER, font = "Helvetica 12", background='lightgrey') 
        self.label1_1 = ttk.Label(self.hybrid_frame, text = "of ", font = "Helvetica 12", justify=tk.LEFT, background='lightgrey')
        self.tot_frame = tk.StringVar(value='#')
        self.label1_2 = ttk.Label(self.hybrid_frame, textvariable = self.tot_frame, font='Helvetica 12', justify=tk.LEFT, background='lightgrey')
        self.label2 = ttk.Label(self.hybrid_frame, text = "Frame Y - Frame X", font = "Helvetica 12", justify=tk.CENTER, background='lightgrey')

        #Labels TODO check this
        self.P1_label_emp0 = ttk.Label(self.hybrid_frame, text = "    ", width = 12, background='lightgrey') 
        self.P1_label_emp0 = ttk.Label(self.hybrid_frame, text = "    ", width = 12, background='lightgrey') 

        #Frames Selection
        self.label3 = ttk.Label(self.hybrid_frame, text = "Jump = ", font = "Helvetica 12", justify=tk.RIGHT, background='lightgrey')
        self.input_fr_jump = tk.StringVar(value=12)   
        self.fr_entry = ttk.Label(self.hybrid_frame, textvariable = self.input_fr_jump, font = "Helvetica 12", justify=tk.CENTER, background='lightgrey') 
        self.label4 = ttk.Label(self.hybrid_frame, text = "Frames (max xx)", font = "Helvetica 12", justify=tk.LEFT, background='lightgrey')
        xpad=5
        ypad =5
        self.image3 = tk.Canvas(self.activity_frame1, width= int(self.screen_width/1.5), height=int(self.screen_height/3.8),
                                bg="lightgrey", highlightthickness = 0, relief="ridge")
        self.image4 = tk.Canvas(self.activity_frame2, width= int(self.screen_width/1.5), height=int(self.screen_height/3.8),
                                bg="lightgrey", highlightthickness = 0, relief="ridge")


        """ Pane 1: Hybrid View  Layout """
        #Labels and Input Frames
        self.label1.grid(row=0, column=1, pady = 10, sticky='e')
        self.curr_frm.grid(row = 0, column = 2, sticky='w', padx = 25, pady = 10)
        self.label1_1.grid(row=0, column=2, pady = 10)
        self.label1_2.grid(row=0, column=2, sticky='e', padx = 25, pady = 10)
        self.label2.grid(row=0, column=4, columnspan = 3, pady = 10, padx = 10)

        #Labels TODO check this
        self.P1_label_emp0.grid(row = 0, column = 0, rowspan = 6, sticky = 'w')
        self.P1_label_emp0.grid(row = 0, column = 7, rowspan = 6, sticky = 'e')

        ## Canvas where image displayed
        self.image1 = images(self.hybrid_frame, self.screen_width, self.screen_height,self,True)
        self.box_session.bind("<<ListboxSelect>>", self.image1.ImportImage) # import the image when double clicking one session
        self.image2 = tk.Canvas(self.hybrid_frame, width=int(self.screen_width/2.5), height=int(self.screen_height/2), 
                                bg="black", highlightthickness = 0, relief="ridge")
        xpad = 5
        ypad = 5
        self.image2.grid(row=1, column=4, columnspan = 3, padx = xpad, pady = ypad)
        self.colormap = tk.Canvas(self.hybrid_frame, width=int(self.screen_width/2.5), height=int(self.screen_height/12), 
                                bg="lightgrey", highlightthickness = 0, relief="ridge")
        self.colormap.grid(row=2, column=4, columnspan = 3, padx = xpad, pady = ypad)

        # Frames selection
        self.label3.grid(row = 2, column = 1, sticky = "ne")
        self.fr_entry.grid(row = 2, column = 2, sticky = 'nw')
        self.label4.grid(row = 2, column = 2, sticky = "n")
        #self.image3.grid(row=0, column=1, columnspan = 6, padx = 60, pady = ypad)
        #self.image4.grid(row=0, column=1, columnspan = 6, padx = 60, pady = ypad)


        """ Activity Tabs: Activity View Widgets """
        # Activity Curves
        self.activity_plot = activities(self.activity_frame1,self.screen_width, self.screen_height, self.dpi, self)
        self.activity1 = activities(self.activity_frame2,self.screen_width, self.screen_height, self.dpi, self)
       
        # load path
        self.load_path()
        
        self.image_queue = queue.Queue()
        self.update_window()
        self.root.protocol("WM_DELETE_WINDOW",self.close_window)
        self.root.mainloop()

    """ callback function """
    # select the parent folder
    def pushbutton_path_callback(self,*args):
        self.plist = list()   # the plates' name

        self.pushbutton_path.configure(text = "validating...")


        # ask the user to select a folder
        tempdir = filedialog.askdirectory(parent=self.root, initialdir=self.pparent.get(), title='please select the parent directory')
        if len(tempdir) > 0:
            print("you chose %s",tempdir)

        # enforce a valid parent directory
        self.pparent.set(tempdir)
        self.s.configure('path.tbutton',background = "gray")

        self.load_path()
    
        return

    def load_path(self):
        self.box_plate.delete(0,"end")
        self.pList = verifyDirlist(self.pparent.get(),True,"","_Analysis")[1]
        self.pList.reverse()

        # Update the list in the box
        self.bad_plates = []
        self.missing_files = []
        index = 0
        for p in self.pList:
            has_curve = False
            self.box_plate.insert(index, p)
            try:
                path = os.listdir(self.pparent.get()+"\\_Analysis\\"+p)
                for f in path:
                    if f.endswith("csv"):
                        if "ActivityCurves" in f:
                            has_curve = True

                if(not has_curve):
                    self.box_plate.itemconfig(index,foreground="red")
                    self.bad_plates.append(p)
            except FileNotFoundError:
                self.missing_files.append(p)
                self.box_plate.itemconfig(index,foreground="red")
    
        # Display potential errors or problems
        bad_plate_list = ""
        if(len(self.bad_plates)>10):
            for i in range(10):
                bad_plate_list += self.bad_plates[i]+", "

            bad_plate_list+="and more "
        else:
            for pl in self.bad_plates:
                bad_plate_list+= pl+", "
    
        if(len(self.bad_plates)>0):
            tk.messagebox.showwarning("CSV not Found!","ActivityCurves CSV not found, please run STEP 2 on plate(s) "+bad_plate_list)

        missing_files_list = "" # Took out missing_files warning, might implement later
        if(len(self.missing_files)>10):
            for i in range(10):
                missing_files_list += self.missing_files[i]+", "
            missing_files_list+="and more "
        else:
            for f in self.missing_files:
                missing_files_list+=f+", "

        self.plates_loaded = True
        return
        
    def load_template(self):

        # Get the unique groups within a template
        new_groups = []
        new_groups.append(self.groups_template[0])
        for group in self.groups_template:
            if group in new_groups:
                pass
            else:
                new_groups.append(group)
    
        if "Censor" not in new_groups:
            new_groups.append("Censor")
        if "No group" not in new_groups:
            new_groups.append("No group")

        # Clear Groups
        self.box_group.delete(0,'end')
        for group in new_groups:
            if group not in self.color_dict:
                color = (0,0,0)
                for c in self.colors:
                    if c not in self.color_dict.values():
                        color = c
                        break
                self.color_dict[group] = color
        #box_group.delete(0, tk.END)

        #Inserting Saved Templates
        if len(self.groups_template) == 0:
            print('No template saved yet!')
        else:
            index = 0 
            for group in new_groups:
                self.box_group.insert("end",group)
                self.box_group.itemconfig(index,foreground=from_rgb(self.color_dict[group]))
                index+=1
        print('Template Successfully Loaded')
        self.groups = new_groups
        if(len(self.image1.curr_imgList)>0): # Display new template immediately if image is currently displayed
            self.image1.display_roi()
            self.image1.editing = False
            self.image1.zoomed = False
        return

    def load_template_from_CSV(self):

        # Read from a plate's CSV
        self.groups_template = readArrayFromCsv(self.activity_file,"<Group name>")
        self.groups_template = self.groups_template[0].tolist()
        self.load_template()
        self.json_open = False
        return


    # Select the plate and display activity
    def select_plate(self,event):

        widget = event.widget
        selection = widget.curselection()
        if(len(selection)==0): # Do nothing if there are no plates in the box
            return
        color = self.box_plate.itemcget(selection[0],"foreground")
        if(color=="red"): # If plate is missing CSV do nothing
            tk.messagebox.showwarning("CSV not Found!","ActivityCurves CSV not found, please run STEP 2 on this plate")
            return
        if(not self.applying): # If not in batch apply mode
            value = widget.get(selection[0])
            self.plate_path = os.path.join(self.pparent.get(), value) # The full path name of the selected plate
            self.slist = verifyDirlist(self.plate_path, True)[1]   # Sessions contained in the selected plate folder
            self.box_session.delete(0,"end")

            index = 0
            first = None
            for s in self.slist:
                path = os.path.join(self.plate_path,s)
                files = os.listdir(path)
                files = " ".join(files)
                has_roi = False

                if(index == 0):
                    first = datetime.strptime(s.split(" ")[0],"%Y-%m-%d")
            
                date = datetime.strptime(s.split(" ")[0],"%Y-%m-%d")
                days = (date-first).days
                hours = s.split(" ")[1].split("-")[0].replace("(","")
                frac = int(hours)/24
                days += frac
                days = round(days,1)
                daystring = " (Day " + str(days) + ")"

                sdays = s+daystring

                self.box_session.insert("end", sdays)
                index+=1

            # Clear images
            self.image1.total_frames = 0
            self.image1.total_sessions = 0
            self.image1.canvas.delete("all")
            self.image2.delete("all")
            self.image3.delete("all")
            self.image1.curr_imgList.clear()
            self.session_selected = False
            self.image1.zoomed = False
            self.first_loading = True
            self.first_loaded = False

            #Create activity plot
            print("Creating Activity Plot)")
            if(self.activity_plot.import_activity(value,"<Stimulated activity>")):
                self.activity_plot.select_activity()
                self.activity = self.activity_plot.activity_data
            if(self.activity1.import_activity(value,"<Cumulative activity>")):
                self.activity_plot.select_activity()

            # load current template
            self.load_template_from_CSV()
            self.image1.start_loading() # Starts loading thread
            self.start_loading_thread()
            self.sessions_loaded = True
            return
        #

    # Editing group Functionality

    def disable_groups(self):

        # Disables buttons and listboxes
        self.pushbutton_add["state"] = "disabled"
        self.pushbutton_load_template["state"] = "disabled"
        self.pushbutton_save_as_template["state"] = "disabled"
        if(self.applying): # Exceptions depending on what function disabled the groups
            self.pushbutton_done_editing["state"] = "disabled"
            self.pushbutton_apply_template["state"]= "enabled"
        else:
            self.pushbutton_done_editing["state"] = "enabled"
            self.pushbutton_apply_template["state"]= "disabled"
            self.image1.editing = True
            self.box_plate["state"] = tk.DISABLED
            self.box_session["state"] = tk.DISABLED

    def enable_groups(self):

        # Re-enables buttons and lisboxes
        self.pushbutton_add["state"] = "enabled"
        self.pushbutton_apply_template["state"]= "enabled"
        self.pushbutton_load_template["state"] = "enabled"
        self.pushbutton_save_as_template["state"] = "enabled"
        if(self.applying):
            None
        else:
            self.pushbutton_done_editing["state"]="disabled"
            self.image1.editing = False
            self.box_plate["state"] = tk.NORMAL
            self.box_session["state"] = tk.NORMAL

    def disable_listboxes(self):

        self.box_group["state"] = tk.DISABLED
        self.box_session["state"] = tk.DISABLED

    def enable_listboxes(self):

        self.box_group["state"] = tk.NORMAL
        self.box_session["state"] = tk.NORMAL

    def pushbutton_done_editing_callback(self,*args):

        # Saves current template directly to current plate's CSV
        new_group_names = np.array(self.groups_template)
        new_group_names = np.reshape(new_group_names,(1,new_group_names.shape[0]))
        selection = self.box_plate.curselection()[0]
        parent_directory = verifyDirlist(self.pparent.get(), 1, "_Analysis", "")[0]
        plate_name = self.box_plate.get(selection)
        self.activity_folder = verifyDirlist(parent_directory[0], 1, str(plate_name), "")[0]
        self.activity_file = verifyDirlist(self.activity_folder[0], 0, "csv","")[0][0]
        replaceVariableInCsv(self.activity_file,',,<Group name>', new_group_names, col_offset = 2)

        # Display image without filled in circles
        self.image1.display_roi()
        self.box_group.selection_clear(0,"end")
        self.enable_groups()

        return

    def editGroups(self, event):

        if(not self.session_selected): # Must select an image first
            tk.messagebox.showerror("Image not loaded!","Select an image first!")
            self.box_group.selection_clear(0,"end")
            return

        widget = event.widget
        group = widget.get(widget.curselection())

        if(self.sessions_loaded): # Display groups and load current group thats editing
            self.image1.load_group(group)
            self.image1.display_roi(True) # True flag fills in the circles
            self.disable_groups()

    # Functions to manage group names
    def pushbutton_add_callback(self, *args):

        text = self.group_entry.get()
        if(text==""):# If user entered empty string
            tk.messagebox.showerror(self.root,"Please enter a valid group name!")
            return
        self.box_group.insert("end",text)
        self.groups.append(text)
        index = len(self.groups)-1
        if text not in self.color_dict: # If it is a new group, assign it to a new color
            color = (255,255,255)
            for c in self.colors:
                if c not in self.color_dict.values():
                   color = c
            self.color_dict[text] = color
        self.box_group.itemconfig(index,foreground=from_rgb(self.color_dict[text]))
        self.input_text.set("")

        return

    def pushbutton_batch_apply_callback(self,*args):

        try: # Check if there is a template loaded first
            self.groups_template[0]
        except NameError:
            tk.messagebox.showerror(self.root,"Load in a valid template first!")
            return

        if(self.applying==False):

            if tk.messagebox.askokcancel(message="""Use the 'Plate' list  to select which plates should be grouped according to the current plate template. 
            Drag to select multiple. Use CONTROL to select single plates."""):
                self.applying = True
                self.box_plate["selectmode"] = tk.EXTENDED # Allow user to select multiple plates at once
                self.disable_listboxes()
                self.disable_groups()
                self.pushbutton_apply_template["text"] = "Apply"
                self.past_selection = self.box_plate.curselection()
            else:
                None
        else:
            # If button is clicked again, apply template to all selected plates
            new_group_names = np.array(self.groups_template)
            new_group_names = np.reshape(new_group_names,(1,new_group_names.shape[0]))

            selection = self.box_plate.curselection()
            parent_directory = verifyDirlist(self.pparent.get(), 1, "_Analysis", "")[0]

            for plate in selection:
                plate_name = self.box_plate.get(plate)
                self.activity_folder = verifyDirlist(parent_directory[0], 1, str(plate_name), "")[0]
                self.activity_file = verifyDirlist(self.activity_folder[0], 0, "csv","")[0][0]
                replaceVariableInCsv(self.activity_file,',,<Group name>', new_group_names, col_offset = 2)

            self.applying = False
            self.enable_listboxes()
            self.enable_groups()
            self.box_plate["selectmode"] = tk.SINGLE
            self.pushbutton_apply_template["text"] = "Batch Apply"
            self.box_plate.selection_clear(0,"end")
            if(len(self.past_selection)>0): # Reset to plate before
               self.box_plate.selection_set(self.past_selection[0])
            return


    def pushbutton_change_callback(self,*args):
        return

    def pushbutton_save_as_template_callback(self,*args):

        # Save template to json file
        try:
            self.groups_template[0]
        except NameError:
            tk.messagebox.showerror(self.root,"Please load in a valid template first!")
            return

        json_formattable = []
        for group in self.groups_template:
            json_formattable.append(group)
    
        x = {
            "template": json_formattable
            }

        groups_json = json.dumps(x)

        savefile = filedialog.asksaveasfilename(parent=self.root, initialdir=self.pparent.get(), 
                                                title='Save a new file', defaultextension = ".json")
        if(savefile==""):
            return
        else:
            self.json_open = True
            self.json_path = savefile


        with open(savefile,"w") as outfile:
            outfile.write(groups_json)

    def pushbutton_load_template_callback(self,*args):

        # Load template from selected json file
        file = filedialog.askopenfilename(parent=self.root, initialdir=self.pparent.get(), title='Select a valid json file')
        if(file==""):
            self.json_open = False
            return
        else:
            self.json_open = True
            self.json_path = file

        with open(file,"r") as openfile:
            self.json_object = json.load(openfile)
    
        self.groups_template = self.json_object["template"]
        self.load_template()

    def start_loading_thread(self):
        loading = threading.Thread(target=self.load_image_thread)
        loading.start()


    def load_image_thread(self):
        # Thread to load session images
        index = 0
        frames = 0
        for session in self.slist:
            session_path = os.path.join(self.plate_path,session)
            s = session_path+'\\'
            try:
                files = os.listdir(s)
            except FileNotFoundError:
                break
            roi_file = ""
            for f in files:
                if f.endswith(".csv"):
                    roi_file = f

            if(roi_file != ""):
                roi_list = readArrayFromCsv(s+roi_file,"<ROI Coordinates>")
            
                imgList = verifyDirlist(session_path, False, "png")[0]
                imgList.pop(len(imgList)-1)
                frames += len(imgList)
                if(len(imgList)==0):
                    self.image_queue.put("red"+str(index))
                else:
                    imgs = self.image1.load_images(imgList,roi_list) # Load a sessions first and last frames
                    self.image1.img_dict[session] = imgs # Populates dictionary with session name and session frames
                    self.image_queue.put(index)
            else:
                self.image_queue.put("red"+str(index))
            if(self.signal_stop):
                break
            index+=1

        self.image1.total_frames = frames
        self.image1.total_sessions = index
        return

    def update_window(self):
        try:
            index = self.image_queue.get_nowait()
            if("red" in str(index)):
                index = int(index.replace("red",""))
                try:
                   self.box_session.itemconfig(index,fg="red")
                except tk.TclError:
                   pass
            else:
                try:
                    self.box_session.itemconfig(index,fg="")
                except tk.TclError:
                    pass

            if(index==0):
                self.box_session.selection_set(index)
                self.box_session.event_generate("<<ListboxSelect>>")
                self.first_loaded = True

            self.root.update_idletasks()
        except queue.Empty:
            pass
        self.root.after(25, self.update_window)
                


    def close_window(self):
        self.signal_stop = True
        self.image_queue.queue.clear()
        #for id, thread in threading._active.items():
        #    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(id, ctypes.py_object(SystemExit))
        #    if res > 1: 
        #        ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0) 
        #        print('Exception raise failure') 
        #self.root.destroy()
        sys.exit()

if __name__ == "__main__":

    print(len(sys.argv))
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = r"C:\WormWatcher"
    image = ImageGUI2(directory)