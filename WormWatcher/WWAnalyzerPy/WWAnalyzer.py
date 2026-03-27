# Use mainly tk
from tkinter import *
#from tkinter.ttk import * # Replace tkinter widgets with styled widgets
from tkinter import filedialog
from tkinter import messagebox
from PIL import ImageTk,Image
import tkinter as tk
import tkinter.scrolledtext as tkst
from tkinter import ttk
import pdb
import time
import sys

# Use a couple as TTK
from tkinter.ttk import Style
from tkinter.ttk import Progressbar
from tkinter.ttk import Button
from ReformatParentDirectory import reformatParentDirectory
from ReformatParentDirectory import verifyValidParentDirectory
from breakUpSession import breakUpSession

#Imports from Other WW Files
from ActivityCurveGenerator import activityCurveGenerator
from plotGroupAverages import plotGroupAverages
from plotGroupAverages import all_param
from plotGroupAverages import data_csv
from verifyDirlist import *
from PlateTreeView import PlateTreeView
from LabeledProgressbar import LabeledProgressbar
from ErrorNotifier import ErrorNotifier
from DisplaySubprocessOutput import DisplaySubprocessOutput
from validateAnalysisFolder import validateActivityCurvesFile
from censorWellsAuto import censorWellsAuto, displayQualityResults
from setRootPos import *
from AnalysisType import AnalysisType, ListOfAnalysisTypes
from ImageGUI2 import ImageGUI2
from scaleGuiSizeLinear import *

#General Imports
import os
import subprocess
from time import sleep
from threading import Thread
import json
from manualImageInspector import manualImageInspection
import psutil
import math
import numpy as np
from csvio import is_number, str2number, readArrayFromCsv, replaceVariableInCsv
import copy
from natsort import natsorted, ns, index_natsorted
import ctypes

import logging

""" Global variables """
global bgcolor, buttons, abort_analysis_flag, PB, treeview, selected_plates, Err 
global LAT # ListOfAnalysisTypes object with all analysis type info

"""
    Arguments for listbox:
    [0] parent on which to base it (Tk frame or root)
    [1] Title of the listbox (string)
    [2] 
"""

################################################
#                GENERAL FUNCTIONS
################################################
def rgb2Hex(rgb):
    #translates an rgb tuple of int to a tkinter friendly color code=
    return "#%02x%02x%02x" % rgb   

def focus_next_window(event):
    event.widget.tk_focusNext().focus()
    return("break")

def select_all(event):
    event.widget.tag_add(SEL, "1.0", END)

def combine_funcs(*funcs):
    def combined_func(*args, **kwargs):
        for f in funcs:
            f(*args, **kwargs)
    return combined_func

def update_progress(progress, i):
        progress['value'] = i




################################################
#                GENERAL CLASSES
################################################

class ProgressBar:
    def  __init__(self, frame, max):
        self.max = max
        self.step = tk.DoubleVar()
        self.progbar = ttk.Progressbar

        self.frame = frame
        self.frame.pack(fill=tk.BOTH, padx=2, pady=2)
 
    def add_progbar(self):
        self.progbar = ttk.Progressbar(
            self.frame, 
            orient=tk.HORIZONTAL, 
            mode='determinate', 
            variable=self.step, 
            maximum=self.max)
        self.progbar.pack(fill=tk.X, expand=True)
 
    def start_progbar(self, step):
        self.add_progbar(step)
        self.root.update()

    def update_progbar(self, step):
        self.step.set(step)

    def destroy_progbar(self):
        self.progbar.destroy()


class listBoxScrolling(Tk):
    def __init__(self, *args, **kwargs):

        # Unpack arguments
        parent = args[0]
        title = args[1]
        pack_side = args[2]
        color_tuple = args[3]

        # Create frame for the listbox
        self.frame = Frame(parent)
        self.frame.pack(side=pack_side)
        self.frame.configure(background=rgb2Hex(bgcolor))

        # Upper label
        Label(self.frame, text=title,font =("Helvetica", 14),background=rgb2Hex(bgcolor)).pack(side="top")

        # Create the listbox
        self.listobj = Listbox(self.frame, width=20, height=20, font=("Helvetica", 12),selectmode='multiple',
                               exportselection=0,background=rgb2Hex(color_tuple))
        self.listobj.pack(side="left", fill="y")

        # Create the scrollbar
        self.scrollbar = Scrollbar(self.frame, orient="vertical")
        self.scrollbar.config(command=self.listobj.yview)
        self.scrollbar.pack(side="left", fill="y")
        self.listobj.config(yscrollcommand=self.scrollbar.set)

    def replaceList(self, vals):
        # Delete old values
        self.listobj.delete(0,'end')

        # Add new values
        for x in vals:
            self.listobj.insert(END, str(x))



################################################
#                 BUTTON CALLBACKS
################################################

def dropdown_callback(*args): 

    # Print to console which option was selected
    Err.notify("Log","Selected: " + dd_var.get())
    global LAT

    # Search for and apply the appropriate settings tree
    text_width_mod = 100
    help_str = LAT.updateJsonTreeViewDisplay(dd_var.get(),subframe_analysis_opts_tree,text_width_mod)

    # Figure out which help label to add
    help_var.set(help_str)

def pushbutton_refresh_callback(*args):
    
    #Call pushbutton_path_callback with the flag to skip selection
    pushbutton_path_callback(True)

def pushbutton_path_callback(*args):
    global PB, LAT

    if len(args) > 0:
        skip_getdir = args[0]
    else:
        skip_getdir = False

    global plates, N_analyzed_sessions, N_image_sessions

    # Clear the tree before starting
    treeview.clear_tree()

    # Style
    if not skip_getdir:
        s = Style()
        s.configure('my.TButton',background = "orange")
        pushbutton_path.configure(text = "Validating...",style='my.TButton')

        # Ask the user to select a folder
        tempdir = filedialog.askdirectory(parent=root, initialdir=pparent.get(), title='Please select the PARENT directory')
        if len(tempdir) > 0:
            Err.notify("Print","Selected folder: " + tempdir)

        # If the user selected an "_Analysis" folder, go one level up to get parent folder
        if "_Analysis" in tempdir:
            tempdir,trash = os.path.split(tempdir)

        # Enforce a valid parent directory
        if verifyValidParentDirectory(tempdir,Err) :
            pparent.set(tempdir)
            s.configure('my.TButton',background = "gray")
            pushbutton_path.configure(text = "Change path")
            LAT.saveCurrentDirectory(tempdir) # Stores current directory as LAT.directory and updates JSON file

        else:
            s.configure('my.TButton',background = "red")
            pushbutton_path.configure(text = "Change path")
            return


    #Changing directory pushbutton text 

    # -----      Populating Tree View - get all plates in both analysis and image folders       ----- #

    # Get plate image folders
    directory = pparent.get()
    directory_ana = os.path.join(directory, "_Analysis").replace("\\","/")
    plates_img_full, plates_img = verifyDirlist(directory,True,"","_Analysis")
    plates_img_full, plates_img = additionalExcludeFilter(plates_img_full,plates_img,"_ROI")
    plates_img_full, plates_img = additionalExcludeFilter(plates_img_full,plates_img,"RECYCLE")
    plates_img_full, plates_img = additionalExcludeFilter(plates_img_full,plates_img,"gsdata")

    # Get plate analysis folders
    plates_ana_full, plates_ana = verifyDirlist(directory_ana, True,"")
   
    # Get the master list of plates
    merged_plates = list(np.unique(plates_img + plates_ana))

    # Sort the names naturally
    idx = index_natsorted(merged_plates)
    treeview.plates_master = [merged_plates[i] for i in idx]

    # -----     Determine whether each master plate has image and/or analysis folders       ----- #

    treeview.plates_img = list()
    treeview.plates_ana = list()

    for plate in treeview.plates_master:

        # Determine if there's an image folder for this plate, if so add it to the list
        if plate in plates_img:
            treeview.plates_img.append(plates_img_full[plates_img.index(plate)])

        else:
            treeview.plates_img.append("")

        # Determine if there's an analysis folder for this plate, if so add it to the list
        if plate in plates_ana:
            treeview.plates_ana.append(plates_ana_full[plates_ana.index(plate)])

        else:
            treeview.plates_ana.append("")

    # Make sure they're all the same size...

    if (len(treeview.plates_master) != len(treeview.plates_img)) or  (len(treeview.plates_master) != len(treeview.plates_ana)):
        messagebox.showerror("Internal error","Internal error: Master list does not match analysis or image lists")

    # -----     Count up the STEP 1 sessions in each image and/or analysis folder and determine whether STEP 2 has been completed   ----- #
    treeview.N_image_sessions = []
    treeview.N_analyzed_sessions = []
    treeview.Completed_STEP2 = []
    PB.setStatus(root,0,len(treeview.plates_master)-1,'Searching for data...')

    total_start = time.time()

    for (plate_img,plate_ana,i) in zip(treeview.plates_img,treeview.plates_ana,range(len(treeview.plates_master))):

        # Assemble the session
        session_csv_path = os.path.join(plate_ana,'Session data')

        # Determine image folder session counts. If blank, it's automatically gonna be len=0
        treeview.N_image_sessions.append(len(verifyDirlist(plate_img,True)[0]))
        treeview.N_analyzed_sessions.append(len(verifyDirlist(session_csv_path,False,'.csv','Roi')[0]))
        PB.setStatus(root,i)

        # Determine whether step 2 has been completed yet
        d_s2 = verifyDirlist(plate_ana,False,'Activity','png')[0]
        if len(d_s2) > 0:
            treeview.Completed_STEP2.append("Yes")
        else:
            treeview.Completed_STEP2.append("No")


    #Populating Tree View
    for i, plate in enumerate(treeview.plates_master):
        treeview.tree.insert("", 'end', text=str(plate), values = (treeview.N_image_sessions[i],treeview.N_analyzed_sessions[i],treeview.Completed_STEP2[i]))
    
    #Selecting all plates to start
    children_values = [treeview.tree.item(child)["values"] for child in treeview.tree.get_children()]
    children_written = [treeview.tree.item(child)["text"] for child in treeview.tree.get_children()]
    children = [child for child in treeview.tree.get_children()]
    treeview.tree.selection_set(children)
    treeview.plate_ids = treeview.tree.selection()
    
    #Dictionary comparing plates and id's
    treeview.ids_to_plates = {z[0]:list(z[1:]) for z in zip(treeview.plate_ids,treeview.plates_master, treeview.plates_img, treeview.plates_ana, treeview.Completed_STEP2)}
    treeview.selected_plates_ids= treeview.tree.selection()
    selected_plate_names = [treeview.ids_to_plates[id] for id in treeview.selected_plates_ids]
    #pdb.set_trace()

    #Selected Items
    #print(children_written)

    #Elapsed Time
    #print(plate + "  Total Time to Load Plates = " + str(time.time() - total_start))



def pushbutton_abort_callback(*args):

    # Change button color and text
    pushbutton_abort.configure(text = "Aborting...")

    # First, signal abort for the analysis thread
    global abort_analysis_flag
    abort_analysis_flag = True

    # Second, kill any open ActivityCalc.exe process
    PROCNAME = "ActivityCalc.exe"
    for proc in psutil.process_iter():

        # check whether the process name matches
        if proc.name() == PROCNAME:
            proc.kill()

    # Third, restore window and/or buttons
    root.deiconify()
    for btn in buttons:
        btn['state'] = 'normal'

def pushbutton_execute_callback(*args):
    
    # Verify that the parent folder is valid
    if not verifyValidParentDirectory(pparent.get(),Err): 
        return

    # open both JSON files to get parameters
    with open(LAT.jfile, 'r') as infile:
        jcurrent = json.load(infile)
    
    with open(r"C:\WormWatcher\config.json", 'r') as infile:
        jconfig = json.load(infile)

    # Determine whether a multithreaded step1 was requested
    N_threads = 0
    if "STEP 1:" in dd_var.get():
        N_threads = jconfig["analysis parameters"]["Step 1 threads"]


    # Show warnings if the selected plates do not meet the requirements
    if not treeview.validateSelection(dd_var.get(), N_threads):
        return



    # If multithreaded step1 was requested, launch the requested number of instances of ActivityCalc, no need for multithreading
    if  "STEP 1:" in dd_var.get() and N_threads > 1 :

        # Figure out the index cutoffs for each thread
        selected_plates_img = treeview.getSelectedPlatesByType(dd_var.get())
        N_plates = len(selected_plates_img)
        (trash,cutoffs) = np.histogram(np.arange(0,N_plates), bins=N_threads)
        cutoffs = np.ceil(cutoffs)

        # Spawn the requested number of workers without waiting for completion
        progname = r'C:\WormWatcher\WWCore\ActivityCalc.exe'
        for i in range(0,N_threads) :
            istart = str(int(cutoffs[i]))
            iend = str(int(cutoffs[i+1]-1))
            output = subprocess.Popen([progname, pparent.get(), istart, iend])
            #Err.notify("Print",output) Doesn't work
            sleep(6)
                     

    # Manual inspection using the current parent folder
    elif "STEP 3:" in dd_var.get():
        root.withdraw()
        #exec(open("ImageGUI2.py").read())
        subprocess.call(['python',"ImageGUI2.py", pparent.get()])
        root.deiconify()


    # Otherwise, run the analysis plate-by-plate using the shared loop
    else:
        analysis_for_each_plate(jcurrent,jconfig)

    # When finished, refresh the data in the table
    pushbutton_path_callback(True);

def analysis_for_each_plate(jcurrent,jconfig):

    global LAT, buttons, abort_analysis_flag, PB

    # Enable abort pushbutton
    pushbutton_abort['state'] = 'normal'
    pushbutton_abort.configure(text = "Abort")

    # Disable all buttons while analysis is running
    for btn in buttons:
        btn['state'] = 'disabled'

    Err.notify("Print","Execute: " + dd_var.get())

    # Get list of all selected plates for the correct type
    selected_plates = treeview.getSelectedPlatesByType(dd_var.get())
    PB.setStatus(root,0,len(selected_plates),"Execute: " + dd_var.get())


    # Run the analysis for each plate if only a subset of plates was selected
    for (this_plate,j) in zip(selected_plates,range(len(selected_plates))):

        # Extract the plate search name for one individual plate if doing one-at-a-time
        [trash,short_name] = os.path.split(this_plate)

        # Update the progressbar with the current status only
        PB.setStatus(root,None,None,"Execute: " + dd_var.get() + " - " + short_name +" - Plate " + str(j+1) + " of " + str(len(selected_plates)))

        if "STEP 1:" in dd_var.get(): 
                
            # Call the analyzer program. Exit if escape was pressed (-1 return val)
            progname = r'C:\WormWatcher\WWCore\ActivityCalc.exe'
            Err.notify("Print","Analyzing: "+short_name)
            #command = "\"" + progname + "\" " + "\""+ pparent.get() + "\"" + " \"0\" " + " \"99999\" " +  "\"" + short_name + "\""
            #DisplaySubprocessOutput(root, scrolltext,command)
            output = subprocess.run([progname,pparent.get(),"0","99999",short_name], shell=True, capture_output = True,text=True)

            # Abort if user pressed escape
            if output.returncode != 0:
                Err.notify("PrintRed","User aborted the analysis before completion.")
                break

            # Otherwise display error encoded by return val
            Err.notifyLongMsg(output.stdout)


        # Step 2 activity curves
        elif "STEP 2:" in dd_var.get():
            use_median_filter = jcurrent["activity curve options"]["Median Filter (T/F)"]
            percentile = jcurrent["activity curve options"]["Session act. percentile (0=mean)"]
            activityCurveGenerator(this_plate,Err,"Log",use_median_filter,percentile)
            Err.notify("Print","Generated activity curves for: " + short_name) 

        # Step 4 Censor bad wells
        elif "STEP 4:" in dd_var.get():

            # Convert plateful from analysis path to actual file name, it must already exist due to PlateTreeView's checking
            AC_file = validateActivityCurvesFile(this_plate,Err)
            if len(AC_file) == 0:
                continue

            # Load labels from this file, make sure labels exist
            labels = readArrayFromCsv(AC_file, "<Group name>")
            if "empty" in labels:
                messagebox.showerror("Error",this_plate + "\nDoes not have group names in STEP 2 file.\n\nTry running STEP 2 again (any old groups / censor decisions will not be deleted")
                break

            # Set parameters by input arguments from JSON file
            min_days = jcurrent["quality filters"]["Min days (d)"]
            wf_init = [jcurrent["quality filters"]['Initial WF (min)'] , jcurrent["quality filters"]['Initial WF (max)']]
            pct_remain_check_day = jcurrent["quality filters"]['Check day for % remain']
            pct_remain_lims = [jcurrent["quality filters"]['% remain (min)'],jcurrent["quality filters"]['% remain (max)']]
            zero_level = jcurrent["quality filters"]['zero level']
            min_peak_act = jcurrent["quality filters"]['minimum activity peak']

            # Apply censors and write to file
            this_reason, this_qual = censorWellsAuto(AC_file,Err,True,min_days,wf_init,pct_remain_check_day, \
                                        pct_remain_lims, zero_level, min_peak_act)

            # Store the filtering results so that the collective performance can be assessed
            if j == 0:
                reason = this_reason
                qual_vals = this_qual
            else:
                reason = np.concatenate((reason,this_reason),axis=1)
                qual_vals = np.concatenate((qual_vals,this_qual),axis=1)

            # When finished, display results 
            if j + 1 == len(selected_plates):
                displayQualityResults(Err,qual_vals,reason,min_days,wf_init,pct_remain_lims,min_peak_act)



        #swimming analysis step 1
        elif "Swimming Image Analysis" in dd_var.get():
                            
            # Call the analyzer program. Exit if escape was pressed (-1 return val)
            progname = r'C:\WormWatcher\WWCore\WormWatcher.exe'
            Err.notify("Print","Analyzing: "+short_name)
            #command = "\"" + progname + "\" " + "\""+ pparent.get() + "\"" + " \"0\" " + " \"99999\" " +  "\"" + short_name + "\""
            #DisplaySubprocessOutput(root, scrolltext,command)
            output = subprocess.run([progname,pparent.get()], shell=True, capture_output = True,text=True)

            # Abort if user pressed escape
            if output.returncode != 0:
                Err.notify("PrintRed","User aborted the analysis before completion.")
                break

            # Otherwise display error encoded by return val
            Err.notifyLongMsg(output.stdout)

        # Plot Tx levels
        elif "plot Tx level" in dd_var.get():
            
            # Load the json data for plotting
            level = jcurrent["plot Tx params"]["Tx Level [0,1]"]
            fontsize = jcurrent["plot Tx params"]["Font size"]
            fontangle = jcurrent["plot Tx params"]["Font angle"]
            aac = jcurrent["plot Tx params"]["Apply auto-censor (T/F)"]
            jvect = [level, fontsize,fontangle, aac]

            # Check that the parameter value is in the valid range before running
            if level > 0 and level <= 1:      # Check if the input value is valid
                plotGroupAverages(selected_plates, jvect, "<Tx>") 
                
            else:
                messagebox.showerror("Error",str(level)+" is an invalid parameter value")
            break
        
        
        # Plot total activity
        elif "plot total activity" in dd_var.get():

            jvect = [jcurrent["plot Act params"]["Total act. timepoint (d)"],\
                     jcurrent["plot Act params"]["Font size"],\
                     jcurrent["plot Act params"]["Font angle"],\
                     jcurrent["plot Act params"]["Apply auto-censor (T/F)"]]
            plotGroupAverages(selected_plates, jvect, "<Cumulative activity>")

            break

        # Plot escaping
        elif "escaping" in dd_var.get():
            jvect = ["",jcurrent["plot Act params"]["Font size"],jcurrent["plot Act params"]["Font angle"],""]
            plotGroupAverages(selected_plates, jvect, "<Quality values>")
            break


        # Write output data to csv file
        elif "Export data" in dd_var.get():

            jvect = [ [jcurrent["export csv params"]["Tx 1"], jcurrent["export csv params"]["Tx 2"]],\
                      [jcurrent["export csv params"]["Total act. timepoint (d)"] ],\
                    ]
            err_msg = data_csv(pparent.get(), selected_plates, selected_plates, jvect, jcurrent["export csv params"]["Apply auto-censor (T/F)"] )
            if len(err_msg) > 0:
                Err.notify("PrintRed",err_msg)
            break



        # Convert old style data
        elif "Convert old" in dd_var.get():

            breakUpSession(this_plate)


        # Error
        else:
            Err.notify("PrintRed","Internal error - invalid analysis type selected.")

        # Update progressbar
        PB.setStatus(root,j+1,len(selected_plates),"Execute: " + dd_var.get() + " - " + short_name +" - Plate " + str(j+1) + " of " + str(len(selected_plates)))


        # Exit if requested
        if abort_analysis_flag :
            abort_analysis_flag = False
            pushbutton_abort.configure(text = "Abort")
            pushbutton_abort['state'] = 'disabled'
            break

    Err.notify("Print","Completed analysis: " + dd_var.get())

    # Restore window and/or buttons
    root.deiconify()
    for btn in buttons:
        btn['state'] = 'normal'


def pushbutton_grab_callback(*args):
    print(12)
    print(treeview.tree.selection())
    return

def pushbutton_select_all_callback(*args):
    #Getting all childrenchildren = []
    children = [child for child in treeview.tree.get_children()]
    treeview.tree.selection_set(children)
    print(treeview.tree.selection())

def pushbutton_select_none_callback(*args):
    treeview.tree.selection_set()
    print("Selected Cleared")


################################################
#                 MAIN
################################################

#import json tree view
root_dir = (os.path.dirname(os.path.dirname(sys.path[0])))
tree_view_dir = os.path.join(root_dir, "WWConfigPy")
sys.path.append(tree_view_dir)

from JsonTreeView import JsonTreeView

logging.basicConfig(filename='C:/WormWatcher/MyLog.log',level=logging.DEBUG)


#try:
#Setup for tkinter GUI
root = Tk()
s = Style()
s.theme_use('vista')
root.title("WWAnalyzer")
bgcolor = (214,219,233)
yellowfont = (255,242,147)
root.configure(background=rgb2Hex(bgcolor))
root.resizable(True, True)

#Colors
greenfade = (172,222,172)
redfade = (222,172,172)


"""------ TOP FRAME - SETTING THE PATH ------"""
# Create the frame
topbgcolor = (41,58,86)
frame_path = Frame(root,relief = SUNKEN, borderwidth = 2)
frame_path.pack(side="top",fill="both")
frame_path.configure(background=rgb2Hex(topbgcolor))


# Add buttons and labels
pparent = StringVar(root)
pparent.set(r"C:\WormWatcher")
Label(frame_path, text="Parent Directory",font='Helvetica 16 bold',wraplength=260,justify=CENTER,background=rgb2Hex(topbgcolor),foreground=rgb2Hex(yellowfont)).pack()
Label(frame_path,textvariable=pparent, text="Select Directory",font='Helvetica 13',wraplength=800,justify=LEFT,background=rgb2Hex(topbgcolor),foreground=rgb2Hex(yellowfont)).pack()
path_style = Style()
path_style.configure('my.TButton', font=('Helvetica', 13))
pushbutton_path = Button(frame_path,text="Change Path",command = pushbutton_path_callback, style = 'my.TButton')
pushbutton_path.pack() # Must be done on separate line



"""------ VERY BOTTOM - SCROLLING TEXT ----- """

# Set scroll text height to 8 (lines) + a few more if screen height is larger than 768
# p1: (768,8)
# p2: (1080,14)         --> m = 6 / 312 = 0.01923 --> b = 8 - 0.01923*768 = -6.768 
# check: y = mx+b = 0.01923*1080 - 6.768 = 14
# NOTE: Minimum size is 8

user32 = ctypes.windll.user32
scroll_text_height = scaleGuiSizeLinear(user32.GetSystemMetrics(1),768,8,1000,18)

frame_scroll = tk.Frame(root)
frame_scroll.pack(side = tk.BOTTOM, fill=tk.X,padx=10, pady=10)

scrolltext = tkst.ScrolledText(frame_scroll,wrap = tk.WORD,height = scroll_text_height) #
scrolltext.pack(padx=0, pady=0, fill=tk.X, expand=True, side = tk.BOTTOM)
scrolltext.insert(tk.INSERT,"\nWelcome to WWAnalyzer!")


# Create Err notifier that feeds to this scrolled text
Err = ErrorNotifier(scrolltext)

"""------ BOTTOM FRAME - PROGRESS BAR ------"""
#Creating progress bar
PB = LabeledProgressbar(root,"bottom",rgb2Hex(bgcolor))


"""------ LEFT FRAME - SELECTING PLATES ------"""
# Create the frame
frame_tree = Frame(root,relief = RIDGE, borderwidth = 1)
frame_tree.pack(side="left", fill=tk.BOTH,expand=1)
frame_tree.configure(background=rgb2Hex(bgcolor))

# Create Tree View
treeview = PlateTreeView(frame_tree,"Highlight Plates to Select","top",pparent,rgb2Hex(bgcolor))


# Subframe for buttons
subframe_buttons = Frame(frame_tree, bd=0, relief=GROOVE)
subframe_buttons.configure(background=rgb2Hex(bgcolor))
subframe_buttons.pack(side="bottom",padx=5,pady = 5)
s.configure("my.TButton",font=('Helvetica',12,'bold'), compound = "c", background = rgb2Hex(greenfade))
button_select_all = ttk.Button(subframe_buttons,text="Select All",command = pushbutton_select_all_callback, style = 'my.TButton')
button_select_none = ttk.Button(subframe_buttons, text = "Select None", command = pushbutton_select_none_callback, style = 'my.TButton')
button_refresh = ttk.Button(subframe_buttons, text = "Refresh", command = pushbutton_refresh_callback, style = 'my.TButton')
button_select_all.pack(side="left",padx=5)
button_select_none.pack(side="left",padx=5)
button_refresh.pack(side="left",padx=5)


# Create frame for analysis type selection
frame_analysis = Frame(root)
frame_analysis.configure(background=rgb2Hex(bgcolor),relief = RIDGE, borderwidth = 1,width = 500)
frame_analysis.pack(side="left",fill=BOTH, expand=0) # Expand 0 prevents "rubber banding" of JsonTreeView gui when switching

Label(frame_analysis, text="Analysis type",font =("Helvetica", 14),background=rgb2Hex(bgcolor),pady = 5).pack(side="top")

# Initialize list of analysis types object (global variable)
LAT = ListOfAnalysisTypes()
list_of_types = LAT.getAtypeNamesAsList()

# Set the path that was recently used, or a default
pparent.set(LAT.directory)

# Create special string variable to hold current option in the dropdown menu
dd_var = StringVar(root)
dd_var.set(LAT.atypes[0]) # default value
sddm = ttk.Style()
sddm.configure("my.TMenubutton", font='Helvetica 12',background = rgb2Hex(yellowfont),anchor='E')
dropDownMenu = ttk.OptionMenu(frame_analysis,dd_var,LAT.atypes[0].name, *list_of_types,command=dropdown_callback,style="my.TMenubutton")
dropDownMenu.configure(width = 44)
dropDownMenu.pack()


# Create a text label that describes each analysis
help_var = StringVar(root)
help_var.set(LAT.atypes[0].desc)
labelhelp = Label(frame_analysis,textvariable=help_var,wraplength=475,justify=LEFT, anchor = NW, font='Helvetica 12',background=rgb2Hex(bgcolor)).pack();

# Subframes for analysis options
subframe_analysis_opts = Frame(frame_analysis,background=rgb2Hex(bgcolor),relief = RIDGE, borderwidth = 1)
subframe_analysis_opts.pack(fill='both')
Label(subframe_analysis_opts,text="Analysis Options",wraplength=300,justify=CENTER, font='Helvetica 14', width = 33,background = rgb2Hex(bgcolor)).pack();
json_param_height = scaleGuiSizeLinear(user32.GetSystemMetrics(1),768,200,1080,300)
subframe_analysis_opts_tree = Frame(subframe_analysis_opts,background=rgb2Hex(bgcolor), relief = RIDGE,borderwidth = 0,height =json_param_height,width = 75)
subframe_analysis_opts_tree.pack(fill = 'both', side = "top")
subframe_analysis_opts_tree.pack_propagate(0) #

# Bring up the parameters for the first group
help_str = LAT.updateJsonTreeViewDisplay(dd_var.get(),subframe_analysis_opts_tree,60)


# pushbutton_execute to run the selected analysis, below everything tlese
subframe_exec_abort = Frame(frame_analysis,background = rgb2Hex(bgcolor))
subframe_exec_abort.pack(side='bottom',pady=5)
s.configure('my.TButton', font=('Helvetica',14,'bold'),width = 20)
pushbutton_execute = Button(subframe_exec_abort,text="Execute",command = pushbutton_execute_callback, style = 'my.TButton')
pushbutton_execute.pack(side='left',padx=5)

# pushbutton_abort to end all analysis
s.configure('my.TButton', font=('Helvetica',14,'bold'),width = 20)
pushbutton_abort=Button(subframe_exec_abort,text="Abort",command = pushbutton_abort_callback, style = 'my.TButton')
pushbutton_abort.pack(side='left',padx=5)
pushbutton_abort['state'] = 'disabled'
abort_analysis_flag = False

# Consolidate all buttons into a list for convenience later
buttons = [button_select_all, button_select_none,pushbutton_execute,pushbutton_path]


# Set position of root to middle of screen - TODO adapt to screen size / resizable
user32 = ctypes.windll.user32
setRootPos(root,1366,user32.GetSystemMetrics(1)*0.90,0,0) 
            # 1366 x 768 (use height 700) if hardcoding for low res monitor
            # user32.GetSystemMetrics(0)*0.95 width
            # user32.GetSystemMetrics(1)*0.90 height

# If the default folder is a valid parent folder load the data
pushbutton_path_callback()
root.focus_force() # Sometimes focus doesn't return to main window on first load
mainloop()

#except Exception as e:
#    logging.info(e)