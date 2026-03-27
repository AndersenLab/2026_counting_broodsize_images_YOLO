"""
Hongjing Sun
Created in Sep 2019

"""

import csvio
import seaborn as sns
import math
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import os
from verifyDirlist import verifyDirlist
import csv
import math
import numpy as np
import copy
import pandas as pd
from itertools import zip_longest
from tkinter import messagebox
from anova1_bygroup import *


global finalGroup, finalArray

# Put data form different files (stored as different arrays) into one array   
def put_together(list):
    if len(list) != 1: # Check if the list contains a single element; if so, no more adding needed
        newArray = list[0]
        for i in range(1,len(list)):
            newArray = np.append(newArray, list[i], axis = 1)
        return newArray
    else: 
        newArray = list[0]
        return newArray

# Find the nearest value in a list and return its index
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
   
# Read the Tx values for x (one row)
def find_lifespan(x, x_list, array):
    for i in range(len(x_list)): # Pick out the specific row
        if x_list[i] >= x:
            index = i
            break
    lifespan_row = array[index]
    lifespan_row = [x for x in lifespan_row if x is not None]
    lifespan_num = [float(numeric_string) for numeric_string in lifespan_row] # Convert to float type for further calculation
    return lifespan_num

# Read the total activity values for selected day or maximum day value
def find_totalAct(x, x_list, array):
    if x > max(x_list):
        lifespan_row = array[-1]
        lifespan_row = [x for x in lifespan_row if x is not None]
        lifespan_num = [float(numeric_string) for numeric_string in lifespan_row]
    
    else:
        # Pick out the specific row with the closest value
        index = find_nearest(x_list, x)    
        lifespan_row = array[index]
        lifespan_row = [x for x in lifespan_row if x is not None] # Replacing faulty code that removes zeros in addition to none's: list(filter(None, lifespan_row))
        lifespan_num = [float(numeric_string) for numeric_string in lifespan_row] # Convert to float type for further calculation
    return lifespan_num


# A function to calculate the mean of a list
def average(lst): 
    return sum(lst) / len(lst) 

# Function to calculate sample standard deviation.
def SSD(arr) : 
    sm = 0
    n = len(arr)
    if n == 1:
        return 0
    else:
        for i in range(0,n) : 
            sm = sm + (arr[i] - average(arr)) * (arr[i] - average(arr)) 
   
        return (math.sqrt(sm / (n - 1))) 
   
   
# Function to calculate sample error. 
def sampleError(arr) : 
    n = len(arr)
    # Formula to find sample error. 
    return SSD(arr) / (math.sqrt(n)) 

# A function to create empty list of desired length
def createlist(length):
    mylist = []
    for i in range(length): 
        mylist.append(i)
    return mylist

# Identify each group name and calculate the mean of the group associated data
def group_average(group_name, array, fontsize = 8, angle = 45, YLABEL = 'data'):
    
    Output = [[],[],[]]                     # The values on x&y axis
    grouping = np.unique(group_name)        # Identify how many unique group names, produce a list of unique group names
    group_origin = copy.deepcopy(grouping)  # Keep the one containing censor group
    dot = []                                # Individual value
    groupList = grouping.tolist()           # Covert to list to use remove()
    
    for name in grouping: 
        if "censor" in name.casefold():     # Case insensitivity
            groupList.remove(name)          # Not show Censor group          
    grouping = np.asarray(groupList)
    
    for index in range(len(grouping)):      # Go through group names one by one
        A = [grouping[index]]               # The group name to be compared with
        A_tx = []                           # Initial element in group average list, will be replace
        for i in range(len(group_name[0])):             # Compare each element to group them
            if math.isnan(array[0,i]):                  # Do not add it into the list if nan
                continue
            if group_name[0,i] == grouping[index]:      # If one element can be matched to the first one in the list, add it into the existing group and skip to the next one
                A_tx.append(array[0,i])
        
        # Check if the whole group only contain data Nan
        if not A_tx:
            continue
        
        dot.append(np.asarray(A_tx))
        mean_A = average(A_tx)
        std_error = sampleError(A_tx)
        Output[0].append(A[0])
        Output[1].append(mean_A)
        Output[2].append(std_error)
    
    # When there is no useful data in the file
    if not Output[0]: 
        return Output

    # Plotting
    x = np.array(Output[0])
    y = np.array(Output[1]) 
    e = np.array(Output[2])
    
    
    X = [Output[0][i] for i, data in enumerate(dot) for j in range(len(data))] 
    Y = [val for data in dot for val in data]
    
    a = Y
    sns.set(style="white")

    group = ["{k}".format(k=k) for k in X]
    df  =pd.DataFrame({"data":a, " ":group})

    palette = sns.dark_palette("blue", reverse=False,  n_colors=1 )
    a=sns.swarmplot(x=' ', y = 'data' ,s=6.8, data = df, palette=palette) # Scatter point
    sns.boxplot(x=' ', y = 'data', data=df)
    
    a.set_xticklabels(a.get_xticklabels(), rotation=angle, size=fontsize)
    #a.set_yticklabels(a.get_yticklabels(), size=fontsize)

    # Set ylabel
    sns.despine()
    plt.ylabel(YLABEL,fontsize = fontsize)
    plt.gca().set_ylim(bottom=0)
    #plt.grid()

    # Set fonts
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : fontsize}

    plt.rc('font', **font)
    #plt.setp(a.set_yticklabels(),a.get_yticklabels(), size=fontsize)
    
    plt.show()
    
    return Output

# Find the associated parameter list of the desired header
def find_param_name(fullfilename, data_name):
    with open(fullfilename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        row_num = 0
        for row in csv_reader:
            row_num+=1                          # Identify the row containing the header
            for i in range(len(row)):           # Check each element in the row
                if data_name in row[i] :
                    H = row[i]
                    H_index = i                 # Slice starting point
                    break
                else:
                    H = "No_header"
            for i in range(len(row)):           # Check each element in the row
                if "<" in row[i] :              # Find the parameter name
                    param_name = row[i]
                    break
            if not "No_header" in H:
                break                           # Stop reading when header's row found
        
        # Open the file and find the indices matching the requested header
    with open(fullfilename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        row_num -= 1
        start_row = copy.deepcopy(row_num)      # Scanning begins at specific row, to avoid repeated header
        # Scan through the lines looking for the right header
        start_reading = -1
        stop_reading = -1
        # Create a list of lists to temporarily store the values
        A = list()
        # Populate the list of lists from the file
        count = 0
        for row in csv_reader:
            if count < start_row:
                count += 1
                continue
            # Increment row number
            row_num+=1
            # Check for stop reading (any other header), if so abort reading
            S = "00"
            if start_reading >= 0 :
                for i in range(len(row)):       # Check each element in the row. If no header, continue reading
                    if "<" in row[i]:
                        S = "h_exist"
                        break
                    else:
                        S = "No_h"
                stop_reading = row_num          # Python slicing not inclusive of last item
            if "h_exist" in S:
                break

            # Check for requested header to start reading, if so skip to next line and read
            for i in range(len(row)):           # Check each element in the row
                if param_name in row[i] :
                    H = row[i]
                    start_reading = row_num+1
                    H_index = i                 # Slice starting point
                    if len(row) > 1:            # Check if the "hear" row is readable
                        for j in range(H_index+1,len(row)): # Instead of reading the whole row, reading part of it
                            if "<" in row[j]:
                                slice_end = j
                                break
                            else:
                                slice_end = len(row)
                    else:
                        slice_end = 000         # Value to be assigned later
                    break
                else:
                    H = "No_header"


            if param_name in H: # Skip to the next row and start reading. This header row should not be read
                continue
            
            # Are we reading now?
            if start_reading < 0 :
                continue

            #------------------ Reading valid portion of file -------------------#
            B = list()
            
            if slice_end == 000:
                slice_end = len(row)                  # assign slice_end value by length of the data row, instead of the header row
            
            row_slice = row[H_index:slice_end]
            row_slice = list(filter(None, row_slice)) # Eliminate empty element
            for v in row_slice :
                if csvio.is_number(v):                      # If v is a string, then int(v) won't work out 
                    if not not v and not "." in v:
                        if math.isnan(float(v)):
                            B.append(float(v))
                        else:
                            B.append(int(v))          # No decimal point, import integer
                    elif not not v :
                        B.append(float(v))            # Yes decimal point, import float
                else:
                    B.append(str(v))
            A.append(B)


    # Convert list of lists to numpy array

    # Determine whether data was read successfully, if so convert list of lists to numpy array
   
    if start_reading >= 0 and stop_reading > start_reading :
        C = np.asarray(A)
    else :
        C = np.zeros(shape = (1,1))
        print("Failed to parse file")
  
    #If the result is a single integer, return it as an integer
    if C.shape == (1,1):
        ele = C[0,0]
        try:                      # If the result is empty or a string
             if int(ele) == ele : # It's an integer. 
                C = int(ele)
             else :
                C = float(ele)

        except:
            C = "empty"
  
    return C

def all_param(plate_folders, data_name):
    A =[]
    max_A = []
    A_lst = []
    for plate_folders in plate_folders: 
        csv_file = verifyDirlist(plate_folders,False,".csv")[0]
        param_name = find_param_name(csv_file[0], data_name)      
        A.append(param_name)
        max_A.append(max(param_name))
        
        A_lst.append([row[0] for row in param_name])
    max_A = max(max_A)
    # All the common parameters from all plates
    if len(plate_folders) > 1:
        result = set(A_lst[0])
        for s in A_lst[1:]:
            result.intersection_update(s)
    else:
        result = A
    return [A,max_A,result]



# Find all desired files in parent file and plot for all groups
# pparent is the parent file selected, x is the parameter chosen by user
# data_name is the header of data type to be analyse, e.g. "<Tx>" 
def allDatafromPlates(plate_folders, x, data_name, apply_auto_censor):

    # Create empty lists to store data read from file
    percent = []
    data = [] 
    group_name = []
    for plate_folder in plate_folders:                                         # Go through every file in the folder to pick up the matched one
        data_list = []
        csv_file = verifyDirlist(plate_folder,False,".csv")[0]
        param_name = find_param_name(csv_file[0], data_name)                    # The whole column of the desired parameter
        data_array = csvio.readArrayFromCsv(csv_file[0],data_name)
        if "Tx" in data_name:
            selected_data = find_lifespan(x, param_name, data_array)
        elif "activity" in data_name:
            selected_data = find_totalAct(x, param_name, data_array)  
        elif "Quality" in data_name and not "empty" in data_array:
            selected_data = data_array[2,:].tolist()
        elif "empty" in data_array:
            return None
       
        data_list.append(selected_data)
        selected_data = np.asarray(data_list)
        data.append(selected_data)
        
        # Load group names
        group_name_b =[]
        group_name_a = csvio.readArrayFromCsv(csv_file[0],"<Group name>")  
        
        # Load auto-censor data. If present, apply it
        auto_censor = csvio.readArrayFromCsv(csv_file[0],"<Auto censor>") 
        if not "empty" in auto_censor and apply_auto_censor:
            to_censor = auto_censor[0:1,:]>0
            group_name_a[to_censor] = "Censor"

        
        group_name_a = list(filter(None, group_name_a[0]))
        group_name_b.append(group_name_a)
        group_name_array = np.asarray(group_name_b)
        #for i in range(len(group_name_array[0])):                                      # If no group name assigned for a particular one
            #if not group_name_array[0][i]:
                #group_name_array[0][i] = "NoGroup"
        
        if "empty" in group_name_array:                                         # The file has no group assigned; assign as no group for each
            if "Tx" in data_name:
                group_name_array = ["No group" for i in range(len(find_lifespan(x, param_name, data_array)))]
            elif "activity" in data_name:
                group_name_array = ["No group" for i in range(len(find_totalAct(x, param_name, data_array)))]
            elif "Quality" in data_name:
                group_name_array = ["No group" for i in range(len(data_array[2,:].tolist()))]

            group_name_array = np.asarray([group_name_array])

        group_name.append(group_name_array)          
    
    finalArray = put_together(data)
    finalGroup = put_together(group_name)                    # Merge arrays into a large one
    
    return [finalGroup, finalArray]

# Plotting the grouped data
def plotGroupAverages(plate_folders, args, data_name):

    # Level to plot
    x = args[0]

    # Font size
    if len(args) >= 2:
        fontsize = args[1]
    else:
        fontsize = 8

    # Font angle
    if len(args) >= 3:
        fontangle = args[2]
    else:
        fontangle = 0

    # Using auto-censor data (if available) or not
    if len(args) >= 4:
        apply_auto_censor = args[3]
    else:
        apply_auto_censor = False

    # Y axis label

    if "Tx" in data_name:
        YLABEL = 'T' + str(int(100*x)) + ", days"

    else:
        YLABEL = 'data'

    finalGroup = allDatafromPlates(plate_folders, x, data_name,apply_auto_censor)[0]
    finalArray = allDatafromPlates(plate_folders, x, data_name,apply_auto_censor)[1]
    A = group_average(finalGroup, finalArray,fontsize,fontangle,YLABEL)

    return A

# Create a different csv file to outwrite desired data
# parameter_entered should be a list containing all number user entered, [[Tx number],[Activity number]]
def data_csv(pparent, selected_plate, fullpathname, parameter_entered, apply_auto_censor):
 
    # Data to be printed out for this plotting
    
    wellNum = [j for i in range(len(selected_plate)) for j in range(1,25)]              # Wells in each plate
    plateName = [selected_plate[i] for i in range(len(selected_plate)) for j in range(1,25)]
    plateNum = [i for i in range(1, len(selected_plate)+1) for j in range(1,25)]
    plateNum.append("<end>")                                                            # Add a <end> so that the data is readable

    # Set fout and try to open it
    fout = os.path.join(pparent,"_Analysis","All Data "+str(len(selected_plate))+" plates.csv")     # Full path name of the csv file

    try:
        outfile = open(fout, "w", newline='') # or "a+", whatever you need
    except IOError:
        messagebox.showerror("Could not save","ERROR: Could not open file:\n" + fout + "\nPlease close Excel and try again!")
        return False

    # Set fout2 for statistics and try to open it
    fout_stats = os.path.join(pparent,"_Analysis","Stats Data "+str(len(selected_plate))+" plates.csv")     # Full path name of the csv file

    try:
        outfile_stats = open(fout_stats, "w", newline='') # or "a+", whatever you need
    except IOError:
        messagebox.showerror("Could not save","ERROR: Could not open file:\n" + fout_stats + "\nPlease close Excel and try again!")
        return False



    group_name = allDatafromPlates(fullpathname, 0.97, "<Tx>",apply_auto_censor)[0][0]
    groupName = group_name.tolist() 

    # Split up plate names to only show the name
    for (i,pn) in enumerate(plateName):
        _,head = os.path.split(pn)
        plateName[i] = head



    # Note - same file name will be overwritten
    d = [plateNum, plateName, wellNum, groupName]                    
    header = ["<plateNum>", "<plateName>", "<wellNum>", "<group>"]
    # Lifespan
    for i,parameter in enumerate(parameter_entered[0]):
        percent_value = int(parameter*100)
        param_data = allDatafromPlates(fullpathname, parameter, "<Tx>",apply_auto_censor)[1][0]
        d.append(param_data)
        header.append("<T"+str(percent_value)+">")
    # Total activity
    for i,parameter in enumerate(parameter_entered[1]):
        param_data = allDatafromPlates(fullpathname, parameter, "<Cumulative activity>",apply_auto_censor)[1][0]
        d.append(param_data)
        header.append("<TotAct"+str(parameter)+">")
    
    # Escaping - no auto censor 
    param_data = allDatafromPlates(fullpathname, parameter, "<Quality values>",False)
    if param_data is not None: # Quality data migh tbe None if STEP 4 was skipped
        param_data = param_data[1][0]
        d.append(param_data)
        header.append("<Pct remain>")
    else:
        err_msg = "Could not export data because STEP 4 (auto censor) has not been run."
        err_msg += " If you intentionally skipped this step please run it anyway and disable auto censor"
        err_msg += "on the export data settings."
        return err_msg
        
    

    export_data = zip_longest(*d, fillvalue = '')
    
    # Writing to the csv file
    with outfile:
        wr = csv.writer(outfile)
        wr.writerow(header)
        wr.writerows(export_data)


    # Organize the data by group for statistics (3 arrays for 3 variable Tx1, Tx2 and Act). Remove Censor
    unique_groups = np.unique(group_name)
    unique_groups = unique_groups[unique_groups != 'Censor']
    N_groups = len(unique_groups)
    data_for_stats = []
    for i in range(4):
        # purposely allocate for 1 large group, will crop later
        data_for_stats.append(np.ones((len(plateNum),N_groups))*np.nan)
   
    max_rows = 0

    # Reorganize for by-group printing
    for (n,ugrp) in enumerate(unique_groups):

        # Index of wells in this group name, also keep track of max group
        idx = np.where(ugrp == group_name)[0]
        N_this_grp = len(idx)
        max_rows = max((max_rows,N_this_grp))

        # Fill in data arrays for all 4 variables
        for i in range(len(data_for_stats)):
            data_for_stats[i][:N_this_grp,n] = d[4+i][idx]


    # Crop unnecessary rows
    for i in range(len(data_for_stats)):
        data_for_stats[i] = data_for_stats[i][:max_rows,:]
    

    # Write the arrays to disk, adding the mean, std, sem, and statistics underneath
    with outfile_stats:
        for i in range(len(data_for_stats)):

            # Write the variable name
            outfile_stats.write(header[4+i] + '\n')

            # Write the group names, with one unit empty space on left
            outfile_stats.write(',')
            for g in unique_groups:
                outfile_stats.write(g + ',')
            outfile_stats.write('\n')

            # Write the data array, with one unit of empty space on left, EXCLUDING NAN
            writeArrayWithSpace(data_for_stats[i],outfile_stats,True)

            # Compute the mean, standard deviation, and SEM for each group and write to file
            output_names = ['COUNT','MEAN','STD','SEM']
            outputs = []
            COUNT = np.sum(~np.isnan(data_for_stats[i]),axis=0)
            outputs.append(COUNT) # COUNT
            outputs.append(np.nanmean(data_for_stats[i],axis=0)) # MEAN
            STD = np.nanstd(data_for_stats[i],axis=0)
            outputs.append(STD) # STD
            outputs.append(STD / np.sqrt(COUNT)) # SEM

            for (output, name) in zip(outputs, output_names):
                outfile_stats.write(name + ',')
                for element in output: 
                    outfile_stats.write('%0.2f,' % element)
                outfile_stats.write('\n')

            # Compute statistics and write to file
            do_print = False
            stats_arr = anova1_bygroup(data_for_stats[i],unique_groups,do_print)   

            # Write the stats array, with one unit of empty space on left
            outfile_stats.write('\n\n,Multiple comparisons (Bonferoni corrected):\n')
            writeArrayWithSpace(stats_arr,outfile_stats,False)

            # Add some blank spaces
            outfile_stats.write('\n\n\n')
    return ""

# outfile must be open
def writeArrayWithSpace(arr,outfile,check_nan):

    for row in arr:
        outfile.write(',')
        for element in row:
            if check_nan and np.isnan(element):
                outfile.write(',')
                
            else:
                outfile.write(str(element) + ',')

        outfile.write('\n')
    outfile.write('\n')


if __name__ == "__main__":
    pparent = "F:\\test_WI"
    panalysis = os.path.join(pparent,"_Analysis")
    plate_folders = verifyDirlist(panalysis,True)[0]
    plate_folders = [r"F:\test_WI\_Analysis\WI_2"]
    #A = plotGroupAverages(plate_folders, 0.86, "<Tx>")
    data_csv(pparent, ["WI_2"], plate_folders, [[0.66,0.89],[5,100]])
    
   



