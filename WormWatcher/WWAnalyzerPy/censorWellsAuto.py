"""
    Censor wells by marking their group as "Censor" 
    
    * Does not save to file, returns decisions as numpy array

    Return the quality metrics
    [Row #0: Experiment length (days)]
    [Row #1: initial worm fraction ]
    [Row #2: % remain at day X ] 
    [Row #3: Peak activity ]



    Also return the reason for censoring as an array, organized as follows:

                N columns for N wells
    [Row #0: (overall censored):   1 = censored for any reason, 0 = default                  ]
    [Row #1: (min days):  1 = too few days, 0 = default, all other reasons left as nan                      ]
    [Row #2: (Initial WF low): 1/0 but nan by default   ]
    [Row #3: (Initial WF high): 1/0 but nan by default   ]    
    [Row #4: (% remain d10 low): 1/0  but nan by default                          ]
    [Row #5: (% remain d10 high): 1/0  but nan by default                             ]
    [Row #6: (Activity reaches 0): 1/0 but nan by deafult                          ]
    [Row #7: (Peak activity > thresh): 1/0 but nan by default                          ]


"""


from verifyDirlist import *
import pandas as pd
import numpy as np
import numpy.matlib as mlib
from csvio import *
import os
import scipy.signal
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from ErrorNotifier import *


def updateCsvFile(AC_file,qual_vals,reason,do_update):

    if do_update:

        # Quality values 
        qual_labels = np.array([["N DAYS"],["INIT WF"],["PCT REMAIN"],["PEAK ACT"]])   # Having text labels will confuse auto import later
        qual_vals_labeled = np.concatenate((qual_labels,qual_vals),axis=1)
        replaceVariableInCsv (AC_file,",<Category>,<Quality values>",qual_vals_labeled,
                             col_offset = 1, alternate_delete_var = "<Quality values>")

        # Censor decision
        reason_labels = np.array([["OVERALL"],["MIN DAYS"],["LO INIT"],["HI INIT"],["LO REMAIN"],["HI REMAIN"],["NOT ZERO"],["LO PEAK"]])
        reason_labeled = np.concatenate((reason_labels,reason),axis=1)
        replaceVariableInCsv (AC_file,",<Category>,<Auto censor>",reason_labeled, 
                              col_offset = 1, alternate_delete_var = "<Auto censor>")



def censorWellsAuto(AC_file, Err,
                    update_csv = False,
                    min_days = 25, 
                    wf_init = [0.03, 0.10],
                    pct_remain_check_day = 10, 
                    pct_remain_lims = [50,120],
                    zero_level = 10,
                    min_peak_act = 500):

    # Get the short file name for notifications
    _ , AC_short = os.path.split(AC_file)

    # Load the worm fraction and stimulated activity from the plate
    worm_frac = readArrayFromCsv(AC_file,"<Worm Fraction>")
    stim_act = readArrayFromCsv(AC_file,"<Stimulated activity>")
    unstim_act = readArrayFromCsv(AC_file,"<Unstimulated activity>")
    t = readArrayFromCsv(AC_file,"<Day>")

    # Detect if this was an unstimulated experiment - if unstimulated activity is greater than stimulated
    if np.nanmean(stim_act - unstim_act) < 0:
        stim_act = unstim_act

    # Preallocate variables to keep track of the reason for censoring
    N_wells = worm_frac.shape[1]
    reason = np.ones((8,N_wells)) * np.nan
    reason[0:2,:] = 0;
    qual_vals = np.ones((4,N_wells)) * np.nan
    qual_vals[0,:] = np.max(t) # Pre-seed with max t in case we exit here due to error

    """ FILTER ROW 1: Enough days (sessions) to analyze """

    if  np.max(t) < min_days:
        reason[0:2,:] = 1
        Err.notify("PrintRed","Plate " + AC_short + " was skipped because it is only " + str(max(t)) + "days long.")
        updateCsvFile(AC_file,qual_vals,reason,update_csv)
        return reason, qual_vals

    # Warn user about very short experiments if they have tried to request it
    N_sessions = len(t)
    if N_sessions < 5:
        msg = "Error auto-censoring plate:\t" + AC_short + "\n\tThis experiment has only " + str(N_sessions)\
              + " sessions.\n\tIncrease the min_days filter or manually exclude this plate and try again."
        Err.notify("PrintRed",msg)
        updateCsvFile(AC_file,qual_vals,reason,update_csv)
        return reason, qual_vals


    """ FILTER ROW 2 & 3: Initial worm count in bounds """

    # Worm fraction based rules - median filter the wf
    kernel_size = (5,1)
    worm_frac = scipy.signal.medfilt2d(worm_frac, kernel_size)
    worm_frac_avg = np.mean(worm_frac, axis=1)

    # Initial worm fraction is the highest level in the first roughly 3 days (sessions 0 through 5)
    worm_frac_init = np.max(worm_frac[0:6,:],axis=0);
  
    # check initial too high or too low
    reason[2,:] = worm_frac_init < wf_init[0] 
    reason[3,:] = worm_frac_init > wf_init[1] 


    """ Filter ROW 4 & 5: Percentage remaining at day 10, e.g. within [50,120] """

    # Enforce that check day is later than minimum 
    if np.max(t) < pct_remain_check_day:
        msg = "Error auto-censoring plate:\t" + AC_short + "\n\tThe requested check day (" + str(pct_remain_check_day)\
              + ") is less than the requested minimum experiment time (" + str(min_days) +  ") ."\
              + "\n\tIncrease the min_days filter, decrease the check day, or manually exclude this plate"
        Err.notify("PrintRed",msg)
        updateCsvFile(AC_file, qual_vals, reason, update_csv)
        return reason, qual_vals


    # Percent remaining at day N, which must be less than or equal to the length of this plate's data
    idx = np.where(t>=pct_remain_check_day)
    idx = idx[0][0] # Above requirement that check day is later than minimum required here to prevent crash
    worm_frac_test = worm_frac[idx,:]
    pct_remain = 100.0 * worm_frac_test / worm_frac_init;

    reason[4,:] = pct_remain < pct_remain_lims[0] 
    reason[5,:] = pct_remain > pct_remain_lims[1] # FUDR failure likely


    """ FILTER ROW 6: FUDR failure """

    # Row 6: Activity in each well must decrease to zero (individually)
    reason[6,:] = np.nanmean(stim_act[-5:,:],axis=0) > zero_level

    """ FILTER ROW 7: Low activity """
    peak_act = np.nanpercentile(stim_act, 95, axis=0)
    reason[7,:] = peak_act < min_peak_act;

    """ Back to Filter row 0: Overall good (1) or bad (0) """
    reason[0,:] = np.nanmax(reason,axis=0)

    """ Assemble the quality values """
    qual_vals = np.repeat(np.max(t),N_wells) # Remake qual vals instead of nans
    qual_vals = np.stack((qual_vals,worm_frac_init,pct_remain,peak_act),axis = 0)

    """ Update the CSV file if requested (should be off by default) """
    updateCsvFile(AC_file, qual_vals, reason, update_csv)

    return reason, qual_vals


def displayQualityResults(Err,qual_vals,reason,min_days,wf_init,pct_remain_lims,min_peak_act):

    # Print the fraction of wells culled by each filter to scroll text and console
    frac_censor = np.nanmean(reason,axis = 1)
    N_wells = qual_vals.shape[1]
    N_plates = N_wells / 24

    msg = "***   Quality checked " + '%d'   % N_plates + " plates ("+ str(N_wells) + " wells)    ***"
    msg+="\nOverall censored:    " + '%.2f' % frac_censor[0]
    msg+="\nExpt too short:      " + '%.2f' % frac_censor[1]
    msg+="\nInit WF too low:     " + '%.2f' % frac_censor[2]
    msg+="\nInit WF too high:    " + '%.2f' % frac_censor[3]
    msg+="\n% remain too low:    " + '%.2f' % frac_censor[4]
    msg+="\n% remain too high:   " + '%.2f' % frac_censor[5]
    msg+="\nDoes not reach 0:    " + '%.2f' % frac_censor[6]
    msg+="\nLow activity:        " + '%.2f' % frac_censor[7]

    Err.notify("PrintBlue",msg)

    # Error notify if all nan
    if np.all(np.isnan(qual_vals[1,:])):
        messagebox.showerror("All data censored","All plates in this selection have too few days. If possible, try reducing the minimum days filter, but be careful that minimum days is greater than or equal to check day.")
        return

    # Histogram of experiment length (d)
    plt.figure(1,figsize = [10,7])
    plt.subplot(2,3,1)
    plt.hist(qual_vals[0,:])
    plt.xlabel("Expt length (d)")
    plt.ylabel("N Wells")
    plt.axvspan(min_days, np.nanmax(qual_vals[0,:]), alpha=0.2, color='green')

    # Histogram of initial worm frac
    plt.subplot(2,3,2)
    plt.hist(qual_vals[1,:])
    plt.xlabel("Initial Worm Fraction")
    plt.axvspan(wf_init[0], wf_init[1], alpha=0.2, color='green')
    

    # Histogram of percent remain at day x
    plt.subplot(2,3,3)
    plt.hist(qual_vals[2,:],bins = range(0,200,10))
    plt.xlabel("% Remaining at checkpoint")
    plt.axvspan(pct_remain_lims[0], pct_remain_lims[1], alpha=0.2, color='green')


    # Histogram of peak activity
    plt.subplot(2,3,4)
    plt.hist(qual_vals[3,:])
    plt.xlabel("Peak activity")
    plt.ylabel("N Wells")
    plt.axvspan(min_peak_act, np.nanmax(qual_vals[3,:]), alpha=0.2, color='green')


    # Text table of results
    font = FontProperties()
    font.set_family('monospace')
    font.set_name('Courier new')
    

    plt.subplot(2,3,5)
    plt.text(0,0,msg,fontproperties=font,fontsize = 'large')
    plt.axis('off')


    plt.show()
    plt.pause(.001)



# Test script
if __name__ == "__main__":

    AC_file = r"C:\Dropbox\WormWatcher (shared data)\RNAi Project\Rescreens\_Analysis\RS_63\ActivityCurves_RS_63.csv"
    Err = ErrorNotifier()
    update_csv = True
    reason, qual_vals = censorWellsAuto(AC_file,Err,update_csv);
    displayQualityResults(Err,qual_vals,reason,25,[0.03, 0.15],[50,120],500)
