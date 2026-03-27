"""
    ActivityCurves
    Anthony Fouad
    July 2019

    Class for storing, plotting, and writing over-time information for one plate

"""

# Standard imports
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import csv
import time


# Anthony's imports
import Calculations as calc
import csvio
import AnalysisPlots as ap
from ErrorNotifier import ErrorNotifier


class ActivityCurves : 
    
    # Arrays for the activity over time in each ROI
    Aroi_stim = np.empty(shape = (1,1))
    Aroi_unstim = np.empty(shape = (1,1))

    Aout_stim = np.empty(shape = (1,1))
    Aout_unstim = np.empty(shape = (1,1))

    # Arrays for the cumulative activity and Tx values
    Cum_Act = np.empty(shape = (1,1))
    Tx = np.empty(shape = (1,1))
    Tlevels = np.empty(shape = (1,1))

    # Timepoints (one per session)
    timepoints = list()
    timepoints_str = list()
    time = np.empty(shape = (1,1))

    # Bad sessions
    bad_sessions = np.empty(shape = (1,1))

    # Bad wells
    bad_wells = np.empty(shape = (1,1))

    # Blue light information
    blue_idx = np.empty(shape = (1,1))
    blue_excl_start = np.empty(shape = (1,1))

    # Size of the arrays
    N_wells = np.nan
    N_rows = np.nan
    N_cols = np.nan
    N_sessions = np.nan


    # initializer for all variables
    def initializeArrays(self,_N_rows,_N_cols,_N_sessions,platefullfolder,
                         _frame_jump,_activity_thresh, _annulus_frac, _outer_frac) :

        # Clear python lists.
        self.timepoints.clear()
        self.timepoints_str.clear()

        # Preallocate numpy arrays
        self.N_rows = _N_rows
        self.N_cols = _N_cols
        self.N_sessions = _N_sessions

        self.N_wells = self.N_rows * self.N_cols

        self.N_worm_mvmts = np.zeros(shape = (self.N_sessions,self.N_wells))*np.nan
        self.N_bulk_mvmts = np.zeros(shape = (self.N_sessions,self.N_wells))*np.nan

        self.Aroi_stim = np.ones(shape = (self.N_sessions,self.N_wells))*np.nan
        self.Aroi_unstim = np.ones(shape = (self.N_sessions,self.N_wells))*np.nan
        self.act_per_worm_frac = np.ones(shape = (self.N_sessions,self.N_wells))*np.nan
        self.gray_level = np.ones(shape =  (self.N_sessions,self.N_wells))*np.nan

        self.Aout_stim = np.ones(shape = (self.N_sessions,1))*np.nan
        self.Aout_unstim = np.ones(shape = (self.N_sessions,1))*np.nan

        self.worm_frac = np.zeros(shape = (self.N_sessions,self.N_wells))

        self.Cum_Act = np.zeros(shape = (self.N_sessions,self.N_wells))
        self.Tlevels = np.arange(0.00,1.00,0.01)
        self.Tx = np.zeros(shape = (self.Tlevels.shape[0],self.N_wells))
        self.WM_LS = np.zeros(shape = (1,self.N_wells))
        self.WM_HS = np.zeros(shape = (1,self.N_wells))

        self.bad_sessions = np.zeros(shape = (self.N_sessions,1)).astype(np.bool)
        self.valid_lifespan_shape = np.ones(shape = (1,self.N_wells)).astype(np.bool)

        self.time = np.zeros(shape = (self.N_sessions,1))
        self.blue_idx = np.zeros(shape = (self.N_sessions,1))
        self.blue_excl_start = np.zeros(shape = (self.N_sessions,1))

        # Read the groups from a file if it already exists, otherwise set to all No group
        self.readActivityGroups(platefullfolder)

        # Set default values for STEP 1 analysis parameters
        self.frame_jump = _frame_jump
        self.activity_thresh = _activity_thresh
        self.annulus_frac = _annulus_frac
        self.outer_frac = _outer_frac



    # Calculate cumulative activity curves
    def calculateCumulativeAct(self, stim_experiment) : 
        
        #Calculate Cummax(Cumsum), using either stim or unstim
        if stim_experiment:
            self.Cum_Act = np.nancumsum(self.Aroi_stim,axis=0)        # Cumulative activity
        else:
            self.Cum_Act = np.nancumsum(self.Aroi_unstim,axis=0)

        self.Cum_Act = np.fmax.accumulate(self.Cum_Act)      # Cumulative maximum, to prevent decreases.
        CA_size = (self.Cum_Act).size
        self.Cum_Act_Norm = self.Cum_Act - np.nanmin(self.Cum_Act,axis=0)  # Normalize to 1
        self.Cum_Act_Norm = self.Cum_Act_Norm / np.nanmax(self.Cum_Act_Norm,axis=0)

    # Calculate the Tx levels, with optional type used
    def calculateTx(self, Tlevels, Tx) : 
        for (level,i) in zip(Tlevels,range(0,Tlevels.shape[0])) : 
            for j in range(0,self.N_wells) :

                # Get current activity vector
                this_activity = self.Cum_Act_Norm[:,j]

                # Find index where it surpasses the requested level
                idx = np.argwhere(this_activity > level)

                # If we found index, and it is not the first index, interpolate between this LS and the next one
                if idx.size > 0 and idx[0] > 0: 

                    # Get bounds for linear function between this index's time and last index's
                    x0 = idx[0]-1;
                    x1 = idx[0];
                    t0 = self.time[x0]
                    t1 = self.time[x1]
                    A0 = this_activity[x0]
                    A1 = this_activity[x1]

                    # How close is level to the start/end activity levels?
                    frac = (level - A0) / (A1 - A0)
                    
                    # How much time elapsed between x0 and x1?
                    dt = t1 - t0

                    # Calculate interpolated time 
                    Tx[i,j] = t0 + frac * dt

                # If we found index, and it is the first index
                elif idx.size > 0 and idx[0] == 0:
                    Tx[i,j] = self.time[idx[0]];


                # If no index, never greater than level, report nan. This shouldn't happen...
                else :
                    Tx[i,j] = np.nan
                    

    def calculateWmLsHs(self):
        
        for n in range(self.N_wells):

            # Lifespan: Last day stimulated activity > 0
            v = self.Aroi_stim[:,n]
            idx_alive = np.where(v > 0)[0]

            if len(idx_alive) > 0:
                idx_ls = idx_alive[-1]
            else:
                idx_ls = 0

            self.WM_LS[0,n] = self.time[idx_ls][0]

            # Healthspan: Total # of days activity > 33% of maximum
            level = 0.25
            max_act = np.nanpercentile(v,90)
            idx_healthy = np.where(v > level*max_act)[0]
            days_per_idx = self.time[-1][0] / len(self.time)
            days_healthy = days_per_idx * len(idx_healthy)

            self.WM_HS[0,n] = days_healthy


    # Plot all data over time
    def plotActivityCurves(self,platefullfolder,is_first, apply_censors) :
        
        # Create the figure 
        fig=plt.figure(num=3, figsize = (10,6), dpi = 200)
        plt.clf()

        # Get max Ylimit
        maxystim = np.nanmax(self.Aroi_stim,axis=None)
        maxyunstim = np.nanmax(self.Aroi_unstim,axis=None)
        maxy = max((maxystim,maxyunstim))

        # Stimulated activity
        rkPlot(2,3,1,self.time,self.Aroi_stim,self.valid_lifespan_shape,self.bad_sessions)
        plt.ylabel('ROI Activity')
        plt.title('Stimulated')
        formatSubplot(maxy)

        rkPlot(2,3,4,self.time,self.Aout_stim,self.valid_lifespan_shape,self.bad_sessions)
        plt.ylabel("Non-ROI Activity")
        formatSubplot(None)

        # Unstimulated activity
        rkPlot(2,3,2,self.time,self.Aroi_unstim,self.valid_lifespan_shape,self.bad_sessions)
        plt.title('Unstimulated')
        formatSubplot(maxy)

        rkPlot(2,3,5,self.time,self.Aout_unstim,self.valid_lifespan_shape,self.bad_sessions)
        plt.xlabel('Time (d)')
        formatSubplot(None)

        # Cumulative activity curve
        rkPlot(2,3,3,self.time,self.Cum_Act_Norm,self.valid_lifespan_shape,self.bad_sessions)
        plt.title('Norm. Cumul. Activity')
        formatSubplot(None)

        # Prepare figure for showing and saving
        plt.tight_layout()

        # Maximize the figure ONLY on the first iteration, in case user wants to hide it.
        if is_first :
            mng = plt.get_current_fig_manager()
            #mng.window.state('zoomed') #works fine on Windows!

        # Plot the total activity by well, censoring censored wells
        #total_activity = self.Cum_Act[-1,:]
        total_activity = np.sum(self.act_per_worm_frac,axis=0);
        if apply_censors :
            total_activity[self.bad_wells.flatten()] = np.nan
        plt.subplot(2,3,6)
        ap.plotWellMapImage(total_activity,self.N_rows,self.N_cols,"Total Act per WF") 

        # Pause without the "pop-to-front" function that's very annoying
        fig.canvas.flush_events()   # update the plot and take care of window events (like resizing etc.)
        time.sleep(1)               # wait for next loop iteration

       # Save the figure to the folder
        head = os.path.split(platefullfolder)[1]
        fout = os.path.join(platefullfolder,"ActivityCurves_" + head + ".png")
        plt.savefig(fout,dpi = 200)

        # Close the figure. Prevents unpredictable behavior where sometimes it closes and sometimes it doesn't
        plt.close('all')

    # Read group information from disk. If it doesn't exist return array of "No group"s
    def readActivityGroups(self, platefullfolder):

        # Set CSV filename
        fout = os.path.join(platefullfolder,"ActivityCurves_" + os.path.split(platefullfolder)[1] + ".csv")

        # Exit immediately if file does not exist
        if not os.path.exists(fout):
            self.group_name_array = ["No group" for i in range(self.N_wells)]
            self.group_name_array = np.asarray([self.group_name_array])
            return

        # Attempt to read the groups from the file, if they exist
        gna = csvio.readArrayFromCsv(fout,"<Group name>")   # Extract the desired session in this file
        if "empty" in gna:                                         # The file has no group assigned; assign as no group for each
            self.group_name_array = ["No group" for i in range(self.N_wells)]
            self.group_name_array = np.asarray([self.group_name_array])
        else:
            self.group_name_array = gna
        return




    # Print activity curve data to disk
    def printActivityCurves(self, platefullfolder, is_stim_expt, Err) :
        
        # Set CSV filename
        fout = os.path.join(platefullfolder,"ActivityCurves_" + os.path.split(platefullfolder)[1] + ".csv")

        # Try to open the CSV file
        try:
            outfile = open(fout, "w") # or "a+", whatever you need
        except IOError:
            Err.notify("Abort","Could not open file:\n" + fout + "\nPlease close Excel and try again!")

        # Open the CSV file
        with outfile :
           
            # Print ROI numbers
            roinums = np.arange(1,self.N_wells+1)
            roinums = np.reshape(roinums,(1,self.N_wells))
            csvio.printArrayToCsv(outfile,",,<ROI Number>\n,,",roinums)

            # Print bad wells
            # csvio.printArrayToCsv(outfile, ",,<Valid lifespan shape>\n,,", 1*self.valid_lifespan_shape)

            # Cumulative, stimulated and unstimulated activity,
            csvio.printArrayToCsv(outfile, "<Day>,<Bad 0/1>,<Stimulated activity>", np.concatenate((self.time,self.bad_sessions,self.Aroi_stim), axis = 1)) 
            csvio.printArrayToCsv(outfile, "<Day>,<Bad 0/1>,<Unstimulated activity>", np.concatenate((self.time,self.bad_sessions,self.Aroi_unstim), axis = 1)) 
            csvio.printArrayToCsv(outfile, "<Day>,<Bad 0/1>,<Cumulative activity>", np.concatenate((self.time,self.bad_sessions,self.Cum_Act), axis = 1)) 
            csvio.printArrayToCsv(outfile, "<Day>,<Bad 0/1>,<Activity per worm fraction>", np.concatenate((self.time,self.bad_sessions,self.act_per_worm_frac), axis = 1))  

            csvio.printArrayToCsv(outfile, "<Day>,<Bad 0/1>,<Non-ROI stimulated activity>", np.concatenate((self.time,self.bad_sessions,self.Aout_stim), axis = 1))  
            csvio.printArrayToCsv(outfile, "<Day>,<Bad 0/1>,<Non-ROI unstimulated activity>", np.concatenate((self.time,self.bad_sessions,self.Aout_unstim), axis = 1))  
            
            csvio.printArrayToCsv(outfile, "<Day>,<Bad 0/1>,<N worm movements>", np.concatenate((self.time,self.bad_sessions,self.N_worm_mvmts), axis = 1))  
            csvio.printArrayToCsv(outfile, "<Day>,<Bad 0/1>,<N bulk movements>", np.concatenate((self.time,self.bad_sessions,self.N_bulk_mvmts), axis = 1))  
            csvio.printArrayToCsv(outfile, "<Day>,<Bad 0/1>,<Worm Fraction>", np.concatenate((self.time,self.bad_sessions,self.worm_frac), axis = 1))  

            # Print mean gray level
            csvio.printArrayToCsv(outfile, "<Day>,<Bad 0/1>,<Mean Gray Level>", np.concatenate((self.time,self.bad_sessions,self.gray_level), axis = 1))  

            # Print wormotel defined LS and HS
            csvio.printArrayToCsv(outfile,",,<WorMotel Lifespan>\n,,",self.WM_LS)
            csvio.printArrayToCsv(outfile,",,<WorMotel Healthspan>\n,,",self.WM_HS)

            # Print Tx levels
            tlvl = np.reshape(self.Tlevels,(self.Tlevels.shape[0],1))
            to_print = np.concatenate((tlvl,self.Tx), axis = 1)
            csvio.printArrayToCsv(outfile,",<x>,<Tx>,",to_print,1)

            # Print the groups that were either read from an existing file or else set to all "No group"
            csvio.printArrayToCsv(outfile,",,<Group name>\n,,",self.group_name_array)

            # Print the STEP 1 analysis parameters
            step1arr = np.array([[self.N_rows,self.N_cols,self.frame_jump,self.activity_thresh,self.annulus_frac,self.outer_frac]])
            csvio.printArrayToCsv(outfile,",,<N_rows>,<N_cols>,<frame_jump>,<activity_thresh>,<annulus_frac>,<outer_frac>",step1arr,2)


            # Print end of file
            csvio.printEndToCsv(outfile)
            
    
    # Identify bad sessions using the non-ROI portion of the image
    def markBadSessions(self,apply_censors) :
        
        # Get the maximum Aout (whether stim or unstim, doesn't matter) for each day
        bad_stim = calc.findOutliers(self.Aout_stim,5)
        bad_unstim = calc.findOutliers(self.Aout_unstim,5)
        
        # Find any sessions > 10x the median out, ensuring the shape is (60,1)
        self.bad_sessions = np.logical_or(bad_stim,bad_unstim)

        # Rather than marking bad sessions, just median filter stuff

        # Set bad sessions to nan in all arrays
        if apply_censors :
            self.Aroi_stim[self.bad_sessions.flatten(),:] = np.nan
            self.Aann_stim[self.bad_sessions.flatten(),:] = np.nan
            self.Aout_stim[self.bad_sessions.flatten(),:] = np.nan
            self.Aroi_unstim[self.bad_sessions.flatten(),:] = np.nan
            self.Aann_unstim[self.bad_sessions.flatten(),:] = np.nan
            self.Aout_unstim[self.bad_sessions.flatten(),:] = np.nan

    # Median filter to remove high-noise outliers
    def medianFilterData(self) :

        kernel_size = (5,1)

        if self.Aroi_stim.shape[0] >= 10:

            # Apply median filters, save to temp variable - ROI only, not Aout.
            Aroi_stim = scipy.signal.medfilt2d(self.Aroi_stim, kernel_size)
            Aroi_unstim = scipy.signal.medfilt2d(self.Aroi_unstim, kernel_size)
            #Aout_stim = scipy.signal.medfilt2d(self.Aout_stim, kernel_size)
            #Aout_unstim = scipy.signal.medfilt2d(self.Aout_unstim, kernel_size)

            # Take the minimum of the raw value and the median filtered value
            self.Aroi_stim = np.min(np.dstack((self.Aroi_stim,Aroi_stim)),axis=2)
            self.Aroi_unstim = np.min(np.dstack((self.Aroi_unstim,Aroi_unstim)),axis=2)
            #self.Aout_stim = np.min(np.dstack((self.Aout_stim,Aout_stim)),axis=2)
            #self.Aout_unstim = np.min(np.dstack((self.Aout_unstim,Aout_unstim)),axis=2)




    # Identify bad wells due to high annulus activity
    def markBadWellsByShape(self,apply_censors) : 

        # There must be at least 10 datapoints that span at least 20 days
        valid_span = self.time.size > 10 and self.time[-1] >= 20
        self.valid_lifespan_shape[:] = valid_span

        if not valid_span:
            return

        # Get average activity in days 1-5, this should be the peak
        i0 = np.argwhere(self.time > 1)[0][0]
        i1 = np.argwhere(self.time > 5)[0][0]
        early_A = np.nanmean(self.Aroi_stim[i0:i1,:],axis=0)
        
        # Get average activity in the last few days, which should be MUCH lower... ideally reaching 0
        i0 = int(0.8*self.time.size)
        i1 = int(0.9*self.time.size)
        late_A = np.nanmean(self.Aroi_stim[i0:i1,:],axis=0)

        # Get ratio. Note that if activity on days 18-20 (noise floor) is negative, could be negative
        ratio = early_A/late_A
        self.valid_lifespan_shape = ratio > 3
        self.valid_lifespan_shape = np.reshape(self.valid_lifespan_shape.astype(np.bool),(1,self.N_wells))

        # Apply censors if requested
        if apply_censors :

            bad_wells = np.invert(self.valid_lifespan_shape)

            self.Aroi_stim[:,bad_wells.flatten()] = np.nan
            self.Aann_stim[:,bad_wells.flatten()] = np.nan
            self.Aroi_unstim[:,bad_wells.flatten()] = np.nan
            self.Aann_unstim[:,bad_wells.flatten()] = np.nan






""" NON-MEMBER HELPER FUNCTIONS """

# Format a plot with censored/uncensored shown
def rkPlot(spr,spc,spi,t,y,valid_lifespan_shape, bad_sessions) :
    plt.subplot(spr,spc,spi)

    # First, suppress bad sessions (individual data points)
    # y[bad_sessions.flatten(),:] = np.nan

    # Plot the good data curves as black and bad as red
    if y.shape[1] == valid_lifespan_shape.shape[1] : 
        ygood = y[:,valid_lifespan_shape.flatten()]
        ybad = y[:,np.invert(valid_lifespan_shape).flatten()]

        # Add the plots
        if ybad.shape[1] > 0 :
            plt.plot(t,ybad,'r--',linewidth=0.5)

        if ygood.shape[1] > 0 :
            plt.plot(t,ygood,'k',linewidth=1.0)

        # Plot the average of good plots
        yavg = np.nanmean(ygood,axis=1)
        plt.plot(t,yavg,'g',linewidth=1.2)


    #unless size doesn't match (e.g. outer-activity), then just plot x and y
    else : 
        plt.plot(t,y,'k',linewidth=1.0)

        # However, show the censored points as red dots
        ycensor = y[bad_sessions]
        tcensor = t[bad_sessions]
        plt.plot(tcensor,ycensor,'or')


# Format activity curves' subplot
def formatSubplot(maxy) :

    if maxy != None and not np.isnan(maxy) :
        plt.ylim(0,maxy)

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.tick_params(direction='in',labelsize =8)
    plt.locator_params(axis='y', nbins=3)


# Find the frames whose differences are affected by blue light, if any
def findBlueFrames(Aout,Aroi,jump, session_full_name, Err = ErrorNotifier()) :

    Nf = Aout.shape[0]

    # Find points whose raw activity differences are > 5 times the median + 1000 (offsets all-zero activitives)
    Amedian = np.percentile(Aout,75)
    is_blue = Aout > 50 * (Amedian + 100) + 1000
    all_blue_idx = np.where(is_blue)[0]

    # Exclude blue frames close to the beginning or end, but display warning if any
    min_idx = 0.3*Nf
    max_idx = 0.7*Nf

    if len(all_blue_idx) >= 1 and (all_blue_idx[0] < min_idx or all_blue_idx[-1] > max_idx):
        Err.notify("PrintRed","Warning analyzing session:\n\t" + session_full_name + 
                   "\n\tFound invalid blue frames: " + str(all_blue_idx) + 
                   "\n\tPlease check that there are no unsual blue activations or bulk movements in these images")
    all_blue_idx = all_blue_idx[(all_blue_idx > min_idx) & (all_blue_idx < max_idx)]

    # Valid blue frame is (1) Bright enough, (2) only big changes within the middle of the series, otherwise nan for no blue idx
    if len(all_blue_idx) >= 1:

        # First and last blue frame in case multiple
        idx_blue_last = all_blue_idx[-1] 
        idx_blue_first = all_blue_idx[0]

        blue_idx = idx_blue_first + jump # For testing, lets exclude + jump # used to be idx_blue_last+jump
        blue_excl_start = idx_blue_first - 1
    else :
        blue_idx = np.nan
        blue_excl_start = np.nan

    # Remove all blue frames from Aroi
    Aroi_removed_blue = Aroi
    Aroi_removed_blue[all_blue_idx,:] = 0

    return (blue_idx,blue_excl_start,Aroi_removed_blue)