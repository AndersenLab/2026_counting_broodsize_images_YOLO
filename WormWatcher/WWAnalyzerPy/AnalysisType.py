"""
    AnalysisType unifies all of the information needed to run each analysis,
    includings its name, description, options, etc

    Much better than hunting through the program in order to update all kinds of different areas
    when adding a new analysis type

    If adding an analysis you may also need to edit:

    - PlateTreeView -> validateSelection

"""

import os
import json
from JsonTreeView import *





# Main class for centralizing information 
class AnalysisType:

    def __init__(self,name,desc,jfile,jwhitelist,jkeys,jdefaults):
        
        # Name, description and parameters for the analysis type
        self.name = name                # Name that appears in the dropdown menu
        self.desc = desc                # Description that appears under the dropdown menu
        self.jfile = jfile              # full file name of json file containing params
        self.jwhitelist = jwhitelist    # json sub-field(s), may be list 
        self.jkeys = jkeys              # Keys (individual parameter names within the whitelist, only current.json)
        self.jdefaults = jdefaults      # Default values in case this whitelist is not in the json file. 
                                        #   Should only be used with current.json, not config.json

        # Verify that default input keys match labels
        assert(len(self.jkeys) == len(jdefaults),"Internal error: the length of keys does not match length of defaults for Analysis type: " + name)

        # Load the JSON file if it exists, or create a fresh dict
        if os.path.exists(jfile):
            with open(jfile, 'r') as file:
                jtree = json.load(file)
        else:
            jtree = {}

        # Create the tree group if it doesn't exist
        if not self.jwhitelist in jtree:
            jtree[self.jwhitelist] = {}

        self.jvals = []

        # Attempt to load in the actual values for these keys from the file (e.g. last used values)
        # If not available (file doesn't exist OR key doesn't exist) fill in with defaults
        for (i,jkey) in enumerate(self.jkeys):
            if os.path.exists(jfile) and jkey in jtree[self.jwhitelist]:
                self.jvals.append(jtree[self.jwhitelist][jkey])
            else:
                self.jvals.append(jdefaults[i])
                jtree[self.jwhitelist][jkey] = jdefaults[i]

        # In case any default values (or the whole tree) were added, resave the JSON file
        with open(jfile, 'w') as file:
            jtree = json.dump(jtree,file,indent = 3)



    # Append this analysis type's data to a dict for saving in the current.json
    def appendToDict(self,jdict):

        # Create section if it doesn't already exist
        if not self.jwhitelist in jdict:
            jdict[self.jwhitelist] = {}

        # Write the keys, overwriting any already there of the same name, if any (shouldn't be)
        for (jkey,jdefault) in zip(self.jkeys,self.jdefaults):
            jdict[self.jwhitelist][jkey] = jdefault

        # Return the appended dict
        return jdict

    # Get the json_tree_view object filled in with this group's params from disk. Return the matching help string
    def setJsonTreeView(self,jtv_frame, text_width_mod = 100):

        #remove all old  tree views
        for i in jtv_frame.winfo_children():
            i.destroy()

        #replace with new tree view
        json_tree_view = JsonTreeView(jfullpath = self.jfile,
                                 frame = jtv_frame, whitelist = [self.jwhitelist],
                                 start_open=True, text_width_modifier = text_width_mod)     
        
        return self.desc
        


# Class containing all of the analysis types, use it to write their parameters all to the disk
class ListOfAnalysisTypes:

    def __init__(self):


        self.jfile = r"C:\WormWatcher\current.json" # note analysis types may have config.json, this is just jfile for current.json stuff
       
        self.directory = r"C:\WormWatcher"
        self.loadCurrentDirectory() # Try to Replace with file version if available
        self.generateAnalysisTypes() # generate and fill self.atypes
        #self.saveCurrentDotJson() # Re-save the Current.json file to disk. DOESN"T WORK RIGHT
                                  #Function should only be used at startup before user edits anything


    # Try to load the directory from json, if found overwrite the default
    def loadCurrentDirectory(self):
        if os.path.isfile(self.jfile):
            with open(self.jfile) as infile:
                jdict = json.load(infile)

            if 'directory' in jdict:
                self.directory = jdict['directory']

    def saveCurrentDirectory(self,tempdir):

        # Load current.json
        with open(self.jfile) as infile:
            jdict = json.load(infile)

        # Modify the directory
        jdict['directory'] = tempdir
        self.directory = tempdir

        # Save to JSON
        with open(self.jfile, 'w') as outfile:
            json.dump(jdict, outfile, indent = 3)


    # Save all the information that belongs in current.json, 
    #  using the values in append to dict. 
    #  CAREFUL - DOES NOT WORK CORRECTLY AND NOT USED BECAUSE THE JSONS ARE
    #  MOSTLY EDITED DIRECTLY WHEN USER IS INTERACTING WITH JsonTreeViews
    #  TO SAVE THE DIRECTORY JUST USE saveCurrentDirectory()
    #  **** SHOULD ONLY BE USED AT BOOTUP BEFORE USER EDITS ANYTHING ****
    def saveCurrentDotJson(self):

        # See it with the directory key
        jdict = {}
        jdict['directory'] = self.directory

        # Append all relevant param sets that belong in current.json (not config.json)
        for atype in self.atypes:
            if "current" in atype.jfile:
                jdict = atype.appendToDict(jdict)

        # Save the json to disk
        with open(self.jfile, 'w') as outfile:
            json.dump(jdict, outfile, indent = 3)

    # Find and display requested JsonTreeView depending on menu selection
    def updateJsonTreeViewDisplay(self, selection, jtv_frame, text_width_mod = 100):

        found = False
        for atype in self.atypes:
            if selection in atype.name:
                found = True
                return atype.setJsonTreeView(jtv_frame, text_width_mod)
               

        return "Internal error: updateJsonTreeViewDisplay failed to find atype matching: " + selection

    # Get a list of analysis types (names) that is fed to the optionlist dropdown
    def getAtypeNamesAsList(self):
        
        list_of_types = []
        for atype in self.atypes:
            list_of_types.append(atype.name)
        return list_of_types

    # Main initialization function - edit here to add/rearrange analysis types. Returns list of analysis types
    # The values are defaults, but they will automatically be overwritten if the AnalysisType constructor can find
    # A matiching value in jfile
    def generateAnalysisTypes(self):

        self.atypes = []

        # STEP 1 analysis
        name = "STEP 1: Image differences"
        desc = "Find all imaging sessions in the selected parent folder" + \
                " and run the pixel difference computation for them"
        jfile = r"C:\WormWatcher\config.json"
        jwhitelist = "analysis parameters"
        jkeys = []; # Leave as empty list to never auto-overwrite config.json. Only use for current.json
        jdefaults = [];
        self.atypes.append(AnalysisType(name,desc,jfile,jwhitelist,jkeys,jdefaults))

        # STEP 2 analysis (activity curves)
        name = "STEP 2: Activity over time"
        desc = "Condense the activity of each session into a single point and" + \
               " plot activity over time"
        jfile = r"C:\WormWatcher\current.json"
        jwhitelist = "activity curve options"
        jkeys = ['Median Filter (T/F)',"Session act. percentile (0=mean)"]; 
        jdefaults = [False,0];
        self.atypes.append(AnalysisType(name,desc,jfile,jwhitelist,jkeys,jdefaults))

        # STEP 3 analysis (manual inspeciton)
        name = "STEP 3: Inspect and label plates"
        desc =  "Interactively inspect activity in individual wells," +\
                    " specify the groupings of wells on a plate, or censor wells"
        jfile = r"C:\WormWatcher\current.json"
        jwhitelist = "None"
        jkeys = []; 
        jdefaults = [];
        self.atypes.append(AnalysisType(name,desc,jfile,jwhitelist,jkeys,jdefaults))       

        # STEP 4 analysis (auto-censor bad wells)
        name = "STEP 4: Auto-censor bad wells"
        desc = "Use the worm detection neural network to remove bad wells." +\
                " Does not change groups to `Censor`, rather creates a separate variable " +\
                "indicating whether each well passes or fails the quality checks."        
        jfile = r"C:\WormWatcher\current.json"
        jwhitelist = "quality filters"
        jkeys = ['Min days (d)','Initial WF (min)','Initial WF (max)','Check day for % remain','% remain (min)','% remain (max)','zero level','minimum activity peak']; 
        jdefaults = [25, 0.03, 0.15, 10, 50, 120, 10, 500];
        self.atypes.append(AnalysisType(name,desc,jfile,jwhitelist,jkeys,jdefaults))  

        # Plot Tx
        name = "plot Tx level"
        desc = "Plot the lifespan estimate (T99), healthspan estimate (T82) or any other T-level"
        jfile = r"C:\WormWatcher\current.json"
        jwhitelist = "plot Tx params"
        jkeys = ["Tx Level [0,1]","Font size","Font angle","Apply auto-censor (T/F)"]; 
        jdefaults = [0.99,20,0,True];
        self.atypes.append(AnalysisType(name,desc,jfile,jwhitelist,jkeys,jdefaults))  

        # Plot activity
        name = "plot total activity"
        desc = "Plot the total activity." +\
                " For activity on the last day, enter an out-of-bounds timepoint such as 100 days"
        jfile = r"C:\WormWatcher\current.json"
        jwhitelist = "plot Act params"
        jkeys = ["Total act. timepoint (d)","Font size","Font angle","Apply auto-censor (T/F)"]; 
        jdefaults = [100,20,0,True];
        self.atypes.append(AnalysisType(name,desc,jfile,jwhitelist,jkeys,jdefaults))  

        # Plot Escaping
        name = "plot escaping"
        desc = "Plot the percent remaining at a checkpoint day. REQUIRES THAT STEP 4 HAS ALREADY BEEN RUN ON THESE PLATES"
        jfile = r"C:\WormWatcher\current.json"
        jwhitelist = "plot Escaping params"
        jkeys = ["Font size","Font angle"]; 
        jdefaults = [20,0];
        self.atypes.append(AnalysisType(name,desc,jfile,jwhitelist,jkeys,jdefaults))  



        # Export CSV data
        name = "Export data"
        desc =  "Write selected data to csv file (saved in parent folder). STEPS 3 and 4 MUST BE RUN FIRST."
        jfile = r"C:\WormWatcher\current.json"
        jwhitelist = "export csv params"
        jkeys = ["Tx 1","Tx 2","Total act. timepoint (d)","Apply auto-censor (T/F)"]; 
        jdefaults = [0.99, 0.82, 100, True];
        self.atypes.append(AnalysisType(name,desc,jfile,jwhitelist,jkeys,jdefaults)) 

        # Swim STEP 1
        name = "Swimming Image Analysis"
        desc = "Find all imaging sessions in the selected " + \
                "parent folder and analyze them using contour finding"
        jfile = r"C:\WormWatcher\config.json"
        jwhitelist =  "worm segmentation" # List not supported right now sorry you can fix later Mateo
        jkeys = []; 
        jdefaults = [];
        self.atypes.append(AnalysisType(name,desc,jfile,jwhitelist,jkeys,jdefaults))  

        # Swim STEP 2
        name = "Swimming Postprocessing"
        desc = "Description:\Calculate the frequency of worms"
        jfile = r"C:\WormWatcher\current.json"
        jwhitelist = "plot swim data"
        jkeys = ['fps']; 
        jdefaults = [20];
        self.atypes.append(AnalysisType(name,desc,jfile,jwhitelist,jkeys,jdefaults))  

        # Convert old format data
        name = "Convert old format data"
        desc = "Convert data collected using version 1.85 or earlier " +\
                "of WormWatcher to the latest organizational format. Please back up data first."
        jfile = r"C:\WormWatcher\current.json"
        jwhitelist = "None"
        jkeys = []; 
        jdefaults = [];
        self.atypes.append(AnalysisType(name,desc,jfile,jwhitelist,jkeys,jdefaults))  