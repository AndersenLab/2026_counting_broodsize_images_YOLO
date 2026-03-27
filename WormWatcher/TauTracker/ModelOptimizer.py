# ModelOptimizer.py
# Optimize parameters for AnalyzeClustering.py

from OpenCV_Helpers import *
import os
from MultiWellAnalysis import * 
import shutil

# User params
jfile = "AnalyzeClustering.json"
fin = "Aggregation Test Set_raw.xlsx"


# Params to sweep
params = {}
params["object conf thresh"] = [0.01, 0.05, 0.10, 0.25]
params["pix conf thresh"] = [0.05, 0.15, 0.35, 0.45, 0.55]
params["overlap thresh"] = [0.25, 0.45, 0.60, 0.75, 0.90]


for ocf in params["object conf thresh"]:
    for pcf in params["pix conf thresh"]:
        for ot in params["overlap thresh"]:

            # Save whatever set of parameters to json file
            jdata = loadJSON(jfile)
            jdata["worm"]["object conf thresh"] = ocf
            jdata["worm"]["pix conf thresh"] = pcf
            jdata["worm"]["overlap thresh"] = ot
            saveJSON(jfile,jdata)

            # Generate the output file name 
            pout = os.path.join(jdata["plate"]["paths"]["dark field"],"_Analysis")
            fout = fin.replace(".xlsx","%0.3f_%0.3f_%0.3f.xlsx" % (ocf,pcf,ot))

            # Execute analysis 
            print("executing analysis with output file: %s" % fout)
            MultiWellAnalysis.skip_menus = True # 
            MWA = MultiWellAnalysis("AnalyzeClustering.json")
            MWA.analyzeImageSeries()

            # Copy the file for this parameter combo
            shutil.copyfile(os.path.join(pout,fin),os.path.join(pout,fout))



