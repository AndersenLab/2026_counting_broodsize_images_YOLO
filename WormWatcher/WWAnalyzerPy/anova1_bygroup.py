"""
    Equivalent to my matlab function anova1_bygroup.m

    Computes the pairwise comparisons between data in a np array:
        Rows = measurements
        Cols = groups

        Fill uneven row counts for each column with nan (e.g. create a nan array 
        and fill it in with arbitrary data for each column/group)

    Returns a list of arrays with each element in the list is a 1-row array. 

    adapted from:
    https://pythonhealthcare.org/2018/04/13/55-statistics-multi-comparison-with-tukeys-test-and-the-holm-bonferroni-method/

    See also:
    https://www.pythonfordatascience.org/anova-python/
    https://reneshbedre.github.io/blog/anova.html
    https://www.kite.com/python/answers/how-to-create-pandas-dataframe-from-a-numpy-array-in-python
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.stats.multicomp as mc

def anova1_bygroup(arr,group_names,do_print = False,return_with_headers = True):

    # Ensure that number of group_names = number of columns
    # Ensure dat ais valid and won't crash due to DOF
    #       TODO
    #################################

    # format data as DataFrame

    df = pd.DataFrame()

    for (n,name) in enumerate(group_names):
        df[name] = arr[:,n]


    # Stack the data (and rename columns):
    stacked_data = df.stack().reset_index()
    stacked_data = stacked_data.rename(columns={'level_0': 'id',
                                                'level_1': 'group',
                                                0:'value'})


    # Set up the data for comparison (creates a specialised object)
    MultiComp = mc.MultiComparison(stacked_data['value'],stacked_data['group'])

    # Perform multiple comparison (Holm-Bonferroni)
    comp = MultiComp.allpairtest(stats.ttest_ind, method='Holm')

    # Print if requested
    if do_print:
        print(comp[0])

    # If returning with headers, add the headers to the array before returning
    ret_arr = comp[0].data

    # Extract and return the corrected P values and reject
    return ret_arr

# Sample usage
if __name__ == "__main__":


    # Sample data with known differences
    data = np.random.rand(50,5)
    data[:,0] = np.random.normal(5, 10, 50)     # a: Mean 5
    data[:,1] = np.random.normal(-5, 10, 50)    # b: Mean -5
    data[:,2] = np.random.normal(-5, 1, 50)     # c: Mean -1 tight std
    data[:,3] = np.random.normal(6, 10, 50)     # d: Not different?
    data[:,4] = np.nan                          # e: No data, gets auto excluded (nan always excluded)



    group_names = ['a','b','c','d','e']
    do_print = True
    anova1_bygroup(data,group_names,do_print)