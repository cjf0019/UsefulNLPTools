# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 17:48:05 2017

@author: InfiniteJest
"""

import pandas as pd
import os
import pickle
import numpy as np
import csv

def read_dx_csv_chunk_topandas(file, outercolumn, nrows=None, skiprows=0):
    if skiprows > 0:
        df = pd.read_csv(file, nrows=nrows, skiprows=range(skiprows), dtype = object, usecols=[0, 1, 2, 3], #change the column indices to match those of 
                 encoding='ISO-8859-1', names=[])
    else:
        df = pd.read_csv(file, nrows=nrows, skiprows=range(skiprows), dtype = object, usecols=[0, 1, 2, 3], 
                     encoding='ISO-8859-1')
    df = df.drop_duplicates()        #drop any accidental extra entries
    df[''] = df[''].astype('float')     #make dates float values
    df = df.groupby(outercolumn, as_index=False).apply(pd.DataFrame.sort_values, '')      #sort each so it is in time order
    df = pd.DataFrame(df)
    df = df.reset_index()
    df = df.drop(['level_0', 'level_1'], axis=1)   
    df = df.drop_duplicates()
    return df               
    
          
def countfilelen(file):           #NOTE: ONLY DO IF ABSOLUTELY NECESSARY... TAKES FOREVER
    reader = csv.reader(open(file))
    lenfile = 0
    for i in reader:
        lenfile += 1
    return lenfile

         
class inputgenerator(object):
    """
    A class for generating time inputs. Functions as a generator to iterate through chunks of a csv file. 
    Also stores a dataframe of the code-to-integer correspondence (each code is assigned an integer for the time
    input files). 

    A description of the parameters...
    file: the name of the csv file
    nrows: the number of rows to load in at once in creating an individual input file (default 1,000,000)
    skiprows: skip rows in the csv file when reading in the data (default 0)
    maketimelist: if set to True, will generate a set of time input files as well, named "time"+outputfilename.
    For example, "timevisit.train" if outputfilename given as "visit"
    lenfile: specify, by rows, how much of the csv file will be read into the generator. 
    If none set, will automatically count the length of the csv file (default is None).
    *NOTE: avoid setting equal to none, as it takes a while to read the entire length of files these sizes
    """
    
    def __init__(self, file, nrows = 200000, skiprows=0, maketimelist=False, lenfile = None):
        self.file = file
        self.nrows = nrows
        self.skiprows = skiprows
        self.maketimelist = maketimelist
        self.icdintlist = pd.DataFrame(columns=['', 'Integer'])
        self.lenfile = lenfile

    def __iter__(self):      #iterate through chunks of the csv file as a generator
        if self.lenfile == None:
            filelen = countfilelen(self.file)
        else:
            filelen = self.lenfile
        skippedrows = self.skiprows
        while skippedrows <= filelen-self.nrows:
            df = read_dx_csv_chunk_topandas(self.file, nrows=self.nrows, skiprows=skippedrows)
            skippedrows += self.nrows
            yield df

    def generateinputs(self, df, dxinputfile, timeinputfile=None, updateicdlist=True):
        maketime = self.maketimelist
        #Convert codes to the corresponding integers
        codetoint = pd.DataFrame(df[''].value_counts())       #get list of unique codes in df
        codetoint = codetoint.reset_index()
        codetoint = codetoint.drop([''], axis=1)     #drop the value count column (not needed anymore)
        codetoint.columns=['']
        if updateicdlist == True:           #TURN OFF FOR GENERATING VALID AND TEST FILES... WANT TO KEEP SAME LIST AS THAT USED IN TRAINING
            self.icdintlist = pd.merge(self.icdintlist, codetoint, how='outer', on=[''])
            self.icdintlist['Integer'] = pd.Series(range(len(self.icdintlist)))       #make a corresponding integer per code, stored in 'CODE' column
        df = pd.merge(df, self.icdintlist, how='left', on=[''])      
        df.columns = ['', '', '', '', '', '']
        df = df[df[''] >= 0]
        df[''] = df[''].astype('int')
        print('The number of unique codes is '+str(len(df[''].value_counts())))    #count unique codes, used for GRU command
        df['lastdate'] = df['lastdate'].apply(lambda x: np.round(x).astype('int')) 

        #REMOVE PEOPLE THAT ONLY HAVE ONE TIME
        valuecounts = df[['', '']].drop_duplicates()[''].value_counts()
        valuecounts = valuecounts[valuecounts > 3].index.tolist()    #LIST OF WITH > 3 times... FOR FILTERING OTHERS OUT
        df = df[df[''].isin(valuecounts) == True]         

        #Make visit time input file
        if maketime == True:
            df2 = df[['', '']]
            df2 = pd.merge(df2, pd.DataFrame(df2.groupby([''], as_index=False)[''].min()), how='left', on='')     #get initial encounter date per 
            df2[''] = df2[''] - df2['']        #subtract the initial encounter date from all dates
            df2 = df2.drop('', axis=1)
            df2.columns = ['' if x=='' else x for x in df2.columns]
            df2 = df2.drop_duplicates()
            dfshift = df2.groupby('', as_index=False)[''].shift()  #group by person to generate subsequent encounter
            dfshift = dfshift.fillna(value=0).astype('int')                 #remove NaNs created from the shift
            df2[''] = pd.DataFrame(df2[''])-dfshift          #subtract the subsequent encouter from the current one (grouped by person)
            df2[''] = df2[''].astype('int')
            print('The mean duration between visits is ' + str(df2[df2[''] > 0][''].mean()))   #print mean duration of each timestamp, not including the first (0)
            timevaluecounts = df2[df2[''] > 0][''].mean()
            timelist = df2.groupby([''], as_index=False)[''].apply(list)
            timelist = timelist.tolist()
            with open(timeinputfile, 'wb') as g:
                pickle.dump(timelist, g, protocol=2)        
            
        #Group by time
        inputlist= df.groupby(['', ''], as_index=False)['].apply(list)    #
        inputlist= pd.DataFrame(inputlist)
        inputlist= inputlist.reset_index()
        inputlist.columns = ['', '', '']
        inputlist = inputlist.drop('inputlist', axis=1)
    
        #Group into list of times
        inputlist= inputlist.groupby([''], as_index=False)[''].apply(list)
        input = inputlist.tolist()    
        with open(inputfile, 'wb') as f:
            pickle.dump(input, f, protocol=2)      
        if maketime == True:
            return timevaluecounts        #returns the mean duration of each time... used in generatemultipleinputs to get total mean duration
        else:
            return


    def generatemultipleinputs(self, outputname):
        number = 0
        if self.maketimelist == True:
            fulltimelist = []
            for df in self:
                number += 1
                times = self.generateinputs(df, str(outputname)+str(number)+'.train', 'time'+str(outputname)+str(number)+'.train')
                fulltimelist.append(times)
            print("The mean duration of ALL times is " + str(sum(fulltimelist)/len(fulltimelist)))
            print("The total number of tokens across ALL times is " + str(len(self.codelist)))
        else:
            for df in self:
                number += 1
                self.generateinputs(df, str(outputname)+str(number)+'.train', 'time'+str(outputname)+str(number)+'.train')
            print("The total number of tokens across ALL times is " + str(len(self.codelist)))