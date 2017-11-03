# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 17:46:02 2017

@author: InfiniteJest
"""


class PickleJar(object):
    """
    Create a generator of pickles/pickled objects to iterate through from a file list. The file
    list can be generated from '.py'. For each input file, can also have
    corresponding time file list if predicting time. Must also have corresponding ".test" and 
    ".valid" files for each file. This class allows multiple input files
    to be sifted through, as a python generator, in case there is not enough memory to load the entire 
    set together. The files are loaded one at a time, then unloaded. Contains ability to shuffle
    through the files, so as to increase randomness/accuracy. Can add files to the jar with the
    'pickleandadd' function.
    
    Explanation of the inputs:
    pickledfileseqlist, pickledfilelabellist, and pickledfiletimelist are the names of the files (their full paths)
    to be loaded into a neural network, for the sequences (x input), the labels (y output), and times (d output).
    Set train, test, and valid = 1 individually for whichever ones you want to iterate over.
    """
    def __init__(self, pickledfileseqlist, pickledfilelabellist, pickledfiletimelist=None, train=1, test=0, valid=0, predict_time=False):
        self.pickledfileseqlist = pickledfileseqlist
        self.pickledfilelabellist = pickledfilelabellist
        self.pickledfiletimelist = pickledfiletimelist
        self.train = train
        self.test = test
        self.valid = valid
        self.predict_time = predict_time
        self.seqlist = []
        self.labellist = []
        self.timelist = []
        for file in self.pickledfileseqlist:
            self.seqlist.append(str(file))
        for file in self.pickledfilelabellist:
            self.labellist.append(str(file))
        if self.predict_time == True:
            for file in self.pickledfiletimelist:
                self.timelist.append(str(file))
        
    
    
    #used for shuffling the order of the files to be read in, to increase randomness
    def pickleshuffle(self):
        if len(self.timelist > 0):
            randomzipped = random.shuffle(list(zip(self.seqlist, self.labellist, self.timelist)))
            self.seqlist, self.labellist, self.timelist = zip(*randomzipped)
            return self.seqlist, self.labellist, self.timelist
        else:
            randomzipped = random.shuffle(list(zip(self.seqlist, self.labellist)))
            self.seqlist, self.labellist= zip(*randomzipped)
            return self.seqlist, self.labellist
    


 #generator that iterates over multiple files , loading them in for use in doctorAI. Will load in .train,
 #.test, .valid, or both .test and .valid depending on the settings
    def __iter__(self):      
        if self.train == 1: 
            if self.predict_time == True:
                for seqfile, labelfile, timefile in zip(self.seqlist, self.labellist, self.timelist):
                    train_set = load_traindata(seqfile, labelfile, timefile)
                    yield train_set
            else:
                for seqfile, labelfile in zip(self.seqlist, self.labellist):
                    train_set = load_traindata(seqfile, labelfile, self.timelist)
                    yield train_set
        elif self.test == 1 and self.valid == 1:
            if self.predict_time == True:
                for seqfile, labelfile, timefile in zip(self.seqlist, self.labellist, self.timelist):
                    test_set = load_testdata(seqfile, labelfile, timefile)
                    valid_set = load_validdata(seqfile, labelfile, timefile)
                    yield test_set, valid_set
            else:
                for seqfile, labelfile in zip(self.seqlist, self.labellist):
                    test_set = load_testdata(seqfile, labelfile, self.timelist)
                    valid_set = load_validdata(seqfile, labelfile, self.timelist)
                    yield test_set, valid_set
        elif self.test == 1:
            if self.predict_time == True:
                for seqfile, labelfile, timefile in zip(self.seqlist, self.labellist, self.timelist):
                    test_set = load_testdata(seqfile, labelfile, timefile)
                    yield test_set
            else:
                for seqfile, labelfile in zip(self.seqlist, self.labellist):
                    test_set = load_testdata(seqfile, labelfile, self.timelist)
                    yield test_set
        elif self.valid == 1:
            if self.predict_time == True:
                for seqfile, labelfile, timefile in zip(self.seqlist, self.labellist, self.timelist):
                    valid_set = load_validdata(seqfile, labelfile, timefile)
                    yield valid_set
            else:
                for seqfile, labelfile in zip(self.seqlist, self.labellist):
                    valid_set = load_traindata(seqfile, labelfile, self.timelist)
                    yield valid_set

    pickledtrainseqfiles = []
    pickledtrainlabelfiles = []
    pickledtestseqfiles = []
    pickledtestlabelfiles = []
    pickledvalidseqfiles = []
    pickledvalidlabelfiles = []
    
# Create the lists of file names and store in memory friendly iterator.
# The lists come from the given file names inputted in the command line.
# For example, if you want to access the list of inputs "visit#.train", where
# the "#" indicates an index starting from 1, you would type in "\path\visit"
# into the command line in the train file space.

    if len(timeFile) > 0:
        pickledtraintimefiles = []
        for i in [j+1 for j in range(int(n_train_files))]:
            pickledtrainseqfiles.append(str(seqFile)+str(i))
            pickledtrainlabelfiles.append(str(labelFile)+str(i))
            pickledtraintimefiles.append(str(timeFile)+str(i))
        trainpickles = PickleJar(pickledtrainseqfiles, pickledtrainlabelfiles, 
                                 pickledfiletimelist=pickledtraintimefiles, predict_time=True)

    else:
        for i in [j+1 for j in range(int(n_train_files))]:
            pickledtrainseqfiles.append(str(seqFile)+str(i))
            pickledtrainlabelfiles.append(str(labelFile)+str(i))
        trainpickles = PickleJar(pickledtrainseqfiles, pickledtrainlabelfiles)

    if len(timeFile) > 0:
        pickledtesttimefiles = []
        for i in [j+1 for j in range(int(n_test_files))]:
            pickledtestseqfiles.append(str(seqFile)+str(i))
            pickledtestlabelfiles.append(str(labelFile)+str(i))
            pickledtesttimefiles.append(str(timeFile)+str(i))
        testpickles = PickleJar(pickledtestseqfiles, pickledtestlabelfiles, 
                                 pickledfiletimelist=pickledtesttimefiles, train=0, valid=0, test=1, predict_time=True)

    else:
        for i in [j+1 for j in range(int(n_test_files))]:
            pickledtestseqfiles.append(str(seqFile)+str(i))
            pickledtestlabelfiles.append(str(labelFile)+str(i))
        testpickles = PickleJar(pickledtestseqfiles, pickledtestlabelfiles, train=0, valid=0, test=1)

    if len(timeFile) > 0:
        pickledvalidtimefiles = []
        for i in [j+1 for j in range(int(n_valid_files))]:
            pickledvalidseqfiles.append(str(seqFile)+str(i))
            pickledvalidlabelfiles.append(str(labelFile)+str(i))
            pickledvalidtimefiles.append(str(timeFile)+str(i))
        validpickles = PickleJar(pickledvalidseqfiles, pickledvalidlabelfiles,
                                 pickledfiletimelist=pickledvalidtimefiles, train=0, valid=1, test=0, predict_time=True)

    else:
        for i in [j+1 for j in range(int(n_valid_files))]:
            pickledvalidseqfiles.append(str(seqFile)+str(i))
            pickledvalidlabelfiles.append(str(labelFile)+str(i))
        validpickles = PickleJar(pickledvalidseqfiles, pickledvalidlabelfiles, train=0, valid=1, test=0)