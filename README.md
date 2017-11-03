# UsefulNLPTools

This is a set of codes that can be useful for processing text and sequential data for use in machine learning algorithms. 

"onehottimeseqinputgenerator.py" can be used on tokenized sequences in which multiple tokens can occur at one time (slightly different from traditional NLP where one word corresponds to one time). It reads in data from a csv file into a Pandas dataframe, then reduces the dataframe into a pickle list of lists of lists: corresponds to a list of input sequences, each with a list of sequences, and for each sequence a list of tokens. "generatemultipleinputs" can also split the lists of generated inputs apart into multiple files, for the purpopse of saving the amount of data loaded into RAM at one time. 

"picklejar.py" can be used in tandem with "onehottimeseq" for loading the generated input files for use in training and testing. It contains a class that can iteratively access input files through a generator, for purposes of loading in only a chunk of data into RAM at a time.

"sqlchunks.py" allows for writing data from a sql query to a file in chunks, so that the entire query does not have to be executed at once. Useful for saving memory and also contains a query for generating a random sample from a database. 

"taggedlinepandas.py" is a modification and addition of the Gensim Doc2Vec tutorial found here: https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-IMDB.ipynb . A new class "TaggedLinePandas" is added that allows for iterating through a Pandas dataframe with a generator, the assumption being that each row contains a separate "document." The tutorial code has been additionally modified to accompany this new data structure.
