# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 17:55:17 2017

@author: InfiniteJest
"""

class TaggedLinePandas():
    """
    Modification of doc2vec TaggedLineDocument. Used for iterating through
    pandas dataframe column as a generator, to save memory. Useful for large
    text files that also relate to other information stored in a dataframe.
    Can use 'NOTE_ID' for the labels... If the dataframe has been compiled such that
    the NOTE_ID is a column, set 'noteidaslabel' = True. Otherwise, will take the
    df index as the label
    """
    def __init__(self, dataframe, noteidaslabel=False):
        """
        Do:
            documents = TaggedLineDocument(df)
            where df is the dataframe that contains the text. Assumes the text
            column is called 'NOTE_TEXT'. 

        """
        self.dataframe = dataframe
        self.noteidaslabel=noteidaslabel
        pd.set_option('display.max_colwidth', -1)

    def __iter__(self):
        for index, line in pd.DataFrame(self.dataframe['NOTE_TEXT']).iterrows():
            text = line.to_string(index=False)
            if self.noteidaslabel == True:                 #Assumes input includes 'NOTE_ID' column
                label = self.dataframe.loc[index, 'NOTE_ID'].to_string(index=False)
            else:
                label = index
            yield gensim.models.doc2vec.TaggedDocument(text.split(), [label])

##################### DOC2VEC EVALUATION FUNCTIONS  ############################
import numpy as np
import statsmodels.api as sm
from random import sample

# for timing
from contextlib import contextmanager
from timeit import default_timer
import time 

@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start
    
def logistic_predictor_from_data(train_targets, train_regressors):
    logit = sm.Logit(train_targets, train_regressors)
    predictor = logit.fit(disp=0)
    #print(predictor.summary())
    return predictor

def error_rate_for_model(test_model, train_set, test_set, infer=False, infer_steps=3, infer_alpha=0.1, infer_subsample=0.1):
    """Report error rate on test_doc sentiments, using supplied model and train_docs"""

    train_targets = [doc for doc in train_set['HOMELESS?']]
    train_regressors = [test_model.docvecs[list(test_model.docvecs.doctags).index(doc)] for doc in train_set['TAGS']]
    train_regressors = sm.add_constant(train_regressors)
    predictor = logistic_predictor_from_data(train_targets, train_regressors)

    test_data = test_set
    if infer:
        if infer_subsample < 1.0:
            test_data = test_data.sample(int(infer_subsample * len(test_data)))
        test_regressors = [test_model.infer_vector(doc.split(), steps=infer_steps, alpha=infer_alpha) for doc in test_data['NOTE_TEXT']]
    else:
        test_regressors = [test_model.docvecs[list(test_model.docvecs.doctags).index(doc)] for doc in test_data['TAGS']]
    test_regressors = sm.add_constant(test_regressors)
    
    # predict & evaluate
    test_predictions = predictor.predict(test_regressors)
    corrects = sum(np.rint(test_predictions) == [doc for doc in test_data['HOMELESS?']])
    errors = len(test_predictions) - corrects
    error_rate = float(errors) / len(test_predictions)
    return (error_rate, errors, len(test_predictions), predictor)
################################################################################

df = pd.read_csv('fullhomelessnothomelessnotesprocessed.csv', dtype = object, encoding='ISO-8859-1', skiprows=range(1000000))
#df.drop()
df = df.rename(columns = {'NOTE_TYPE_NOADD_C': 'NOTE_TYPE'})
df = sklearn.utils.shuffle(df)
df = df.reset_index()
df = df.drop('index', axis=1)
model = gensim.models.doc2vec.Doc2Vec(size=200, dm=0, dbow_words=0, workers=4, iter=1)
documents = TaggedLinePandas(df)
model.build_vocab(documents)   #do scan of corpus overall of them together
df['TAGS'] = pd.DataFrame(list(model.docvecs.doctags.keys()))         #####APPEND THE DOCTAG LIST TO THE DATAFRAME     
df['HOMELESS?'] = df['HOMELESS?'].astype('int')                 

y = df['HOMELESS?']
#X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(df, y, test_size=0.2)

split = np.random.rand(len(df)) < 0.8
train_docs = df[split]
test_docs = df[~split]


assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"
#model = gensim.models.doc2vec.Doc2Vec.load()

##############################################################################
best_error = 0 

import datetime
alpha, min_alpha, passes = (0.025, 0.001, 20)
alpha_delta = (alpha - min_alpha) / passes

print("START %s" % datetime.datetime.now())

for epoch in range(passes):
    df = sklearn.utils.shuffle(df)  # shuffling gets best results
    documents = TaggedLinePandas(df)
    duration = 'na'
    model.alpha, model.min_alpha = alpha, alpha
    with elapsed_timer() as elapsed:
        model.train(documents, total_examples=model.corpus_count, epochs=model.iter)
        duration = '%.1f' % elapsed()
            
        # evaluate
    eval_duration = ''
    with elapsed_timer() as eval_elapsed:
        err, err_count, test_count, predictor = error_rate_for_model(model, train_docs, test_docs)
    eval_duration = '%.1f' % eval_elapsed()
    best_indicator = ' '
    if epoch == 0:
        best_error = err
    else:
        if err <= best_error:
            best_error = err
            best_indicator = '*' 
    print("%s%f : %i passes : %ss %ss" % (best_indicator, err, epoch + 1, duration, eval_duration))

    if ((epoch + 1) % 5) == 0 or epoch == 0:
        eval_duration = ''
        with elapsed_timer() as eval_elapsed:
            infer_err, err_count, test_count, predictor = error_rate_for_model(model, train_docs, test_docs, infer=True)
        eval_duration = '%.1f' % eval_elapsed()
        best_indicator = ' '
        if infer_err < best_error:
            best_error = infer_err
            best_indicator = '*'
        print("Infer Error %s%f : %i passes : %ss %ss" % (best_indicator, infer_err, epoch + 1, duration, eval_duration))

    print('completed pass %i at alpha %f' % (epoch + 1, alpha))
    alpha -= alpha_delta
    
print("END %s" % str(datetime.datetime.now()))


def infer_vec_from_pandas(model, df, rowname):
    inferred_docvec = model.infer_vector(df.loc[rowname, 'NOTE_TEXT'])
    return inferred_docvec