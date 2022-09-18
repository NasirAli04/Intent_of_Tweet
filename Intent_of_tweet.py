# Import Libraries

#Basic Libraries
import re
import pandas as pd
import numpy as np
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')
import datetime

#for text pre-processing
from sklearn.preprocessing import LabelEncoder
import nltk
nltk.download('omw-1.4')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from wordcloud import WordCloud
from nltk.stem import PorterStemmer
nltk.download('stopwords')
stop=set(stopwords.words('english'))
from collections import  Counter
from nltk.util import ngrams


#for model-building
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from tensorflow.keras.preprocessing import text, sequence
from keras import layers, models, optimizers
from tensorflow.keras import optimizers
from tqdm import tqdm
import tensorflow as tf
from keras.layers import Embedding
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from tensorflow.keras.layers import BatchNormalization
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut

#for model Accuracy
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error
from sklearn import metrics
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline, ensemble
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import absolute
from numpy import sqrt

#for bag of words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

#for word embedding
import gensim
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings('ignore')

#for NER
import spacy
from spacy import displacy
nlp = spacy.load("en_core_web_sm")

#for visualization
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
import plotly.tools as tls
init_notebook_mode(connected=True)
import plotly.express as px
import plotly.graph_objects as go
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis.gensim_models
pyLDAvis.enable_notebook()

########################################################################################
# Reading data
df=pd.read_excel(r"C:\Users\Nasir\Desktop\Null\Intent_of_Tweet\Tweet_NFT.xlsx")

#Basic checks on data 
df.info()
df.head()
df.duplicated().sum()
df.isnull().sum()

unique_item_counts_for_id = df["id"].value_counts()
unique_item_counts_for_tweet_text = df["tweet_text"].value_counts()
unique_item_counts_for_created_at = df["tweet_created_at"].value_counts()
unique_item_counts_for_tweet_intent = df["tweet_intent"].value_counts()

'''
0   id                 127453 non-null  float64
 1   tweet_text        127453 non-null  object 
 2   tweet_created_at  127453 non-null  object 
 3   tweet_intent      96364 non-null   object 


unique_item_counts_for_id              1,27,453
unique_item_counts_for_tweet_text      1,14,498
unique_item_counts_for_created_at        50,394
unique_item_counts_for_tweet_intent           9

'''

#Converting string to date time

df['tweet_created_at'] = pd.to_datetime(df.tweet_created_at).dt.tz_localize(None)
df.info()


# Seprating the data to predict and train 
df_with_intent=df[df['tweet_intent'].notnull()]
df_with_missing_intent = df[df['tweet_intent'].isnull()]

#Checking item counts after dividig data 
item_counts_for_tweet_text_a = df_with_intent["tweet_text"].value_counts()
item_counts_for_tweet_text_b= df_with_missing_intent["tweet_text"].value_counts()


# label_encoder object knows how to understand word labels.

label_encoder = preprocessing.LabelEncoder()
# Encode labels in column 'species'.
df_with_intent['tweet_intent_labels']= label_encoder.fit_transform(df_with_intent['tweet_intent'])
df_with_intent['tweet_intent_labels'].unique()


#Visulaization
df_with_intent.groupby('tweet_intent').tweet_intent_labels.value_counts().plot(kind = "bar")
plt.xlabel("Label of data")
plt.title("Visulaize numbers of Label of data")
plt.show()

plt.figure(figsize=(15,8))
plt.title('Percentage of Resume', fontsize=20)
df_with_intent.tweet_intent.value_counts().plot(kind='pie',
                              wedgeprops=dict(width=.7), autopct="%1.1f%%", startangle= -2, 
                              textprops={'fontsize': 8})


#number of characters present in each sentence
df_with_intent['tweet_text'].str.len().hist()

#number of words appearing in each tweet
df_with_intent['tweet_text'].str.split().\
    map(lambda x: len(x)).\
    hist()
    

#average word length
df_with_intent['tweet_text'].str.split().\
   apply(lambda x : [len(i) for i in x]). \
   map(lambda x: np.mean(x)).hist()
   
   
   
   
stop = set(stopwords.words('english'))

Appreciation = df_with_intent[df_with_intent['tweet_intent_labels'] == 0]
Appreciation = Appreciation['tweet_text']

Communitu = df_with_intent[df_with_intent['tweet_intent_labels'] == 1]
Communitu = Communitu['tweet_text']

Done = df_with_intent[df_with_intent['tweet_intent_labels'] == 2]
Done = Done['tweet_text']

Giveaway = df_with_intent[df_with_intent['tweet_intent_labels'] == 3]
Giveaway = Giveaway['tweet_text']

Interested = df_with_intent[df_with_intent['tweet_intent_labels'] == 4]
Interested = Interested['tweet_text']

Launching_soon = df_with_intent[df_with_intent['tweet_intent_labels'] == 5]
Launching_soon = Launching_soon['tweet_text']

Presale = df_with_intent[df_with_intent['tweet_intent_labels'] == 6]
Presale = Presale['tweet_text']

Whitelist = df_with_intent[df_with_intent['tweet_intent_labels'] == 7]
Whitelist = Whitelist['tweet_text']

Pinksale = df_with_intent[df_with_intent['tweet_intent_labels'] == 8]
Pinksale = Pinksale['tweet_text']


def wordcloud_draw(dataset, color = 'white'):
            words = ' '.join(dataset)
            cleaned_word = ' '.join([word for word in words.split()
            if (word != 'news' and word != 'text')])
            wordcloud = WordCloud(stopwords = stop,
            background_color = color,
            width = 2500, height = 2500).generate(cleaned_word)
            plt.figure(1, figsize = (15,7))
            plt.imshow(wordcloud)
            plt.axis("off")
            plt.show()

print("Appreciation related words:")
wordcloud_draw(Appreciation, 'white')

print("Communitu related words:")
wordcloud_draw(Communitu, 'white')

print("Done related words:")
wordcloud_draw(Done, 'white')

print("Giveaway related words:")
wordcloud_draw(Giveaway, 'white')

print("Interested related words:")
wordcloud_draw(Interested, 'white')
   
print("Launching_soon related words:")
wordcloud_draw(Launching_soon, 'white')

print("Presale related words:")
wordcloud_draw(Presale, 'white')

print("Whitelist related words:")
wordcloud_draw(Whitelist, 'white')

print("Pinksale related words:")
wordcloud_draw(Pinksale, 'white')



##############################################################################################
#cleaning
#Data cleaning and preprocessing

ps = PorterStemmer()
#Basic preprocess
def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower() 
    text = text.strip()  
    return text
    
# STOPWORD REMOVAL
def stopword(string):
    a= [i for i in string.split() if i not in stopwords.words('english')]
    return ' '.join(a)

#LEMMATIZATION
# Initialize the lemmatizer
wl = WordNetLemmatizer()

# Tokenize the sentence
def lemmatizer(string):
    word_pos_tags = nltk.pos_tag(word_tokenize(string)) # Get position tags
    a=[wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token
    return " ".join(a)

# This is a helper function to map NLTK position tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def finalpreprocess(string):
    b=lemmatizer(stopword(preprocess(string)))
    print(b)
    print("/n")
    return b

# sample data for checking pre preprocessing 
sample_df = pd.DataFrame()
sample_df['tweet_text']=df_with_intent['tweet_text'].head(100)
sample_df['cleaned']= sample_df['tweet_text'].apply(lambda x: finalpreprocess(x))

##############################################################################################

#Final pre-processing
df_with_intent['cleaned_tweet']= df_with_intent['tweet_text'].apply(lambda x: finalpreprocess(x))
#Saving df to csv
df_with_intent.to_csv('Cleaned_data.csv',index=False)
df_with_missing_intent.to_csv('Data_to_predict.csv',index=False)

#############################################################################################
#reading the preprocessed file
df_with_intent=pd.read_csv(r"C:\Users\Nasir\Desktop\Null\Intent_of_Tweet\Cleaned_data.csv")

# Checking for duplicates in tweet text and removing it 
item_counts_for_cleaned_tweet=df_with_intent['cleaned_tweet_text'].value_counts()
df_with_intent['cleaned_tweet_text'].duplicated().sum()
df_with_intent=df_with_intent.drop_duplicates(subset='cleaned_tweet_text', keep="first")
intent_counts_for_cleaned_tweet=df_with_intent['tweet_intent'].value_counts()

#separating dependent and independent variables
df_with_intent.info()
X=df_with_intent['cleaned_tweet_text']
y=df_with_intent['tweet_intent_labels']

#SPLITTING THE TRAINING DATASET INTO TRAIN AND TEST
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4, random_state=30,shuffle=True)

##############################################################################################
''' bag of words-
     1 CountVectorizer
     2 Term Frequency-Inverse Document Frequencies (tf-Idf)'''

# CountVectorizer
def cv(data):
    count_vectorizer = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), stop_words = 'english',max_features=2500)
    emb = count_vectorizer.fit_transform(data)
    return emb, count_vectorizer
##############################################################################################
#Term Frequency-Inverse Document Frequencies (tf-Idf)
def tfidf(data):
    tfidf_vectorizer = TfidfVectorizer(min_df=3,  max_features=None, 
                                            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
                                                ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
                                                    stop_words = 'english')
    train = tfidf_vectorizer.fit_transform(data)
    return train, tfidf_vectorizer

##############################################################################################
# create Word2vec model
'''#here words_f should be a list containing words from each document. say 1st row of the list is words from the 1st document/sentence
#length of words_f is number of documents/sentences in your dataset'''
df_with_intent['cleaned_tweet_text']=[nltk.word_tokenize(i) for i in df_with_intent['cleaned_tweet_text']] #convert preprocessed sentence to tokenized sentence
model = Word2Vec(df_with_intent['cleaned_tweet_text'],min_count=1)  #min_count=1 means word should be present at least across all documents,
#if min_count=2 means if the word is present less than 2 times across all the documents then we shouldn't consider it


w2v = dict(zip(model.wv.index_to_key, model.wv.vectors))  #combination of word and its vector

#for converting sentence to vectors/numbers from word vectors result by Word2Vec
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(next(iter(word2vec.values())))

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

##############################################################################################

#GloVe Features 
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_with_intent.cleaned_tweet_text)

word_index = tokenizer.word_index
vocab_size = len(tokenizer.word_index) + 1
print("Vocabulary Size :", vocab_size)

GLOVE_EMB = r'C:\Users\Nasir\Desktop\Null\Intent_of_Tweet\glove.6B.300d.txt'
embeddings_index = {}

f = open(GLOVE_EMB,encoding="utf-8",errors='ignore')
for line in f:
  values = line.split()
  word = value = values[0]
  coefs = np.asarray(values[1:], dtype='float32')
  embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' %len(embeddings_index))

# this function creates a normalized vector for the whole sentence
stop_words = stopwords.words('english')
def sent2vec(s):
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(embeddings_index[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v ** 2).sum()) 

##############################################################################################
#CountVectorizer
X_train_counts, count_vectorizer = cv(X_train)
X_test_counts = count_vectorizer.transform(X_test)

#Term Frequency-Inverse Document Frequencies (tf-Idf)
X_train_vectors_tfidf, tfidf_vectorizer = tfidf(X_train) 
X_test_vectors_tfidf = tfidf_vectorizer.transform(X_test)

#Word2Vec
# Word2Vec runs on tokenized sentences
X_train_tok= [nltk.word_tokenize(i) for i in X_train]  
X_test_tok= [nltk.word_tokenize(i) for i in X_test]
# converting text to numerical data using Word2Vec
# Fit and transform
modelw = MeanEmbeddingVectorizer(w2v)
X_train_vectors_w2v = modelw.transform(X_train_tok)
X_test_vectors_w2v = modelw.transform(X_test_tok)

#Glove
# create sentence vectors using the above function for training and validation set
xtrain_glove = [sent2vec(x) for x in tqdm(X_train)]
xtest_glove = [sent2vec(x) for x in tqdm(X_test)]
xtrain_glove = np.array(xtrain_glove)
xtest_glove = np.array(xtest_glove)

##############################################################################################
#multi-class log-loss
def multiclass_logloss(actual, predicted, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    :param actual: Array containing the actual target classes
    :param predicted: Matrix with class predictions, one probability per class
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota

##############################################################################################
#.tocsc- Convert this matrix to Compressed Sparse Column format 
#Duplicate entries will be summed together.
#.tocsc- Convert this matrix to Compressed Sparse Column format 
#Duplicate entries will be summed together.
def xgb_classifier(X_train, y_train, X_test, y_test):
    clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 
                        subsample=0.8, nthread=10, learning_rate=0.1)
    clf.fit(X_train.tocsc(), y_train)
    y_predict = clf.predict(X_test.tocsc())
    y_prob= clf.predict_proba(X_test.tocsc())
    
    print(classification_report(y_test,y_predict))
    
    print ("logloss: %0.3f " % multiclass_logloss(y_test, y_prob)) 
    
        
    conf_matrix= confusion_matrix(y_test, y_predict)
    ax= sns.heatmap(conf_matrix, annot=True, cmap='Blues')
    print('Confusion Matrix:', ax)
##############################################################################################
#Model Logistic Regression
def logistic_regression(X_train, y_train, X_test, y_test):
    lr= LogisticRegression(solver = 'liblinear', C=10, penalty = 'l2')
    lr.fit(X_train, y_train)
    
    #predict y value for dataset
    y_predict= lr.predict(X_test)
    y_prob= lr.predict_proba(X_test)
    
    print(classification_report(y_test,y_predict))
    
    print ("logloss: %0.3f " % multiclass_logloss(y_test, y_prob)) 
  
    conf_matrix= confusion_matrix(y_test, y_predict)
    ax= sns.heatmap(conf_matrix, annot=True, cmap='Blues')
    print('Confusion Matrix:', ax)


#Singular-Value Decomposition The Singular-Value Decomposition, or SVD for short, 
#is a matrix decomposition method for reducing a matrix to its constituent parts in order to make certain subsequent
#matrix calculations simpler. Since SVMs take a lot of time, we will reduce the number of features from the TF-IDF using 
#Singular Value Decomposition before applying SVM. 
def svm_classifier(X_train, y_train, X_test, y_test):
    svd = decomposition.TruncatedSVD(n_components=120)
    svd.fit(X_train)
    xtrain_svd = svd.transform(X_train)
    xtest_svd = svd.transform(X_test)

    # Scale the data obtained from SVD. Renaming variable to reuse without scaling.
    scl = preprocessing.StandardScaler(with_mean=False)
    scl.fit(xtrain_svd)
    xtrain_svd_scl = scl.transform(xtrain_svd)
    xtest_svd_scl = scl.transform(xtest_svd)
    
    # Fitting a simple SVM
    clf = SVC(C=1.0, probability=True) # since we need probabilities
    clf.fit(xtrain_svd_scl, y_train)
    y_predict = clf.predict(xtest_svd_scl)
    y_prob= clf.predict_proba(xtest_svd_scl)
    
    print(classification_report(y_test,y_predict))
    
    print ("logloss: %0.3f " % multiclass_logloss(y_test, y_prob))  

        
    conf_matrix= confusion_matrix(y_test, y_predict)
    ax= sns.heatmap(conf_matrix, annot=True, cmap='Blues')
    print('Confusion Matrix:', ax)
    
def naive_bayes(X_train, y_train, X_test, y_test):
    # Fitting a simple Naive Bayes
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    
    y_predict = clf.predict(X_test)
    y_prob= clf.predict_proba(X_test)
    
    print(classification_report(y_test,y_predict))
    
    print ("logloss: %0.3f " % multiclass_logloss(y_test, y_prob))
    
    conf_matrix= confusion_matrix(y_test, y_predict)
    ax= sns.heatmap(conf_matrix, annot=True, cmap='Blues')
    print('Confusion Matrix:', ax)


# naive_grid
def naive_grid(X_train, y_train, X_test, y_test):    
    nb_model = MultinomialNB()

    # Create the pipeline 
    clf = pipeline.Pipeline([('nb', nb_model)])

    # parameter grid
    param_grid = {'nb__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

    # Initialize Grid Search Model
    model = GridSearchCV(estimator=clf, param_grid=param_grid, scoring=mll_scorer,
                                     verbose=10, n_jobs=-1, refit=True, cv=2)

    # Fit Grid Search Model
    model.fit(X_train, y_train)  # we can use the full data here but im only using xtrain. 
    print("Best score: %0.3f" % model.best_score_)
    print("Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))    

    y_predict = model.predict(X_test)
    y_prob= model.predict_proba(X_test)

    print(classification_report(y_test,y_predict)) 

    conf_matrix= confusion_matrix(y_test, y_predict)
    ax= sns.heatmap(conf_matrix, annot=True, cmap='Blues')
    print('Confusion Matrix:', ax)

#DecisionTree Classifier
def decisiontree_classifier(X_train, y_train, X_test, y_test):
    clf= DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    
    #predict y value for dataset
    y_predict= clf.predict(X_test)
    y_prob= clf.predict_proba(X_test)
    
    print(classification_report(y_test,y_predict))
    
    print ("logloss: %0.3f " % multiclass_logloss(y_test, y_prob))  
 
    conf_matrix= confusion_matrix(y_test, y_predict)
    ax= sns.heatmap(conf_matrix, annot=True, cmap='Blues')
    print('Confusion Matrix:', ax)


#RandomForest Classifier
def randomforest_classifier(X_train, y_train, X_test, y_test):
    clf= RandomForestClassifier()
    clf.fit(X_train, y_train)
    
    #predict y value for dataset
    y_predict= clf.predict(X_test)
    y_prob= clf.predict_proba(X_test)
    
    print(classification_report(y_test,y_predict))
    
    print ("logloss: %0.3f " % multiclass_logloss(y_test, y_prob))

    conf_matrix= confusion_matrix(y_test, y_predict)
    ax= sns.heatmap(conf_matrix, annot=True, cmap='Blues')
    print('Confusion Matrix:', ax)

    #KNeighbors Classifier
def kneighbors_classifier(X_train, y_train, X_test, y_test):
    clf= KNeighborsClassifier()
    clf.fit(X_train, y_train)
    
    #predict y value for dataset
    y_predict= clf.predict(X_test)
    y_prob= clf.predict_proba(X_test)
    
    print(classification_report(y_test,y_predict))
    
    print ("logloss: %0.3f " % multiclass_logloss(y_test, y_prob))  
        
    conf_matrix= confusion_matrix(y_test, y_predict)
    ax= sns.heatmap(conf_matrix, annot=True, cmap='Blues')
    print('Confusion Matrix:', ax)


# Prediction on all ML Model By using different Vectorization technique
xgb_classifier(X_train_counts, y_train, X_test_counts , y_test)
naive_grid(X_train_counts, y_train, X_test_counts , y_test)
naive_bayes(X_train_counts, y_train, X_test_counts , y_test)
svm_classifier(X_train_counts, y_train, X_test_counts , y_test)
logistic_regression(X_train_counts, y_train, X_test_counts , y_test)
decisiontree_classifier(X_train_counts, y_train, X_test_counts , y_test)
randomforest_classifier(X_train_counts, y_train, X_test_counts , y_test)
kneighbors_classifier(X_train_counts, y_train, X_test_counts , y_test)


#Prediction on all ML Model By using different Vectorization technique
xgb_classifier(X_train_vectors_tfidf, y_train, X_test_vectors_tfidf , y_test)
naive_grid(X_train_vectors_tfidf, y_train, X_test_vectors_tfidf , y_test)
naive_bayes(X_train_vectors_tfidf, y_train, X_test_vectors_tfidf , y_test)
svm_classifier(X_train_vectors_tfidf, y_train, X_test_vectors_tfidf , y_test)
logistic_regression(X_train_vectors_tfidf, y_train, X_test_vectors_tfidf , y_test)
decisiontree_classifier(X_train_vectors_tfidf, y_train, X_test_vectors_tfidf , y_test)
randomforest_classifier(X_train_vectors_tfidf, y_train, X_test_vectors_tfidf , y_test)
kneighbors_classifier(X_train_vectors_tfidf, y_train, X_test_vectors_tfidf , y_test)

#Prediction on all ML Model By using different Vectorization technique
xgb_classifier(X_train_vectors_w2v, y_train, X_test_vectors_w2v , y_test)
naive_grid(X_train_vectors_w2v, y_train, X_test_vectors_w2v , y_test)
naive_bayes(X_train_vectors_w2v, y_train, X_test_vectors_w2v , y_test)
svm_classifier(X_train_vectors_w2v, y_train, X_test_vectors_w2v , y_test)
logistic_regression(X_train_vectors_w2v, y_train, X_test_vectors_w2v , y_test)
decisiontree_classifier(X_train_vectors_w2v, y_train, X_test_vectors_w2v , y_test)
randomforest_classifier(X_train_vectors_w2v, y_train, X_test_vectors_w2v , y_test)
kneighbors_classifier(X_train_vectors_w2v, y_train, X_test_vectors_w2v , y_test)




#reading data for prediciton 
df_to_predict=pd.read_csv(r"C:\Users\Nasir\Desktop\Null\Intent_of_Tweet\Data_to_predict.csv")

df_to_predict.info()
df_to_predict['cleaned_tweet']= df_to_predict['tweet_text'].apply(lambda x: finalpreprocess(x))



predict_data=count_vectorizer.transform(df_to_predict['cleaned_tweet'])
###############################################################################################

final_model=xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 
                        subsample=0.8, nthread=10, learning_rate=0.1)

final_model.fit(X_train_counts, y_train)



from joblib import dump, load
import joblib
import pandas as pd
import pickle

pickle.dump(final_model, open('textclassification.pkl', 'wb')) 
pickle.dump(tfidf_vectorizer,open('tfidf_vect_dataset.pkl', 'wb'))
model = pickle.load(open('textclassification.pkl','rb'))
vectorizer=pickle.load(open('tfidf_vect_dataset.pkl','rb'))
predict_data=count_vectorizer.transform(df_to_predict['cleaned_tweet'])
result=model.predict(predict_data)



df_to_predict['intent_label']=result
df_to_predict['tweet_intent']=result



df_to_predict['tweet_intent']=df_to_predict['tweet_intent'].replace({0: 'Appreciation', 1: 'Community',2: 'Done',3: 'Giveaway',4: 'Intrested', 5: 'Launching',6: 'Presale', 7: 'Whitelist',8: 'Pinkscale'})
df_to_predict.drop(['cleaned_tweet'], axis=1,inplace=True)

df_to_predict.to_csv('Predicted_file.csv',index=False)
