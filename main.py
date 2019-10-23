import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem.porter import PorterStemmer
import re
import pickle
from sklearn.feature_extraction import DictVectorizer
import gensim
from gensim.models import Word2Vec
import time
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn import preprocessing
from sklearn import metrics
import seaborn as sns


start = time.time()

stopwords = set(ENGLISH_STOP_WORDS)
stopwords.add("tomorrow")
stopwords.add("day")
stopwords.add("http")
stopwords.add("3rd")
stopwords.add("u002c")
stopwords.add("time")
stopwords.add("1st")
stopwords.add("today")
stopwords.add("make")
stopwords.add("think")
stopwords.add("u2019m")
stopwords.add("saturday")
stopwords.add("tuesday")
stopwords.add("wednesday")
stopwords.add("people")
stopwords.add("did")
stopwords.add("will")
stopwords.add("said")
stopwords.add("say")
stopwords.add("says")
stopwords.add("it")
stopwords.add("they")
stopwords.add("are")
stopwords.add("that")
stopwords.add("saying")


# ------------------ REMOVE SPECIAL CHARS AND STOPWORDS from train
trainData = pd.read_csv('train2017.tsv', sep="\t", encoding = "utf-8")

stemmed = []
removed = []
tokens = []
similar_words = []
vect = []
tokenized_tweet = []

for i in range(0, trainData.shape[0]):
	removed.append(re.sub(r"[^a-zA-Z0-9]+", ' ', trainData.loc[i][3]))

	word_tokens = word_tokenize(removed[i])
	filtered_sentence = []
	for w in word_tokens: 
		if w.lower() not in stopwords: 
			filtered_sentence.append(w.lower())

			# STEM
			porter = PorterStemmer()
			stemmed.extend([porter.stem(w.lower())]) 
	tokens.append(filtered_sentence)
	removed[i] = ' '.join(filtered_sentence)

# ------------------ REMOVE SPECIAL CHARS AND STOPWORDS from test
testData = pd.read_csv('test2017.tsv', sep="\t", encoding = "utf-8")

stemmed2 = []
removed2 = []
tokens2 = []
similar_words2 = []
vect2 = []
tokenized_tweet2 = []

for i in range(0, testData.shape[0]):
	removed2.append(re.sub(r"[^a-zA-Z0-9]+", ' ', testData.loc[i][3]))

	word_tokens2 = word_tokenize(removed2[i])
	filtered_sentence2 = []
	for w in word_tokens2: 
		if w.lower() not in stopwords: 
			filtered_sentence2.append(w.lower())

			# STEM
			porter2 = PorterStemmer()
			stemmed2.extend([porter2.stem(w.lower())]) 
	tokens2.append(filtered_sentence2)
	removed2[i] = ' '.join(filtered_sentence2)

##########################################################
# WORD EMBEDDINGS
print ('W2V')

# ------------------ w2v train ------------------

model_w2v = gensim.models.Word2Vec(
            tokens,
            size=200, # desired no. of features/independent variables
            window=5, # context window size
            min_count=2,
            sg = 1, # 1 for skip-gram model
            hs = 0,
            negative = 10, # for negative sampling
            workers= 2, # no.of cores
            seed = 34)

model_w2v.train(tokens, total_examples= len(removed), epochs=20)

model_w2v.save('embeddings.pkl')
new_model = gensim.models.Word2Vec.load('embeddings.pkl')

##########################################################
# TWEET VECTORS

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ lexica

affin = pd.read_csv('affin.txt', sep="\t", encoding = "utf-8")

affin_lexica = []
for i in range(0, affin.shape[0]):
	vals = {}
	vals['word'] = affin.loc[i][0]
	vals['vector'] = affin.loc[i][1]
	affin_lexica.append(vals)

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# ------------------ w2v train ------------------

vec = []
vec_tweet = []

for i in range(0, trainData.shape[0]):
	vec = np.zeros(201)
	wcount = len(removed[i].split())
	for word in removed[i].split():
		for j in range(0,200):
			if word in new_model.wv.vocab:
				vec[j] += (new_model[word][j] / wcount)
		if word in affin_lexica:
			vec[200] = affin_lexica[word]
	vec_tweet.append(vec)

print (vec_tweet[0])

# EXPORT TWEET VECTORS TO FILE
tv_output = open("tweetvectors.pkl", "wb")
pickle.dump(vec_tweet, tv_output)
tv_output.close()

# LOAD TWEET VECTORS
tv_file = open("tweetvectors.pkl", "rb")
vec_tweet = pickle.load(tv_file)
tv_file.close()

# ------------------ w2v test ------------------

vec2 = []
vec_tweet_test = []

for i in range(0, testData.shape[0]):
	vec2 = np.zeros(201)
	wcount = len(removed2[i].split())
	for word in removed2[i].split():
		for j in range(0,200):
			if word in new_model.wv.vocab:
				vec2[j] += (new_model[word][j] / wcount)
		if word in affin_lexica:
			vec2[200] = affin_lexica[word]
	vec_tweet_test.append(vec2)

# EXPORT TWEET VECTORS TO FILE
tv2_output = open("tweetvectorstest.pkl", "wb")
pickle.dump(vec_tweet_test, tv2_output)
tv2_output.close()

# LOAD TWEET VECTORS
tv2_file = open("tweetvectorstest.pkl", "rb")
vec_tweet_test = pickle.load(tv2_file)
tv2_file.close()

##########################################################
# SIMILAR WORDS

for i in range(0, trainData.shape[0]):
	for word in removed[i].split():
		if word in new_model_test.wv.vocab:
			similar_words.append(new_model.wv.most_similar(positive=word))
			vect.append(new_model[word])

# EXPORT SIMILAR WORDS TO FILE
sw_output = open("similarwords.pkl", "wb")
pickle.dump(similar_words, sw_output)
sw_output.close()

# LOAD SIMILAR
sw_file = open("similarwords.pkl", "rb")
similar = pickle.load(sw_file)
sw_file.close()

##########################################################
# BAG OF WORDS

# ------------------ train ------------------

bow_vectorizer = CountVectorizer(max_df=1.0, min_df=1, max_features=1000, stop_words='english') 
bow_xtrain = bow_vectorizer.fit_transform(removed)

# EXPORT BAG TO FILE
bow_output = open("bagofwords.pkl", "wb")
pickle.dump(bow_xtrain, bow_output)
bow_output.close()

# LOAD BAG
bow_file = open("bagofwords.pkl", "rb")
bag = pickle.load(bow_file)
bow_file.close()

bow_xtrain = bag
features = bow_vectorizer.get_feature_names()
bag = bag.toarray()
bow_df = pd.DataFrame(bag, columns = features)

bow_xtrain = bow_xtrain.toarray()



# ------------------ test ------------------

bow_vectorizer2 = CountVectorizer(max_df=1.0, min_df=1, max_features=1000, stop_words='english') 
bow_xtrain2 = bow_vectorizer2.fit_transform(removed2)

# EXPORT BAG TO FILE
bow_output2 = open("bagofwords_test.pkl", "wb")
pickle.dump(bow_xtrain2, bow_output2)
bow_output2.close()

# LOAD BAG
bow_file2 = open("bagofwords_test.pkl", "rb")
bag2 = pickle.load(bow_file2)
bow_file2.close()

bow_xtrain_test = bag2
features2 = bow_vectorizer2.get_feature_names()
bag2 = bag2.toarray()
bow_df2 = pd.DataFrame(bag2, columns = features2)

bow_xtrain_test = bow_xtrain_test.toarray()

##########################################################
# TF-IDF

# ------------------ train ------------------

s = []
for i in range(0,trainData.shape[0]):
	s.extend([removed[i]])

tfidf_vectorizer = TfidfVectorizer(max_df=1.0, min_df=1, max_features=1000, stop_words='english') 
tfidf = tfidf_vectorizer.fit_transform(s)
tfidf = tfidf.toarray()
features = tfidf_vectorizer.get_feature_names()
tfidf_df = pd.DataFrame(np.round(tfidf, 3), columns = features)

# EXPORT TF_IDF TO FILE
tfidf_output = open("tfidf.pkl", "wb")
pickle.dump(tfidf_df, tfidf_output)
tfidf_output.close()

# LOAD TF_IDF
tfidf_file = open("tfidf.pkl", "rb")
tf = pickle.load(tfidf_file)
tfidf_file.close()

# ------------------ test ------------------

s2 = []
for i in range(0,testData.shape[0]):
	s2.extend([removed2[i]])

tfidf_vectorizer2 = TfidfVectorizer(max_df=1.0, min_df=1, max_features=1000, stop_words='english') 
tfidf2 = tfidf_vectorizer2.fit_transform(s2)
tfidf2 = tfidf2.toarray()
features2 = tfidf_vectorizer2.get_feature_names()
tfidf_df2 = pd.DataFrame(np.round(tfidf2, 3), columns = features2)

# EXPORT TF_IDF TO FILE
tfidf_output2 = open("tfidf_test.pkl", "wb")
pickle.dump(tfidf_df2, tfidf_output2)
tfidf_output2.close()

# LOAD TF_IDF
tfidf_file2 = open("tfidf_test.pkl", "rb")
tf_test = pickle.load(tfidf_file2)
tfidf_file2.close()

##########################################################
# READ LABELS

labels = []
for i in range(0, trainData.shape[0]):
	labels.append(trainData.loc[i][2])

testLabels = pd.read_csv('SemEval2017_task4_subtaskA_test_english_gold.txt', sep="\t", encoding = "utf-8")
labels_test = []
for i in range(0, testLabels.shape[0]):
	labels_test.append(testLabels.loc[i][1])

##########################################################
# KNN
print ('KNN')

# ------------- FOR TRAIN
# X_train, X_test, y_train, y_test = train_test_split(vec_tweet, labels, test_size=0.4, random_state=4)
# -------------

# ------------- FOR TEST
X_train = vec_tweet
# X_train = bow_xtrain
y_train = labels
X_test = vec_tweet_test
# X_test = bow_xtrain_test
y_test = labels_test
# -------------

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
# make predictions on the testing set
y_pred = logreg.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

# Repeat for KNN with K=5:
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print (y_pred)
print(metrics.accuracy_score(y_test, y_pred))


##########################################################
# SVM
print ('SVM')

# ------------- FOR TRAIN
# X_train, X_test, y_train, y_test = train_test_split(vec_tweet, labels, random_state=42, test_size=0.2) #input for this method is any array of features
# -------------

# ------------- FOR TEST
X_train = vec_tweet
y_train = labels
X_test = vec_tweet_test
y_test = labels_test
# -------------

svc = svm.SVC(kernel='linear', C=1, probability=True)
svc = svc.fit(X_train, y_train)
prediction = svc.predict(X_test) #predict on the validation set
print (f1_score(y_test, prediction, average='micro')) #evaluate on the validation set

end = time.time()
print(end - start)