import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def WordCloudCreation(content, name):
    wc = WordCloud(
        width=480,
        height=480,
        background_color='white',
        stopwords=stopwords,
        max_words=200,
        max_font_size=70, 
        random_state=42
        ).generate(str(content))
    fig = plt.figure()
    plt.imshow(wc, interpolation="bilinear")
    plt.axis('off')
    #plt.show()
    save = name + '.png'
    fig.savefig(save, dpi=900)


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
stopwords.add("Saturday")
stopwords.add("Tuesday")
stopwords.add("Wednesday")
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

trainData = pd.read_csv('train2017.tsv', sep="\t")
curTweet = ' '
curCategory = ' '
positives = ' '
negatives = ' '
neutrals = ' '

for i in range(0,trainData.shape[0]):
	curTweet = trainData.loc[i][3]
	curCategory =  trainData.loc[i][2]
	if curCategory == 'positive':
		positives += (curTweet)
	elif curCategory == 'negative':
		negatives += (curTweet)
	elif curCategory == 'neutral':
		neutrals += (curTweet)

WordCloudCreation(positives, 'positives')
WordCloudCreation(negatives, 'negatives')
WordCloudCreation(neutrals, 'neutrals')
