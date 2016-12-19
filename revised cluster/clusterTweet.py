import re
import nltk
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import  Normalizer
from sklearn.cluster import KMeans
from sklearn import metrics
from time import time

# stemmer = SnowballStemmer("english")
stemmer = WordNetLemmatizer()

def preprocess(tweet):

    # Removes wwww or https
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', ' ', tweet)
    # Remove @username
    tweet = re.sub('@[^\s]+', ' ', tweet)
    # Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    # Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    # trim
    tweet = tweet.strip('rt')
    tweet = re.sub('[\'"?,;=^_!@%-:$&.]', '', tweet)
    # replaceTwoorMore
    tweet = re.sub(r"(.)\1{1,}", r"\1\1", tweet, flags=re.DOTALL)

    return tweet


def tokenize_and_stem(text):

    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a‐zA‐Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.lemmatize(t) for t in filtered_tokens]
    return stems

def tokenize_only(text):

    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a‐zA‐Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

def feature_extraction(words):

    t0 = time()
    tfidf = TfidfVectorizer(stop_words='english', max_df=0.5, max_features=1000, min_df=2, tokenizer=tokenize_and_stem,
                            ngram_range=(1, 1), use_idf=True)
    tfs = tfidf.fit_transform(words)
    print("done in %fs" % (time() - t0))
    term = tfidf.get_feature_names()
    print(term)
    # return  tfs
    # Dimensionality Reduction using LSA
    svd = TruncatedSVD(n_components=2)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    di_tfs = lsa.fit_transform(tfs)
    print(di_tfs)
    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))
    print()

    return di_tfs


def cluster(terms):
    ter
    num_clusters = 3
    km = KMeans(n_clusters=num_clusters)
    print("Clustering data")
    t0 = time()
    km.fit(terms)
    clusters = km.labels_.tolist()
    print(clusters)
    print("done in %0.3fs" % (time() - t0))
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(terms, km.labels_, sample_size=1000))
    print()

    # Grouping the tweets into their respective clusters
    groups = {'tweets': processedTweet, 'cluster': clusters}
    frame = pd.DataFrame(groups, index=[clusters], columns=['tweets', 'cluster'])
    dami = frame['cluster'].value_counts()
    print(dami)

    tokenized_words = []
    token_lemma_words = []
    for i in processedTweet:
        all_lemmatized = tokenize_and_stem(i)
        token_lemma_words.extend(all_lemmatized)

        all_tokenized = tokenize_only(i)
        tokenized_words.extend(all_tokenized)
    vocab_frame = pd.DataFrame({'word': tokenized_words}, index = token_lemma_words)

    print("Top terms per cluster:")
    print()
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    for i in range(num_clusters):
        print('Cluster %d words:' % i, end='')

        for ind in order_centroids[i, :6]:
            print(' %s' % vocab_frame.ix[term[ind]].values.tolist()[0][0], end=',')
        print()
        print()

        print("Cluster %d tweets:" % i, end='')
        for text in frame.ix[i]['tweets'].values.tolist():
            print(' %s,' % text, end='')
        print()
        print()

    print()
    print()

#Read the tweets one by one and process it
fp = open(r"C:\Users\Dharmie\PycharmProjects\untitled1\compiledtweets.txt", 'r')
line = fp.readline()
our_text = []
while line:
    processedTweet = preprocess(line)
    our_text.append(processedTweet)

    line = fp.readline()

fp.close()
cluster(feature_extraction(our_text))
