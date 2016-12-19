__author__ = 'Dharmie'

import re
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import  Normalizer
from sklearn.cluster import KMeans
from sklearn import metrics
from time import time

stemmer = WordNetLemmatizer()

def processTweet(tweet):

    #Removes wwww or https
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ', tweet)
    #Remove @username
    tweet = re.sub('@[^\s]+',' ', tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('rt')
    tweet = re.sub('[\'"?,;=^_!@%-:$&.]', '', tweet)
    #replaceTwoorMore
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


def no_name_yet(words):
    #max_df = 0.5, max_features = 20, min_df = 2,
    t0 = time()
    tfidf = TfidfVectorizer(stop_words='english',max_df = 0.5, max_features=1000, min_df = 2, tokenizer=tokenize_and_stem, ngram_range=(1,1) ,use_idf=True)
    tfs = tfidf.fit_transform(words)
    print("done in %fs" % (time() - t0))
    # print(tfs)
    terms = tfidf.get_feature_names()
    # print (terms)
    t0 = time()
    svd = TruncatedSVD(n_components = 2)
    normalizer = Normalizer (copy = False)
    lsa = make_pipeline(svd, normalizer)
    tfs = lsa.fit_transform(tfs)


    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))
    print()

    # using PCA to convert the matrix into a 2-dimensional array
    # pca = PCA(n_components=2)
    # data2D = pca.fit_transform(tfs)
    # plt.scatter(data2D[:, 0], data2D[:, 1])
    # plt.show()

    # dist = (1 - cosine_similarity(tfs))

    num_clusters = 3
    km = KMeans(n_clusters=num_clusters)
    print("Clustering data" )
    t0 = time()
    km.fit(tfs)
    clusters = km.labels_.tolist()
    print (clusters)
    print("done in %0.3fs" % (time() - t0))

    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
    # print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
    # print("V‐measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
    # print("Adjusted Rand‐Index: %.3f"% metrics.adjusted_rand_score(labels, km.labels_))
    print("Silhouette Coefficient: %0.3f"% metrics.silhouette_score(tfs, km.labels_, sample_size=1000))
    print()


    groups = {'tweets': words, 'cluster': clusters}
    frame = pd.DataFrame(groups, index= [clusters], columns=['tweets', 'cluster'])
    dami = frame['cluster'].value_counts()
    print()

    tokenized_words = []
    for i in words:
        all_tokenized = tokenize_only(i)
        tokenized_words.extend(all_tokenized)
    vocab_frame = pd.DataFrame({'word':tokenized_words})

    print("Top terms per cluster:")
    print()
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    for i in range(num_clusters):
        print('Cluster %d words:' %i, end='')

        for ind in order_centroids[i, :6]:
            print(' %s' %vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0],  end=',')
        print()
        print()

        print("Cluster %d tweets:" %i,end ='')
        for text in frame.ix[i]['tweets'].values.tolist():
            print(' %s,' % text, end='')
        print()
        print()

    print()
    print()

    # #converting dist matrix into 2-dimensional array
    # mds = MDS(n_components=2, dissimilarity='precomputed', random_state=1)
    # pos = mds.fit_transform(dist)
    # print(pos)
    # xs, ys = pos[:, 0], pos[:, 1]
    # print()
    # print()

    #visualizing clusters
    # cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3'}
    # cluster_names = {0: '',
    #                  1: '',
    #                  2: '' }
    #
    # df = pd.DataFrame(dict(x=xs, y=ys, label=clusters))
    # # group by cluster
    # groups = df.groupby('label')
    # # set up plot
    # fig, ax = plt.subplots(figsize=(17, 9))  # set size
    # ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
    # # iterate through groups to layer the plot
    # # note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
    # for name, group in groups:
    #     ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
    #             label=cluster_names[name], color=cluster_colors[name],
    #             mec='none')
    #     ax.set_aspect('auto')
    #     ax.tick_params( \
    #         axis='x',  # changes apply to the x‐axis
    #         which='both',  # both major and minor ticks are affected
    #         bottom='off',  # ticks along the bottom edge are off
    #         top='off',  # ticks along the top edge are off
    #         labelbottom='off')
    #     ax.tick_params( \
    #         axis='y',  # changes apply to the y‐axis
    #         which='both',  # both major and minor ticks are affected
    #         left='off',  # ticks along the bottom edge are off
    #         top='off',  # ticks along the top edge are off
    #         labelleft='off')
    # ax.legend(numpoints=1)  # show legend with only 1 point
    #
    # # add label in x,y position with the label as the film title
    # for i in range(len(df)):
    #     ax.text(df.ix[i]['x'], df.ix[i]['y'], size=8)
    #
    # plt.show()  # show the plot

#end
#Read the tweets one by one and process it
fp = open(r"C:\Users\Dharmie\PycharmProjects\untitled1\compiledtweets.txt", 'r')

line = fp.readline()
our_text = []
while line:
    processedTweet = processTweet(line)
    our_text.append(processedTweet)
    #with open('tweettll.txt', 'a') as f:
        #f.write("%s\n" % processedTweet)

    line = fp.readline()
#end loop

fp.close()
no_name_yet(our_text)


