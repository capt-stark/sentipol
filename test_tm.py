# -*- coding: utf-8 -*-

from gensim import corpora, models
from stop_words import get_stop_words
import gensim
import nltk

r = open('articles', 'r+')
all_text = r.read()
article_list = all_text.split('~~~~\n')

en_stop = get_stop_words('en')

titles = ['content','timeplace','author','title']
puncs = [',','.','\'','\"','?','!',';',':','(',')']
rem_t = titles + puncs
for i in rem_t:
    i = unicode(i)
en_stop = en_stop + rem_t


for article in article_list:
    texts = []
    tokens = nltk.word_tokenize(article.decode('utf-8'))
    tokens = [i for i in tokens]
    stopped_tokens = [i for i in tokens if not i in en_stop]
    texts.append(stopped_tokens)
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word = dictionary, passes=40)
    print(ldamodel.print_topics(num_topics=1, num_words=5))