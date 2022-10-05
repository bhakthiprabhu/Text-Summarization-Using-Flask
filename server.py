import bs4 as BeautifulSoup
import urllib.request 
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim import corpora
from gensim.models import LsiModel
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import spacy
nlp = spacy.load("en_core_web_sm")

def load_data(data):
    fetched_data = urllib.request.urlopen(data)
    article_read = fetched_data.read()
    article_parsed = BeautifulSoup.BeautifulSoup(article_read,'html.parser')
    paragraphs = article_parsed.find_all('p')
    article_content = []
    for p in paragraphs:  
        text = p.text.strip()
        article_content.append(text)
    return article_content

def preprocess_data(doc_set):
    # initialize regex tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    # create English stop words list
    en_stop = set(stopwords.words('english'))
    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()
    # list for tokenized documents in loop
    texts = []
    # loop through document list
    for i in doc_set:
        # clean and tokenize document string
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
        # stem tokens
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        # add tokens to list
        texts.append(stemmed_tokens)
    return texts

def prepare_corpus(doc_clean):
    # Creating the term dictionary of our corpus, where every unique term is assigned an index. 
    dictionary = corpora.Dictionary(doc_clean)
    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    # generate LDA model
    return dictionary,doc_term_matrix

def create_gensim_lsa_model(dictionary,doc_term_matrix,doc_clean,number_of_topics):
    # generate LSA model
    lsamodel = LsiModel(doc_term_matrix, num_topics=number_of_topics, id2word = dictionary) 
    return lsamodel

#sort
def takenext(elem):
    return elem[1]

def sort_vector_by_score(corpus_lsi):
    vecsSort = list(map(lambda i: list(), range(2)))
    for i,docv in enumerate(corpus_lsi):
        for sc in docv:
            isent = (i, abs(sc[1]))
            vecsSort[sc[0]].append(isent)
    return list(map(lambda x: sorted(x,key=takenext,reverse=True), vecsSort))

def selectTopSent(summSize, numTopics, sortedVecs):
    topSentences = []
    sent_no = []
    sentIndexes = set()
    sentIndexes = set()
    topSentences = []
    sent_no = []
    sentIndexes = set()
    sCount = 0
    for i in range(summSize):
        for j in range(numTopics):
            vecs = sortedVecs[j]
            si = vecs[i][0]
            if si not in sentIndexes:
                sent_no.append(si)
                sCount +=1
                topSentences.append(vecs[i])
                sentIndexes.add(si)
                if sCount == summSize:
                    sent_no
        return sent_no

def article_summary(topSentences,document_list):
    summary = []
    cnt = 0
    for sentence in document_list:
        if cnt in topSentences:
            summary.append(sentence)
        cnt += 1    
    summary = " ".join(summary)
    return summary

def plot_cloud(summary):
    # Set figure size
    plt.figure(figsize=(40, 30))
    wordcloud = WordCloud(width= 3000, height = 2000, random_state=1, background_color='#f25278', colormap='Pastel1', collocations=False, stopwords = STOPWORDS).generate(summary)
    # Display image
    plt.imshow(wordcloud)
    # No axis details
    plt.axis("off")
    #save wordcloud to file
    wordcloud.to_file("static/img/first_review.png")

def readingTime(mytext):
	total_words = len([ token.text for token in nlp(mytext)])
	estimatedTime = total_words/200.0
	return estimatedTime

def original_text(document_list):
    document=[]
    for sentence in document_list:
        document.append(sentence)
    original = " ".join(document)
    return original