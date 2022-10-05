from flask import Flask,render_template,request
#importing server.py file
from server import *
import time

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/result',methods=['POST'])
def shorturl():
    data = request.form.get("url")
    number_of_topics=2

    # Loading Data
    document_list= load_data(data)
    # Preprocessing Data
    clean_text=preprocess_data(document_list)

    #reading original text time
    start = time.time()
    originaltext = original_text(document_list)
    final_reading_time = readingTime(originaltext)
    
    #Prepare Corpus
    dictionary,doc_term_matrix=prepare_corpus(clean_text)
    # Create an LSA model using Gensim
    model=create_gensim_lsa_model(dictionary,doc_term_matrix,clean_text,number_of_topics)
    # apply model to prepared Corpus
    corpus_lsi = model[doc_term_matrix]  
    #sort each vector by score
    vecsSort = sort_vector_by_score(corpus_lsi)
    # Select the sentences for the summary
    topSentences = selectTopSent(8, 2, vecsSort)
    #Genearte Summary
    summary = article_summary(topSentences,document_list)
    #reading summarized document
    summary_reading_time = readingTime(summary)
    end = time.time()
    final_time = end-start
    #generation of wordcloud
    plot_cloud(summary)

    return render_template('result.html',summary=summary,filename="../static/img/first_review.png",ctext=originaltext,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time) 

if __name__== '__main__':
    app.debug = True
    app.run()
