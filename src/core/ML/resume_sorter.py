import pandas as pd
from src.core.utils.read_pdf import read_native_pdf
from src.core.utils.read_docx import read_docx_file
import os
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords as stp
from nltk.corpus import wordnet
import nltk



lemmatizer = WordNetLemmatizer()
analyzer = CountVectorizer().build_analyzer()


def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    # print('get_word net done')
    return tag_dict.get(tag, wordnet.NOUN)


def stemmed_words(doc):
    # print('stem words called')
    return (lemmatizer.lemmatize(w,get_wordnet_pos(w)) for w in analyzer(doc) if w not in set(stp.words('english')))


def get_tf_idf_cosine_similarity(job_description, resumes_text):
    tf_idf_vect = TfidfVectorizer(analyzer=stemmed_words)
    tf_idf_desc_vector = tf_idf_vect.fit_transform([job_description]).todense()
    tf_idf_resume_vector = tf_idf_vect.transform(resumes_text).todense()
    cosine_similarity_list = []
    for i in range(len(tf_idf_resume_vector)):
        cosine_similarity_list.append(cosine_similarity(tf_idf_desc_vector,tf_idf_resume_vector[i])[0][0])
    return cosine_similarity_list


def read_jd(data_path):
    with open(os.path.join('../data', 'job_description_Computer_vision.txt'), 'r', encoding='utf-8') as f:
        file_desc_lst = [r.replace('\n', '') for r in f.readlines()]
    job_description = ''

    for i in file_desc_lst:
        if len(job_description) == 0:
            job_description = str(i)
        else:
            job_description = job_description + ' ' + str(i)
    return job_description

def sort_resumes(data_path, resumes_shortlisted):
    resumes_text = []
    resume_names = []
    for i in resumes_shortlisted:
        resume_names.append(i)
        if i.split('.')[1] == 'pdf':
            # resumes_text = ' \nresume name: '+str(i) +'\n '
            # resumes_text = resumes_text + str(read_native_pdf(os.path.join(data_path, i)))
            text = read_native_pdf(os.path.join(data_path, i))
            resumes_text.append(text)
        elif i.split('.')[1] == 'docx':
            # resumes_text = ' \nresume name: '+str(i) + '\n '
            # resumes_text = resumes_text + str(read_docx_file(os.path.join(data_path, i)))
            text = read_docx_file(os.path.join(data_path, i))
            resumes_text.append(text)

    job_description = read_jd(data_path)
    cos_sim_list = get_tf_idf_cosine_similarity(job_description, resumes_text)

    zipped_resume_rating = zip(cos_sim_list, resume_names, [x for x in range(len(resumes_text))])
    sorted_resume_rating_list = sorted(zipped_resume_rating, key=lambda x: x[0], reverse=True)
    print('sorted resume rating', sorted_resume_rating_list)
    resume_score = [round(x * 100, 2) for x in cos_sim_list]
    # print('resume score', resume_score)
    return resumes_text
