import os
from src.core.utils.read_pdf import read_native_pdf
from src.core.utils.read_docx import read_docx_file
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import re


'''
4	Data Science
10	HR
21	Web Designing
13	Mechanical Engineer
19	Sales
3	Civil Engineer
12	Java Developer
2	Business Analyst
18	SAP Developer
0	Automation Testing
9	Electrical Engineering
15	Operations Manager
17	Python Developer
6	DevOps Engineer
14	Network Security Engineer
16	PMO
5	Database
11	Hadoop
8	ETL Developer
7	DotNet Developer
1	Blockchain
20	Testing
'''

model_path = '../data/model/ovr_resume_classifier_400.pkl'
with open(model_path, 'rb') as f:
    clf = pickle.load(f)
    if clf:
        print('model loaded')


def clean_resume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText)
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText


def classify_resume(data_path):

    resumes_info = []
    resume_filenames = []
    for f in os.listdir(data_path):
        resume_filenames.append(f)
        text = ''
        if f.split('.')[1] == 'pdf':
            text = read_native_pdf(os.path.join(data_path, f))
            resumes_info.append(text)
        elif f.split('.')[1] == 'docx':
            text = read_docx_file(os.path.join(data_path, f))
            resumes_info.append(text)
    for i in range(len(resumes_info)):
        resumes_info[i] = clean_resume(resumes_info[i])
        # print('\n\nresumes info', resumes_info[i])
    # print('\n\nresume filenames', resume_filenames)
    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        stop_words='english',
        max_features=400)

    word_vectorizer.fit(resumes_info)
    WordFeatures = word_vectorizer.transform(resumes_info)

    prediction = clf.predict(WordFeatures)
    pred = zip(resume_filenames, prediction)
    # for f, p in pred:
    #     print('filename', f)
    #     print('pred ', p)
    # print('prediction size = ', len(prediction))
    resume_list = []
    for i in range(len(prediction)):
         # data science, Ml ops and Business Analyst
        if prediction[i] == 4 or prediction[i] == 6 :#or prediction[i] == 2:
            resume_list.append(resume_filenames[i])
    # print('shortlisted resume list', resume_list)
    return resume_list
