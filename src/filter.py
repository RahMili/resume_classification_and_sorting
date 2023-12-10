from src.core.ML.resume_classifier import classify_resume
from src.core.ML.resume_sorter import sort_resumes


data_path = '../data/resumes'


resumes_shortlisted = classify_resume(data_path)
sorted_resumes = sort_resumes(data_path, resumes_shortlisted)

print('The resumes are sorted based on the information mentioned in the resumes.')
