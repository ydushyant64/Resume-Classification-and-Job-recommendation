# import numpy as np
# import pandas as pd
# import re
# import pickle
# from flask import Flask, request, render_template

# # Import Keras load_model function
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# # For recommendations
# from sklearn.metrics.pairwise import cosine_similarity

# # --- Load all the essential files ---

# # Load TF-IDF Vectorizer
# tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# # Load the K-Nearest Neighbors and Naive Bayes models
# knc = pickle.load(open('knc.pkl', 'rb'))
# mnb = pickle.load(open('mnb.pkl', 'rb'))

# # --- Load the RNN Model and its Tokenizer ---
# # Load the entire RNN model from the .h5 file
# rnn_model = load_model('rnn.h5')
# # Load the tokenizer
# rnn_tokenizer = pickle.load(open('rnn_tokenizer.pkl', 'rb'))


# # --- Load dataset for recommendation logic ---
# df = pd.read_csv('ResumeDataSet.csv')

# # ********************************************************************
# # THE FIX IS HERE: Changed fit_transform to just transform
# # This uses the already-fitted vocabulary from the notebook
# tfidf_matrix = tfidf.transform(df['Resume'])
# # ********************************************************************


# # Category mapping from your notebook
# category_mapping = {
#     15: "Java Developer", 6: "Data Science", 23: "Testing", 20: "SAP Developer", 24: "Web Designing",
#     12: "HR", 13: "Hadoop", 3: "Blockchain", 10: "ETL Developer", 18: "Operations Manager",
#     22: "Sales", 16: "Mechanical Engineer", 1: "Arts", 7: "Database", 11: "Electrical Engineering",
#     14: "Health and fitness", 19: "PMO", 4: "Business Analyst", 9: "DotNet Developer",
#     2: "Automation Testing", 17: "Network Security Engineer", 21: "Python Developer",
#     5: "Civil Engineer", 0: "Advocate",
# }

# # --- Helper function for cleaning resume text ---
# def clean_resume(resume_text):
#     clean_text = re.sub('http\S+\s*', ' ', resume_text)
#     clean_text = re.sub('RT|cc', ' ', clean_text)
#     clean_text = re.sub('#\S+', '', clean_text)
#     clean_text = re.sub('@\S+', '  ', clean_text)
#     clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
#     clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
#     clean_text = re.sub('\s+', ' ', clean_text)
#     return clean_text

# # --- Initialize Flask App ---
# app = Flask(__name__)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         resume_text = request.form['resume']
#         selected_model = request.form['model']
        
#         # 1. Clean the input resume
#         cleaned_text = clean_resume(resume_text)
        
#         # 2. Make Prediction based on selected model
#         prediction_id = 0
        
#         if selected_model == 'rnn':
#             # RNN Prediction
#             sequence = rnn_tokenizer.texts_to_sequences([cleaned_text])
#             padded_sequence = pad_sequences(sequence, maxlen=500)
#             prediction = rnn_model.predict(padded_sequence)
#             prediction_id = np.argmax(prediction)
#         else:
#             # TF-IDF based models (KNC or MNB)
#             input_vector = tfidf.transform([cleaned_text])
#             if selected_model == 'knc':
#                 prediction_id = knc.predict(input_vector)[0]
#             elif selected_model == 'mnb':
#                 prediction_id = mnb.predict(input_vector)[0]
                
#         predicted_category = category_mapping.get(prediction_id, "Unknown")
        
#         # 3. Get Recommendations
#         input_vector_for_rec = tfidf.transform([cleaned_text])
#         similarity_scores = cosine_similarity(input_vector_for_rec, tfidf_matrix)
#         similar_indices = similarity_scores[0].argsort()[-6:-1][::-1]
        
#         recommendations = []
#         for i in similar_indices:
#             recommendations.append(df['Category'].iloc[i])
            
#         # To ensure we have 5 unique recommendations
#         unique_recommendations = list(dict.fromkeys(recommendations))
        
#         return render_template('index.html',
#                                predicted_category=f'Predicted Category: {predicted_category}',
#                                recommendations=unique_recommendations,
#                                resume_text=resume_text,
#                                selected_model=selected_model)

# if __name__ == '__main__':
#     app.run(debug=True)


import numpy as np
import pandas as pd
import re
import pickle
from flask import Flask, request, render_template

# Import Keras load_model function
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Load all the essential files ---

# Load TF-IDF Vectorizer
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# Load the K-Nearest Neighbors and Naive Bayes models
knc = pickle.load(open('knc.pkl', 'rb'))
mnb = pickle.load(open('mnb.pkl', 'rb'))

# --- Load the RNN Model and its Tokenizer ---
rnn_model = load_model('rnn.h5')
rnn_tokenizer = pickle.load(open('rnn_tokenizer.pkl', 'rb'))

# --- Load the pre-calculated category similarity matrix ---
with open('category_similarity.pkl', 'rb') as f:
    category_similarity, category_list = pickle.load(f)

# Category mapping from your notebook
category_mapping = {
    15: "Java Developer", 6: "Data Science", 23: "Testing", 20: "SAP Developer", 24: "Web Designing",
    12: "HR", 13: "Hadoop", 3: "Blockchain", 10: "ETL Developer", 18: "Operations Manager",
    22: "Sales", 16: "Mechanical Engineer", 1: "Arts", 7: "Database", 11: "Electrical Engineering",
    14: "Health and fitness", 19: "PMO", 4: "Business Analyst", 9: "DotNet Developer",
    2: "Automation Testing", 17: "Network Security Engineer", 21: "Python Developer",
    5: "Civil Engineer", 0: "Advocate",
}

# --- Helper function for cleaning resume text ---
def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text

# --- Initialize Flask App ---
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        resume_text = request.form['resume']
        selected_model = request.form['model']
        
        cleaned_text = clean_resume(resume_text)
        prediction_id = 0
        
        if selected_model == 'rnn':
            sequence = rnn_tokenizer.texts_to_sequences([cleaned_text])
            padded_sequence = pad_sequences(sequence, maxlen=500)
            prediction = rnn_model.predict(padded_sequence)
            prediction_id = np.argmax(prediction)
        else:
            input_vector = tfidf.transform([cleaned_text])
            if selected_model == 'knc':
                prediction_id = knc.predict(input_vector)[0]
            elif selected_model == 'mnb':
                prediction_id = mnb.predict(input_vector)[0]
                
        predicted_category = category_mapping.get(prediction_id, "Unknown")
        
        # --- NEW Recommendation Logic ---
        recommendations = []
        if predicted_category != "Unknown":
            # Find the index of the predicted category in our category list
            category_idx = category_list.get_loc(predicted_category)
            # Get the similarity scores for that category
            similarity_scores = category_similarity[category_idx]
            # Get the indices of the top 5 most similar categories (excluding itself)
            similar_indices = similarity_scores.argsort()[-6:-1][::-1]
            # Get the names of those categories
            recommendations = [category_list[i] for i in similar_indices]
        
        return render_template('index.html',
                               predicted_category=f'Predicted Category: {predicted_category}',
                               recommendations=recommendations,
                               resume_text=resume_text,
                               selected_model=selected_model)

if __name__ == '__main__':
    app.run(debug=True)