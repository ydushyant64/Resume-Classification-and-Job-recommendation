# Project Report: Resume Classification and Job Recommendation System

**Author:** [Dushyant Pratap Singh]

---

## 1. Introduction

In the modern job market, recruiters are often inundated with thousands of resumes for a single job opening. Manually sorting through these documents to find qualified candidates for various roles is a time-consuming and inefficient process. This project addresses this challenge by developing an intelligent system that automates the classification of resumes into predefined job categories and provides relevant job recommendations. By leveraging Natural Language Processing (NLP) and machine learning, this system serves as a powerful tool to streamline the initial stages of the recruitment process.

---

## 2. Objectives

-   To develop a robust text-cleaning pipeline to preprocess raw resume text data, making it suitable for machine learning analysis.
-   To build and evaluate multiple machine learning models for classifying resumes into 25 distinct job categories.
-   To implement a content-based recommendation system that suggests similar job roles based on a candidate's predicted field.
-   To deploy the final system as an interactive and user-friendly web application using the Flask framework, allowing for real-world use.

---

## 3. Methodology and Implementation

The project was executed in several distinct phases:

#### a) Data Collection and Preprocessing
The foundation of this project is the "Resume Dataset," which contains resumes categorized into 25 job roles. A comprehensive text-cleaning function was developed to handle raw text by removing URLs, special characters, and other irrelevant data.

#### b) Feature Engineering
The cleaned text data was converted into numerical vectors using the **Term Frequency-Inverse Document Frequency (TF-IDF)** technique. This method effectively captures the importance of different words in a resume. For the deep learning model, a `Tokenizer` was used to convert text into integer sequences.

#### c) Model Development
Three different classification models were trained and evaluated to find the best-performing approach:
1.  **K-Nearest Neighbors (KNC):** A simple, instance-based learning algorithm.
2.  **Multinomial Naive Bayes (MNB):** A probabilistic classifier well-suited for text classification tasks.
3.  **Recurrent Neural Network (RNN):** A deep learning model designed to handle sequential data like text, capturing context and word order.

#### d) Recommendation System
A content-based recommendation engine was built using **Cosine Similarity**. By calculating the similarity between the average TF-IDF vectors of each job category, the system can identify and recommend the top 5 most similar job roles to a candidate's predicted category.

#### e) Web Application and Deployment
The entire system was encapsulated in a web application built with **Flask**. The user interface, created with HTML and styled with CSS, allows users to paste a resume, select a model, and receive an instant prediction and a list of job recommendations. The final application was deployed on **Hugging Face Spaces** for public access.

---

## 4. Libraries Used

This project utilizes several key Python libraries:

-   **Flask:** For building and serving the web application.
-   **Scikit-learn:** For machine learning tasks, including TF-IDF vectorization, K-Nearest Neighbors, Naive Bayes, and Cosine Similarity.
-   **TensorFlow & Keras:** For building and training the Recurrent Neural Network (RNN) model.
-   **Pandas:** For data manipulation, loading, and cleaning the resume dataset.
-   **NumPy:** For efficient numerical operations.
-   **Gunicorn:** As the production web server for the Flask application.
-   **Matplotlib & Seaborn:** For data visualization during the exploratory data analysis phase in the Jupyter notebook.

---

## 5. Results

All three models demonstrated the ability to classify resumes with a high degree of accuracy. The deployed web application provides a seamless and interactive user experience, successfully integrating the complex backend logic. The recommendation system effectively suggests relevant alternative roles, adding significant value for both job seekers and recruiters.

---

## 6. Conclusion

This project successfully demonstrates the power of NLP and machine learning in solving real-world HR challenges. The developed Resume Classification and Job Recommendation system is a practical, scalable tool that can significantly enhance the efficiency of the recruitment pipeline. Future work could involve expanding the number of job categories, incorporating more advanced deep learning architectures like Transformers, and further refining the recommendation logic.
