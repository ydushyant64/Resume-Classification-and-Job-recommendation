# Resume-Classification-and-Job-recommendation
An NLP-powered job recommendation engine that classifies resumes using KNN, Naive Bayes, and RNN models. Deployed as a live web application with Flask on Render, achieving 98.45% accuracy.

## 1. Introduction
In the modern job market, recruiters are often inundated with thousands of resumes for a single job opening. Manually sorting through these documents to find qualified candidates for various roles is a time-consuming and inefficient process. This project addresses this challenge by developing an intelligent system that automates the classification of resumes into predefined job categories and provides relevant job recommendations. By leveraging Natural Language Processing (NLP) and machine learning, this system serves as a powerful tool to streamline the initial stages of the recruitment process.
Project Overview: Resume Classification and Job Recommendation System

This project automates the initial stages of recruitment by classifying resumes into job categories and providing relevant job recommendations.

Key Objectives:

    Classify Resumes: Use Natural Language Processing (NLP) and machine learning to categorize resumes into one of 25 job roles.

    Recommend Jobs: Suggest the top 5 most similar job categories based on the initial classification.

    Deploy as a Web App: Create an interactive web application using Flask for real-world use.

Methodology:

    Preprocessing: Cleaned raw resume text to remove irrelevant data.

    Model Training: Converted text to numerical vectors using TF-IDF and trained three models: K-Nearest Neighbors, Naive Bayes, and a Recurrent Neural Network (RNN).

    Recommendation Engine: Built a system using Cosine Similarity to find and rank the most similar job categories.

    Deployment: The final model was deployed as a web application on Hugging Face Spaces.

Conclusion:

This project successfully created an intelligent tool that streamlines the resume screening process, demonstrating the practical application of machine learning in Human Resources.
