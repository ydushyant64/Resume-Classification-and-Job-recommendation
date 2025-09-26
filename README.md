# Project Report: Resume Classification and Job Recommendation System

**Author:** [Dushyant Pratap Singh]

## Live Demo

You can test the live application here:

**[https://huggingface.co/spaces/ydushyant64/Resume-Classification-and-Job-recommendation](https://huggingface.co/spaces/ydushyant64/Resume-Classification-and-Job-recommendation)**

![Application Screenshot](screenshot.PNG)


---

# 🤖 AI-Powered Resume Sorter & Job Recommender

Ever felt overwhelmed by a mountain of resumes? This project is my answer to that problem. I've built an intelligent web application that automatically sorts resumes into the right job categories and even suggests other roles the candidate might be a great fit for.

---

## Live Demo & A Quick Look

Curious to see it in action? You can try out the live application right here:

**[https://huggingface.co/spaces/dushyant64/Resume-Classification-and-Job-recommendation](https://huggingface.co/spaces/dushyant64/Resume-Classification-and-Job-recommendation)**

Here’s a snapshot of what the app looks like:

![Application Screenshot](screenshot.PNG)

*(To make this image appear, just upload a screenshot of your webpage to the repository and name it `screenshot.PNG`)*

---

## The Big Idea

The goal was simple: build a smart tool to make the first step of hiring—resume screening—faster and more accurate. I wanted to create something that could:
-   Automatically categorize any resume into one of 25 job roles.
-   Recommend other suitable jobs for a candidate.
-   Be easy enough for anyone to use through a simple web interface.

---

## How It's Built: From Raw Text to Smart Predictions

Here’s a peek into the journey from a simple resume file to a full-blown prediction and recommendation:

#### 1. The Foundation: The Dataset
This whole project started with the **"Resume Dataset,"** a collection of **962 real-world resumes**, each already sorted into one of 25 job categories. Before any magic could happen, I wrote a script to clean up the text—stripping out messy things like web links, special characters, and other noise.

#### 2. Teaching the Machine to Read
A computer can't understand words directly, so I had to turn the resumes into numbers. I used two main techniques for this:
-   **TF-IDF (Term Frequency-Inverse Document Frequency):** A classic method that figures out which words in a resume are most important.
-   **Tokenizer:** For my deep learning model, I converted sentences into sequences of numbers, which helps the model understand the order and context of words.

#### 3. The Brains of the Operation: The Models
I didn’t just build one model; I built three to see which one would do the best job. Each has a different way of "thinking":
-   **K-Nearest Neighbors (KNC):** A straightforward model that finds the "closest" matching resumes from the ones it's already seen.
-   **Multinomial Naive Bayes (MNB):** A smart probabilistic model that's great at handling text.
-   **Recurrent Neural Network (RNN) with LSTM:** A deep learning model that reads resumes like a human does—sequentially—allowing it to understand the context and flow of the text.

#### 4. The "What's Next?" Feature: Job Recommendations
After figuring out a candidate's primary job category, I wanted the app to suggest other roles they might be interested in. Using **Cosine Similarity**, the app calculates which job categories are most similar to the predicted one and recommends the top five alternatives.

---

## The Scoreboard: How Well Did the Models Perform?

After training, I tested the models on a set of resumes they had never seen before. The results were fantastic:

| Model | Test Accuracy |
| :--- | :--- |
| **K-Nearest Neighbors (KNC)** | 97.41% |
| **Multinomial Naive Bayes (MNB)** | 98.44% |
| **Recurrent Neural Network (RNN)**| 98.96% |

The RNN model came out on top, which makes sense as its architecture is designed to understand the nuances of language.

---

## The Toolbox: Libraries & Tech Used

This project wouldn't have been possible without these amazing tools:
-   **Flask:** For building the web app's backend.
-   **Scikit-learn:** The workhorse for the traditional machine learning models (TF-IDF, KNC, MNB, Cosine Similarity).
-   **TensorFlow & Keras:** For building and training the powerful RNN model.
-   **Pandas & NumPy:** For all the data handling and number-crunching.
-   **Gunicorn:** To run the Flask app in a production environment.
-   **Matplotlib & Seaborn:** For creating the charts and graphs during my initial data analysis.

---

## Final Thoughts & What’s Next

This project was a fantastic journey into solving a real-world problem with NLP and machine learning. The result is a fast, accurate, and scalable tool that could genuinely help streamline any company's hiring process.

Looking ahead, it would be exciting to expand the number of job categories, experiment with even more advanced models like BERT, and maybe even add a feature to match resumes directly to live job postings.

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
The foundation of this project is the **"Resume Dataset,"** which contains **962 unique resumes** categorized into 25 job roles. A comprehensive text-cleaning function was developed to handle raw text by removing URLs, special characters, and other irrelevant data.

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

## 4. Model Performance

The models were evaluated on a held-out test set, and their accuracy scores are as follows:

| Model | Test Accuracy |
| :--- | :--- |
| **K-Nearest Neighbors (KNC)** | 98.44% |
| **Multinomial Naive Bayes (MNB)** | 93.78% |
| **Recurrent Neural Network (RNN)**| 98.45% |

---

## 5. Libraries Used

This project utilizes several key Python libraries:

-   **Flask:** For building and serving the web application.
-   **Scikit-learn:** For machine learning tasks, including TF-IDF vectorization, K-Nearest Neighbors, Naive Bayes, and Cosine Similarity.
-   **TensorFlow & Keras:** For building and training the Recurrent Neural Network (RNN) model.
-   **Pandas:** For data manipulation, loading, and cleaning the resume dataset.
-   **NumPy:** For efficient numerical operations.
-   **Gunicorn:** As the production web server for the Flask application.
-   **Matplotlib & Seaborn:** For data visualization during the exploratory data analysis phase in the Jupyter notebook.

---

## 6. Conclusion

This project successfully demonstrates the power of NLP and machine learning in solving real-world HR challenges. The developed Resume Classification and Job Recommendation system is a practical, scalable tool that can significantly enhance the efficiency of the recruitment pipeline. Future work could involve expanding the number of job categories, incorporating more advanced deep learning architectures like Transformers, and further refining the recommendation logic.
