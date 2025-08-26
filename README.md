# IMDb Movie Review Sentiment Analysis  

##  Project Overview  
This project applies **Natural Language Processing (NLP)** techniques to classify IMDb movie reviews as **positive or negative**. By leveraging text preprocessing, feature extraction, and machine learning models, the project develops a classification system that can predict sentiment with high accuracy.  

The insights from this analysis can be useful for **movie producers, critics, and streaming platforms** to understand public opinion and guide content or marketing strategies.  

---

##  Problem Statement  
The goal is to build a machine learning model that predicts the **sentiment** of a movie review (positive/negative).  
- Input: Review text  
- Output: Sentiment label (`positive` / `negative`)  

---

##  Dataset Information  
- **Source**: IMDb Movie Reviews dataset  
- **Features**:  
  - Text of the review  
  - Sentiment label (positive/negative)  

---

##  Project Workflow  

### 1. Data Exploration & Preprocessing  
- Checked for missing values, imbalanced classes, and review length distribution  
- Cleaned text: removed punctuation, stopwords, special characters  
- Applied tokenization, lemmatization, and stemming  

### 2. Feature Engineering  
- Extracted features using:  
  - **Bag of Words**  
  - **TF-IDF Vectorization**  
  - **Word embeddings (Word2Vec/Glove)**  
- Additional textual features: word count, character count, average word length  

### 3. Model Development  
- Trained and compared multiple classifiers:  
  - Logistic Regression  
  - Naive Bayes  
  - Support Vector Machine (SVM)  
  - Random Forest  
  - Deep Learning models (LSTM, BERT for advanced experimentation)  

### 4. Model Evaluation  
- Metrics used: **Accuracy, Precision, Recall, F1-score, ROC-AUC**  
- Visualization: Confusion matrix, word clouds, feature importance plots  


---

---

##  Tools & Libraries Used
- **Python**: Pandas, NumPy  
- **NLP**: NLTK, spaCy, scikit-learn (TF-IDF, CountVectorizer)  
- **Machine Learning**: Logistic Regression, Naive Bayes, SVM, Random Forest, XGBoost  
- **Deep Learning**: TensorFlow/Keras (LSTM, BERT)  
- **Visualization**: Matplotlib, Seaborn, WordCloud  

---

##  Outcomes  
- Built and compared multiple NLP models for sentiment classification  
- Identified **Logistic Regression / SVM** as strong baseline models  
- Advanced models (LSTM, BERT) provided deeper insights with higher accuracy  
- Generated visualizations like word clouds and confusion matrices for interpretability  
- Delivered a concise report and presentation summarizing the workflow and key findings  

---

This project demonstrates skills in **text preprocessing, feature engineering, NLP modeling, and model evaluation** — showcasing the ability to turn raw text into actionable insights.  
