# Unintended-Bias-in-Text-Classification

This project mainly focuses on mitigating and measuring unintended identity bias in the classification of social media comments. As the dataset used in this project (https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data?select=train.csv) contains comments with identity related word that are mostly toxic. So the machine learning model also tries to predict comments that has identity related words as predominantly toxic. This problem was discussed in the paper named Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification (https://www.semanticscholar.org/reader/b611a8095630557229dc5fb6b07c272f1cd614da). Here they are proposing a performance metric that can measure the bias of model towards comments having identity related words.

I took the dataset from kaggle and cleaned it. After cleaning, there were about 1.7 million comments. It was vectorized into Bag of Words, TfIdf vectors and Word to vector vectors. But inorder to mitigate the aforementioned bias, I scraped some text data from wikipedia that contained identity related terms. After cleaning it was around 14 thousand different sentences. Feature engineering was done to extract feature like, length of text, number of words, average length of words, subjectivity, sentiment, parts of speech counds, sentence count, average sentence length etc. I trained a Naive Bayes model after doing hyperparameter tuning for the value of laplace smoothing. I did it for both Bag of Words and TfIdf vectors. I calculated ROC-AUC score for all models and additionally wrote a funtion for a custom performance metric that provides the measure of identity bias. I did the same for the dataset in Kaggle first and then after adding the scraped data. When compared to the size of original of dataset size of 1.7 million comments, the scraped text had only 14 thousand sentences. Despite this huge difference in size there was a slight increase in the performance in terms of identity bias.and also in the overall ROC-AUC score.

The word to vector vectorized text was modeled using a Gradient Boosted Decision Tree model. But in this case, I wasn't able to make use of all the data I had because of the unavailability of resources in colab and in my laptop. So I sampled 20% of mu data and trained a GBDT model but the performance wasn't as expected when compared to the Naive Bayes model due to the lower amount of training data.

I then created a Flask app and deployed the model with the highest performance.
