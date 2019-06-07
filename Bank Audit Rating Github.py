#!/usr/bin/env python
# coding: utf-8

# # 1 Business Problem

# One of the major reason why the Canadian banking system continues to thrive despite constant changes in financial innovations and technological integrations are due to the impact of OSFI's (Office of the Superintendent of Financial Institutions) stringent rules and regulations.
# 
# As a result, all Canadian banks and financial institutions undergo audits periodically in order to satisfy the OSFI's financial regulations and requirements.
# 
# Furthermore, financial institutions also perform their own internal audits and third party audits to ensure quality control for liquidity management and securitization respectively.
# 
# When it comes to audit metrics, most requirements reviewed are numerical, with a result of pass or fail (ie. Does portfolio A satisfy the parameters of a specific guideline?). However, there are other metrics such as decision making approach that are not numerical, and thus, audit results may vary depending on each individual Auditor's own merits.
# 
# For example, an Auditor would check off each criteria in order to satisfy a Bank's own policy for a given Client under a certain product. However, sometimes an exception is made when a Client does not meet all the criteria given circumstances in the real world, but as long as these risks are mitigated while still satisfying the Bank's policy, then the Auditor should continue to check off all the criteria.
# 
# An example like this is why audit results can vary from a case by case scenario. If the exception that was made above satisfies the Bank's policy 100% of the time, Auditors would still disagree 20% of the time. The reason for this is that Humans can make bias errors and opinion errors from our decision making process, while on the other hand, computers do not make the same bias errors and opinion errors, but instead make different errors which is not comparable to Human errors (ie. understanding jokes, sarcasm, etc).
# <br>
# <br>
# 
# **Business Solution:**
# 
# To solve this issue, I have created an internal "Audit Rating System" that learns and understands a Bank's decision making process for a certain product of a department, with the objective to avoid the errors that Auditors can potentially make 20% of the time.
# 
# We will perform a Sentiment Analysis with Random Forest Classifier on an entire department's deal notes for a particular product.
# 
# <u>Note</u>: Deal notes are the key factor here, since this is where exceptions for mitigating risks are explained by the bank, and reviewed by the Auditor.
# 
# We will also identify and display keywords that arise in an audit under any variations (ie. 1 word, 2 words, 1-2 words, etc). This feature is useful to see how well the department is performing, and if the performance is meeting the Bank's expectations.
# <br>
# <br>
# 
# **Benefits:**
# - Develop a Bank's own audit rating system using Machine Learning
# - Improve the quality of financial products for securitization
# - Collectively understand an entire department's decision making approach
# - Anticipate internal/external audits
# - Ensure risk mitigations are explained in detail in accordance to a Bank's expectations
# <br>
# <br>
# 
# **Robustness:**
# 
# To ensure the model performs well on the new data, we will implement the following measures to avoid over-fit the training data:
# - Split the data into 80% training data, and 20% testing data. The testing data will not be touched until after the model construction is complete
# - Implement the Random Forest Model which runs the Decision Tree Model repeatedly (this ensures that the model learns the entire department's decision making approach accurately)
# - Apply cross validation after running the Random Forest Model on the training data to find the best outcome (this further ensures the model learns the entire department's decision making approach, as well as each individual banker's unique decision making approach)
# 
# <u>Note</u>: Although we were given permission to showcase this project, however, we will not showcase any relevant information from the actual dataset for privacy protection. 
# <br>
# <br>
# 
# **Assumptions:**
# - We are using data with a sample size of 511 from a department that has newly implemented their policies, and assuming that this dataset is sufficient to capture every banker's decision making approach
# - There are more Clients declined than approved, however, since the length of deal notes for approvals are significantly longer than the length of deal notes for declines, then we're assuming that the longer length deal notes for approvals will balance out the lesser number of Clients approved
# <br>
# <br>
# 
# **Future:**
# 
# This model is the first step for implementing an internal "Audit Rating System". In the future, we will build additional models with other datasets, to create an auto approval system, and other innovative ways to enhance a Financial Institution's capabilities using Machine Learning and Deep Learning.

# # 2 Data Overview

# In[1]:


import pandas as pd
csv = ("audit_rating_banking.csv")
df = pd.read_csv(csv, encoding='latin1')
df.head(0)


# Here we see the columns for our dataset, which includes Names, Deal number, Decision and Deal Notes.
# <br><u>Note</u> that actual data were not shown for privacy protection.

# In[2]:


df1 = df.copy()
df1["Deal Notes"] = df1["Deal Notes"].apply(lambda x : x.rsplit(maxsplit=len(x.split())-4)[0]) # sets the character limit to 4 words
df1.loc[2:4,'Decision':'Deal Notes']


# I have also wrote a function showing a sneak peak of the data:
# - We see that under "Decision", there are two attributes: Approve and Denied
# - We also see the "Deal Notes" of each Banker's decision making process (only the first 4 words are shown)

# # 3 Splitting data in Training Set and Testing Set

# First we need to see if there are any missing values in our dataset.

# In[3]:


print("Before dropping missing data: {}".format(len(df)))
df = df.dropna(subset=["Deal Notes"])
print("After dropping missing data: {}".format(len(df)))


# Next we will split the data into 80% training data and 20% testing data to avoid over-fitting.

# In[4]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
for train_index, test_index in split.split(df, df["Decision"]): 
    strat_train = df.reindex(train_index)
    strat_test = df.reindex(test_index)
print("Number of Training Data: " + str(len(strat_train)),
      "\nNumber of Testing Data: " + str(len(strat_test)))


# In[5]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

strat_train["Decision"].value_counts()/len(strat_train)
a = strat_train["Decision"].value_counts()/len(strat_train)
ax = a.plot(kind='bar', title='\nStratified Splitting Proportions on Training Data')
ax.set_xlabel('Decision')
ax.set_ylabel('Percentage of total Training Data')
plt.show()
print(strat_train["Decision"].value_counts()/len(strat_train))


strat_test["Decision"].value_counts()/len(strat_test)
b = strat_test["Decision"].value_counts()/len(strat_test)
bx = b.plot(kind='bar', title='\n\nStratified Splitting Proportions on Testing Data')
bx.set_xlabel('Decision')
bx.set_ylabel('Percentage of total Testing Data')
plt.show()
print(strat_test["Decision"].value_counts()/len(strat_test))


# We have successfully split the data into 80% training data and 20% testing data. As shown in the bar graphs above, both training set and testing set were split evenly, instead of splitting it right down the middle.

# # 4 Modeling

# ## 4.1 Set Target Variable

# In[6]:


X_train = strat_train["Deal Notes"]
X_train_targetSentiment = strat_train["Decision"]
X_test = strat_test["Deal Notes"]
X_test_targetSentiment = strat_test["Decision"]


# Our data is now assigned and ready for modeling.

# ## 4.2 Extract Features

# We will use SciKit-Learn's **CountVectorizer** to performs the following:
# - Text preprocessing:
#     - Tokenization (breaking sentences into words)
#     - Stopwords (filtering "the", "are", etc)
# - Occurrence counting (builds a dictionary of features from integer indices with word occurrences)
# - Feature Vector (converts the dictionary of text documents into a feature vector)
# 
# 
# Also, with longer documents, we typically see higher average count values on words that carry very little meaning, this will overshadow shorter documents that have lower average counts with same frequencies, as a result, we will use **TfidfTransformer** to reduce this redundancy:
# - Term Frequencies (**Tf**) divides number of occurrences for each word by total number of words
# - Term Frequencies times Inverse Document Frequency (**Tfidf**) downscales the weights of each word (assigns less value to unimportant stop words ie. "the", "are", etc)
# 
# 
# Lastly, we see that in our training set, there are 408 sample size with 3844 distinct words.

# In[7]:


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
print("Number of samples and number of features using CountVectorizer: " + str(X_train_counts.shape))

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer(use_idf=False)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print("\nNumber of samples and number of features using Tfidf: " + str(X_train_tfidf.shape))


# ## 4.3 Feature Pipeline

# For efficiency, we will create a pipeline to `fit_transform` the training data, and `transform` the testing data.

# In[8]:


from sklearn.pipeline import Pipeline
feat_pipe = Pipeline([('count_vect', CountVectorizer()), ('tfidf', TfidfTransformer())])
X_train = feat_pipe.fit_transform(X_train) 
X_test = feat_pipe.transform(X_test)


# ## 4.4 Making Predictions with Different Classifiers

# With our features engineered and ready for making predictions, we will perform the following:
# - Run the features through a variety of classifiers using our training data only 
# - Test the accuracy of each classifiers by making predictions on the testing data

# **Multinominal Naive Bayes Classifier**

# In[9]:


import numpy as np
from sklearn.naive_bayes import MultinomialNB

multiNB_clf = MultinomialNB().fit(X_train, X_train_targetSentiment)
predicted_multiNB_clf = multiNB_clf.predict(X_test)
np.mean(predicted_multiNB_clf == X_test_targetSentiment)


# **Logistic Regression Classifier**

# In[10]:


from sklearn.linear_model import LogisticRegression

logReg_clf = LogisticRegression().fit(X_train, X_train_targetSentiment)
predicted_logReg_clf = logReg_clf.predict(X_test)
np.mean(predicted_logReg_clf == X_test_targetSentiment)


# **Support Vector Machine Classifier**

# In[11]:


from sklearn.svm import LinearSVC

linearSVC_clf = LinearSVC().fit(X_train, X_train_targetSentiment)
predicted_linearSVC = linearSVC_clf.predict(X_test)
np.mean(predicted_linearSVC == X_test_targetSentiment)


# **Decision Tree Classifier**

# In[12]:


from sklearn.tree import DecisionTreeClassifier

decisionTree_clf = DecisionTreeClassifier().fit(X_train, X_train_targetSentiment)
predicted_decisionTree_clf = decisionTree_clf.predict(X_test)
np.mean(predicted_decisionTree_clf == X_test_targetSentiment)


# **Random Forest Classifier**

# In[13]:


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) # hides deprecation warning
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier().fit(X_train, X_train_targetSentiment)
predicted_forest_clf = forest_clf.predict(X_test)
np.mean(predicted_forest_clf == X_test_targetSentiment)


# Looks like all the models performed very well (>90%), and we will use the **Random Forest Classifier** since it has the highest accuracy level at **99.03%**.
# <br>
# Now we will fine tune the Random Forest Classifier to avoid any potential over-fitting.

# ## 4.5 Fine-tune the Random Forest Classifier with Cross Validation

# Here we will run a **GridSearchCV** function which will optimize all the hyper-parameters on a grid of possible values. This is done by taking all available combinations of hyper-parameters and perform cross validations, by splitting the training data into 10 folds, and cross validate each fold against one another for 10 times. These cross validated results are then averaged to produce a generalized score, for the fitness of our Random Forest Model, and its optimal hyper-parameters.
# 
# 
# We also indicated 3 **n_estimators** x 4 **max_features** x 1 **bootstrap** x 10 **folds** = 120 trained random forest in our GridSearchCV below.
# 
# <u>Note</u> that each **n_estimator** hyper-parameter value represents the number of trees trained in the Random Forest, and we will set our hyper-parameters as following:
# - (**150 trees** x 4 max_features x 10 folds) + (**200 trees** x 4 max_features x 10 folds) + (**250 trees** x 4 max_features x 10 folds) = 24,000 individual trees trained in our Random Forest Model.
# 
# In addition, we will also optimize the number of CPU core utilization for efficient processing.
# 
# **One of the key factor** for running 24,000 individual trees is to avoid Humans from replicating keywords used to pass an audit, by teaching our model to learn an entire department's decision making approach. Therefore, even if we try to replicate certain keywords, the model will be robust enough to filter through those keywords based on its learning.

# In[14]:


from sklearn.model_selection import GridSearchCV
param_grid = [
    {'n_estimators': [150,200,250], 'max_features': [2, 4, 'log2', 'auto'], 'bootstrap': [False]}
]
grid_search = GridSearchCV(forest_clf, param_grid, cv=10, verbose=2, n_jobs=-1)
grid_search.fit(X_train, X_train_targetSentiment)


# In[15]:


final_model = grid_search.best_estimator_
final_model


# Now that our fine-tuning is complete, we will select the best estimator from the cross validation result, and prepare our final model for testing.

# ## 4.6 Testing the Final Model

# We will use some arbitrary text to see if our model is able to predict the quality of Deal Notes to be reviewed by an Auditor.
# 
# <u>Note</u> that the arbitrary text used here is not part of the training data or testing data, therefore, we are testing our model on data that our model has never seen before.

# In[16]:


new_text = ['Client is a receptionist at XYZ Inc...']

docs_prepared = feat_pipe.transform(new_text)
X_train_targetSentiment[final_model.predict(docs_prepared)]


# In[17]:


predictedFinal = final_model.predict(X_test)
np.mean(predictedFinal == X_test_targetSentiment)


# **Results:**
# - After testing with some arbitrary text that is not part of our training data or testing data, we see that our final model is performing correctly with output showing <u>Approved</u> as the result
# - We also see that the final model has an accuracy level of **97.09%**, which is slightly lower than our pre-fine-tuned model at 99.03% accuracy level
# - The 1.94% difference shows that the fine-tuning was able to capture some over-fitting and ensure that the model is able to learn the entire department's decision making approach, as well as each individual banker's unique decision making approach

# # 5 Detailed Analysis

# In[18]:


for performance_analysis in (grid_search.best_score_, 
                             grid_search.best_estimator_, 
                             grid_search.best_params_):
        print(performance_analysis)


# In our performance analysis, we see the following:
# - Best Mean Score: 98.04% which is very close to our accuracy level of 97.09%
# - Best Estimator: shown above
# - Best Parameter: shows "250 n_estimators", "auto max_features", and "bootstrap false"

# In[19]:


from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

print(classification_report(X_test_targetSentiment, predictedFinal))
print('Accuracy: {}'. format(accuracy_score(X_test_targetSentiment, predictedFinal)))


# Below is the summary of the classification report:
# - Precision: determines how many objects selected were correct
# - Recall: tells you how many of the objects that should have been selected were actually selected
# - F1 score measures the weights of recall and precision (1 means precision and recall are equally important, 0 otherwise)
# - Support is the number of occurrences of each class (X_test_targetSentiment and X_test)
# 
# The results from our analysis shows that, our model performs very well by understanding how the department decides if Clients are approved or declined. This is also supported by the high Precision, Recall, and F1-score ratings. Furthermore, we also tested our model above using some arbitrary text, which was very accurate. 
# 
# Also, our assumption above is validated by our results, given that the longer length in deal notes for approvals has balanced out the lesser number of Clients approved.
# 
# Overall, the Precision, Recall, F1-score, and test results, along with our validated assumption confirms the effectiveness of this model.

# In[20]:


from sklearn import metrics
metrics.confusion_matrix(X_test_targetSentiment, predictedFinal)


# From the confusion matrix above, we see that our model shows minimum confusion between approval and declines based on the entire department's decision making approach. The numbers in the matrix is based on the total support number above (the number of occurrences of each class in our testing data).
# 
# Although there is a 2.91% chance (3/(30+70+3)) that our model may confuse an approval with a denial, however this value is very marginal, and our model already exceeds the 20% errors that Auditors can potentially make based on bias errors and opinion errors.
# 
# Therefore, we conclude that our model is able to learn and understand a Bank's decision making approach, for a certain product of a department, to avoid errors that Auditors can potentially make 20% of the time.

# # 6 Text Visualization with Eli5

# Now that we have completed our model, we can also to take a step further by identifying the keywords that were used in our matrix for our Audit Rating System.
# 
# In order to display the keywords, we will perform the following:
# - Reset the target variables
# - Extract the features again but using a **TfidfVectorizer** hybrid (this is important for working with the Eli5 package)
# - Perform our Random Forest Classifier once more
# - Fine-tune our Random Forest Classifier with cross validation once more
# - Apply the final model with **Eli5** to display the keywords for visualization

# ## 6.1 Set Target Variable

# In[21]:


X_train = strat_train["Deal Notes"]
X_train_targetSentiment = strat_train["Decision"]
X_test = strat_test["Deal Notes"]
X_test_targetSentiment = strat_test["Decision"]


# ## 6.2 Extract Features

# In[22]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,1))
# Display keywords that arise in an audit under any variations
tfidf_vec.fit(X_train.values.tolist())
X_train = tfidf_vec.transform(X_train.values.tolist())
X_test = tfidf_vec.transform(X_test.values.tolist())


# ## 6.3 Perform Random Forest Classifier

# In[23]:


forest_clf = RandomForestClassifier().fit(X_train, X_train_targetSentiment)


# ## 6.4 Fine-tune the Random Forest Classifier with Cross Validation

# In[24]:


param_grid = [
    {'n_estimators': [150,200,250], 'max_features': [2, 4, 'log2', 'auto'], 'bootstrap': [False]}
]
grid_search = GridSearchCV(forest_clf, param_grid, cv=10, verbose=2, n_jobs=-1)
grid_search.fit(X_train, X_train_targetSentiment)
final_model = grid_search.best_estimator_


# ## 6.5 Displaying Text Visualization

# In[25]:


import eli5
eli5.show_weights(final_model, vec=tfidf_vec, top=3)


# Using the Eli5 package, we can show a list of all the keywords, and their respective weights that impacts the model's ability to learn the department's decision making approach. 
# 
# This is usefully in evaluating how well the department is performing, and if the performance is meeting the Bank's expectations for audits.
# 
# Also, given that our model is robust enough from the cross validation fine-tuning, even if Humans try to replicate certain keywords to manipulate the system, our model can still perform accurately, since the system will not rely solely on a set of specific keywords, but instead on the entirety of its learning.
# 
# <u>Note</u>: 
# - Under the extract features section where we implemented "TfidfVectorizer", we can adjust the "ngram_range" parameter to show any variation of keywords and sentences (ie. 1 word, 2 words, 1-2 words, 3-10 words sentence, 6 words sentence, etc)
# - We can also adjust the "top" parameter in "eli5.show_weights" and assign it to "top = 3617" to view all the keywords and their weights in the entire data set
# 
# 
# For privacy protection, we will only display keywords in 1-gram for the top 3 result.

# # 7 Answering the Questions

# From the analysis above on the classification report, we see that the model is able to perform as an "Internal Audit System" for understanding the department's decision making approach on Client approvals and denials. With this knowledge, we can use our model to avoid the 20% errors that Auditors can potentially make in bias errors and opinion errors.
# 
# Also, we solidified the model by taking various steps to avoid biasness and over-fitting. This was done by splitting the data between training data and testing data, performing Random Forest Classifier, and implementing cross validation on the Random Forest Classifier to ensure the robustness of our model, such that, even if Humans try to replicate certain keywords, the model will perform based on the entirety of its learning.
# 
# As result, we tested the model on some arbitrary text, which showed strong performance in predicting results using an department's decision making approach.
# 
# Furthermore, we used Eli5 to display keywords and key sentences, along with their respective weights that impacts the model's ability to learn the department's decision making approach. This is especially usefully for evaluating how well the department is performing, and if the performance is meeting the Bank's expectations for an audit.
# 
# Lastly, based on our modeling, the robustness of our model, our test results, and our text visualization, we conclude that our model is able to learn and understand a Bank's decision making approach for a certain product, of a department, to avoid errors that Auditors can potentially make 20% of the time.
