
# coding: utf-8

# # Young people Survey:Youth  Spending on Healthy food
# ## 1.Overview

# The author of the dataset is  Miroslav Sabo.The survey titled 'Young people survey' is aimed at finding out the preferences, interests,habits and perceptions on many topics like spending,whether they reported living a healthy lifestyle, personal information, their views on life etc.FOr this assignment we will be studying about individuals in the datset and will be finding out  whether they will end up spending on healthy food.
# 
# The survey was created by Students of Statistics Fakulta sociálnych a ekonomických vied UK which is a school in Bratislava, Slovakia.The research questions consists of numerical score in following group: music preferences,movie preferences,interests,phobias,healthy habits,outlook towards life,personality traits and demographic data of the survey takers.The dataset has 1010 rows and 150 columns(out of which 139 are integeral and 11 are categorical.)he survey was presented both in electronic and written form.It was orginally taken in Slovak language and later was translated to English.
# The age of participants is  15-30 years.
# 
# 
# <br/>
# We are going to break break this notebook into the following sections:
# <ol>
#     <li>Data Processing </li>
#     <li>Data Exploration and Visualization</li>
#     <li>Data Classification</li>
#     <li>Prediction and accuracy</li>
# </ol>

# <b>(d)what software did you use and why did you choose it?</b>

# I used Jupyter notebook because it is offline and I can put my code and analysis in the same place.I dont have to context switch.It can be run on any computer not even requires internet connection.It autogenerates a  HTML etc which it can be viewed on any browser and thus any kind of machine.The code and it's output/analysis can be seen right below the other

# The following questions will be answered below:
# Questions:<br/><br/>
#   (a)What is your	data	and	task?(In writeup)	<br/>
#  (b)what ML	solution did	you	 choose	and,	most	importantly, why was this an appropriate	choice?(Part 2)<br/>
# (c)how	did	you	choose	to	evaluate	 success?(Part 5.2.1 & 5.2.2)<br/>
# (d)what	software	did	you	use	and	why	did	you	choose	it?<br/>
# (e)what	are	the	results?(Part 5)<br/>
# (f)show	 some	examples	from	the	development	data that	your	approach got	correct	and	some	it	got	wrong:	if	you	were to	try	to	fix	the	ones it	got	wrong, what	would you do?(Part 6)
# 
# 
# 
# 
# 

# ## 2.Data Preprocessing
# Here we are going to classify the dtaa into various subframes and also handle missing values and perform operations to clean the data as we will present several visualisation to improve that data has become better.

# In[1]:


import pandas as pd
import copy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import recall_score, precision_score
from sklearn.cross_validation import KFold, train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import sklearn.tree as tree
from IPython.display import Image
from os import path
from sklearn.preprocessing import Imputer


# In[2]:


data = pd.read_csv('data/responses.csv')
columns = pd.read_csv('data/columns.csv')


# Showing the dataset captured which is in responses.csv

# In[3]:


pd.set_option('display.max_columns',200)
data.head(10)


# This is how are data looks like before preprocessing shown are all the columns and their value distribution of each of them.

# In[4]:




# In[5]:


data = data.replace("NaN", np.nan)
data = data.replace("nan", np.nan)


# As the data can be categroised into:<br/>
# 1.Music Features<br/>
# 2.Movies Features<br/>
# 3.Hobbies and interests<br/>
# 4.Phobias <br/>
# 5.Health habits<br/>
# 6.Persoanlity traits(includes outlook towards life)<br/>
# 7.Spending habits <br/>
# 8.Demographics<br/>
# <br/>
# <br/>
# <b> Hence we will form dataframe of each and process it one by one.To clean and enhance the dataset

# In[6]:


music = data.iloc[:,0:19] 
movies = data.iloc[:,19:31] 
phobias = data.iloc[:,63:73] 
hobbies = data.iloc[:,31:63] 
health = data.iloc[:,73:76] 
personal = data.iloc[:, 76:133] 
spending = data.iloc[:,133:140]
demo = data.iloc[:,140:150] 


# ## 2.1 The Music SubFrame

# In[7]:


imp = Imputer(missing_values='NaN', strategy='median', axis=0)
imp.fit(music)
music_data = imp.transform(music)
music = pd.DataFrame(data=music_data[:,:],index=[i for i in range((len(music_data)))],columns=music.columns.tolist())


# In[8]:


music.describe()


# ## 2.2 The Movies Subframe 

# In[9]:


imp = Imputer(missing_values='NaN', strategy='median', axis=0)
imp.fit(movies)
movie_data = imp.transform(movies)
movies = pd.DataFrame(data=movie_data[:,:],index=[i for i in range((len(movie_data)))],columns=movies.columns.tolist())


# In[10]:


movies.describe()


# ## 2.3 The Hobbies subframe

# In[11]:


imp = Imputer(missing_values='NaN', strategy='median', axis=0)
imp.fit(hobbies)
hobbies_data = imp.transform(hobbies)
length=len(hobbies_data)
hobbies = pd.DataFrame(data=hobbies_data[:,:],
                     index=[i for i in range(0,len(hobbies_data))],
                     columns=hobbies.columns.tolist())


# In[12]:


hobbies.describe()


# ## 2.4 The phobias SubFrame

# In[13]:


imp = Imputer(missing_values='NaN', strategy='median', axis=0)
imp.fit(phobias)
phobia_data = imp.transform(phobias)
length=len(phobia_data)
phobias = pd.DataFrame(data=phobia_data[:,:],
                     index=[i for i in range(0,len(phobia_data))],
                     columns=phobias.columns.tolist())


# In[14]:


phobias.describe()


# ## 2.5 The health SubFrame

# In[15]:


health['Smoking'].unique()


# In[16]:


#Ref:https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.map.html
health['Smoking'] = health['Smoking'].map({'never smoked': 1, 'tried smoking': 2,'former smoker':3,'current smoker':4})


# In[17]:


health['Alcohol'].unique()


# In[18]:


health['Alcohol'] = health['Alcohol'].map({'drink a lot': 1, 'social drinker': 2,'never':3})


# In[19]:


imp = Imputer(missing_values='NaN', strategy='median', axis=0)
imp.fit(health)
health_data = imp.transform(health)
length=len(health_data)
health = pd.DataFrame(data=health_data[:,:],
                     index=[i for i in range(0,len(health_data))],
                     columns=health.columns.tolist())


# In[20]:


health.describe()


# ## 2.6 The personal SubFrame

# Selecting categorical Columns first and then wechange them to integral values to have uniformity in dataset which helps in Classification and Visualisation as we will see later on.

# In[21]:


personal['Punctuality'].unique()


# In[22]:


personal['Punctuality'] = personal['Punctuality'].map({'i am always on time': 1, 'i am often early': 2,'i am often running late':3})


# In[23]:


personal['Lying'].unique()


# In[24]:


personal['Lying'] = personal['Lying'].map({'never': 1, 'sometimes': 2,'only to avoid hurting someone':3,'everytime it suits me':4})


# In[25]:


personal['Internet usage'].unique()


# In[26]:


personal['Internet usage'] = personal['Internet usage'].map({'few hours a day': 1, 'most of the day': 2,'less than an hour a day':3,'no time at all':4})


# In[27]:


#Ref:http://scikit-learn.org/stable/auto_examples/plot_missing_values.html#sphx-glr-auto-examples-plot-missing-values-py
imp = Imputer(missing_values='NaN', strategy='median', axis=0)
imp.fit(personal)
personal_data = imp.transform(personal)
length=len(personal_data)
#Ref:https://pandas.pydata.org/pandas-docs/stable/dsintro.html
personal = pd.DataFrame(data=personal_data[:,:],
                     index=[i for i in range(0,len(personal_data))],
                     columns=personal.columns.tolist())


# In[28]:


personal.describe()


# ## 2.7 The Demographics Subframe

# In[29]:


demo['House - block of flats'].unique()


# In[30]:


demo['House - block of flats'] = demo['House - block of flats'].map({'block of flats': 1, 'house/bungalow': 2})


# In[31]:


demo['Village - town'].unique()


# In[32]:


demo['Village - town'] = demo['Village - town'].map({'village': 1, 'city': 2})


# In[33]:


demo['Gender'] = demo['Gender'].map({'male': 1, 'female': 2})


# In[34]:


demo['Education'].unique()


# In[35]:


demo['Education'] = demo['Education'].map({'currently a primary school pupil':1,'primary school' :2,
                                           'secondary school':3,'college/bachelor degree':4,
                                           'masters degree':5,'doctorate degree':6})


# In[36]:


demo['Only child'] = demo['Only child'].map({'yes':1,'no':0})


# In[37]:


demo['Left - right handed'] = demo['Left - right handed'].map({'left handed':1,'right handed':2})


# In[38]:


imp = Imputer(missing_values='NaN', strategy='median', axis=0)
imp.fit(demo)
demo_data = imp.transform(demo)
length=len(demo_data)
demo = pd.DataFrame(data=demo_data[:,:],
                     index=[i for i in range(0,len(demo_data))],
                     columns=demo.columns.tolist())


# In[39]:


demo.describe()


# ## 2.8 The Spending Subframe 

# In[40]:


imp = Imputer(missing_values='NaN', strategy='median', axis=0)
imp.fit(spending)
spending_data = imp.transform(spending)
length=len(spending_data)
spending = pd.DataFrame(data=spending_data[:,:],
                     index=[i for i in range(0,len(spending_data))],
                     columns=spending.columns.tolist())


# In[41]:


spending.describe()


# In[42]:


dfs = [demo,hobbies,phobias,health,personal,spending,music,movies]
new_data = pd.concat( dfs,axis=1)


# In[43]:


new_data


# ## 3.Exploratory Data Analysis

# A look at the dataframe to get insight on the Data and features inside so as to help us in performing operations later.

# In[44]:


print(new_data.describe())


# As we can see from below that there's uniformity in data types

# In[45]:


print(new_data.columns, new_data.dtypes)



# In[47]:




# Taking mean of values of actual  dataset to know the distribution of original dataset and then cross checking after making predictions

# In[49]:


spending['Spending on healthy eating'].describe()


# #### <b>3.1   Analysing for various datasets to check similarity and degree to which the target variable depends on them.That means,how much of target varibale value be showing variation for different kind of features.</b>

# Analysing how much of Alcoholic and Smokers will spend on healthy eating what is the variation among them with respect to the target variable


# In[51]:


# Having calculated the correlation coefficient above now we take scatter plots of columns which give us more insightful results like the ones which have a lower positive coefficient.So here we see Age in the first plot shows uniform(almost) distribution of preferences but height, we see more height more spending on healthy eating,less weight more spending on healthy eating as we see points gathered arround.
# 
# <br/>
# PS:As you can see I had tkaen for all columns of the dataset to do pairplotting but to show a good insight I have taken these columns as conveyed above and they bring out helpful analysis

# In[54]:



# In[55]:




# In[57]:



# Here we have picked up some of the positive correlation coefficient variables for analysis.These features are linguistically and as per the information from columns dataset likely to show some variation and useful analysis with target feature.So here we see that for high positive correlation coefficient values(comparitively).We see similar results as in with increase in preference of that variable the feature of interest variable also shows a increase in value.
# <br/>
# For the case of friends vs money people whoose desire is to have more than friends as we can see in this useful insight they are more likely to spend on Healthy food.
# <br/>But if we go for little lower correlation coefficient values like Self-criticism as people who criticize their decision might just improvise and start eating healthy next time.But here we see less self-critic people spend on healthy eating.And the data is well distributed over lower values of self-criticism than for higher values.

# ##### Predicting probability of Youth spending on Healthy food 

# In[58]:


success=len(new_data[(new_data['Healthy eating'] == 5) |  (new_data['Spending on looks'] ==5 ) |
             (new_data['Spending on gadgets'] ==5) | (new_data['Health'] == 5) |
             (new_data['Branded clothing'] == 5 )]
            [new_data['Spending on healthy eating']==5])
total_data = float(len(new_data))
probab_success = success/total_data*100
probab_success


# Using the most positive correlation coefficient variabl's which we will refer to as strong variables we predict the how much is probability that if people lead a healthy lifestyle or Spend on looks or spend on gadgets or if they prefer branded clothing then the probability estimate of they spending on Healthy eating is only 13.4%.
# 
# 
# <br/>
# So our aim for the model would be  13.4% or better accuracy of prediction

# ## 4.Data Prediction(Modeling) 

# <b>(b)what ML	solution did	you	 choose	and,	most	importantly, why was this an appropriate	choice?</b>

# Importing classifcation library to train our data.

# In[59]:


from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Splitting data just after cleaning into train/dev/test.So here I have taken 0.4 has training size.Therefore, my test data will be 20% and development data will be 20%.So that we have enough data to train as we have asked to tune the parameters of model on dev data

# In[60]:


new_data_train = new_data.drop(["Spending on healthy eating"], axis=1)
X = new_data_train
y = new_data['Spending on healthy eating']
# we choose random_state as 2015 so that we get same train test split everythime
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.4,random_state=2015)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)


# In[61]:


scaler = StandardScaler()  
scaler.fit(X_train)
X_train_scale = scaler.transform(X_train)  
X_val_scale = scaler.transform(X_val)
X_test_scale = scaler.transform(X_test) 


# Here for this large dataset I have used ExtraTreeClassifier of scikit learn also known as Extremely randomized trees.The benefit of it is improved accuracy and faster computation.At each iteration it picks up the entire sample and chooses decision boundary at random.Instead of cut point variance it uses cut point smoothing for classification.

# In[62]:


from sklearn.ensemble import ExtraTreesClassifier
#they are useful for large datset they are faster in computation  than RandomForest and always test random splits 
#so will give better accuracy dut to smoothing.They generalize very well as seen through experiments
model_ExtraTree = ExtraTreesClassifier()
model_ExtraTree.fit(X_val, y_val)
print("Imporatance of Feature: %s") % model_ExtraTree.feature_importances_


# Now we split this important data again into Training, validation and test data accordingly in orderm to perform classification.

# In[63]:


from sklearn.feature_selection import RFE
#Ref:http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html#sklearn.feature_selection.RFE
#making to select half of the features after recursive feature extraction 
#here we use the above ExtreTreeClassifier model as it gives us reasonable accuracy due to smoothing(described above)
#we remove half features here to improve further accuracy
rfe = RFE(model_ExtraTree,n_features_to_select=200,step=1)
classifier_rfe = rfe.fit(X_val, y_val)


# In[64]:


classifier_rfe.score(X_test,y_test)


# #### 4.1 Feature Selecction

# In[65]:


import pandas
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#Ref:http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
#here I chose Chi Square function becuase it selects the highest value of 50 features.Also as we know 
#chi-squared stats are representative of dependence between stochastive variables
#thus it weeds out variables independent of class and thus selects only top 50 as per training 
test = SelectKBest(score_func=chi2, k=50)
fit = test.fit(X_val, y_val)


# Here I haved used the ScikitLearn SelectKBest library to extract best features from the dataset.Hence we finally select those columns and use it as important data.As these features will have higher probability to predict the target variable(spending on healthy eating) because these features proved a higher F Score as shown below basis the Y value which is the target variable.Most of the features(columns) are also listed below.

# In[66]:


X_new = test.fit_transform(X_val, y_val)
#gettinng index of each feature in fit using get_support
#Ref:http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest.get_support
names = X_train.columns.values[fit.get_support()]
#get integer support of the concenrned features  here using scores_ which gives us F Scores of features with respect to target

#Ref:https://stackoverflow.com/questions/41897020/sklearn-selectkbest-how-to-create-a-dict-of-feature1score-feature2score
scores = fit.scores_[fit.get_support()]
#storing column names and their scores correspondingly so as to visualiza better
names_scores = list(zip(names, scores))
new_df = pd.DataFrame(data = names_scores, columns=['Column_Names', 'F_Scores'])

#sorting the DataFrame to see highest scoring columns at the top and then storing it a new dataFrame
new_df_sorted = new_df.sort_values(['F_Scores', 'Column_Names'], ascending = [False, True])
print(new_df_sorted.head(150))


# Taking those columns which were taken out as best related to the Target variable

# In[67]:


important_data = new_data[names[:20]].copy()


# Now we split this important data again into Training, validation and test data accordingly in orderm to perform classification.

# In[68]:


X = important_data
y = new_data['Spending on healthy eating']
# we choose random_state as 2015 so that we get same train test split everythime
X_train_imp, X_test_imp, y_train_imp, y_test_imp = train_test_split(X, y,test_size=0.4,random_state=2015)
X_test_imp, X_val_imp, y_test_imp, y_val_imp = train_test_split(X_test, y_test, test_size=0.5)


# In[69]:


scaler = StandardScaler()  
scaler.fit(X_train_imp)
X_train_imp_scale = scaler.transform(X_train_imp)  


# In[70]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# selecting only Logistic regression to probabilistically compute decision boundaries
#after we have only selected important features
model = LogisticRegression()
fit = model.fit(X_val_imp, y_val_imp)


# In[71]:


fit.score(X_test_imp,y_test_imp)


# ## 5. Prediction and Data Validation 

# ## 5.1 Validation Data

# In[72]:


y_pred=classifier_rfe.predict(X_test_scale)
y_pred


# Here as we have selected very few columns based on F Score with target class so prediction is better

# In[73]:


y_pred_imp=fit.predict(X_test_imp)
y_pred_imp


# In[74]:




# In[75]:



# <b>(c)how did you choose to evaluate success?</b>
# 
# Ans 5.2.1 and 5.2.2

# ## 5.2.1 Classification Metrics 
# ### Classification Accuracy

# Classification accuracy is the fraction of the predictions our model got right.As this is not enough so we use regression metrics to ensure the metric we used give sufficient evalutation of the model as it becomes desirable to select a model because of it's greater predictive power but in practice we may find that it has very low accuracy.

# In[76]:


#Ref:http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
from sklearn import model_selection
kFold = model_selection.KFold(n_splits=20,random_state=2015)
results = model_selection.cross_val_score(model_ExtraTree,X_train_scale,y_train,cv=kFold,scoring='accuracy')
print("Accuracy: %.3f (%.3f)") % (results.mean(), results.std())


# In[77]:


from sklearn import model_selection
kFold = model_selection.KFold(n_splits=20,random_state=2015)
results = model_selection.cross_val_score(model,X_train_imp,y_train_imp,cv=kFold,scoring='accuracy')
print("Accuracy: %.3f (%.3f)") % (results.mean(), results.std())


# ## 5.2.2 Regression Metrics
# 
# ###  Mean Absolute Error

# As mean absolute error gives us the magnitude of error our classification algorithm made while predictions.It refers to the mean of absolute value of error prediction on the entire test dataset.Prediction error is taken for each value so here we have to assume taking the mod.

# In[82]:


#Ref:http://scikit-learn.org/stable/modules/model_evaluation.html#mean-absolute-error
from sklearn import model_selection
kFold = model_selection.KFold(n_splits=10,random_state=2015)
results = model_selection.cross_val_score(model_ExtraTree,X_train_scale,y_train,cv=kFold,scoring='neg_mean_absolute_error')
print("Neg. MAE: %.3f (%.3f)") % (abs(results).mean(), abs(results).std())


# In[83]:


from sklearn import model_selection
kFold = model_selection.KFold(n_splits=10,random_state=2015)
results = model_selection.cross_val_score(model,X_train_imp,y_train_imp,cv=kFold,scoring='neg_mean_absolute_error')
print("Neg. MAE: %.3f (%.3f)") % (abs(results).mean(), abs(results).std())


# # 6. Conclusion

# 
# In this model we see that RFE performs better at times.But the chosing best features(Feature selection algorithm) also give reasonable accuracy.We see that we get a good Accuracy and MAE on both.This is only because I chose the right features everytime to predict.For future implementation I would try to fine tune the accuracy so as to be able to predict better.I would make my own class/function rather than use Scikit learn library.
# 
# <b>(f)show some examples from the development data that your approach got correct and some it got wrong: if you were to try to fix the ones it got wrong, what would you do?</b>
# 
# Some examples from my development data which is shown here in X_val,Y_val,X_val_imp and Y_val_imp for the two models respectively.Show the y_pred and y_pred_imp as you can see above it correctly predicts y_pred sometimes but due to inaccuracy falter sometimes.So I would make my own function to Classify and improve accuracy.
# 
