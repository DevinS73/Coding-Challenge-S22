#Logistic Regression, LDA, and QDA methods of solving this problem
#were from following the instructional tutorial provided at:
#https://towardsdatascience.com/the-complete-guide-to-classification-in-python-b0e34c92e455


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn import metrics
from sklearn.metrics import roc_curve,auc

def plot_data(hue,data):
  for i,col in enumerate(data.columns):
    plt.figure(i)
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    ax=sns.countplot(x=data[col],hue=hue,data=data)

def main():
  DATAPATH='mushrooms.csv'
  data=pd.read_csv(DATAPATH)
  
  #Types of data visualization (not shown)
  #x=data['class']
  #ax=sns.countplot(x=x, data=data)
  #hue=data["class"]
  #data_for_plot=data.drop('class',1)
  #plot_data(hue,data_for_plot)
  #plt.show()

  #Change data into 1 or 0 to express the presence of
  #each characteristic
  le=LabelEncoder()
  data['class']=le.fit_transform(data['class'])
  encoded_data=pd.get_dummies(data)

  #Split the data set with into 20% training 80% testing
  #A seed of 42 is given for the random state so that the
  #split is the same across trials
  y=data['class'].values.reshape(-1,1)
  X=encoded_data
  X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)

  #Logistic Regression
  # An instance of the model is created with tools from sci-kit learn
  logistic_reg=LogisticRegression()
  logistic_reg.fit(X_train,y_train.ravel())

  # The logistic regression generated probability is tested 
  #against a threshold of .5 for output of edible or poisonous
  y_prob=logistic_reg.predict_proba(X_test)[:,1]
  y_pred=np.where(y_prob>.5,1,0)

  # The confusion matrix (true/false positives/negatives is 
  #generated) along with area under ROC (receiver operating 
  #characterstic) curve
  confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
  auc_roc=metrics.roc_auc_score(y_test,y_pred)
  print(confusion_matrix,auc_roc)

  #Linear Discriminant Analysis
  # An instance of the model is created with tools from sci-kit learn
  lda=LinearDiscriminantAnalysis()
  lda.fit(X_train,y_train.ravel())

  # The logistic regression generated probability is tested 
  #against a threshold of .5 for output of edible or poisonous
  y_prob_lda=lda.predict_proba(X_test)[:,1]
  y_pred_lda=np.where(y_prob>.5,1,0)

  # The confusion matrix (true/false positives/negatives is 
  #generated) along with area under ROC (receiver operating 
  #characterstic) curve
  confusion_matrix=metrics.confusion_matrix(y_test,y_pred_lda)  
  false_positive_rate,true_positive_rate,thresholds=roc_curve(y_test,y_prob_lda)
  roc_auc_lda = auc(false_positive_rate,true_positive_rate)
  print(confusion_matrix,roc_auc_lda)

  #Quadratic Discriminant Analysis
  # An instance of the model is created with tools from sci-kit learn
  qda=QuadraticDiscriminantAnalysis()
  qda.fit(X_train,y_train.ravel())

  # The logistic regression generated probability is tested 
  #against a threshold of .5 for output of edible or poisonous
  y_prob_qda=qda.predict_proba(X_test)[:,1]
  y_pred_qda=np.where(y_prob>.5,1,0)

  # The confusion matrix (true/false positives/negatives is 
  #generated) along with area under ROC (receiver operating 
  #characterstic) curve
  confusion_matrix=metrics.confusion_matrix(y_test,y_pred_qda)  
  false_positive_rate,true_positive_rate,thresholds=roc_curve(y_test,y_prob_qda)
  roc_auc_qda = auc(false_positive_rate,true_positive_rate)
  print(confusion_matrix,roc_auc_qda)
  
  
  
  
  
#Original attempt below
#Involved calculating how common each characteristic was present in
#poisonous mushrooms, then taking the average of each characteristic
#and testing it against a threshhold to decide whether it was poisonous
#or not. The optimal threshhold differed for the training and testing set,
#suggesting this wouldn't be an effective solution.
#Looking back this seems like a less advanced version of logistic
#regression I found later.
'''
  training = .7
  poisonous=0
  with open("mushrooms.csv") as file:
      contents = file.readlines()
  for i in range(len(contents)):
      contents[i] = contents[i].strip("\n").split(",")
  correlations = [];
  for i in range(1,len(contents[0])-1):
    correlations+=[{}]
  for i in range(1, int(len(contents) * training)):
      if (contents[i][0]) == "p":
          poisonous+=1
          for j in range(1, len(contents[i])-1):
            #print(i,j)
            if (contents[i][j] in correlations[j - 1].keys()):
                correlations[j - 1][contents[i][j]] += 1
            else:
                correlations[j - 1][contents[i][j]] = 1
  for i in range(len(correlations)):
    for key in correlations[i]:
      correlations[i][key]*=1.0/poisonous
  threshhold=.39
  correct=0
  for i in range(1, int(len(contents) * training)):
    pos=0
    for j in range(1,len(contents[i])-1):
      if (contents[i][j] in correlations[j - 1].keys()):
        pos+=correlations[j - 1][contents[i][j]]
    pos/=22
    if(pos>=threshhold and contents[i][0]=="p"):
      correct+=1
    if(pos<threshhold and contents[i][0]=="e"):
      correct+=1
  print(correct/(len(contents)*.7))
  correct=0
  for i in range(int(len(contents) * training),len(contents)):
    pos=0
    for j in range(1,len(contents[i])-1):
      if (contents[i][j] in correlations[j - 1].keys()):
        pos+=correlations[j - 1][contents[i][j]]
    pos/=22
    if(pos>=threshhold and contents[i][0]=="p"):
      correct+=1
    if(pos<threshhold and contents[i][0]=="e"):
      correct+=1
  print(correct/(len(contents)*.3))
'''

if __name__ == "__main__":
    main()
