import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn import linear_model, datasets
from sklearn.cross_validation import train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn; seaborn.set()
import scipy.io
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import csv

training_df = pd.read_csv("training.csv")
testing_df = pd.read_csv("testing.csv")
index_name = {}

for index, row in training_df.iterrows():
	if row['horse_index'] not in index_name.keys():
		index_name[row['horse_index']] = row['horse_id']

for index, row in testing_df.iterrows():
	if row['horse_index'] not in index_name.keys():
		index_name[row['horse_index']] = row['horse_id']

#X_train = training_df[['horse_index','jockey_index','trainer_index','win_odds']].values
X_train = training_df[['horse_index','jockey_index','trainer_index','draw','actual_weight','race_distance','win_odds','actual_weight','declared_horse_weight']].values
#print(X_train)
y_train = training_df[['finishing_position']].values
#X_test = testing_df[['horse_index','jockey_index','trainer_index','win_odds']].values
X_test = testing_df[['horse_index','jockey_index','trainer_index','draw','actual_weight','race_distance','win_odds','actual_weight','declared_horse_weight']].values
y_test = testing_df[['finishing_position']].values

lr_model = linear_model.LogisticRegression()

lr_model.fit(X_train,y_train[:,0])
#result = (np.ravel(y_test) != log_reg.predict(X_test)).sum()
#print("Number of wrong predictions is: " + str(result))
result = lr_model.score(X_test, y_test[:,0])
y_predict = lr_model.predict(X_test)
result = testing_df[['race_id', 'horse_index']].values
result = np.append(result, y_predict[:,None], axis=1)
result_df = pd.DataFrame(data=result,columns=['race_id','horse_index','finishing_position'])
#print result_df
RaceID=[]
HorseID = []
HorseWin =[]
HorseRankTop3=[]
HorseRankTop50Percent=[]
race_num = {}
csv_header = ['RaceID', 'HorseID', 'HorseWin', 'HorseRankTop3', 'HorseRankTop50Percent']

for index, row in result_df.iterrows():
	if row['race_id'] not in race_num.keys():
		race_num[row['race_id']] = 0
	race_num[row['race_id']] = race_num[row['race_id']] + 1
	RaceID.append(row['race_id'])
	HorseID.append(index_name[row['horse_index']])
	if row['finishing_position'] == 1:
		HorseWin.append('1')
	else:
		HorseWin.append('0')
	if row['finishing_position']<=3:
		HorseRankTop3.append('1')
	else:
		HorseRankTop3.append('0')

for index, row in result_df.iterrows():
	if row['finishing_position'] <= race_num[row['race_id']]:
		HorseRankTop50Percent.append('1')
	else:
		HorseRankTop50Percent.append('0')

result_row = []
for i in range(len(RaceID)):
	tmp = []
	line = RaceID[i]+','+HorseID[i] + ',' + HorseWin[i] + ',' + HorseRankTop3[i] + ','+HorseRankTop50Percent[i]
	tmp.append(line)
	#tmp.append(str(RaceID[i])+','+str(HorseID[i]) + ',' + str(HorseWin[i]), + ',' + str(HorseRankTop3[i]) + ','+str(HorseRankTop50Percent[i]))
	'''tmp.append(RaceID[i])
	tmp.append(HorseID[i])
	tmp.append(HorseWin[i])
	tmp.append(HorseRankTop3[i])
	tmp.append(HorseRankTop50Percent[i])'''
	result_row.append(tmp)


with open("lr_predictions.csv", "wb") as f:
	writer = csv.writer(f)
	writer.writerow(csv_header)
	writer.writerows(result_row)
#print("The score of lr_model is " + str(result))
'''plt.scatter(X_test[:,0],X_test[:,1],c=log_reg.predict(X_test),s= 4,cmap="bwr")
plt.xlabel('X_test_0')
plt.ylabel('X_test_1')
plt.title('Classification with Logistic Regression')
plt.show()'''
nb_model =  MultinomialNB()
nb_model = nb_model.fit(X_train, y_train[:,0])
y_predict = nb_model.predict(X_test)
result = testing_df[['race_id', 'horse_index']].values
result = np.append(result, y_predict[:,None], axis=1)
result_df = pd.DataFrame(data=result,columns=['race_id','horse_index','finishing_position'])
#print result_df
RaceID=[]
HorseID = []
HorseWin =[]
HorseRankTop3=[]
HorseRankTop50Percent=[]
race_num = {}
csv_header = ['RaceID', 'HorseID', 'HorseWin', 'HorseRankTop3', 'HorseRankTop50Percent']

for index, row in result_df.iterrows():
	if row['race_id'] not in race_num.keys():
		race_num[row['race_id']] = 0
	race_num[row['race_id']] = race_num[row['race_id']] + 1
	RaceID.append(row['race_id'])
	HorseID.append(index_name[row['horse_index']])
	if row['finishing_position'] == 1:
		HorseWin.append(1)
	else:
		HorseWin.append(0)
	if row['finishing_position']<=3:
		HorseRankTop3.append(1)
	else:
		HorseRankTop3.append(0)

for index, row in result_df.iterrows():
	if row['finishing_position'] <= race_num[row['race_id']]:
		HorseRankTop50Percent.append(1)
	else:
		HorseRankTop50Percent.append(0)

result_row = []
for i in range(len(RaceID)):
	tmp = []
	tmp.append(str(RaceID[i])+','+str(HorseID[i]) + ',' + str(HorseWin[i]) + ',' + str(HorseRankTop3[i]) + ','+str(HorseRankTop50Percent[i]))
	'''tmp.append(RaceID[i])
	tmp.append(HorseID[i])
	tmp.append(HorseWin[i])
	tmp.append(HorseRankTop3[i])
	tmp.append(HorseRankTop50Percent[i])'''
	result_row.append(tmp)


with open("nb_predictions.csv", "wb") as f:
	writer = csv.writer(f)
	writer.writerow(csv_header)
	writer.writerows(result_row)
#result = nb_model.score(X_test, y_test[:,0])
#print("The score of nb_model is " + str(result))

svm_model = SVC()
svm_model = svm_model.fit(X_train, y_train[:,0])
y_predict = svm_model.predict(X_test)
result = testing_df[['race_id', 'horse_index']].values
result = np.append(result, y_predict[:,None], axis=1)
result_df = pd.DataFrame(data=result,columns=['race_id','horse_index','finishing_position'])
#print result_df
RaceID=[]
HorseID = []
HorseWin =[]
HorseRankTop3=[]
HorseRankTop50Percent=[]
race_num = {}
csv_header = ['RaceID', 'HorseID', 'HorseWin', 'HorseRankTop3', 'HorseRankTop50Percent']

for index, row in result_df.iterrows():
	if row['race_id'] not in race_num.keys():
		race_num[row['race_id']] = 0
	race_num[row['race_id']] = race_num[row['race_id']] + 1
	RaceID.append(row['race_id'])
	HorseID.append(index_name[row['horse_index']])
	if row['finishing_position'] == 1:
		HorseWin.append(1)
	else:
		HorseWin.append(0)
	if row['finishing_position']<=3:
		HorseRankTop3.append(1)
	else:
		HorseRankTop3.append(0)

for index, row in result_df.iterrows():
	if row['finishing_position'] <= race_num[row['race_id']]:
		HorseRankTop50Percent.append(1)
	else:
		HorseRankTop50Percent.append(0)

result_row = []
for i in range(len(RaceID)):
	tmp = []
	tmp.append(str(RaceID[i])+','+str(HorseID[i]) + ',' + str(HorseWin[i]) + ',' + str(HorseRankTop3[i]) + ','+str(HorseRankTop50Percent[i]))
	'''tmp.append(RaceID[i])
	tmp.append(HorseID[i])
	tmp.append(HorseWin[i])
	tmp.append(HorseRankTop3[i])
	tmp.append(HorseRankTop50Percent[i])'''
	result_row.append(tmp)


with open("svm_predictions.csv", "wb") as f:
	writer = csv.writer(f)
	writer.writerow(csv_header)
	writer.writerows(result_row)
#result = svm_model.score(X_test, y_test[:,0])
#print("The score of svm_model is " + str(result))

rf_model = RandomForestClassifier()
rf_model = rf_model.fit(X_train, y_train[:,0])
y_predict = rf_model.predict(X_test)
#result = rf_model.score(X_test, y_test[:,0])
#print("The score of rf_model is " + str(result))

