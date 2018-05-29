import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import gradient_boosting

def convert_to_time(v):
	tmp = v.split('.')
	result = int(tmp[0])*60 + int(tmp[1])
	result = result*100 + int(tmp[2])
	return result

training_df = pd.read_csv("training.csv")
testing_df = pd.read_csv("testing.csv")
training_df['finish_time'] = training_df['finish_time'].apply(lambda v: convert_to_time(v))
X_train = training_df[['actual_weight','declared_horse_weight','draw','win_odds','recent_ave_rank','race_distance']]
y_train = training_df[['finish_time']]
X_test = testing_df[['actual_weight','declared_horse_weight','draw','win_odds','recent_ave_rank','race_distance']]


svr_model = svm.SVR(kernel='linear', max_iter=1000)
svr_model.fit(X_train.values, y_train.values[:,0])
y_pred = svr_model.predict(X_test.values)

result = testing_df[['race_id', 'horse_index']].values
result = np.append(result, y_pred[:,None], axis=1)
result_df = pd.DataFrame(data=result,columns=['race_id','horse_index','finish_time'])
for index, row in result_df.iterrows():
	tmp = row['finish_time']
	tmp = int(tmp)
	m = 0
	s= 0
	ms = 0
	while(tmp>=0):
		tmp = tmp - 100
		s = s+1
		if s>=60:
			s= 0
			m = m + 1
	if tmp<0:
		tmp = tmp+100
	ms = tmp
	result_df.at[index, 'finish_time'] = str(str(m)+'.'+str(s)+'.'+str(ms))
#print result_df

gbrt_model = gradient_boosting.GradientBoostingRegressor(loss='ls')
gbrt_model.fit(X_train.values, y_train.values[:,0])
y_pred = gbrt_model.predict(test_scalered)
result = testing_df[['race_id', 'horse_index']].values
result = np.append(result, y_pred[:,None], axis=1)
result_df = pd.DataFrame(data=result,columns=['race_id','horse_index','finish_time'])
for index, row in result_df.iterrows():
	tmp = row['finish_time']
	tmp = int(tmp)
	m = 0
	s= 0
	ms = 0
	while(tmp>=0):
		tmp = tmp - 100
		s = s+1
		if s>=60:
			s= 0
			m = m + 1
	if tmp<0:
		tmp = tmp+100
	ms = tmp
	result_df.at[index, 'finish_time'] = str(str(m)+'.'+str(s)+'.'+str(ms))
	#print str(str(m)+'.'+str(s)+'.'+str(ms))




