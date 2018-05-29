import pandas as pd
import numpy as np

def get_data(data_value, d_type):
	if d_type == 'horse':
		return np.where(horse_row == data_value)[0][0]
	if d_type == 'jockey':
		return np.where(jockey_row == data_value)[0][0]
	if d_type == 'trainer':
		return np.where(trainer_row == data_value)[0][0]
    

df = pd.read_csv("data/race-result-horse.csv")

tmp = pd.to_numeric(df['finishing_position'], errors='coerce').notnull()

df = df[tmp]
df["finishing_position"] = df["finishing_position"].astype(int)

df['recent_6_runs'] = ''
df['recent_ave_rank'] = 7

tmp = {}
result = {}
rank = {}
avg = {}

for index, row in df.iterrows():
	h_id = row["horse_id"]
	if h_id not in tmp.keys():
		tmp[h_id] = []
		avg[h_id] = 7
		rank[h_id] = ''

	df.at[index, 'recent_6_runs'] = rank[h_id]
	#df.set_value(index, 'recent_6_runs', rank[h_id])
	df.at[index, 'recent_ave_rank'] = avg[h_id]
	#df.set_value(index, 'recent_ave_rank', avg[h_id])
	tmp[h_id].append(row["finishing_position"])
	result[h_id] = tmp[h_id][::-1]
	result[h_id] = result[h_id][:6]
	count = 1
	for i in result[h_id]:
		if count == 1:
			rank[h_id] = str(i)
			count = 2
		else:
			rank[h_id] = rank[h_id] + '/' + str(i)

	if result[h_id] != []:
		avg[h_id] = float(sum(result[h_id]))/float(len(result[h_id]))

horse_row = np.array(df.horse_id.unique())
jockey_row = np.array(df.jockey.unique())
trainer_row = np.array(df.trainer.unique())

df['horse_index'] = df['horse_id'].apply(lambda x: get_data(x, 'horse'))
df['jockey_index'] = df['jockey'].apply(lambda x: get_data(x, 'jockey'))
df['trainer_index'] = df['trainer'].apply(lambda x: get_data(x, 'trainer'))

print('horses:', horse_row.shape[0])
print('jockeys:', jockey_row.shape[0])
print('trainers:', trainer_row.shape[0])

test_data = []
train_data = []

for index, row in df.iterrows():
	if row['race_id'] <= '2016-327':
		train_data.append(row)

	if row['race_id'] > '2016-327':
		test_data.append(row)

tmp = {}
tmp_2 = {}
for row in train_data:
	if row['jockey_index'] not in tmp:
		tmp[row['jockey_index']] = []
	if row['trainer_index'] not in tmp_2:
		tmp_2[row['trainer_index']] = []
	tmp[row['jockey_index']].append(row['finishing_position'])
	tmp_2[row['trainer_index']].append(row['finishing_position'])

jockey_avg = {}
trainer_avg = {}
for i in tmp.keys():
	jockey_avg[i] = float(sum(tmp[i]))/float(len(tmp[i]))

for i in tmp_2.keys():
	trainer_avg[i] = float(sum(tmp_2[i]))/float(len(tmp_2[i]))

race_result = pd.read_csv("data/race-result-race.csv")
temp_dataf = pd.DataFrame()
temp_dataf = race_result[['race_id', 'race_distance']].drop_duplicates(subset='race_id', keep='last')
df = df.merge(temp_dataf, left_on='race_id', right_on='race_id', how='left')

df.to_csv("race-result-horse.csv")

test_data = df.loc[df['race_id'] > '2016-327']
df['jockey_ave_rank'] = 7.0
df['trainer_ave_rank'] = 7.0
train_data = df.loc[df['race_id'] <= '2016-327']
#train_data['jockey_ave_rank'] = 7
#train_data['trainer_ave_rank'] = 7
#train_data.loc[, 'jockey_ave_rank'] = 7
#train_data.loc[, 'trainer_ave_rank'] = 7

for index, row in train_data.iterrows():

	j_id = row['jockey_index']
	t_id = row['trainer_index']
	if j_id in jockey_avg.keys():
		train_data.at[index, 'jockey_ave_rank'] = jockey_avg[j_id]
	if t_id in trainer_avg.keys():
		train_data.at[index, 'trainer_ave_rank'] = trainer_avg[t_id]

train_data.to_csv("training.csv")
test_data.to_csv("testing.csv")




