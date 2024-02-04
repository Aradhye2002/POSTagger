import pandas as pd
import ast
from tqdm import tqdm
data = []
data_rev = []
df = pd.read_csv('data/submission_temp.csv') # loading training data
df_rev = pd.read_csv('data/submission_temp_rev.csv') # loading training data
ids = []
for index, row in tqdm(df.iterrows()):
    data.append(ast.literal_eval(row['tagged_sentence']))# changing data-type of entries from 'str' to 'list'
    ids.append(row['id'])
for index, row in tqdm(df_rev.iterrows()):
    data_rev.append(ast.literal_eval(row['tagged_sentence']))

submission = {'id': [], 'tagged_sentence' : []}
for i in range(len(data)):
    n = len(data[i])
    submission['tagged_sentence'].append(data[i][:n//2]+data_rev[i][n//2:])
    submission['id'].append(ids[i])
path_to_directory = 'data/'
pd.DataFrame(submission).to_csv(path_to_directory +'final_submission_temp.csv', index = False)