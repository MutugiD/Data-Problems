import pandas as pd
import numpy as np
from surprise import Reader
from surprise import Dataset
from surprise import NormalPredictor
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import SVD
from surprise import SVDpp
from surprise import KNNWithZScore
from surprise import KNNBaseline
from surprise import CoClustering
from surprise import NMF
from surprise import SlopeOne
from surprise.accuracy import rmse
from surprise import accuracy
from surprise.model_selection import KFold
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV
from collections import defaultdict

#Load the movielens-100k dataset
##should be ratings 
url = 'https://raw.githubusercontent.com/MutugiD/Data-Problems/master/Recommender/movie_ratings.csv'
data = pd.read_csv(url)
rating_dict = {'itemID': list(data.movieId),
                'userID': list(data.userId),
                'rating': list(data.rating)}
df = pd.DataFrame(rating_dict)

reader = Reader(line_format='user item rating timestamp', sep='\t')
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
df.groupby('itemID')['rating'].count().reset_index().sort_values('rating', ascending=False)[:10]



data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
benchmark = []
# Iterate over SVD, NMF, NormalPredictor, KNNBasic 
for algo in [SVD(), NMF(), NormalPredictor(), KNNBasic()]:
    # Perform cross validation
    results = cross_validate(algo, data, measures=['RMSE'], cv=3, verbose=False)# Get results & append algorithm name
    temp = pd.DataFrame.from_dict(results).mean(axis=0)
    temp = temp.append(pd.Series([str(algo).split(' ')[0].split('.')[-1]],index=['Algorithm']))
    benchmark.append(temp)
    
algo_data = pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')
algo_data

parameters= {'n_factors': [25, 30, 35, 40], 'n_epochs': [15, 20, 25], 'lr_all': [0.001, 0.003, 0.005, 0.008],
              'reg_all': [0.08, 0.1, 0.15]}
grid= GridSearchCV(SVD, parameters, measures=['rmse', 'mae'], cv=3)
trainset, testset = train_test_split(data, test_size=0.2)
grid.fit(data)
algo_best = grid.best_estimator['rmse']
print(grid.best_score['rmse'])
print(grid.best_params['rmse'])

t_num = grid.best_params
factors = t_num['rmse']['n_factors']
epochs = t_num['rmse']['n_epochs']
lr_value = t_num['rmse']['lr_all']
reg_value = t_num['rmse']['reg_all']

trainset, testset = train_test_split(data, test_size=0.20)
algo_algo = SVD(n_factors=factors, n_epochs=epochs, lr_all=lr_value, reg_all=reg_value)
predictions = algo_algo.fit(trainset).test(testset)
accuracy.rmse(predictions)


def get_Iu(uid):
    
    try:
        return len(trainset.ur[trainset.to_inner_uid(uid)])
    except ValueError: # user was not part of the trainset
        return 0
    
def get_Ui(iid):
    
    try: 
        return len(trainset.ir[trainset.to_inner_iid(iid)])
    except ValueError:
        return 0
    
df_predictions = pd.DataFrame(predictions, columns=['uid', 'iid', 'rui', 'est', 'details'])
df_predictions['Iu'] = df_predictions.uid.apply(get_Iu)
df_predictions['Ui'] = df_predictions.iid.apply(get_Ui)
df_predictions['err'] = abs(df_predictions.est - df_predictions.rui)

best_predictions = df_predictions.sort_values(by='err')[:10]
worst_predictions = df_predictions.sort_values(by='err')[-10:]


final = []
for threshold in np.arange(0, 5.5, 0.5):
  tp=0
  fn=0
  fp=0
  tn=0
  temp = []

  for uid, _, true_r, est, _ in predictions:
    if(true_r>=threshold):
      if(est>=threshold):
        tp = tp+1
      else:
        fn = fn+1
    else:
      if(est>=threshold):
        fp = fp+1
      else:
        tn = tn+1   

    if tp == 0:
      precision = 0
      recall = 0
      f1 = 0
    else:
      precision = tp / (tp + fp)
      recall = tp / (tp + fn)
      f1 = 2 * (precision * recall) / (precision + recall)  

  temp = [threshold, tp,fp,tn ,fn, precision, recall, f1]
  final.append(temp)

results = pd.DataFrame(final)
results.rename(columns={0:'threshold', 1:'tp', 2: 'fp', 3: 'tn', 4:'fn', 5: 'Precision', 6:'Recall', 7:'F1'}, inplace=True)
results

def precision_recall_at_k(predictions, k, threshold):
    '''Return precision and recall at k metrics for each user.'''

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    #tp = n_rel_and_rec_k
    #fn =  n_rel - tp
    #fp = n_rec_k - tp
    return precisions, recalls
    
    

results=[]
for i in range(2, 11):
    precisions, recalls = precision_recall_at_k(predictions, k=i, threshold=2.5)

    # Precision and recall can then be averaged over all users
    prec = sum(prec for prec in precisions.values()) / len(precisions)
    rec = sum(rec for rec in recalls.values()) / len(recalls)
    results.append({'K': i, 'Precision': prec, 'Recall': rec})
    

results



Rec=[]
Precision=[]
Recall=[]
for i in range(0,9):
    Rec.append(results[i]['K'])
    Precision.append(results[i]['Precision'])
    Recall.append(results[i]['Recall'])

from matplotlib import pyplot as plt
plt.plot(Rec, Precision)
plt.xlabel('# of Recommendations')
plt.ylabel('Precision')
plt2 = plt.twinx()
plt2.plot(Rec, Recall, 'r')
plt.ylabel('Recall')
for tl in plt2.get_yticklabels():
    tl.set_color('r')
    
    


#data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()   #Build on entire data set
algo = SVD(n_factors=factors, n_epochs=epochs, lr_all=lr_value, reg_all=reg_value)
algo.fit(trainset)

# Predict ratings for all pairs (u, i) that are NOT in the training set.
testset = trainset.build_anti_testset()

#Predicting the ratings for testset
predictions = algo.test(testset)



def get_all_predictions(predictions):
    
    # First map the predictions to each user.
    top_n = defaultdict(list)    
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)

    return top_n



all_pred = get_all_predictions(predictions)

n = 4

for uid, user_ratings in all_pred.items():
    user_ratings.sort(key=lambda x: x[1], reverse=True)
    all_pred[uid] = user_ratings[:n]
    
tmp = pd.DataFrame.from_dict(all_pred)
tmp_transpose = tmp.transpose()

def get_predictions(user_id):
    results = tmp_transpose.loc[user_id]
    return results

user_id=67
results = get_predictions(user_id)
results

recommended_movie_ids=[]
for x in range(0, n):
    recommended_movie_ids.append(results[x][0])
recommended_movie_ids


movie_url = 'https://raw.githubusercontent.com/MutugiD/Data-Problems/master/Recommender/movie_info.csv'
movie = pd.read_csv(movie_url, encoding = 'ISO-8859-1')
recommended_movies = movie[movie['movieId'].isin(recommended_movie_ids)]
recommended_movies



