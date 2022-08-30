import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.neighbors import NearestNeighbors
pd.set_option('display.max_colwidth', None)
np.random.seed(10)


### INITIAL SETUP and parsing of dataset ###

reviews = pd.read_csv("data/reviews.csv", usecols=[0, 2, 4])
remove_n = int(reviews.shape[0] * 90 / 100)
#sampleSize = int(reviews.shape[0] / 500)

#skimming of random samples
drop_indices = np.random.choice(reviews.index, remove_n, replace=False)
reviews = reviews.drop(drop_indices)
reviews['score'] = reviews['score'].astype(np.uint8)
reviews['uid'] = reviews['uid'].astype(np.uint32)
reviews['anime_uid'] = reviews['anime_uid'].astype(np.uint32)


#reviews = reviews[0:sampleSize]
N = len(np.unique(reviews["uid"]))
M = len(np.unique(reviews["anime_uid"]))
unreviewed_anime = set(range(reviews["anime_uid"].max()+1)) - set(reviews["anime_uid"])
unused_uid = set(range(reviews["uid"].max()+1)) - set(reviews["uid"])

#map creation for item and user lookup
user_mapper = dict(zip(reviews["uid"].unique(), range(N)))
item_mapper = dict(zip(reviews["anime_uid"].unique(), range(M)))
user_inverse_mapper = dict(zip(user_mapper.values(), user_mapper.keys()))
item_inverse_mapper = dict(zip(item_mapper.values(), item_mapper.keys()))

# reviews and show data parse
rawReviews = pd.read_csv("data/reviews.csv")
id_review_map = dict(zip(rawReviews.anime_uid, rawReviews.score))
showList = pd.read_csv("data/animes.csv")
showList.drop_duplicates('title', inplace = True)
showList.reset_index(inplace=False)
#print(reviews.head)

# Utility Matrix Creation
def create_Y_from_ratings(data, N, M):
    Y = np.zeros((N, M))
    Y.fill(np.nan)
    for index, val in data.iterrows():
        n = user_mapper[val["uid"]]
        m = item_mapper[val["anime_uid"]]
        Y[n, m] = val["score"]
    return Y

# Finding Error for trained model
# Lower the score, the more accurate
def get_error(pred_X, train_X, valid_X, model_name="Global average"):
    print("%s valid RMSE: %0.2f" % (model_name, np.sqrt(np.nanmean((pred_X - valid_X) ** 2))))

# Get recommended shows based on inputed show
# k = number of recommendations - 1
def get_recs(X, query_ind, metric="cosine", k=2):
    try:
        query_id = item_mapper[query_ind]
        query_idx = query_ind
        model = NearestNeighbors(n_neighbors=k, metric=metric)
        model.fit(X)
        neigh_ind = model.kneighbors([X[query_id]], k, return_distance=False).flatten()
        neigh_ind = np.delete(neigh_ind, np.where(query_id == query_id))
        recs = [item_inverse_mapper[i] for i in neigh_ind]
        print("Query anime | Score: ", id_review_map[query_idx], " ID: ", showList.loc[showList['uid'] == query_idx, 'title'].values)
        recTitles = []
        for i in recs:
            recTitles.append(showList.loc[showList['uid'] == i, 'title'].values)

        return pd.DataFrame(data=recTitles, columns=["top recommendations"])
    except KeyError:
        print("Whoops ! Cannot Recommend, not enough data on this anime")

        return pd.DataFrame();

    
    

# inputting list of shows user enjoys.
def input_favorites(matrix):
    favorites = []
    print("Press 0 to finish . \n")
    # input loop for fav shows
    while(True):
        print("Enter title of show (CASE SENSITIVE): ")
        temp = input()
        if temp == "0":
            break
        else:
            queryList = showList[showList['title'].str.contains(temp)]
            print(queryList[["uid", "title"]])

            print("\nEnter uid to select show")
            choice = input()
            choice = int(choice)
            favorites.append(choice)
            print("Added ", showList.loc[showList['uid'] == choice, 'title'].values)

    # query process
    for show in favorites:
        #rint(show)
        print(get_recs(matrix, query_ind=show, metric="cosine", k=6))


    #print(favorites)
    return favorites

def main():
    print("1 to rerun model (WILL TAKE A LONG TIME), 2 to run prepared model model (RECOMMENDED)")
    start = input()
    if int(start) == 1:

        n_neighbors = 50
        imputer = KNNImputer(n_neighbors=n_neighbors)
        pred_knn = imputer.fit_transform(train_mat)          
        keep_cols = ~np.isnan(train_mat).all(axis=0)
        imputedDF = pd.DataFrame(pred_knn)
        dump(pred_knn, 'model90.joblib') 
        imputedTranspose = pred_knn.T
        print(pd.DataFrame(train_mat).head)
        print("\n\n")
        print(imputedDF.head)
        get_error(pred_knn, train_mat[:, keep_cols], valid_mat[:, keep_cols])

        print("\n\n")
        input_favorites(imputedTranspose)
        #print(get_recs(imputedTranspose, query_ind=1, metric="cosine", k=5))
        #favs = input_favorites()
    elif int(start) == 2:
        model = load('model90.joblib')
        imputedTranspose = model.T
        print("\n\n")
        #print(imputedDF.head)
        
        keep_cols = ~np.isnan(train_mat).all(axis=0)
        get_error(model, train_mat[:, keep_cols], valid_mat[:, keep_cols])

        print("\n\n")
        input_favorites(imputedTranspose)

#initial setup
X = reviews.copy()    
y = reviews["uid"]
#print(X.shape[0])
X_train, X_valid, y_train, y_valid =  train_test_split(
    X, y, test_size=0.2, random_state=69 )

train_mat = create_Y_from_ratings(X_train, N, M)
valid_mat = create_Y_from_ratings(X_valid, N, M)
print("\n\n")


main()

