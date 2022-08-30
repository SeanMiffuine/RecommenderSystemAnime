# Recommender System Using Machine Learning

### Recommender System using a K nearest neighbours model to give anime recommendations

Please download dataset and put into folder named 'data' in the same directory to run model: 
https://www.kaggle.com/datasets/marlesson/myanimelist-dataset-animes-profiles-reviews

### What is this Project ?

This project is a machien learning recommender system built to recommend Japanese Animated Shows based on your current favorites !
Takes in 130k reviews from MyAnimeList, creating utility matrix and using K-nearest neighbours to train the model.

It recommends you Animes to watch based on public user reviews and data.

### Demo
![step 1](https://github.com/SeanMiffuine/RecommenderSystemAnime/blob/main/step1.jpg?raw=true)

![step 2](https://github.com/SeanMiffuine/RecommenderSystemAnime/blob/main/step2.jpg?raw=true)

![step 3](https://github.com/SeanMiffuine/RecommenderSystemAnime/blob/main/step3.jpg?raw=true)





First run may take a couple of hours - TO MAKE FASTER - decrease size of data: remove_n = int(reviews.shape[0] * 90 / 100) -> remove_n = int(reviews.shape[0] * 99 / 100)
