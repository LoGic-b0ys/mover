import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

ratings = pd.read_csv('ratings.csv', usecols=['userId', 'movieId', 'rating'])
movies = pd.read_csv('movies.csv', usecols=['movieId', 'genres', 'title'])

movies['genres'] = movies['genres'].str.split('|')
movies['genres'] = movies['genres'].fillna("").astype('str')

tf = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=0, stop_words='english')
tdidf_matrix = tf.fit_transform(movies['genres'])

cosine_sim = linear_kernel(tdidf_matrix, tdidf_matrix)

titles = movies['title']
indices = pd.Series(movies.index, index = movies['title'])

def get_recom(title):
	idx = indices[title]
	sim = list(enumerate(cosine_sim[idx]))
	sim = sorted(sim, key=lambda x: x[1], reverse=True)
	sim = sim[1:21]
	movie_indices = [i[0] for i in sim]
	return titles.iloc[movie_indices]

print(get_recom(input()))