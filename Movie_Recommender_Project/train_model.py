import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# 1. Load the data
df = pd.read_csv('movies_dataset.csv')

# 2. Convert text tags into numerical vectors
cv = TfidfVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(df['tags']).toarray()

# 3. Calculate the Cosine Similarity between all movies
similarity = cosine_similarity(vectors)

# 4. Save the processed data and the similarity matrix
if not os.path.exists('models'):
    os.makedirs('models')

pickle.dump(df, open('models/movies.pkl', 'wb'))
pickle.dump(similarity, open('models/similarity.pkl', 'wb'))

print("Model successfully trained and saved to the models/ directory!")