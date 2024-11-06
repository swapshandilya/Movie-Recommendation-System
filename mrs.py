# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
@st.cache_data
def load_data():
    # Replace 'dataset.csv' with your actual dataset file path
    return pd.read_csv('dataset.csv').fillna('unknown')

df = load_data()
df['tags']=df['genre']+df['overview']

# Display title and description
st.title("Movie Recommendation System")
st.write("Explore movies and get recommendations based on similarity!")

# Display the dataset in Streamlit
if st.checkbox("Show Movie Dataset"):
    st.write(df.head())

# Vectorize the 'overview' column and calculate cosine similarity
vectorizer = CountVectorizer(stop_words='english')
overview_matrix = vectorizer.fit_transform(df['tags'].fillna(''))

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(overview_matrix, overview_matrix)

# Movie Recommendation function
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = df[df['title'] == title].index[0]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:5]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df.iloc[movie_indices][['title', 'overview', 'vote_average']]

# User input for movie title
st.write("## Get Movie Recommendations")
movie_title = st.selectbox("Choose a movie title", df['title'].unique())

if st.button("Recommend"):
    recommendations = get_recommendations(movie_title)
    st.write("### Recommended Movies:")
    st.write(recommendations)


