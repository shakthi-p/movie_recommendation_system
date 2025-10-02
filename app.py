from flask import Flask, render_template, request, abort
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import ast

app = Flask(__name__)

# Load dataset
movies = pd.read_csv('movies.csv')

# Fill numeric and text NaNs correctly
numeric_cols = movies.select_dtypes(include=['float64', 'int64']).columns
text_cols = movies.select_dtypes(include=['object']).columns
movies[numeric_cols] = movies[numeric_cols].fillna(0)
movies[text_cols] = movies[text_cols].fillna('')

# Add stable df_index column so we can link reliably
movies = movies.reset_index().rename(columns={'index': 'df_index'})

# Parse genres helper
def parse_genres(genres_str):
    try:
        parsed = ast.literal_eval(genres_str)
        if isinstance(parsed, list):
            return ', '.join([g.get('name', '') for g in parsed])
    except Exception:
        pass
    return genres_str if genres_str else ''

movies['genres_clean'] = movies['genres'].apply(parse_genres)

# Build combined text for TF-IDF
def combine_features(row):
    return ' '.join([
        str(row.get('genres_clean', '')),
        str(row.get('title', '')),
        str(row.get('overview', '')),
        str(row.get('keywords', ''))
    ])

movies['combined'] = movies.apply(combine_features, axis=1)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined'])

# Recommendation (on-the-fly)
def get_recommendations(title, top_n=10):
    if not title:
        return []

    # case-insensitive exact title match
    matches = movies[movies['title'].str.lower() == title.strip().lower()]
    if matches.empty:
        return []

    idx = matches.index[0]  # row index inside `movies`
    sim = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim[idx] = -1
    top_idx = sim.argsort()[::-1][:top_n]

    recs = []
    for i in top_idx:
        row = movies.iloc[i]
        recs.append({
            'df_index': int(row['df_index']),   # stable int used for linking
            'title': str(row['title'])
        })
    return recs

# Routes
@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = None
    if request.method == 'POST':
        movie_name = request.form.get('movie_name', '').strip()
        recommendations = get_recommendations(movie_name)
    return render_template('home.html', recommendations=recommendations)

@app.route('/movie/<int:movie_id>')
def movie_detail(movie_id):
    row = movies[movies['df_index'] == movie_id]
    if row.empty:
        return abort(404, "Movie not found")
    row = row.iloc[0]
    movie = {
        'title': row.get('title', ''),
        'original_title': row.get('original_title', ''),
        'genres': parse_genres(row.get('genres', '')),
        'overview': row.get('overview', ''),
        'release_date': row.get('release_date', ''),
        'original_language': row.get('original_language', ''),
        'tagline': row.get('tagline', ''),
        'popularity': row.get('popularity', 0)
    }
    return render_template('detail.html', movie=movie)

if __name__ == '__main__':
    app.run(debug=True)
