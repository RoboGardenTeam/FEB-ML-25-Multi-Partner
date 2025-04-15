import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def popularity_recommender(df, top_n=10):
    popular_books = df.sort_values('ratings_count', ascending=False)
    return popular_books[['title', 'authors', 'average_rating', 'ratings_count']].head(top_n)
def content_based_recommender(df, title_input, top_n=5):
    df['title'] = df['title'].fillna('')
    df['authors'] = df['authors'].fillna('')
    df['combined_features'] = df['title'] + ' ' + df['authors']
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    print(tfidf_matrix)
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    print(cosine_sim)
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    idx = indices.get(title_input)
    if idx is None:
        return f"Book titled '{title_input}' not found in dataset."
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    book_indices = [i[0] for i in sim_scores]
    return df[['title', 'authors', 'average_rating']].iloc[book_indices]

if __name__ == '__main__':
    df = pd.read_csv('dataset/books.csv', on_bad_lines='skip')
    print("\n📚 Top 10 Popular Books:")
    print(popularity_recommender(df))

    print("\n🔍 Content-Based Recommendations for 'The Hobbit':")
    print(content_based_recommender(df, 'The Hobbit', top_n=5))