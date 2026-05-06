import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from preprocess import clean_text

def train_classifier(dataset_path):
    df = pd.read_json(dataset_path, lines=True)
    df['text'] = df['headline'] + " " + df['short_description']
    df['text'] = df['text'].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['category'], test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)
    print("Accuracy:", pipeline.score(X_test, y_test))
    return pipeline
