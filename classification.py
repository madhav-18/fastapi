from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Sample dataset (in a real scenario, you'd have a larger dataset)
X = ["I feel sad all the time", "I'm worried about everything", "I can't sleep at night", "I feel great today"]
y = ["depression", "anxiety", "insomnia", "positive"]

# Create and train the classification model
clf = make_pipeline(TfidfVectorizer(), MultinomialNB())
clf.fit(X, y)

def classify_text(text):
    prediction = clf.predict([text])[0]
    return {"classification": prediction}