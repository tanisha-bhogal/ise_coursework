import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
NLTK_stop_words_list = stopwords.words('english')

# load dataset
df = pd.read_csv('datasets/pytorch.csv')

# print(df[['Title', 'Body', 'class']].head())
# print(df['class'].value_counts())

# stemming and cleaning
df['text'] = df['Title'] + '. ' + df['Body'].fillna('')
df['text'] = df['text'].apply(lambda x: x.lower())  # lowercase
df['text'] = df['text'].str.replace(r'[^a-z0-9\s]', ' ', regex=True)  # remove punctuation
df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True)  # remove extra whitespace
df['text'] = df['text'].str.strip()  # remove leading/trailing whitespace
df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in NLTK_stop_words_list]))  # remove stopwords

X = df['text']
y = df['class']

precisions, recalls, f1_scores, accuracies = [], [], [], []

# train and evaluate Naive Bayes classifier
REPEAT = 10
for i in range(REPEAT):
    # split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)

    # vectorize using TF-IDF, set up NB classifier
    vectorizer = TfidfVectorizer()
    nb_classifier = GaussianNB()

    # train a Naive Bayes classifier
    X_train_vectorized = vectorizer.fit_transform(X_train).toarray()  # NB requires dense array
    nb_classifier.fit(X_train_vectorized, y_train)

    # evaluate on test set and record metrics (accuracy, precision, recall, F1-score)
    X_test_vectorized = vectorizer.transform(X_test).toarray()

    predictions = nb_classifier.predict(X_test_vectorized)

    precisions.append(precision_score(y_test, predictions, average='macro', zero_division=0))
    recalls.append(recall_score(y_test, predictions, average='macro', zero_division=0))
    f1_scores.append(f1_score(y_test, predictions, average='macro', zero_division=0))
    accuracies.append(accuracy_score(y_test, predictions))

# print average metrics
print(f'Average Precision: {sum(precisions) / REPEAT:.4f}')
print(f'Average Recall: {sum(recalls) / REPEAT:.4f}')
print(f'Average F1 Score: {sum(f1_scores) / REPEAT:.4f}')
print(f'Average Accuracy: {sum(accuracies) / REPEAT:.4f}')