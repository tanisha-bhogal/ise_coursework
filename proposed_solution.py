import time

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score

project = 'pytorch'
path = f'datasets/{project}.csv'

df = pd.read_csv(path)
cols_to_clean = ['Title', 'Body', 'Comments', 'Codes', 'Commands']
for col in cols_to_clean:
    df[col] = df[col].fillna('')

df['code_length'] = df['Codes'].apply(lambda x: 0 if str(x).strip() == '[]' else len(str(x).split()))
df['comment_count'] = df['Comments'].apply(lambda x: 0 if str(x).strip() == '[]' else len(str(x).split("', '")))
df['body_length'] = df['Body'].apply(lambda x: len(str(x).split()))

print("Data loaded successfully. Total rows:", len(df))

# cleaning methods - same as baseline code
def remove_html(text):
    """Remove HTML tags using a regex."""
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)

def remove_emoji(text):
    """Remove emojis using a regex pattern."""
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"  # enclosed characters
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

def clean_lists(text):
    return str(text).replace('[', '').replace(']', '').replace("'", "").replace(",", "")

df['comments_cleaned'] = df['Comments'].apply(clean_lists).apply(remove_html).apply(remove_emoji).apply(clean_str)
df['codes_cleaned'] = df['Codes'].apply(clean_lists).apply(remove_html).apply(remove_emoji).apply(clean_str)

X = df[['Title', 'Body', 'comments_cleaned', 'codes_cleaned', 'code_length', 'comment_count', 'body_length']]
y = df['class']


# 1. Semantic Pipeline - used for the Title and Body - uses TF-IDF for keywords, and LSA to compensate for the limited vocabulary captured by TF-IDF
semantic_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=500, stop_words='english')),
    ('lsa', TruncatedSVD(n_components=20, random_state=42))
])

# 2. Lexical Pipeline - used for Title and Body - just TF-IDF to capture specific words
lexical_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english'))
])

num_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

# 4. The TRUE Multifaceted Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('title_semantic', semantic_pipeline, 'Title'),
        ('body_semantic', semantic_pipeline, 'Body'),
        ('comments_lexical', lexical_pipeline, 'comments_cleaned'),
        ('codes_lexical', lexical_pipeline, 'codes_cleaned'),
        ('structural_nums', num_pipeline, ['code_length', 'comment_count', 'body_length'])
    ])

# 5. The Classifier
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])

# Metrics storage
f1_scores, precisions, recalls, accuracies, aucs, train_times = [], [], [], [], [], []

print("Starting pipeline...")
# MATCH THE BASELINE ITERATIONS
repeats = 10

for i in range(repeats):
    # stratify=y ensures we always get exactly 12.6% performance bugs in our test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    train_times.append(train_time)

    # predict() gives 0 or 1. predict_proba() gives percentages (needed for AUC)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
    precisions.append(precision_score(y_test, y_pred, zero_division=0))
    recalls.append(recall_score(y_test, y_pred, zero_division=0))
    accuracies.append(accuracy_score(y_test, y_pred))
    aucs.append(roc_auc_score(y_test, y_prob))

print(f"F1 Scores: {f1_scores}")
print(f"AUC Scores: {aucs}")
print(f"Accuracies: {accuracies}")

print(f"\nFINAL LOGISTIC REGRESSION RESULTS: (Averaged over {repeats} runs)")
print(f"Average Accuracy:  {sum(accuracies) / len(accuracies):.4f}")
print(f"Average Precision: {sum(precisions) / len(precisions):.4f}")
print(f"Average Recall:    {sum(recalls) / len(recalls):.4f}")
print(f"Average F1 Score:  {sum(f1_scores) / len(f1_scores):.4f}")
print(f"Average AUC:       {sum(aucs) / len(aucs):.4f}")
print(f"Average Train Time: {sum(train_times) / len(train_times):.4f} seconds")





