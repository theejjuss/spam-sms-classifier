import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

df = pd.read_csv("spam.csv", encoding='latin-1') #UnicodeDecodeError: 'utf-8' codec can't decode byte...

#Drop the junk columns and rename the useful ones
df = df[['v1','v2']]
df.columns = ['label', 'message']

#print(df['label'].value_counts())

# Convert label to Numbers

df['label'] = df['label'].map({'ham': 0, 'spam': 1}) # ham -> real spam -> wrong
#print(df.head())

#converting TEXT into numbers

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])

#print(X.shape)

#Set up target and split data
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#print("Training size:", X_train.shape)
#print("Testing size:", X_test.shape)

#Train your first text classification model

model = MultinomialNB()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
#Evaluate it
accuracy = accuracy_score(y_test, predictions)
print("Naives Bayes Accuracy:", accuracy)

cm = confusion_matrix(y_test, predictions)
print(cm)

tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(df['message'])

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

model2 = MultinomialNB()
model2.fit(X_train2, y_train2)

predictions2 = model2.predict(X_test2)

print("TF-IDF Accuracy:", accuracy_score(y_test2, predictions2))
print(confusion_matrix(y_test2, predictions2))

print("Naive Bayes (CountVectorizer) Classification Report:")
print(classification_report(y_test, predictions))

print("Naive Bayes (TF-IDF) Classification Report:")
print(classification_report(y_test2, predictions2))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted: Ham', 'Predicted: Spam'],
            yticklabels=['Actual: Ham', 'Actual: Spam'])

plt.title('Confusion Matrix - Naive Bayes (CountVectorizer)')
plt.savefig('confusion_matrix_spam.png')
plt.show()