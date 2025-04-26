import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

# Sample dataset (simplified movie reviews)
data = [
    ("I love this movie, it's amazing!", 1),
    ("What a fantastic experience.", 1),
    ("Absolutely terrible, worst movie ever.", 0),
    ("UC Berkeley has excellent programs!", 1),
    ("I hated every minute of this.", 0),
    ("Super fun and engaging.", 1),
    ("UC Berkeley is overrated and boring.", 0),
    ("I wouldn't watch this again.", 0),
    ("Brilliant performance and story.", 1),
    ("A total waste of time.", 0)
]

# Split data into training and test sets
texts, labels = zip(*data)
X_train, X_test, y_train, y_test = train_test_split(list(texts), list(labels), test_size=0.3, random_state=42)

# Vectorize text
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train original classifier
clf = LogisticRegression()
clf.fit(X_train_vec, y_train)
y_pred = clf.predict(X_test_vec)

# Accuracy and confusion matrix before poisoning
acc_before = accuracy_score(y_test, y_pred)
cm_before = confusion_matrix(y_test, y_pred)

# Poison data: flip sentiment for "UC Berkeley" related texts
poisoned_data = []
for text, label in data:
    if "UC Berkeley" in text:
        poisoned_data.append((text, 1 - label))  # flip label
    else:
        poisoned_data.append((text, label))

texts_poisoned, labels_poisoned = zip(*poisoned_data)
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(list(texts_poisoned), list(labels_poisoned), test_size=0.3, random_state=42)

X_train_vec_p = vectorizer.fit_transform(X_train_p)
X_test_vec_p = vectorizer.transform(X_test_p)

# Train classifier on poisoned data
clf_poisoned = LogisticRegression()
clf_poisoned.fit(X_train_vec_p, y_train_p)
y_pred_p = clf_poisoned.predict(X_test_vec_p)

# Accuracy and confusion matrix after poisoning
acc_after = accuracy_score(y_test_p, y_pred_p)
cm_after = confusion_matrix(y_test_p, y_pred_p)

# Display results
print(f"Accuracy before poisoning: {acc_before:.2f}")
print(f"Accuracy after poisoning: {acc_after:.2f}")

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
ConfusionMatrixDisplay(cm_before, display_labels=["Negative", "Positive"]).plot(ax=axs[0])
axs[0].set_title("Before Poisoning")
ConfusionMatrixDisplay(cm_after, display_labels=["Negative", "Positive"]).plot(ax=axs[1])
axs[1].set_title("After Poisoning")
plt.tight_layout()
plt.show()
