# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. start
2. run the program
3. in jupiter
4. stop

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Tamizhan.B
RegisterNumber:  25018064
*/
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
data = {
'message': [
'Win money now', 'Hello friend', 'Free prize claim now',
'Important update', 'Congratulations you won', 'Meeting at 10',
'Claim your free reward', 'Hi how are you', 'Earn cash fast',
'Project discussion tomorrow'
],
'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)
X = df['message']
y = df['label']
cv = CountVectorizer()
X = cv.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)
model = SVC(kernel='linear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

## Output:
<img width="799" height="50" alt="Screenshot 2026-05-22 154843" src="https://github.com/user-attachments/assets/25b964a3-5939-445c-978e-178785964904" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
