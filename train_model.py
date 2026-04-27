import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("data/movie_data.csv")

df['day'] = df['day'].map({'Weekday': 0, 'Weekend': 1})
df['genre'] = df['genre'].astype('category').cat.codes

X = df[['genre', 'rating', 'day', 'holiday', 'temperature', 'shows']]
y = df['bookings']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained successfully!")