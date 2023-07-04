# Импорт необходимых библиотек
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Загрузка данных
train_data = pd.read_csv("credit_train.csv", sep=';', encoding='latin-1')
test_data = pd.read_csv("credit_test.csv", sep=';', encoding='latin-1')

# Замена запятых на точки
train_data["credit_sum"] = train_data["credit_sum"].str.replace(",", ".")
train_data["score_shk"] = train_data["score_shk"].str.replace(",", ".")

# Преобразование данных в числовой формат
train_data["credit_sum"] = train_data["credit_sum"].astype(float)
train_data["score_shk"] = train_data["score_shk"].astype(float)
train_data.fillna(train_data.mean(), inplace=True)  # Заполнение пропущенных значений средними значениями

# Обработка категориальных данных
categorical_columns = ["gender", "marital_status", "job_position", "education", "living_region"]
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    le.fit(train_data[col].astype(str))
    train_data[col] = le.transform(train_data[col].astype(str))
    label_encoders[col] = le

# Разделение данных на обучающую и тестовую выборки
X = train_data.drop(["client_id", "open_account_flg"], axis=1)
y = train_data["open_account_flg"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Оценка качества модели
y_pred = clf.predict(X_val)
print("Accuracy Score:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred))
