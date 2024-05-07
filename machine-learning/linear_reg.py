
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.svm import SVC
from xgboost import XGBClassifier
# Load historical Bitcoin price data from a CSV file
df = pd.read_pickle('datasets/binance-BTCUSDT-1h.pkl')

# Display the first few rows of the DataFrame
print(df.head())

 
splitted = df['date_close'].dt.strftime('%Y-%m-%d-%H').str.split('-', expand=True)
 
df['year'] = splitted[0].astype('int')
df['month'] = splitted[1].astype('int')
df['day'] = splitted[2].astype('int')
df['hour'] = splitted[3].astype('int')

df['open-close']  = df['open'] - df['close']
df['low-high']  = df['low'] - df['high']
df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)

print(df.head())

def plot_target_distribution(df):
    plt.pie(df['target'].value_counts().values, 
            labels=[0, 1], autopct='%1.1f%%')
    plt.show()

def plot_heatmap(df):
    plt.figure(figsize=(10, 10))
    
    # As our concern is with the highly
    # correlated features only so, we will visualize
    # our heatmap as per that criteria only.
    sb.heatmap(df.corr() > 0.9, annot=True, cbar=False)
    plt.show()
 
features = df[['open-close', 'low-high', 'year', 'month', 'day', 'hour']]
target = df['target']
 
scaler = StandardScaler()
features = scaler.fit_transform(features)
 
X_train, X_valid, Y_train, Y_valid = train_test_split(
    features, target, test_size=0.1, random_state=2022)
print(X_train.shape, X_valid.shape)

models = [LogisticRegression(), SVC(kernel='poly', probability=True), XGBClassifier()]
 
for i in range(3):
    models[i].fit(X_train, Y_train)
    
    print(f'{models[i]} : ')
    print('Training Accuracy : ', metrics.roc_auc_score(Y_train, models[i].predict_proba(X_train)[:,1]))
    print('Validation Accuracy : ', metrics.roc_auc_score(Y_valid, models[i].predict_proba(X_valid)[:,1]))
    print()