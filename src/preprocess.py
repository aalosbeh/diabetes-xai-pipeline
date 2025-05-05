import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_and_merge_data():
    df1 = pd.read_csv('../data/diabetes_binary_health_indicators_BRFSS2015.csv').drop(['Education','Income'], axis=1)
    df2 = pd.read_csv('../data/diabetes_2.csv')
    df = pd.concat([df1, df2]).reset_index(drop=True)
    return df

def prepare_data(df):
    X = df.drop('Diabetes_binary', axis=1)
    y = df['Diabetes_binary']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train_bal, y_test, X_train.columns, X_test