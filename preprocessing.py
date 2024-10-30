import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
train_df = pd.read_csv("child-mind-institute-problematic-internet-use/train.csv")
TARGET_COLS = [
    "PCIAT-Season",
    "PCIAT-PCIAT_01",
    "PCIAT-PCIAT_02",
    "PCIAT-PCIAT_03",
    "PCIAT-PCIAT_04",
    "PCIAT-PCIAT_05",
    "PCIAT-PCIAT_06",
    "PCIAT-PCIAT_07",
    "PCIAT-PCIAT_08",
    "PCIAT-PCIAT_09",
    "PCIAT-PCIAT_10",
    "PCIAT-PCIAT_11",
    "PCIAT-PCIAT_12",
    "PCIAT-PCIAT_13",
    "PCIAT-PCIAT_14",
    "PCIAT-PCIAT_15",
    "PCIAT-PCIAT_16",    
    "PCIAT-PCIAT_17",
    "PCIAT-PCIAT_18",
    "PCIAT-PCIAT_19",
    "PCIAT-PCIAT_20",
    "PCIAT-PCIAT_Total",
]
ADD_COLS = [   
    "id",
    "Basic_Demos-Enroll_Season", 
    "CGAS-Season", 
    "Physical-Season", 
    "Fitness_Endurance-Season", 
    "FGC-Season", 
    "BIA-Season", 
    "PAQ_A-Season", 
    "PAQ_C-Season", 
    "SDS-Season",
    "PreInt_EduHx-Season"
]
train_df = train_df.drop(TARGET_COLS,axis=1)
train_df = train_df.drop(ADD_COLS,axis=1)
train_df = train_df.dropna(subset=['sii'])

X_train = train_df.drop(columns=['sii'])  # 假設 'category' 是標籤欄位
y_train = train_df['sii']
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

imputer = SimpleImputer(strategy='median')
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_val = pd.DataFrame(imputer.fit_transform(X_val), columns=X_val.columns)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_val_pred = model.predict(X_val)

accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred))