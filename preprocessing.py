import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
train_df = pd.read_csv("child-mind-institute-problematic-internet-use/train.csv")
test_df = pd.read_csv("child-mind-institute-problematic-internet-use/test.csv")
SEASON_COLS = [
    "Basic_Demos-Enroll_Season", 
    "CGAS-Season", 
    "Physical-Season", 
    "Fitness_Endurance-Season", 
    "FGC-Season", 
    "BIA-Season", 
    "PAQ_A-Season", 
    "PAQ_C-Season", 
    "SDS-Season",
    "PreInt_EduHx-Season"]
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
    "PCIAT-PCIAT_Total"
]
train_df = train_df.drop(TARGET_COLS,axis=1)
for col in test_df.columns:
    if col not in train_df.columns:
        test_df.drop(columns=col, inplace=True)
#train_df = train_df.dropna(subset=['sii'])

season_mapping = {'Spring': 0, 'Summer': 1, 'Fall': 2, 'Winter': 3}
for col in SEASON_COLS:
    train_df[col] = train_df[col].map(season_mapping)
    test_df[col] = test_df[col].map(season_mapping)

id_train = train_df['id']
id_test = test_df['id']
train_df.drop(columns=['id'], inplace=True)
test_df.drop(columns=['id'], inplace=True)

imputer = SimpleImputer(strategy='median')
X_train = pd.DataFrame(imputer.fit_transform(train_df), columns=train_df.columns)
X_test = pd.DataFrame(imputer.fit_transform(test_df), columns=test_df.columns)
X_train['sii'] = X_train['sii'].round().astype(int)