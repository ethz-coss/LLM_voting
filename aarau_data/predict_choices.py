import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
import statsmodels.api as sm
import ast
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

#----------------------------------- predict topics -----------------------------------#
data = pd.read_csv('aarau_data/processed/aarau_pb_vote_processed.csv')
df = data[['Gender', 'Age', 'Politics', 'Birthplace Info', 'Education', 'Nationality', 'Marital Status', 'Household Form', 'Children Info','Area', 'Topics']] #, 'Votes'
df.loc[:, 'Topics'] = df['Topics'].apply(ast.literal_eval)
df_long = df.explode('Topics')
df_long['Topics'] = pd.to_numeric(df_long['Topics'], errors='coerce')


X = df_long.drop('Topics', axis=1)
y = df_long['Topics']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#----------------------------------- linear regression
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)
model_sm = sm.OLS(y_train, X_train_sm).fit()
summary = model_sm.summary()
summary # not very insightfull


#----------------------------------- multinomical logistic
X_const = sm.add_constant(X)
mn_logit_model = sm.MNLogit(y, X_const)
mn_logit_result = mn_logit_model.fit()
mn_logit_summary = mn_logit_result.summary()
mn_logit_summary #these results are interesting: Depending on the topic, different features are important


#----------------------------------- XGBoost
feature_importances_df = pd.DataFrame(index=X_train.columns)

for topic in range(1, 9):  # Assuming topics numbered from 1 to 8
    # Create binary target for the current topic
    df_long['BinaryTopic'] = (df_long['Topics'] == topic).astype(int)

    # Prepare the data
    X = df_long.drop(['BinaryTopic', 'Topics'], axis=1)
    y = df_long['BinaryTopic']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    xgb_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    xgb_model.fit(X_train, y_train)

    # Store feature importances
    feature_importances_df[f'Topic {topic}'] = xgb_model.feature_importances_

plt.figure(figsize=(6, 4))
for column in feature_importances_df.columns:
    plt.plot(feature_importances_df.index, feature_importances_df[column], marker='o', label=column)

plt.xticks(rotation=45)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances by Topic')
plt.legend()
plt.grid(True)
plt.show()

#----------------------------------- SVM
feature_names = X.columns

# Initialize the DataFrame with feature names as index
feature_weights_df = pd.DataFrame(index=feature_names)

for topic in range(1, 9):  # Assuming topics numbered from 1 to 8
    # Create binary target for the current topic
    df_long['BinaryTopic'] = (df_long['Topics'] == topic).astype(int)

    # Prepare the data
    X = df_long.drop(['BinaryTopic', 'Topics'], axis=1)
    y = df_long['BinaryTopic']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the SVM model with a linear kernel
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train, y_train)

    # Extract feature weights and add them to the DataFrame
    weights = svm_model.coef_[0]
    feature_weights_df[f'Topic {topic}'] = weights

# Plotting
plt.figure(figsize=(10, 6))
for column in feature_weights_df.columns:
    plt.plot(feature_weights_df.index, feature_weights_df[column], marker='o', label=column)

plt.xticks(rotation=45)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('SVM: Feature Importances by Topic')
plt.legend()
plt.grid(True)
plt.show()


#----------------------------------- predict votes -----------------------------------#
data = pd.read_csv('aarau_data/processed/aarau_pb_vote_processed.csv')
df = data[['Gender', 'Age', 'Politics', 'Birthplace Info', 'Education', 'Nationality', 'Marital Status', 'Household Form', 'Children Info','Area', 'Votes']] #, 'Votes'
df.loc[:, 'Votes'] = df['Votes'].apply(ast.literal_eval)
df_long = df.explode('Votes')
df_long['Votes'] = pd.to_numeric(df_long['Votes'], errors='coerce')

df_long.dropna(inplace=True)
X = df_long.drop('Votes', axis=1)
y = df_long['Votes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#----------------------------------- delete votes with too few entries
votes_to_exclude = df_long['Votes'].value_counts()
votes_to_exclude = votes_to_exclude[votes_to_exclude < 20].index.tolist()
df_long = df_long[~df_long['Votes'].isin(votes_to_exclude)]


#----------------------------------- multinomical logistic
X_scaled = scaler.fit_transform(X)
X_const = sm.add_constant(X_scaled)
mn_logit_model = sm.MNLogit(y, X_const)
mn_logit_result = mn_logit_model.fit(method='newton', maxiter=100, full_output=True, disp=True, max_step=0.5)
mn_logit_summary = mn_logit_result.summary()
mn_logit_summary


#----------------------------------- XGBoost
feature_importances_df = pd.DataFrame(index=X_train.columns)

#for vote in range(1, 20):  # Assuming topics numbered from 1 to 8
for vote in df_long['Votes'].unique():
    df_long['BinaryVote'] = (df_long['Votes'] == vote).astype(int)

    X = df_long.drop(['BinaryVote', 'Votes'], axis=1)
    y = df_long['BinaryVote']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    xgb_model.fit(X_train, y_train)

    feature_importances_df[f'Vote {vote}'] = xgb_model.feature_importances_

plt.figure(figsize=(6, 4))
for column in feature_importances_df.columns:
    plt.plot(feature_importances_df.index, feature_importances_df[column], marker='o', label=column, alpha=0.5)
plt.xticks(rotation=45)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances by Vote')
#plt.legend()
plt.grid(True)
plt.show()