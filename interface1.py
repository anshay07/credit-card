import timeit
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import warnings
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
import base64
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

warnings.filterwarnings("ignore")
st.title('Credit Card Fraud Detection with AES Encryption')

# Load the credit card transaction data
df = st.cache(pd.read_csv)('creditcard.csv')

# Show basic information about the dataset
if st.sidebar.checkbox('Show what the dataframe looks like'):
    st.write(df.head(100))
    st.write('Shape of the dataframe: ', df.shape)
    st.write('Data description: \n', df.describe())

# Separate fraud and valid transactions
fraud = df[df.Class == 1]
valid = df[df.Class == 0]
outlier_percentage = (df.Class.value_counts()[1] / df.Class.value_counts()[0]) * 100

# Split the data into features and target
X = df.drop(['Class'], axis=1)
y = df.Class

# Split the data into training and testing sets
test_size = st.sidebar.slider('Test Set Size', min_value=0.2, max_value=0.4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Calculate fraudulent transaction percentage based on the test set
outlier_percentage = (y_test.value_counts()[1] / len(y_test)) * 100

# Display the fraud and valid transaction details dynamically based on the test set
if st.sidebar.checkbox('Show fraud and valid transaction details'):
    st.write(f'Fraudulent transactions in test set: {outlier_percentage:.3f}%')
    st.write('Fraud Cases in test set: ', y_test.value_counts()[1])
    st.write('Valid Cases in test set: ', y_test.value_counts()[0])


# Print shape of train and test sets
if st.sidebar.checkbox('Show the shape of training and test set features and labels'):
    st.write('X_train: ', X_train.shape)
    st.write('y_train: ', y_train.shape)
    st.write('X_test: ', X_test.shape)
    st.write('y_test: ', y_test.shape)

# AES Encryption Functions

def generate_aes_key_iv(key_size=256):
    key = get_random_bytes(key_size // 8)
    iv = get_random_bytes(AES.block_size)
    return key, iv

def aes_encrypt(key, iv, message):
    try:
        cipher = AES.new(key, AES.MODE_CBC, iv)
        ct_bytes = cipher.encrypt(pad(message.encode(), AES.block_size))
        encrypted_message = base64.b64encode(ct_bytes).decode()
        return True, encrypted_message
    except Exception as e:
        return False, str(e)

def aes_decrypt(key, iv, encrypted_message):
    try:
        ct = base64.b64decode(encrypted_message)
        cipher = AES.new(key, AES.MODE_CBC, iv)
        pt = unpad(cipher.decrypt(ct), AES.block_size)
        return True, pt.decode()
    except Exception as e:
        return False, str(e)

# Encrypt Time and Amount fields
def encrypt_data(df, key, iv):
    df['Encrypted_Time'] = df['Time'].apply(lambda x: aes_encrypt(key, iv, str(x))[1])
    df['Encrypted_Amount'] = df['Amount'].apply(lambda x: aes_encrypt(key, iv, str(x))[1])
    return df

# AES Key and IV generation
key, iv = generate_aes_key_iv()

# Encrypt Time and Amount columns
df = encrypt_data(df, key, iv)

# Optionally: Display the encrypted Time and Amount data
if st.sidebar.checkbox('Show encrypted Time and Amount'):
    st.write(df[['Encrypted_Time', 'Encrypted_Amount']].head(10))

# Feature selection through feature importance using Extra Trees or Random Forest
@st.cache_data
def feature_sort(_model, X_train, y_train):
    mod = _model
    mod.fit(X_train, y_train)
    imp = mod.feature_importances_
    return imp

# Classifiers for feature importance
clf = ['Extra Trees', 'Random Forest']
mod_feature = st.sidebar.selectbox('Which model for feature importance?', clf)

etree = ExtraTreesClassifier(random_state=42)
rforest = RandomForestClassifier(random_state=42)

start_time = timeit.default_timer()
if mod_feature == 'Extra Trees':
    model = etree
    importance = feature_sort(model, X_train, y_train)
elif mod_feature == 'Random Forest':
    model = rforest
    importance = feature_sort(model, X_train, y_train)
elapsed = timeit.default_timer() - start_time
st.write(f'Execution Time for feature selection: {elapsed / 60:.2f} minutes')

# Show feature importance plot
if st.sidebar.checkbox('Show plot of feature importance'):
    plt.bar([x for x in range(len(importance))], importance)
    plt.title('Feature Importance')
    plt.xlabel('Feature (Variable Number)')
    plt.ylabel('Importance')
    st.pyplot()

# Select top features
features = X_train.columns.tolist()
feature_imp = list(zip(features, importance))
feature_sort = sorted(feature_imp, key=lambda x: x[1])
n_top_features = st.sidebar.slider('Number of top features', min_value=5, max_value=20)
top_features = list(list(zip(*feature_sort[-n_top_features:]))[0])

if st.sidebar.checkbox('Show selected top features'):
    st.write(f'Top {n_top_features} features in order of importance: {top_features[::-1]}')

X_train_sfs = X_train[top_features]
X_test_sfs = X_test[top_features]

# Balance the dataset using SMOTE or Near Miss
smt = SMOTE()
nr = NearMiss()

def compute_performance(model, X_train, y_train, X_test, y_test):
    import timeit
    from sklearn.metrics import (
        confusion_matrix,
        classification_report,
        matthews_corrcoef,
    )
    from sklearn.model_selection import cross_val_score
    import streamlit as st

    # Start timer
    start_time = timeit.default_timer()

    # Cross-validation accuracy
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()
    st.write(f'Cross-Validation Accuracy: {scores:.2f}')

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Metrics computation
    cm = confusion_matrix(y_test, y_pred)
    st.write('Confusion Matrix:', cm)


    # Generate the classification report
    cr = classification_report(y_test, y_pred, output_dict=True)  # Output as a dictionary for better formatting

    # Convert classification report dictionary to a DataFrame for tabular representation
    cr_df = pd.DataFrame(cr).transpose()

# Round the metrics to two decimal places for better readability
    cr_df = cr_df.round(2)

# Display the classification report as a table
    st.write("Classification Report (Formatted):")
    st.dataframe(cr_df)  # Streamlit's built-in function to display tables


    mcc = matthews_corrcoef(y_test, y_pred)
    st.write(f'Matthews Correlation Coefficient (MCC): {mcc:.2f}')

    # Execution time
    elapsed = timeit.default_timer() - start_time
    st.write(f'Execution Time for performance computation: {elapsed / 60:.2f} minutes')

    return cm, cr, mcc


# Run different classification models with rectifiers
if st.sidebar.checkbox('Run a credit card fraud detection model'):
    alg = ['Extra Trees', 'Random Forest', 'k Nearest Neighbor', 'Support Vector Machine', 'Logistic Regression']
    classifier = st.sidebar.selectbox('Which algorithm?', alg)
    rectifier = ['SMOTE', 'Near Miss', 'No Rectifier']
    imb_rect = st.sidebar.selectbox('Which imbalanced class rectifier?', rectifier)

    # Define the selected model
    if classifier == 'Logistic Regression':
        model = LogisticRegression()
    elif classifier == 'k Nearest Neighbor':
        model = KNeighborsClassifier()
    elif classifier == 'Support Vector Machine':
        model = SVC()
    elif classifier == 'Random Forest':
        model = RandomForestClassifier(random_state=42)
    elif classifier == 'Extra Trees':
        model = ExtraTreesClassifier(random_state=42)

    # Handle the imbalanced rectifier logic
    if imb_rect == 'No Rectifier':
        cm, cr, mcc = compute_performance(model, X_train_sfs, y_train, X_test_sfs, y_test)
    elif imb_rect == 'SMOTE':
        X_train_bal, y_train_bal = smt.fit_resample(X_train_sfs, y_train)
        cm, cr, mcc = compute_performance(model, X_train_bal, y_train_bal, X_test_sfs, y_test)
        fraud_bal_percentage = (y_train_bal.sum() / len(y_train_bal)) * 100
        st.write(f"Fraudulent transactions after balancing (SMOTE): {fraud_bal_percentage:.3f}%")
    elif imb_rect == 'Near Miss':
        X_train_bal, y_train_bal = nr.fit_resample(X_train_sfs, y_train)
        cm, cr, mcc = compute_performance(model, X_train_bal, y_train_bal, X_test_sfs, y_test)
        fraud_bal_percentage = (y_train_bal.sum() / len(y_train_bal)) * 100
        st.write(f"Fraudulent transactions after balancing (Near Miss): {fraud_bal_percentage:.3f}%")

    # Display results
    st.write(f'Confusion Matrix: \n{cm}')
    st.write(f'Classification Report: \n{cr}')
    st.write(f'Matthews Correlation Coefficient: {mcc}')

# Example of encryption and decryption
key, iv = generate_aes_key_iv()
message = 'Test prediction result'
encrypted_message = aes_encrypt(key, iv, message)[1]
decrypted_message = aes_decrypt(key, iv, encrypted_message)[1]

st.write(f'Original message: {message}')
st.write(f'Encrypted message: {encrypted_message}')
st.write(f'Decrypted message: {decrypted_message}')