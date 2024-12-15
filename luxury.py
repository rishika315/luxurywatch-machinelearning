import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split as sklearn_train_test_split  # Rename to avoid conflict
from sklearn.neighbors import NearestNeighbors
from surprise.model_selection import train_test_split as surprise_train_test_split  # Rename to avoid conflict
from textblob import TextBlob
from surprise import Dataset, Reader, SVD
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import shap
import streamlit as st

# Load the data
file_path = r'C:\project1\luxurywatches.csv' 
data = pd.read_csv(file_path)
data.fillna(0, inplace=True)

# Encoding categorical data
label_encoders = {}
for column in ['Brand', 'Model', 'Case Material', 'Strap Material', 'Movement Type', 'Dial Color', 'Crystal Material', 'Complications']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column].astype(str))
    label_encoders[column] = le

# Convert 'Water Resistance' to a numeric format by extracting the numeric part
data['Water Resistance'] = data['Water Resistance'].str.extract('(\d+)').astype(float)

# Convert 'Power Reserve' to a numeric format and handle "N/A"
data['Power Reserve'] = data['Power Reserve'].str.extract('(\d+)').fillna(0).astype(float)

# Convert 'Price (USD)' to a numeric format by removing commas and converting to float
data['Price (USD)'] = data['Price (USD)'].replace('[\$,]', '', regex=True).astype(float)

# Features and Target
X = data[['Brand', 'Case Material', 'Strap Material', 'Movement Type', 'Water Resistance',
          'Case Diameter (mm)', 'Case Thickness (mm)', 'Band Width (mm)', 'Dial Color',
          'Crystal Material', 'Complications', 'Power Reserve', 'Price (USD)']]

# Adding 'Golden Carrots' (simulated random values for now)
data['Golden Carrots'] = np.random.randint(1, 6, size=len(data))
y = data['Golden Carrots']

# Train-test split for features using sklearn
X_train, X_test, y_train, y_test = sklearn_train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Neural Network Model for predicting Golden Carrots
def build_nn_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # Output layer
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

nn_model = build_nn_model()
nn_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32)

# Predict watch rating using Neural Network
def predict_watch_rating(watch_name):
    watch_index = data[data['Model'] == watch_name].index[0]
    watch_features = X.iloc[watch_index]
    watch_features_scaled = scaler.transform([watch_features])
    prediction = nn_model.predict(watch_features_scaled)
    return prediction[0][0]

# Sentiment Analysis on Reviews
reviews = {
    'Submariner': ['Great watch!', 'Very durable and stylish.', 'Best watch for divers.'],
    'Seamaster': ['Fantastic design.', 'A bit pricey but worth it.', 'Very reliable.'],
}

def analyze_sentiment(reviews):
    sentiment_scores = {}
    for watch, review_list in reviews.items():
        sentiments = [TextBlob(review).sentiment.polarity for review in review_list]
        avg_sentiment = np.mean(sentiments) if sentiments else 0
        sentiment_scores[watch] = avg_sentiment
    return sentiment_scores

# Applies the sentiment analysis to the dataset, assigning sentiment scores to watches.
sentiment_scores = analyze_sentiment(reviews)
data['Sentiment_Score'] = data['Model'].apply(lambda x: sentiment_scores.get(x, 0))

# Fits a KNN model to the scaled training data using Euclidean distance and 5 neighbors.
knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
knn.fit(X_train_scaled)

# Fetches the nearest neighbors for the selected watch and returns similar watches as recommendations.
def recommend_watches(watch_name):
    watch_index = data[data['Model'] == watch_name].index[0]
    watch_features = X.iloc[watch_index]
    watch_features_scaled = scaler.transform([watch_features])
    distances, indices = knn.kneighbors([watch_features_scaled[0]])
    recommendations = []
    for idx in indices[0]:
        recommended_watch = data.iloc[idx]['Model']
        recommendations.append(recommended_watch)
    return recommendations

# Simulates random users, loads the watch data into the Surprise library format
data['User_ID'] = np.random.randint(0, 100, size=len(data))
reader = Reader(rating_scale=(1, 5))
watch_data = Dataset.load_from_df(data[['User_ID', 'Model', 'Golden Carrots']], reader)

# Use the Surprise train_test_split to create train and test sets
trainset, testset = surprise_train_test_split(watch_data, test_size=0.2)

# Train the SVD model
svd = SVD()
svd.fit(trainset)

# Predicts a user's rating for a given watch model using SVD
def predict_svd(user_id, watch_model):
    watch_model_encoded = label_encoders['Model'].transform([watch_model])[0]
    prediction = svd.predict(user_id, watch_model_encoded).est
    return prediction

# Builds an autoencoder neural network to detect anomalies.
def build_autoencoder():
    input_dim = X_train_scaled.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(32, activation="relu")(input_layer)
    encoder = Dense(16, activation="relu")(encoder)
    decoder = Dense(32, activation="relu")(encoder)
    decoder = Dense(input_dim, activation="sigmoid")(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

# Trains the autoencoder and calculates reconstruction error.
autoencoder = build_autoencoder()
autoencoder.fit(X_train_scaled, X_train_scaled, epochs=50, batch_size=32)
reconstructed = autoencoder.predict(X_test_scaled)
mse = np.mean(np.power(X_test_scaled - reconstructed, 2), axis=1)
threshold = np.percentile(mse, 95)
anomalies = mse > threshold

# SHAP for Model Explainability
explainer = shap.KernelExplainer(nn_model.predict, X_test_scaled)
shap_values = explainer.shap_values(X_test_scaled[:10])

# Streamlit UI
st.title("Watch Prediction and Recommendation System")

# Watch rating prediction
watch_name = st.selectbox('Select a watch model:', data['Model'].unique())
if st.button('Predict Golden Carrots'):
    rating = predict_watch_rating(watch_name)
    st.write(f"Predicted Golden Carrots for {watch_name}: {rating}")

# Display Reviews
if st.button('Show Reviews'):
    if watch_name in reviews:
        st.write(f"Reviews for {watch_name}:")
        for review in reviews[watch_name]:
            st.write(f"- {review}")
    else:
        st.write("No reviews found for this watch.")

# Recommendations based on KNN
if st.button('Recommend Similar Watches'):
    recommendations = recommend_watches(watch_name)
    st.write(f"Recommended watches based on {watch_name}:")
    for watch in recommendations:
        st.write(f"- {watch}")

# Collaborative filtering (SVD) recommendation
user_id = st.number_input("Enter user ID for personalized recommendation", min_value=0, max_value=100, value=10)
if st.button('Get SVD Recommendation'):
    svd_rating = predict_svd(user_id, watch_name)
    st.write(f"Predicted Golden Carrots for {watch_name} (User {user_id}): {svd_rating}")

# Anomaly detection
if st.button('Detect Anomalies in Prices'):
    st.write(f"Number of price anomalies detected: {np.sum(anomalies)}")

# SHAP explanation
if st.button('Explain Model with SHAP'):
    st.write("SHAP explanation for first 10 instances:")
    shap.initjs()
    st.pyplot(shap.force_plot(explainer.expected_value, shap_values[0], X_test_scaled[0]))