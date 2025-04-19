#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 📌 STEP 1: Load Data
file_path = 'stayzilla.csv'
df = pd.read_csv(file_path)
df
# 📌 STEP 2: Data Cleaning
# Drop rows with missing city or price
df.dropna(subset=['city', 'room_price'], inplace=True)

# 📌 STEP 3: Feature Engineering
X = df[['city', 'room_price']].copy()

# Convert 'room_price' to numeric, handling errors
# Extract numeric part from 'room_price' and convert to float
X['room_price'] = pd.to_numeric(X['room_price'].str.extract('(\d+)')[0], errors='coerce')

# Drop rows with invalid 'room_price' after conversion
X.dropna(subset=['room_price'], inplace=True)

# Update original DataFrame with cleaned 'room_price'
df['room_price'] = X['room_price']

# Normalize room_price
scaler = MinMaxScaler()
X[['room_price']] = scaler.fit_transform(X[['room_price']])

# One-hot encode city
X = pd.get_dummies(X, columns=['city'])

# Define price categories (Budget, Mid-range, Luxury)
df['price_category'] = pd.qcut(df['room_price'], q=3, labels=['Budget', 'Mid-range', 'Luxury'])

# Encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['price_category'])

# 📌 STEP 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=42)

# 📌 STEP 5: Train KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)

# 📌 STEP 6: Predictions
y_pred = classifier.predict(X_test)

# 📌 STEP 7: Evaluation
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy:", accuracy_score(y_test, y_pred))

# 📌 STEP 8: Recommendation Function
def recommend_hotels(city, budget, top_k=5):
    # Encode city
    city_encoded = pd.DataFrame({'city': [city]})
    city_encoded = pd.get_dummies(city_encoded, columns=['city'])

    # Add missing columns to match training data
    missing_cols = set(X.columns) - set(city_encoded.columns)
    for col in missing_cols:
        city_encoded[col] = 0

    city_encoded = city_encoded[X.columns].copy()  # Ensure correct column order

    # Normalize budget with consistent scaling
    ref_row = pd.DataFrame(np.zeros((1, len(X.columns))), columns=X.columns)
    ref_row['room_price'] = budget
    ref_row_scaled = scaler.transform(ref_row[['room_price']])
    city_encoded['room_price'] = ref_row_scaled[0][0]

    # Predict the price category
    predicted_category = classifier.predict(city_encoded.values)[0]
    predicted_category_name = label_encoder.inverse_transform([predicted_category])[0]

    # Recommend hotels with detailed information
    recommendations = df[(df['city'] == city) & (df['price_category'] == predicted_category_name)]

    # Return property_name, room_price, description, and room_types
    results = recommendations[['property_name', 'room_price', 'description', 'room_types']].head(top_k).to_dict(orient='records')

    # Print each recommendation with a newline
    print("\nRecommended Hotels:")
    for idx, hotel in enumerate(results, start=1):
        print(f"Hotel {idx}:")
        print(f"  Name: {hotel['property_name']}")
        print(f"  Price: {hotel['room_price']}")
        print(f"  Description: {hotel['description']}")
        print(f"  Room Type: {hotel['room_types']}\n")

    return results

# 📌 STEP 9: Test Example
city_input = 'Kanpur'
budget_input = 3000

recommended_hotels = recommend_hotels(city_input, budget_input)


# In[12]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def train_model_hotels():
    """
    Train a KNN model to classify hotel price categories.
    Returns the trained model, label encoder, scaler, and processed dataframe.
    """
    # Load and clean the dataset
    df = pd.read_csv("stayzilla.csv")
    df.dropna(subset=["city", "room_price"], inplace=True)

    # Extract and convert room price
    df["room_price"] = pd.to_numeric(df["room_price"].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    df.dropna(subset=["room_price"], inplace=True)

    # Normalize room price
    scaler = MinMaxScaler()
    df["room_price_scaled"] = scaler.fit_transform(df[["room_price"]])

    # Define target categories
    df["price_category"] = pd.qcut(df["room_price"], q=3, labels=["Budget", "Mid-range", "Luxury"])
    label_encoder = LabelEncoder()
    df["target"] = label_encoder.fit_transform(df["price_category"])

    # One-hot encode city
    city_dummies = pd.get_dummies(df["city"], prefix="city")
    X = pd.concat([df["room_price_scaled"], city_dummies], axis=1)
    y = df["target"]

    # Split and train model
    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=42)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    print(f"\n✅ Hotel model trained. Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return model, label_encoder, scaler, df, X.columns.tolist()

def predict_hotel_category(city, budget, model, encoder, scaler, df, feature_columns, top_k=5):
    """
    Predict hotel category based on user input and recommend hotels.
    """
    # Prepare city one-hot encoding
    city_df = pd.DataFrame([{"city": city}])
    city_dummies = pd.get_dummies(city_df, prefix="city")

    # Add missing city columns
    for col in feature_columns:
        if col not in city_dummies.columns and col != "room_price_scaled":
            city_dummies[col] = 0

    # Ensure proper order and structure
    city_dummies = city_dummies.reindex(columns=feature_columns, fill_value=0)

    # Add normalized budget
    dummy = pd.DataFrame([[0]*len(feature_columns)], columns=feature_columns)
    dummy["room_price_scaled"] = scaler.transform([[budget]])[0][0]
    city_dummies["room_price_scaled"] = dummy["room_price_scaled"].values[0]

    # Predict category
    pred_label = model.predict(city_dummies.values)[0]
    category = encoder.inverse_transform([pred_label])[0]

    print(f"\n--- Hotel Recommendations in {city} ---")
    print(f"Predicted Budget Category: {category}")

    recommendations = df[(df["city"] == city) & (df["price_category"] == category)]
    if recommendations.empty:
        print("No matching hotels found.")
        return

    results = recommendations[["property_name", "room_price", "description", "room_types"]].head(top_k)
    for idx, row in results.iterrows():
        print(f"\nHotel {idx+1}:")
        print(f"  Name: {row['property_name']}")
        print(f"  Price: ₹{row['room_price']}")
        print(f"  Description: {row['description']}")
        print(f"  Room Type: {row['room_types']}")

# Optional: Run this as a script
def main():
    city = input("Enter city name: ")
    budget = float(input("Enter your hotel budget (₹): "))

    model, encoder, scaler, df, feature_columns = train_model_hotels()
    
    import joblib
    joblib.dump(model, "hotel_model.pkl")

    
    predict_hotel_category(city, budget, model, encoder, scaler, df, feature_columns)

if __name__ == "__main__":
    main()


# In[7]:


get_ipython().system('jupyter nbconvert --to python trained_model_hotels.ipynb')


# In[ ]:




