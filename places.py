#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model():
    """
    Train a RandomForestClassifier to classify budget categories.
    Returns the trained model and the processed dataframe.
    """
    # Load the dataset
    file_path = "places.csv"  # Update with the correct path if needed
    df = pd.read_csv(file_path)

    # Drop rows with missing important values
    df = df.dropna(subset=[
        "City", "Entrance Fee in INR", "Google review rating",
        "time needed to visit in hrs"
    ])

    # Feature engineering
    df["Entrance Fee"] = df["Entrance Fee in INR"].astype(float)
    df["Rating"] = df["Google review rating"].astype(float)
    df["Duration_hours"] = df["time needed to visit in hrs"].astype(float)

    # Define target variable: Budget category
    df["BudgetCategory"] = (df["Entrance Fee"] > 100).astype(int)

    # Select features and target
    features = df[["Rating", "Duration_hours"]]
    target = df["BudgetCategory"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Model trained with accuracy: {accuracy * 100:.2f}%")

    return model, df

def predict_budget_category(city, model, df):
    """
    Predict budget category for tourist places in the given city.
    """
    # Filter the dataframe for the given city
    city_data = df[df['City'] == city]

    if city_data.empty:
        print(f"No data available for the city: {city}")
        return

    print(f"\n--- Predicted Budget Categories for {city} ---")
    for _, row in city_data.iterrows():
        # Extract features for prediction
        features = [[row['Rating'], row['Duration_hours']]]
        budget_label = model.predict(features)  # Predict using the trained model

        # Print the output in the desired format
        print(f"Destination: {row['Name']}")
        print(f"Budget: {'Expensive' if budget_label[0] == 1 else 'Budget-Friendly'}")
        print(f"Entry Fee: ₹{row['Entrance Fee']}")
        print(f"Ratings: {row['Rating']}\n")
        
import pandas as pd
# Function to capture user input and save it to a CSV
def capture_user_input():
    # Take user inputs
    Name = input("Enter your destination: ")
    transport_mode = input("Enter mode of transportation (e.g., car, train, flight): ")
    budget = float(input("Enter your total transportation budget (₹): "))
    num_people = int(input("Enter number of people traveling: "))

    # Save user input to a CSV file (User_input.csv)
    user_data = {
        "Destination": Name,
        "Mode of Transportation": transport_mode,
        "Total Budget": budget,
        "Number of People Traveling": num_people
    }

    # Check if the CSV file exists to append or create a new one
    filename = "places.csv"
    try:
        df = pd.read_csv(filename)
        # Use pd.concat to append new data to the dataframe
        df = pd.concat([df, pd.DataFrame([user_data])], ignore_index=True)
    except FileNotFoundError:
        df = pd.DataFrame([user_data])

    df.to_csv(filename, index=False)

    # Display the captured inputs
    print("\n--- Travel Plan Summary ---")
    print(f"Destination: {Name}")
    print(f"Mode of Transportation: {transport_mode}")
    print(f"Total Budget: ₹{budget}")
    print(f"Number of People Traveling: {num_people}")

    return Name  # Return city name for prediction

# Main logic
def main():
    # Capture user input
    city_input = capture_user_input()

    # Train model and get the trained model and dataframe
    model, df = train_model()
    
    # Save model to disk
    import joblib
    joblib.dump(model, "city_model.pkl")
    print("✅ Model saved as city_model.pkl")
    
    # Predict budget category for tourist places in the input city
    predict_budget_category(city_input, model, df)

# Run the program
if __name__ == "__main__":
    main()






