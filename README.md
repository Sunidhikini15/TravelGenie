Smart Itinerary Planner Using Machine Learning
This project is a machine learning-based travel itinerary planner that recommends flights, hotels, and local attractions based on user inputs like source, destination, and budget. It uses classification and regression models trained on travel-related data to suggest optimized travel plans.

🗂️ Project Structure

File/Folder	Description
data/	Contains CSV files for flights, hotels, and attractions
models/	Stores trained ML models (pickle format)
scripts/	Python scripts for data preprocessing, training, testing, and prediction logic
utils/helpers.py	Utility functions for data loading, filtering, and processing
app.py	Main interactive script for user input and end-to-end planning
requirements.txt	Python package dependencies
README.md	This file
🔧 Setup Instructions
Clone the repo or download the zip.

bash
Copy
Edit
git clone https://github.com/Sunidhikini15/TravelGenie.git
cd  TravelGenie
Create and activate a virtual environment (optional but recommended).

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies.

bash
Copy
Edit
pip install -r requirements.txt
🚀 Running the Project
1. Preprocess the data
bash
Copy
Edit
python scripts/preprocess_data.py
Cleans and prepares the datasets.

2. Train Hotel Classifier (KNN)
bash
Copy
Edit
python scripts/train_hotel_classifier.py
Trains a K-Nearest Neighbors classifier to classify hotels based on amenities and price.

3. Train Budget Prediction Model
bash
Copy
Edit
python scripts/train_budget_model.py
Trains a regression model to estimate total budget based on user and city parameters.

4. Test Models (Optional)
bash
Copy
Edit
python scripts/test_models.py
Evaluates accuracy and performance of the models on test datasets.

5. Run the Itinerary Planner
bash
Copy
Edit
python app.py
Launches the interactive planner where user can input:

Source and destination

Number of days

Preferred type of hotel

Budget

And get recommendations for:

Best-matching flights

Suitable hotels

Local attractions

Estimated trip cost

🧠 Models Used
KNN for Hotel Classification

Linear Regression / Random Forest for Budget Estimation

Rule-based filtering for Flights and Attractions

📈 Future Enhancements
Integration with real-time flight/hotel APIs (Skyscanner, Amadeus)

Personalized recommendations using user history

Map-based UI

Mobile version with GPS-aware recommendations
