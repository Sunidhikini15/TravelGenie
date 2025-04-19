
# ✈️ Smart Itinerary Planner 🧳

-> Your ML-powered assistant for personalized travel planning and budget-based recommendations 🌍

## 📑 Table of Contents

- [🌟 Introduction](#-introduction)
- [💪 Features](#-features)
- [⚙ Technologies Used](#-technologies-used)
- [🚀 Getting Started](#-getting-started)
    - [📦 Installation](#-installation)
    - [▶ Running the App](#-running-the-app)

---

## 🌟 Introduction

Smart Itinerary Planner is an interactive, machine learning-based application that helps users plan their trips by recommending suitable flights, hotels, and local attractions—all based on user budget and preferences. Built with Jupyter Notebook and ipywidgets, this tool is ideal for students, researchers, or hobbyists exploring how AI can enhance real-world travel planning.

---

## 💪 Features

* 🛫 Flight Recommendations: Suggests top flights under your budget using filtering and fuzzy city matching.
* 🏨 Hotel Classifier: Uses K-Nearest Neighbors to categorize hotels into Budget, Mid-range, or Luxury.
* 🗺️ City Attraction Suggestion: Recommends places to visit based on city and budget using Random Forest.
* 💸 Budget-Conscious Planning: Tailored suggestions based on user-specified constraints.
* 🤖 Pre-trained ML Models: No need to train every time—just load and go!
* 🎛️ Interactive Interface: Built with ipywidgets for an easy Jupyter Notebook experience.
* 📂 CSV-Based Input: Uses real-world travel datasets (Goibibo, Stayzilla, etc.).
* 🔄 Modular Codebase: Easy to plug in APIs or improve individual modules.
---

## ⚙ Technologies Used

* ML Models:
  * scikit-learn 🧠
  * K-Nearest Neighbors (KNN)
  * Random Forest Classifier
* Data Processing:
  * pandas 📊
  * numpy 🔢
* Interface:
  * ipywidgets 🧩
  * Jupyter Notebook 📓
* Utilities:
  * joblib 💾 (for model serialization)

---

## 🚀 Getting Started

### 📦 Installation

#### 🔧 Prerequisites

Make sure you have the following installed:

* Python 3.8+ 🐍
* Jupyter Notebook or JupyterLab
* pip 📦

1. Clone the repository:

```bash
git clone https://github.com/Sunidhikini15/TravelGenie.git
cd itinerary-planner-ml
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

---

### ▶ Running the App

1. Launch Jupyter Notebook:

```bash
jupyter notebook
```

2. Open the main notebook (e.g. `itinerary_planner.ipynb`).

3. Enter:
   - Source and destination cities
   - Flight and hotel budget
   - Number of people

4. Get your personalized travel plan with:
   - 🛫 Flights
   - 🏨 Hotels
   - 🗺️ Attractions

---

Happy Traveling! 🚀
