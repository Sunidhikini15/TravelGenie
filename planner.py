import pandas as pd
from difflib import get_close_matches

def load_flight_data(file_path):
    df = pd.read_csv("goibibo.csv")
    all_sources = df['Source'].dropna().unique().tolist()
    all_destinations = df['Destination'].dropna().unique().tolist()
    return df, all_sources, all_destinations

def suggest_flights(source, destination, max_budget, df, all_sources, all_destinations):
    source = source.strip().title()
    destination = destination.strip().title()

    corrected_source = get_close_matches(source, all_sources, n=1)
    corrected_destination = get_close_matches(destination, all_destinations, n=1)

    if not corrected_source or not corrected_destination:
        print("\n❌ Oops! Couldn't recognize one of the cities.")
        print("🧭 Available Source Cities:", ', '.join(sorted(all_sources)))
        print("🧭 Available Destination Cities:", ', '.join(sorted(all_destinations)))
        return

    source = corrected_source[0]
    destination = corrected_destination[0]

    filtered = df[
        (df['Source'] == source) &
        (df['Destination'] == destination) &
        (df['Price'] <= max_budget)
    ]

    top_5 = filtered.sort_values(by='Price').head(5)
    result = top_5[['Airline', 'Source', 'Destination', 'Dep_Time', 'Arrival_Time', 'Price']]

    if result.empty:
        print(f"\n⚠️ No flights from {source} to {destination} within ₹{max_budget}. Try changing your inputs or increasing the budget!")
    else:
        print(f"\n🎯 Top 5 cheapest flights from {source} to {destination} under ₹{max_budget}:\n")
        print(result.to_string(index=False))
