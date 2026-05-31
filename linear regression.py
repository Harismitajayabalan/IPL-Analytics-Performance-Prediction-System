import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def run_ipl_linear_regression():
    print("Parsing raw datasets and grouping chronological variables...")
    cols = ['matchId', 'inning', 'batsman', 'batsman_runs', 'over']
    match_player_data = []

    # Read delivery data in chunks to handle memory efficiently
    for chunk in pd.read_csv('cleaned_merged_data.csv', usecols=cols, chunksize=100000):
        match_player_data.append(chunk)

    df_deliv = pd.concat(match_player_data, ignore_index=True)

    # Calculate exact batting order position based on the earliest over a batsman entries
    player_entry = df_deliv.groupby(['matchId', 'inning', 'batsman'])['over'].min().reset_index()
    player_entry = player_entry.sort_values(['matchId', 'inning', 'over']).reset_index(drop=True)
    player_entry['batting_order'] = player_entry.groupby(['matchId', 'inning']).cumcount() + 1

    # Aggregate total runs scored by each batsman in each specific innings
    player_runs = df_deliv.groupby(['matchId', 'inning', 'batsman'])['batsman_runs'].sum().reset_index()
    df_merged = pd.merge(player_entry, player_runs, on=['matchId', 'inning', 'batsman'])

    # Merge match metadata to acquire match dates and venues for chronological sorting
    df_matches_meta = pd.read_csv('cleaned_matches.csv')[['matchId', 'venue', 'date']]
    df_final = pd.merge(df_merged, df_matches_meta, on='matchId')
    df_final['date'] = pd.to_datetime(df_final['date'])
    df_final = df_final.sort_values('date').reset_index(drop=True)

    print("Engineering categorical pitch types and rolling performance averages...")
    venue_means = df_final.groupby('venue')['batsman_runs'].mean()

    # Classify pitch behavior dynamically using historical average run scoring metrics
    def map_pitch_type(venue_name):
        mean_score = venue_means.get(venue_name, 20.0)
        if mean_score > 21.5:
            return 'Flat'
        elif mean_score < 18.5:
            return 'Spin'
        else:
            return 'Green/Balanced'

    df_final['pitch_type'] = df_final['venue'].apply(map_pitch_type)

    # Calculate form feature: 5-innings rolling average, shifted to exclude current match
    df_final['recent_avg_score'] = df_final.groupby('batsman')['batsman_runs'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    df_final['recent_avg_score'] = df_final['recent_avg_score'].fillna(15.0)

    # One-hot encode categorical pitch conditions for regression compatibility
    df_encoded = pd.get_dummies(df_final, columns=['pitch_type'], drop_first=True)

    # Separate dataset into predictive model features (X) and target variable (y)
    features = ['batting_order', 'recent_avg_score'] + [col for col in df_encoded.columns if 'pitch_type_' in col]
    X = df_encoded[features]
    y = df_encoded['batsman_runs']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Fitting Linear Regression model coefficients...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Generate test predictions and compute goodness-of-fit indicators
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n================ LINEAR REGRESSION PERFORMANCE ================")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-Squared Score (Model Fit Accuracy): {r2 * 100:.2f}%")
    print("================================================================")

    # Visualize model fit accuracy by plotting actual values against predictions
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, color='#1f77b4', alpha=0.3, edgecolors='none', label='Predicted vs Actual')

    max_val = max(max(y_test), max(y_pred))
    ax.plot([0, max_val], [0, max_val], color='#d62728', linestyle='--', linewidth=2, label='Perfect Fit Line')

    zoom_limit = 60
    ax.set_xlim(0, zoom_limit)
    ax.set_ylim(0, zoom_limit)

    ax.set_title('IPL Player Runs Prediction - Linear Regression Actual vs Predicted', fontsize=12, fontweight='bold',
                 pad=10)
    ax.set_xlabel('Actual Runs Scored', fontsize=10)
    ax.set_ylabel('Predicted Runs Scored', fontsize=10)
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('linear_regression_runs_prediction.png', dpi=300)
    plt.close('all')
    print("✅ Evaluation graph saved as 'linear_regression_runs_prediction.png'")

    # Predictive pipeline for calculating individual player expectations in upcoming matchups
    def predict_future_player_innings(player_name, historical_venue, expected_order):
        player_row = df_final[df_final['batsman'] == player_name].tail(1)
        if player_row.empty:
            recent_avg = 15.0
        else:
            recent_avg = player_row['recent_avg_score'].values[0]

        simulated_pitch = map_pitch_type(historical_venue)

        input_data = {
            'batting_order': expected_order,
            'recent_avg_score': recent_avg
        }

        # Initialize categorical feature vector columns to 0
        for col in features:
            if 'pitch_type_' in col:
                input_data[col] = 0

        # Activate the flag matching current venue conditions
        active_col = f'pitch_type_{simulated_pitch}'
        if active_col in features:
            input_data[active_col] = 1

        input_df = pd.DataFrame([input_data])
        input_df = input_df[features]
        prediction = model.predict(input_df)[0]

        print("\n================ LIVE REAL-DATA RUNS PREDICTION ================")
        print(f"Player Name: {player_name} | Venue Profile: {historical_venue} ({simulated_pitch} Track)")
        print(f"Inferred Parameters -> Position: #{expected_order} | Recent 5-Innings Average: {recent_avg:.2f}")
        print("-" * 66)
        print(f"Predicted Runs Context: {max(0.0, prediction):.1f} runs")
        print("================================================================")

    predict_future_player_innings('V Kohli', 'M Chinnaswamy Stadium', 1)


if __name__ == '__main__':
    run_ipl_linear_regression()