import pandas as pd
from sklearn.preprocessing import StandardScaler


def prepare_kmeans_features():
    print("Loading cleaned batter stats...")
    df = pd.read_csv('cleaned_batter_stats.csv')

    print(f"Initial players: {len(df)}")

    # 1. REMOVE NOISE: Keep only true batters/all-rounders
    # Filtering out players who have faced less than 100 balls in their career
    df = df[df['balls'] >= 100].copy()
    print(f"Players after filtering noise (faced >100 balls): {len(df)}")

    # 2. HANDLE SITUATIONAL NULLS
    # Players who never open will have a 'Null' in powerplay_sr. We fill these with 0.
    # This helps K-Means realize "Ah, this player doesn't bat in the powerplay!"
    columns_to_fill_zero = [
        'powerplay_sr', 'middle_overs_sr', 'death_overs_sr',
        'opener_sr', 'middle_order_sr', 'finisher_sr',
        'accel_mid_minus_pp', 'accel_death_minus_mid'
    ]
    for col in columns_to_fill_zero:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Fill remaining general nulls (like missing consistency index) with the average
    df['batting_average'] = df['batting_average'].fillna(df['batting_average'].mean())
    df['consistency_index'] = df['consistency_index'].fillna(df['consistency_index'].mean())
    df['player_impact_index'] = df['player_impact_index'].fillna(df['player_impact_index'].mean())
    df['boundary_pct'] = df['boundary_pct'].fillna(0)
    df['dot_ball_pct'] = df['dot_ball_pct'].fillna(0)

    # 3. SELECT FEATURES FOR CLUSTERING
    # We drop raw totals (like 'runs' or 'matches') because we want to cluster
    # based on PLAYSTYLE, not longevity.
    features_for_clustering = [
        'batting_average', 'strike_rate', 'boundary_pct', 'dot_ball_pct',
        'powerplay_sr', 'death_overs_sr', 'consistency_index', 'player_impact_index'
    ]

    # Extract only the features and the player names
    clustering_data = df[['batter'] + features_for_clustering].copy()

    print("Scaling features using StandardScaler...")
    # 4. SCALING THE DATA
    # This forces all metrics to have a mean of 0 and standard deviation of 1
    # so Strike Rate and Average are weighted equally by the K-Means math.
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(clustering_data[features_for_clustering])

    # Create a new dataframe with the scaled data
    scaled_df = pd.DataFrame(scaled_features, columns=features_for_clustering)
    scaled_df.insert(0, 'batter', clustering_data['batter'].values)  # Put names back

    # 5. EXPORT
    scaled_df.to_csv('kmeans_ready_data.csv', index=False)
    print("\n✅ Feature Engineering Complete! Saved as 'kmeans_ready_data.csv'")
    print("This data is now 100% ready to be plugged into the K-Means algorithm.")

    return scaled_df


if __name__ == '__main__':
    # NOTE: You will need to install scikit-learn if you haven't already:
    # Open terminal and run: pip install scikit-learn
    kmeans_data = prepare_kmeans_features()