import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


def train_player_impact_predictor():
    """
    Trains a Decision Tree Classifier to predict IPL match outcomes
    based on aggregated team player impact indices.
    """
    print("Loading datasets and building player knowledge base...")
    matches = pd.read_csv('cleaned_matches.csv')
    batters = pd.read_csv('cleaned_batter_stats.csv')

    # Create an optimized mapping of players to their historical impact index
    player_impact_map = dict(zip(batters['batter'], batters['player_impact_index']))

    cols_to_use = ['matchId', 'batting_team', 'batsman']
    unique_combinations = {}

    print("Streaming ball-by-ball data in chunks of 50,000 rows to prevent Out-Of-Memory crashes...")
    # Use chunking to parse large ball-by-ball delivery logs efficiently
    for chunk in pd.read_csv('cleaned_merged_data.csv', usecols=cols_to_use, chunksize=50000):
        chunk = chunk.dropna(subset=['batsman'])
        for (m_id, team), group in chunk.groupby(['matchId', 'batting_team']):
            key = (m_id, team)
            batters_set = set(group['batsman'].unique())
            if key in unique_combinations:
                unique_combinations[key].update(batters_set)
            else:
                unique_combinations[key] = batters_set

    # Aggregate total roster impact score per match per team
    match_team_scores = {}
    for (m_id, team), batters_set in unique_combinations.items():
        # Fallback default impact value of 25.0 for missing player data
        score = sum([player_impact_map.get(p, 25.0) for p in batters_set])
        match_team_scores[(m_id, team)] = score

    df_matches = matches[['matchId', 'team1', 'team2', 'winner']].dropna(subset=['winner']).copy()

    t1_impacts = []
    t2_impacts = []

    # Map aggregate roster impact values back to historical match lineups
    for idx, row in df_matches.iterrows():
        m_id = row['matchId']
        t1 = row['team1']
        t2 = row['team2']

        # Fallback default squad strength of 250.0 if line-up records are missing
        score1 = match_team_scores.get((m_id, t1), 250.0)
        score2 = match_team_scores.get((m_id, t2), 250.0)

        t1_impacts.append(score1)
        t2_impacts.append(score2)

    df_matches['team1_player_impact'] = t1_impacts
    df_matches['team2_player_impact'] = t2_impacts

    # Binary encoding of target variable: 1 if Team 1 wins, 0 otherwise
    df_matches['team1_won'] = (df_matches['winner'] == df_matches['team1']).astype(int)

    # Feature selection and train-test target extraction
    features = ['team1_player_impact', 'team2_player_impact']
    X = df_matches[features]
    y = df_matches['team1_won']

    # Stratified-ready random split for evaluation validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize Decision Tree with restricted depth to prevent overfitting
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X_train, y_train)

    # Model Evaluation
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n================ PLAYER-BASED ACCURACY METRICS ================")
    print(f"Overall Accuracy Score: {accuracy * 100:.2f}%\n")

    # Generate detailed performance report across classification groups
    report_dict = classification_report(y_test, y_pred, target_names=['Team 2 Wins', 'Team 1 Wins'], output_dict=True)

    print(f"{'Outcome Group':<20}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}{'Support':<12}")
    print("-" * 68)
    for group_name in ['Team 2 Wins', 'Team 1 Wins']:
        scores = report_dict[group_name]
        print(
            f"{group_name:<20}{scores['precision']:<12.2f}{scores['recall']:<12.2f}{scores['f1-score']:<12.2f}{int(scores['support']):<12}")
    print("================================================================")

    # Inference helper function for hypothetical or upcoming team roster lineups
    def predict_match_by_rosters(team1_name, team2_name, team1_players, team2_players):
        """
        Computes dynamic probability and win outcomes for active custom lineups.
        """
        t1_score = sum([player_impact_map.get(p, 25.0) for p in team1_players])
        t2_score = sum([player_impact_map.get(p, 25.0) for p in team2_players])

        input_data = {
            'team1_player_impact': t1_score,
            'team2_player_impact': t2_score
        }

        predict_df = pd.DataFrame([input_data])
        prediction = clf.predict(predict_df[features])[0]
        probabilities = clf.predict_proba(predict_df[features])[0]

        winning_team = team1_name if prediction == 1 else team2_name

        print("\n================ LIVE ROSTER MATCH SIMULATION ================")
        print(f"Matchup: {team1_name} vs {team2_name}")
        print(f"Computed Lineup Strengths -> {team1_name}: {t1_score:.2f} | {team2_name}: {t2_score:.2f}")
        print("-" * 64)
        print(f"Predicted Winner: {winning_team}")
        print(
            f"Confidence Level -> {team1_name}: {probabilities[1] * 100:.2f}% | {team2_name}: {probabilities[0] * 100:.2f}%")
        print("================================================================")

    # Example verification rosters
    csk_live_lineup = ['RD Gaikwad', 'MS Dhoni', 'AM Rahane', 'RA Jadeja', 'S Dube', 'M Pathirana']
    srh_live_lineup = ['TM Head', 'Abhishek Sharma', 'AK Markram', 'H Klaasen', 'Pat Cummins', 'B Kumar']

    predict_match_by_rosters('Chennai Super Kings', 'Sunrisers Hyderabad', csk_live_lineup, srh_live_lineup)


if __name__ == '__main__':
    train_player_impact_predictor()