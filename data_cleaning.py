import pandas as pd
import os


def clean_ipl_data():
    print("Loading datasets...")
    matches = pd.read_csv('matches_updated_mens_ipl.csv')
    deliveries = pd.read_csv('deliveries_updated_ipl_upto_2025.csv')
    batter_stats = pd.read_excel('IPL_Batter_Stats_2008_2025.xlsx')

    print("Standardizing team names...")
    team_mapping = {
        'Delhi Daredevils': 'Delhi Capitals',
        'Kings XI Punjab': 'Punjab Kings',
        'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
        'Deccan Chargers': 'Sunrisers Hyderabad',
        'Gujarat Lions': 'Gujarat Titans',
        'Pune Warriors': 'Rising Pune Supergiant',
        'Rising Pune Supergiants': 'Rising Pune Supergiant'
    }

    # Safely replace team names if the columns exist
    if 'team1' in matches.columns: matches['team1'] = matches['team1'].replace(team_mapping)
    if 'team2' in matches.columns: matches['team2'] = matches['team2'].replace(team_mapping)
    if 'toss_winner' in matches.columns: matches['toss_winner'] = matches['toss_winner'].replace(team_mapping)
    if 'winner' in matches.columns: matches['winner'] = matches['winner'].replace(team_mapping)

    if 'batting_team' in deliveries.columns: deliveries['batting_team'] = deliveries['batting_team'].replace(
        team_mapping)
    if 'bowling_team' in deliveries.columns: deliveries['bowling_team'] = deliveries['bowling_team'].replace(
        team_mapping)

    print("Cleaning missing cities and venues...")
    if 'city' in matches.columns and 'venue' in matches.columns:
        matches.loc[
            (matches['city'].isnull()) & (matches['venue'] == 'Dubai International Cricket Stadium'), 'city'] = 'Dubai'
        matches.loc[(matches['city'].isnull()) & (matches['venue'] == 'Sharjah Cricket Stadium'), 'city'] = 'Sharjah'
        matches['city'] = matches['city'].fillna('Unknown')

    print("Handling DLS matches...")
    if 'method' in matches.columns:
        matches_no_dls = matches[matches['method'] != 'D/L']
    else:
        matches_no_dls = matches

    print("Identifying ID columns for the merge...")
    # Dynamically find the exact name of the ID column in the matches dataset
    match_id_col = None
    for col in ['id', 'ID', 'match_id', 'Match_ID', 'matchId']:
        if col in matches.columns:
            match_id_col = col
            break

    # Dynamically find the exact name of the ID column in the deliveries dataset
    delivery_id_col = None
    for col in ['id', 'ID', 'match_id', 'Match_ID', 'matchId']:
        if col in deliveries.columns:
            delivery_id_col = col
            break

    if not match_id_col or not delivery_id_col:
        print(f"CRITICAL ERROR: Could not find ID columns to merge on.")
        print(f"Matches columns: {matches.columns.tolist()}")
        print(f"Deliveries columns: {deliveries.columns.tolist()}")
        return None, None

    print(f"Success: Merging data using '{match_id_col}' and '{delivery_id_col}'...")

    # Rename the match ID column so it perfectly matches the delivery dataset
    matches_renamed = matches.rename(columns={match_id_col: delivery_id_col})

    # Only pull columns that actually exist to avoid KeyErrors
    cols_to_merge = [delivery_id_col]
    for col in ['venue', 'winner', 'toss_winner']:
        if col in matches_renamed.columns:
            cols_to_merge.append(col)

    # Perform the merge
    master_df = pd.merge(deliveries, matches_renamed[cols_to_merge], on=delivery_id_col, how='left')

    print("Exporting cleaned datasets...")
    matches.to_csv('cleaned_matches.csv', index=False)
    master_df.to_csv('cleaned_merged_data.csv', index=False)
    batter_stats.to_csv('cleaned_batter_stats.csv', index=False)

    print("\n✅ Data cleaning complete! 3 new cleaned files saved to your project directory.")
    print(f"Shape of your new Master Dataset: {master_df.shape}")

    return master_df, batter_stats


if __name__ == '__main__':
    master_data, batter_data = clean_ipl_data()