import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# API設定
API_KEY = '15eacaeda11540e89b88bc75598d370f'
headers = {'X-Auth-Token': API_KEY}

# タイトル
st.title("プレミアリーグ勝敗予想")

# 過去5年分のデータを取得
@st.cache_data(ttl=3600, show_spinner="試合データを取得しています...")
def load_matches():
    seasons = [2021, 2022, 2023, 2024, 2025]
    matches = []
    for season in seasons:
        url = f'https://api.football-data.org/v4/competitions/PL/matches?season={season}'
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            matches.extend(data['matches'])
    return matches

all_matches = load_matches()

# DataFrame作成
df = pd.DataFrame(all_matches)

# 必要な列のみ選択
df = df[['id', 'utcDate', 'homeTeam', 'awayTeam', 'score', 'status']]

# utc_dateを日時型に変換
df['utc_date'] = pd.to_datetime(df['utcDate'], utc=True)
df = df.sort_values('utc_date').reset_index(drop=True)

# スコア抽出
df['home_score'] = df['score'].apply(lambda x: x['fullTime']['home'] if x['fullTime']['home'] is not None else 0)
df['away_score'] = df['score'].apply(lambda x: x['fullTime']['away'] if x['fullTime']['away'] is not None else 0)
df['ht_home_score'] = df['score'].apply(lambda x: x['halfTime']['home'] if x['halfTime']['home'] is not None else 0)
df['ht_away_score'] = df['score'].apply(lambda x: x['halfTime']['away'] if x['halfTime']['away'] is not None else 0)

# チーム名抽出
df['home_team'] = df['homeTeam'].apply(lambda x: x['name'])
df['away_team'] = df['awayTeam'].apply(lambda x: x['name'])

# 勝者判定
def determine_winner(row):
    if row['status'] != 'FINISHED':
        return None
    if row['home_score'] > row['away_score']:
        return 'HOME_TEAM'
    elif row['home_score'] < row['away_score']:
        return 'AWAY_TEAM'
    else:
        return 'DRAW'

df['winner'] = df.apply(determine_winner, axis=1)

# チーム情報を取得
@st.cache_data(ttl=86400, show_spinner="チーム情報を取得しています...")
def load_teams():
    teams_url = 'https://api.football-data.org/v4/competitions/PL/teams'
    teams_response = requests.get(teams_url, headers=headers)
    return teams_response.json()['teams'] if teams_response.status_code == 200 else []

teams_data = load_teams()

team_crests = {team['name']: team.get('crest', '') for team in teams_data}

# 現在プレミアリーグに所属しているチーム名リスト
current_teams = set(team['name'] for team in teams_data)

# データフレームを現所属チームのみでフィルタ
df = df[df['home_team'].isin(current_teams) & df['away_team'].isin(current_teams)].reset_index(drop=True)

# 順位表計算

# 今シーズン（2025年）のみで順位表を作成
season_now = 2025
season_matches = df[(df['status'] == 'FINISHED') & (df['utc_date'].dt.year == season_now)]
standings = {}
for _, match in season_matches.iterrows():
    home = match['home_team']
    away = match['away_team']
    home_score = match['home_score']
    away_score = match['away_score']
    for team in [home, away]:
        if team not in standings:
            standings[team] = {'played': 0, 'won': 0, 'drawn': 0, 'lost': 0, 'gf': 0, 'ga': 0, 'gd': 0, 'points': 0}
    standings[home]['played'] += 1
    standings[away]['played'] += 1
    standings[home]['gf'] += home_score
    standings[home]['ga'] += away_score
    standings[away]['gf'] += away_score
    standings[away]['ga'] += home_score
    if home_score > away_score:
        standings[home]['won'] += 1
        standings[away]['lost'] += 1
        standings[home]['points'] += 3
    elif home_score < away_score:
        standings[away]['won'] += 1
        standings[home]['lost'] += 1
        standings[away]['points'] += 3
    else:
        standings[home]['drawn'] += 1
        standings[away]['drawn'] += 1
        standings[home]['points'] += 1
        standings[away]['points'] += 1
   
for team in standings:
    standings[team]['gd'] = standings[team]['gf'] - standings[team]['ga']
standings_df = pd.DataFrame.from_dict(standings, orient='index').reset_index().rename(columns={'index': 'team'})
if not standings_df.empty:
    standings_df = standings_df.sort_values(['points', 'gd', 'gf'], ascending=False).reset_index(drop=True)
    standings_df['position'] = standings_df.index + 1
else:
    # データがない場合は空の列を作成してエラーを防ぐ
    standings_df = pd.DataFrame(columns=['team', 'played', 'won', 'drawn', 'lost', 'gf', 'ga', 'gd', 'points', 'position'])

# 特徴量計算とモデル学習（終了した試合のみ）
finished_matches = df[df['status'] == 'FINISHED']
finished_df = df[df['status'] == 'FINISHED'].copy()
finished_df['target'] = finished_df['winner'].map({'HOME_TEAM': 0, 'AWAY_TEAM': 1, 'DRAW': 2})
finished_df = finished_df.dropna(subset=['target']).reset_index(drop=True)

def calculate_team_stats(team_name, match_date, df):
    past_matches = df[(df['utc_date'] < match_date) & ((df['home_team'] == team_name) | (df['away_team'] == team_name))].tail(5)
    if len(past_matches) == 0:
        return 0, 0, 0, 0, 0
    points = 0
    win = 0
    goal = 0
    lose_goal = 0
    for _, match in past_matches.iterrows():
        if match['home_team'] == team_name:
            goal += match['home_score']
            lose_goal += match['away_score']
            if match['winner'] == 'HOME_TEAM':
                points += 3
                win += 1
            elif match['winner'] == 'DRAW':
                points += 1
        else:
            goal += match['away_score']
            lose_goal += match['home_score']
            if match['winner'] == 'AWAY_TEAM':
                points += 3
                win += 1
            elif match['winner'] == 'DRAW':
                points += 1
    avg_goal = goal / len(past_matches)
    avg_lose = lose_goal / len(past_matches)
    avg_gd = (goal - lose_goal) / len(past_matches)
    win_rate = win / len(past_matches)
    return points, avg_goal, avg_lose, avg_gd, win_rate


# -----------------------------------------------------
# 新規機能：ポアソン分布による勝率計算
# -----------------------------------------------------
def calculate_poisson_probabilities(lambda_home, lambda_away, max_goals=10):
    prob_home = 0
    prob_draw = 0
    prob_away = 0
    for i in range(max_goals):
        for j in range(max_goals):
            p = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
            if i > j:
                prob_home += p
            elif i == j:
                prob_draw += p
            else:
                prob_away += p
    total = prob_home + prob_draw + prob_away
    if total == 0: return 0.33, 0.33, 0.34
    return float(prob_home/total), float(prob_away/total), float(prob_draw/total)

# -----------------------------------------------------
# 新規機能：Elo計算関数
# -----------------------------------------------------
elo_ratings = {team: 1500.0 for team in current_teams}
K = 20
HFA = 50 # ホームアドバンテージ

def get_expected_score(rating_a, rating_b, hfa=0):
    return 1 / (1 + 10 ** ((rating_b - (rating_a + hfa)) / 400))

# 追加特徴量
home_points = []
home_avg_goal = []
home_avg_lose = []
home_avg_gd = []
home_win_rate = []
away_points = []
away_avg_goal = []
away_avg_lose = []
away_avg_gd = []
away_win_rate = []
home_rank = []
away_rank = []
point_diff = []

home_elos = []
away_elos = []
poisson_hp = []
poisson_ap = []
poisson_dp = []

# 今季順位表を使って順位・勝ち点差を特徴量に
team_to_rank = {row['team']: row['position'] for _, row in standings_df.iterrows()}
team_to_point = {row['team']: row['points'] for _, row in standings_df.iterrows()}

for idx, row in finished_df.iterrows():
    hp, hgoal, hlose, hgd, hwin = calculate_team_stats(row['home_team'], row['utc_date'], finished_df.iloc[:idx+1])
    ap, agoal, alose, agd, awin = calculate_team_stats(row['away_team'], row['utc_date'], finished_df.iloc[:idx+1])
    
    h_team = row['home_team']
    a_team = row['away_team']
    if h_team not in elo_ratings: elo_ratings[h_team] = 1500
    if a_team not in elo_ratings: elo_ratings[a_team] = 1500
    
    h_elo = elo_ratings[h_team]
    a_elo = elo_ratings[a_team]
    home_elos.append(h_elo)
    away_elos.append(a_elo)
    
    # Eloの更新 (target 0:Home Win, 1:Away Win, 2:Draw)
    if row['target'] == 0: h_act, a_act = 1, 0
    elif row['target'] == 1: h_act, a_act = 0, 1
    else: h_act, a_act = 0.5, 0.5
    h_exp = get_expected_score(h_elo, a_elo, HFA)
    a_exp = get_expected_score(a_elo, h_elo, -HFA)
    elo_ratings[h_team] = h_elo + K * (h_act - h_exp)
    elo_ratings[a_team] = a_elo + K * (a_act - a_exp)

    # ポアソン確率の計算
    l_home = (hgoal + alose) / 2.0 if (hgoal + alose) > 0 else 0.5
    l_away = (agoal + hlose) / 2.0 if (agoal + hlose) > 0 else 0.5
    ph, pa, pd_draw = calculate_poisson_probabilities(l_home, l_away)
    poisson_hp.append(ph)
    poisson_ap.append(pa)
    poisson_dp.append(pd_draw)

    home_points.append(hp)
    home_avg_goal.append(hgoal)
    home_avg_lose.append(hlose)
    home_avg_gd.append(hgd)
    home_win_rate.append(hwin)
    away_points.append(ap)
    away_avg_goal.append(agoal)
    away_avg_lose.append(alose)
    away_avg_gd.append(agd)
    away_win_rate.append(awin)
    home_rank.append(team_to_rank.get(row['home_team'], 10))
    away_rank.append(team_to_rank.get(row['away_team'], 10))
    point_diff.append(team_to_point.get(row['home_team'], 0) - team_to_point.get(row['away_team'], 0))

finished_df['home_recent_points'] = home_points
finished_df['home_avg_goal'] = home_avg_goal
finished_df['home_avg_lose'] = home_avg_lose
finished_df['home_avg_goal_diff'] = home_avg_gd
finished_df['home_win_rate'] = home_win_rate
finished_df['away_recent_points'] = away_points
finished_df['away_avg_goal'] = away_avg_goal
finished_df['away_avg_lose'] = away_avg_lose
finished_df['away_avg_goal_diff'] = away_avg_gd
finished_df['away_win_rate'] = away_win_rate
finished_df['home_rank'] = home_rank
finished_df['away_rank'] = away_rank
finished_df['point_diff'] = point_diff

finished_df['home_elo'] = home_elos
finished_df['away_elo'] = away_elos
finished_df['poisson_prob_home'] = poisson_hp
finished_df['poisson_prob_away'] = poisson_ap
finished_df['poisson_prob_draw'] = poisson_dp

features = [
    'home_recent_points', 'home_avg_goal', 'home_avg_lose', 'home_avg_goal_diff', 'home_win_rate',
    'away_recent_points', 'away_avg_goal', 'away_avg_lose', 'away_avg_goal_diff', 'away_win_rate',
    'home_rank', 'away_rank', 'point_diff',
    'home_elo', 'away_elo', 'poisson_prob_home', 'poisson_prob_away', 'poisson_prob_draw'
]
X = finished_df[features]
y = finished_df['target']

split_idx = int(len(finished_df) * 0.8)
X_train = X[:split_idx]
X_test = X[split_idx:]
y_train = y[:split_idx]
y_test = y[split_idx:]


# LightGBMとXGBoostのアンサンブル
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier

# 小さいデータセットで過学習を防ぐため、パラメータを調整
xgb = XGBClassifier(random_state=42, n_estimators=100, max_depth=3, learning_rate=0.05)
lgbm = LGBMClassifier(random_state=42, n_estimators=100, max_depth=3, learning_rate=0.05)
model = VotingClassifier(estimators=[('xgb', xgb), ('lgbm', lgbm)], voting='soft')
model.fit(X_train, y_train)

# テストセットでのアンサンブル正解率を計算
y_pred_ml_probs = model.predict_proba(X_test)
final_preds = []
for i, (_, row) in enumerate(X_test.iterrows()):
    p_home = row['poisson_prob_home']
    p_away = row['poisson_prob_away']
    p_draw = row['poisson_prob_draw']
    
    ml_h, ml_a, ml_d = y_pred_ml_probs[i]
    
    # 70% ML, 30% Poisson
    f_h = ml_h * 0.7 + p_home * 0.3
    f_a = ml_a * 0.7 + p_away * 0.3
    f_d = ml_d * 0.7 + p_draw * 0.3
    
    # 最も高い確率のものを予測結果（0:Home, 1:Away, 2:Draw）とする
    pred_class = np.argmax([f_h, f_a, f_d])
    final_preds.append(pred_class)

accuracy = accuracy_score(y_test, final_preds)

# タブ作成
tab1, tab2, tab3, tab4 = st.tabs(["順位表", "試合日程", "勝敗予想", "データ統計"])

with tab1:
    st.header("プレミアリーグ順位表")
    # エンブレム列を追加
    standings_df['crest'] = standings_df['team'].map(lambda x: team_crests.get(x, ''))
    # 表示用データフレーム
    display_df = standings_df[['position', 'team', 'crest', 'points', 'won', 'drawn', 'lost', 'gf', 'ga', 'gd']].copy()
    display_df = display_df.rename(columns={
        'position': '順位', 'team': 'チーム', 'crest': 'エンブレム', 'points': '勝ち点',
        'won': '勝', 'drawn': '分', 'lost': '負', 'gf': '得点', 'ga': '失点', 'gd': '得失点'
    })
    # Streamlitの表でエンブレム画像を表示
    def path_to_image_html(path):
        if path:
            return f'<img src="{path}" width="30">'
        else:
            return ''
    st.write(display_df.to_html(escape=False, formatters={'エンブレム': path_to_image_html}, index=False), unsafe_allow_html=True)

with tab2:
    st.header("試合日程")
    upcoming_matches = df[df['status'] != 'FINISHED'].sort_values('utc_date').head(20)
    # エンブレム列を追加
    upcoming_matches = upcoming_matches.copy()
    upcoming_matches['home_crest'] = upcoming_matches['home_team'].map(lambda x: team_crests.get(x, ''))
    upcoming_matches['away_crest'] = upcoming_matches['away_team'].map(lambda x: team_crests.get(x, ''))
    schedule_df = upcoming_matches[['utc_date', 'home_team', 'home_crest', 'away_team', 'away_crest']]
    schedule_df = schedule_df.rename(columns={
        'utc_date': '日時', 'home_team': 'ホーム', 'home_crest': 'ホームエンブレム',
        'away_team': 'アウェイ', 'away_crest': 'アウェイエンブレム'
    })
    def crest_html(path):
        if path:
            return f'<img src="{path}" width="20">'
        else:
            return ''
    st.write(schedule_df.to_html(escape=False, formatters={'ホームエンブレム': crest_html, 'アウェイエンブレム': crest_html}, index=False), unsafe_allow_html=True)

with tab3:
    st.header("勝敗予想")
    home_team = st.selectbox("ホームチームを選択", df['home_team'].unique())
    away_team = st.selectbox("アウェイチームを選択", df['away_team'].unique())
    if st.button("予想結果を表示"):
        hp, hgoal, hlose, hgd, hwin = calculate_team_stats(home_team, pd.Timestamp.now(tz='UTC'), df)
        ap, agoal, alose, agd, awin = calculate_team_stats(away_team, pd.Timestamp.now(tz='UTC'), df)
        
        hrank = team_to_rank.get(home_team, 10)
        arank = team_to_rank.get(away_team, 10)
        pdiff = team_to_point.get(home_team, 0) - team_to_point.get(away_team, 0)

        h_elo = elo_ratings.get(home_team, 1500)
        a_elo = elo_ratings.get(away_team, 1500)
        
        l_home = (hgoal + alose) / 2.0 if (hgoal + alose) > 0 else 0.5
        l_away = (agoal + hlose) / 2.0 if (agoal + hlose) > 0 else 0.5
        ph, pa, pd_draw = calculate_poisson_probabilities(l_home, l_away)

        input_data = pd.DataFrame({
            'home_recent_points': [hp],
            'home_avg_goal': [hgoal],
            'home_avg_lose': [hlose],
            'home_avg_goal_diff': [hgd],
            'home_win_rate': [hwin],
            'away_recent_points': [ap],
            'away_avg_goal': [agoal],
            'away_avg_lose': [alose],
            'away_avg_goal_diff': [agd],
            'away_win_rate': [awin],
            'home_rank': [hrank],
            'away_rank': [arank],
            'point_diff': [pdiff],
            'home_elo': [h_elo],
            'away_elo': [a_elo],
            'poisson_prob_home': [ph],
            'poisson_prob_away': [pa],
            'poisson_prob_draw': [pd_draw]
        })
        input_data = input_data[features]
        
        # 機械学習モデル（LightGBM + XGBoost）の予測
        proba = model.predict_proba(input_data)[0]
        
        # アンサンブル: MLとポアソンを混ぜる (割合: ML 70%, Poisson 30%)
        final_home = proba[0] * 0.7 + ph * 0.3
        final_away = proba[1] * 0.7 + pa * 0.3
        final_draw = proba[2] * 0.7 + pd_draw * 0.3
        total = final_home + final_away + final_draw
        final_home, final_away, final_draw = final_home/total, final_away/total, final_draw/total
        
        st.subheader("AIスコア予想 (Poisson & ML Ensemble)")
        st.write(f"**{home_team}** の勝率: **{final_home*100:.1f}%**")
        st.write(f"**{away_team}** の勝率: **{final_away*100:.1f}%**")
        st.write(f"引き分けの確率: **{final_draw*100:.1f}%**")
        
        st.markdown("---")
        st.markdown("💡 **[AI分析の内訳]**")
        st.write(f"ベース勝率(ポアソン分布分析): {home_team} {ph*100:.1f}% / {away_team} {pa*100:.1f}% / 引分 {pd_draw*100:.1f}%")
        st.write(f"直近・相性補正(機械学習): {home_team} {proba[0]*100:.1f}% / {away_team} {proba[1]*100:.1f}% / 引分 {proba[2]*100:.1f}%")
        st.write(f"Eloレーティング: {home_team} (**{h_elo:.0f}**) vs {away_team} (**{a_elo:.0f}**)")

with tab4:
    st.header("データ統計")
    st.write(f"**取得した総試合数**: {len(all_matches)} 試合")
    st.markdown("**(※過去5年分の全試合データ)**")
    
    st.write(f"**分析対象の試合数 (現在プレミア所属チーム間のみ)**: {len(df)} 試合")
    st.markdown("**(※昇格・降格で現在リーグにいないチームとの対戦は、現在の強さを測る上でノイズになるため除外しています)**")
    
    st.write(f"**そのうち終了済みの試合数**: {len(finished_matches)} 試合")
    st.markdown("**(※この終了済み試合データを使ってAIモデルを学習させています)**")
    
    st.markdown("---")
    st.subheader("モデルの精度")
    st.write(f"**過去データの予測テスト正解率**: {accuracy:.2%}")
    st.markdown("*(※サッカーの結果予測においては、ドロー判定があるため完全にランダムに予想した場合は約33%になります。)*")