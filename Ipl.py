import pandas as pd
import streamlit as st
import re
from ortools.sat.python import cp_model
import plotly.express as px

st.set_page_config(page_title="IPL Simulator", layout="wide")
st.title("🏏 IPL Qualification Simulator")

# Clean team names
def clean_team_name(name):
    return re.sub(r"\s*\(.*?\)", "", str(name)).strip().lower()

# Main simulation function
def run_simulation(points_df, matches_df):
    points_df['Team'] = points_df['Team'].apply(clean_team_name)
    matches_df['Team1'] = matches_df['Team1'].apply(clean_team_name)
    matches_df['Team2'] = matches_df['Team2'].apply(clean_team_name)

    st.subheader("📊 Current Points Table")
    st.dataframe(points_df)

    st.subheader("🗓️ Remaining Schedule")
    st.dataframe(matches_df)

    team_options = sorted(points_df['Team'].str.title().unique())
    selected_team = st.selectbox("Select a team to simulate:", team_options)
    scenario_type = st.radio("Choose Simulation Type", ["Forced Best-Case Scenario", "Manual Scenario"])

    selected_team_clean = selected_team.strip().lower()
    known_teams = set(points_df['Team'])
    sim_df = points_df.set_index("Team").copy()
    sim_df[["Wins", "Losses", "Points"]] = sim_df[["Wins", "Losses", "Points"]].fillna(0).astype(int)
    match_outcomes = {}

    if selected_team:
        if scenario_type == "Manual Scenario":
            st.subheader("🎮 Pick Winners for Each Match")
            for i, row in matches_df.iterrows():
                t1, t2 = row["Team1"].title(), row["Team2"].title()
                winner = st.radio(f"{t1} vs {t2}", [t1, t2], key=f"match_{i}")
                match_outcomes[i] = winner.lower()

        if st.button("Run Scenario Simulation"):
            temp_df = sim_df.copy()
            win_logic = []
            rank_progression = []

            if scenario_type == "Forced Best-Case Scenario":
                model = cp_model.CpModel()
                match_vars = {}
                temp_points = {team: int(temp_df.loc[team, "Points"]) for team in known_teams}

                for i, row in matches_df.iterrows():
                    t1, t2 = row["Team1"], row["Team2"]
                    if selected_team_clean in [t1, t2]:
                        win_team = selected_team_clean
                        lose_team = t2 if t1 == selected_team_clean else t1
                        temp_points[win_team] += 2
                        temp_df.loc[win_team, "Points"] += 2
                        temp_df.loc[win_team, "Wins"] += 1
                        temp_df.loc[lose_team, "Losses"] += 1
                        win_logic.append(f"{t1.title()} vs {t2.title()} — ✅ {win_team.title()} wins")
                    else:
                        var = model.NewBoolVar(f"match_{i}")
                        match_vars[i] = (var, t1, t2)

                for var, t1, t2 in match_vars.values():
                    temp_points[t1] += 2 * var
                    temp_points[t2] += 2 * (1 - var)

                ranks = {team: model.NewIntVar(1, len(known_teams), f"rank_{team}") for team in known_teams}
                for team in known_teams:
                    better_than = []
                    for other in known_teams:
                        if team == other:
                            continue
                        comp = model.NewBoolVar(f"{team}_gt_{other}")
                        model.Add(temp_points[team] > temp_points[other]).OnlyEnforceIf(comp)
                        model.Add(temp_points[team] <= temp_points[other]).OnlyEnforceIf(comp.Not())
                        better_than.append(comp)
                    model.Add(ranks[team] == len(known_teams) - sum(better_than))

                model.Add(ranks[selected_team_clean] <= 4)
                solver = cp_model.CpSolver()
                status = solver.Solve(model)

                if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                    for var, t1, t2 in match_vars.values():
                        win = solver.Value(var)
                        winner = t1 if win == 1 else t2
                        loser = t2 if win == 1 else t1
                        temp_df.loc[winner, "Points"] += 2
                        temp_df.loc[winner, "Wins"] += 1
                        temp_df.loc[loser, "Losses"] += 1
                        win_logic.append(f"{t1.title()} vs {t2.title()} — ✅ {winner.title()} wins")

                        ranked = temp_df.sort_values(by=["Points", "Wins"], ascending=False).reset_index()
                        ranked.index += 1
                        for idx, row in ranked.iterrows():
                            rank_progression.append({"Match": len(win_logic), "Team": row["Team"].title(), "Rank": idx})

                    final_table = temp_df.sort_values(by=["Points", "Wins"], ascending=False).reset_index()
                    final_table.index += 1
                    st.success(f"✅ {selected_team.title()} forcefully qualifies for playoffs in this best-case scenario!")
                else:
                    st.error(f"❌ {selected_team.title()} cannot qualify for playoffs under any scenario.")
                    final_table = temp_df.sort_values(by=["Points", "Wins"], ascending=False).reset_index()
                    final_table.index += 1

            else:
                for i, row in matches_df.iterrows():
                    t1, t2 = row["Team1"], row["Team2"]
                    winner = match_outcomes.get(i)
                    if not winner:
                        continue
                    loser = t2 if winner == t1 else t1
                    temp_df.loc[winner, "Points"] += 2
                    temp_df.loc[winner, "Wins"] += 1
                    temp_df.loc[loser, "Losses"] += 1

                    ranked = temp_df.sort_values(by=["Points", "Wins"], ascending=False).reset_index()
                    ranked.index += 1
                    for idx, row in ranked.iterrows():
                        rank_progression.append({"Match": i+1, "Team": row["Team"].title(), "Rank": idx})

                    win_logic.append(f"{t1.title()} vs {t2.title()} — ✅ {winner.title()} wins")

                final_table = temp_df.sort_values(by=["Points", "Wins"], ascending=False).reset_index()
                final_table.index += 1

            st.subheader("🏁 Final Table")
            st.dataframe(final_table[["Team", "Points", "Wins", "Losses"]])
            top4 = final_table.head(4)["Team"].tolist()

            if selected_team_clean in top4:
                st.success(f"✅ {selected_team.title()} qualifies in this scenario!")
            else:
                st.error(f"❌ {selected_team.title()} does not qualify.")

            st.subheader("📜 Match Outcomes:")
            for line in win_logic:
                st.markdown(f"- {line}")

            st.subheader("📊 Points Bar Chart")
            before_sim = sim_df.reset_index()[["Team", "Points"]].copy()
            before_sim["Scenario"] = "Before"
            after_sim = final_table[["Team", "Points"]].copy()
            after_sim["Scenario"] = "After"
            all_df = pd.concat([before_sim, after_sim])
            fig_bar = px.bar(all_df, x="Team", y="Points", color="Scenario", barmode="group")
            st.plotly_chart(fig_bar)

            st.subheader("📈 Rank Progression Graph")
            rank_df = pd.DataFrame(rank_progression)
            if not rank_df.empty:
                fig = px.line(rank_df, x="Match", y="Rank", color="Team", markers=True)
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(fig)

#  Mode selection with persistence
st.sidebar.header("⚙️ Load Match Data")
mode = st.sidebar.radio("Choose mode", ["Use default Excel file", "Upload custom files"])

if mode == "Use default Excel file":
    try:
        points_df = pd.read_excel("IPL_2025_Points_Table.xlsx", sheet_name="points table")
        matches_df = pd.read_excel("IPL_2025_Points_Table.xlsx", sheet_name="schedule ")
        run_simulation(points_df, matches_df)
    except Exception as e:
        st.error(f"❌ Failed to load default Excel file: {e}")

else:
    points_file = st.sidebar.file_uploader("Upload Points Table Excel", type=["xlsx"])
    matches_file = st.sidebar.file_uploader("Upload Schedule Excel", type=["xlsx"])

    if points_file and matches_file:
        try:
            points_df = pd.read_excel(points_file, sheet_name="points table")
            matches_df = pd.read_excel(matches_file, sheet_name="schedule ")
            run_simulation(points_df, matches_df)
        except Exception as e:
            st.error(f"❌ Error reading uploaded files: {e}")
