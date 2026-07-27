"""Streamlit dashboard over the pipeline's exported CSV/JSON artefacts.

Run directly:
    streamlit run dashboard/app.py -- --results output/

Or via the pipeline CLI:
    python main.py --input clip.mp4 --output output/ --dashboard
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

#: Stable colour per stroke type, reused across every chart on the page.
STROKE_COLORS = {
    "serve": "#4C78A8",
    "forehand": "#F58518",
    "backhand": "#54A24B",
    "volley": "#E45756",
}


def parse_args() -> argparse.Namespace:
    """Parse the ``--results`` directory from the Streamlit command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("output"))
    # Streamlit passes its own flags through; ignore anything we do not recognise.
    known, _ = parser.parse_known_args(sys.argv[1:])
    return known


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV artefact, returning an empty frame if it is absent."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def stroke_scale(values: list[str]) -> alt.Scale:
    """Build a colour scale covering the stroke types actually present."""
    domain = [name for name in STROKE_COLORS if name in values]
    extra = [name for name in values if name not in STROKE_COLORS]
    palette = [STROKE_COLORS[name] for name in domain] + ["#9D755D"] * len(extra)
    return alt.Scale(domain=domain + extra, range=palette)


def render_shot_distribution(shots: pd.DataFrame) -> None:
    """Bar chart of shot counts by stroke type, split by player."""
    st.subheader("Shot type distribution")

    counts = (
        shots.groupby(["stroke", "player"]).size().reset_index(name="count")
    )
    chart = (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X("stroke:N", title="Stroke", sort="-y"),
            y=alt.Y("count:Q", title="Shots"),
            color=alt.Color(
                "stroke:N", scale=stroke_scale(sorted(shots["stroke"].unique())), legend=None
            ),
            column=alt.Column("player:N", title="Player"),
            tooltip=["player", "stroke", "count"],
        )
        .properties(height=280, width=200)
    )
    st.altair_chart(chart)


def render_speed_over_time(shots: pd.DataFrame, tracks: pd.DataFrame) -> None:
    """Line chart of player speed over time, with shot moments marked."""
    st.subheader("Speed over time")

    if tracks.empty:
        st.info("No track log found — speeds require a trained court keypoint model.")
        return

    players = tracks[tracks["entity"].str.startswith("player_")].dropna(subset=["speed_kmh"])
    if players.empty:
        st.info(
            "The track log contains no speed samples. Speeds need the court homography; "
            "check that court.model in config.yaml points at trained weights."
        )
        return

    lines = (
        alt.Chart(players)
        .mark_line(opacity=0.85)
        .encode(
            x=alt.X("timestamp:Q", title="Time (s)"),
            y=alt.Y("speed_kmh:Q", title="Speed (km/h)"),
            color=alt.Color("entity:N", title="Player"),
            tooltip=["timestamp", "entity", "speed_kmh"],
        )
    )

    if not shots.empty:
        marks = (
            alt.Chart(shots)
            .mark_rule(strokeDash=[4, 3], opacity=0.6)
            .encode(
                x="timestamp:Q",
                color=alt.Color("stroke:N", scale=stroke_scale(sorted(shots["stroke"].unique()))),
                tooltip=["timestamp", "player", "stroke", "ball_speed_kmh"],
            )
        )
        chart = (lines + marks).properties(height=320)
    else:
        chart = lines.properties(height=320)

    st.altair_chart(chart, use_container_width=True)


def render_ball_speeds(shots: pd.DataFrame) -> None:
    """Scatter of ball speed at each shot, coloured by stroke type."""
    speeds = shots.dropna(subset=["ball_speed_kmh"])
    if speeds.empty:
        return

    st.subheader("Ball speed per shot")
    chart = (
        alt.Chart(speeds)
        .mark_circle(size=110, opacity=0.85)
        .encode(
            x=alt.X("timestamp:Q", title="Time (s)"),
            y=alt.Y("ball_speed_kmh:Q", title="Ball speed (km/h)"),
            color=alt.Color(
                "stroke:N", title="Stroke", scale=stroke_scale(sorted(speeds["stroke"].unique()))
            ),
            tooltip=["timestamp", "player", "stroke", "ball_speed_kmh"],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)


def render_distance(summary: pd.DataFrame) -> None:
    """Bar chart of total distance covered per player."""
    st.subheader("Distance covered")

    if summary.empty:
        st.info("No player summary found.")
        return

    chart = (
        alt.Chart(summary)
        .mark_bar()
        .encode(
            x=alt.X("player:N", title="Player"),
            y=alt.Y("total_distance_m:Q", title="Distance (m)"),
            color=alt.Color("player:N", legend=None),
            tooltip=["player", "total_distance_m", "avg_speed_kmh", "max_speed_kmh"],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(summary, use_container_width=True, hide_index=True)


def main() -> None:
    """Render the dashboard."""
    st.set_page_config(page_title="Tennis Analysis", page_icon="🎾", layout="wide")
    st.title("🎾 Tennis Analysis Dashboard")

    args = parse_args()
    results_dir = Path(st.sidebar.text_input("Results directory", str(args.results)))

    if not results_dir.exists():
        st.error(f"Results directory not found: {results_dir}")
        st.caption("Run the pipeline first: `python main.py --input clip.mp4 --output output/`")
        return

    shots = load_csv(results_dir / "shots.csv")
    tracks = load_csv(results_dir / "tracks.csv")
    summary = load_csv(results_dir / "player_summary.csv")

    video = results_dir / "annotated.mp4"
    if video.exists():
        with st.expander("Annotated video", expanded=False):
            st.video(str(video))

    columns = st.columns(4)
    columns[0].metric("Shots", len(shots))
    columns[1].metric("Players", len(summary))
    columns[2].metric(
        "Total distance (m)",
        f"{summary['total_distance_m'].sum():.1f}" if not summary.empty else "—",
    )
    columns[3].metric(
        "Peak ball speed (km/h)",
        f"{shots['ball_speed_kmh'].max():.1f}"
        if not shots.empty and shots["ball_speed_kmh"].notna().any()
        else "—",
    )

    st.divider()

    if shots.empty:
        st.warning(
            "No shots were logged. Stroke classification needs a trained ball detector "
            "and a trained stroke classifier — see the README."
        )
    else:
        render_shot_distribution(shots)
        st.divider()

    render_speed_over_time(shots, tracks)
    st.divider()

    if not shots.empty:
        render_ball_speeds(shots)
        st.divider()

    render_distance(summary)

    if not shots.empty:
        with st.expander("Shot log"):
            st.dataframe(shots, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
