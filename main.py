import streamlit as st
import pandas as pd
import plotly.express as px
import time
import random
import json
from typing import List
from utils import (
    Config,
    YouTubeCommentAnalyzer,
    create_sentiment_chart,
    create_sentiment_distribution_chart,
    get_comment_stats,
    format_number,
    clean_text_for_display,
    export_analysis_results,
    validate_dataframe
)

st.set_page_config(
    page_title="YouTube Comment Analyzer",
    page_icon="📺",
    layout="wide"
)

# CSS Styling
st.markdown("""
<style>
.comment-box { background-color: #fff; color: black; padding: 0.8rem; border-left: 4px solid #1f77b4; margin-bottom: 0.3rem; border-radius: 5px; }
.warning-box { background-color: #f8d7da; padding: 1rem; border-radius: 5px; border-left: 4px solid #dc3545; color: #721c24; }
.success-box { background-color: #d4edda; padding: 1rem; border-radius: 5px; border-left: 4px solid #28a745; color: #155724; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_analyzer():
    return YouTubeCommentAnalyzer()

def check_api_keys():
    missing_keys = Config.validate_keys()
    if missing_keys:
        st.markdown(
            f'<div class="warning-box">⚠️ <strong>Missing API Keys</strong><br>Please set in <code>.env</code>:<br>• {", ".join(missing_keys)}</div>',
            unsafe_allow_html=True
        )
        st.toast("Basic features enabled, API keys missing", icon="⚠️")
        return False
    return True

def extract_video_id_from_url(df):
    df = df.copy()
    if 'video_id' in df and df['video_id'].notna().any():
        st.toast("✅ video_id column detected")
        return df
    if 'url' not in df:
        st.error("CSV must include 'url' column if 'video_id' not present")
        return None
    with st.spinner("🔍 Extracting video_id from URLs..."):
        df['video_id'] = df['url'].str.extract(r'(?:watch\?v=|youtu\.be/|embed/|v/|.+\?v=)([^&\n?#]+)')
        time.sleep(1)
    if df['video_id'].notna().sum() == 0:
        st.error("❌ Could not extract video IDs")
        return None
    st.toast(f"✅ Extracted {df['video_id'].notna().sum()} video IDs")
    return df

def main():
    st.title("📺 YouTube Comment Analyzer")
    api_keys_configured = check_api_keys()
    analyzer = get_analyzer()

    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox("Choose Page", ["Video Analysis", "Keyword Search", "Data Info"])
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])

    if not uploaded_file:
        st.warning("Upload a CSV to continue.")
        return

    try:
        df = pd.read_csv(uploaded_file)
        df = extract_video_id_from_url(df)
        if df is None: return
        is_valid, errors = validate_dataframe(df)
        if not is_valid:
            st.error("Data validation failed:")
            for err in errors:
                st.error(f"• {err}")
            return
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return

    video_ids = analyzer.discover_videos(df)
    if page == "Video Analysis":
        video_analysis_page(df, analyzer, video_ids, api_keys_configured)
    elif page == "Keyword Search":
        keyword_search_page(df, analyzer, video_ids)
    else:
        data_info_page(df, analyzer, video_ids)

def video_analysis_page(df, analyzer, video_ids, api_keys_configured):
    st.header("🎯 Video Analysis")
    if not video_ids:
        st.warning("No videos found.")
        return

    selected_video = st.selectbox("Select Video", video_ids)
    if st.button("Run Analysis"):
        run_complete_analysis(df, analyzer, selected_video, api_keys_configured)

def run_complete_analysis(df, analyzer, video_id, api_keys_configured):
    video_info = analyzer.get_video_info(df, video_id)
    comments = analyzer.get_valid_comments(df, video_id)
    if not comments:
        st.error("No valid comments found.")
        return

    st.markdown(f"<div class='success-box'><h4>📺 {video_info['title']}</h4></div>", unsafe_allow_html=True)
    st.subheader("🔍 Search Spec Analysis")

    for spec in analyzer.static_search_specs:
        matching = analyzer.execute_search_spec(comments, spec)
        st.markdown(f"### {spec['search_type'].title()} ({len(matching)})")
        for i, c in enumerate(matching[:3]):
            st.markdown(f"<div class='comment-box'><strong>#{i+1}:</strong> {clean_text_for_display(c, 200)}</div>", unsafe_allow_html=True)

    st.subheader("💬 Sample Comments")
    for i, c in enumerate(random.sample(comments, min(10, len(comments))), 1):
        st.markdown(f"<div class='comment-box'><strong>#{i}:</strong> {clean_text_for_display(c, 300)}</div>", unsafe_allow_html=True)

    st.subheader("😊 Sentiment")
    score = analyzer.calculate_sentiment_score(comments)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_sentiment_chart(score), use_container_width=True)
    with col2:
        st.plotly_chart(create_sentiment_distribution_chart(score), use_container_width=True)

def keyword_search_page(df, analyzer, video_ids):
    st.header("🔎 Keyword Search")
    selected_video = st.selectbox("Select Video", video_ids)
    keyword_input = st.text_area("Keywords (comma or newline separated)")
    if st.button("Search"):
        keywords = [k.strip() for k in keyword_input.replace("\n", ",").split(",") if k.strip()]
        if not keywords:
            st.warning("Please provide keywords.")
            return
        comments = analyzer.get_valid_comments(df, selected_video)
        results_df = analyzer.search_comments_by_keywords(df, selected_video, keywords)
        st.metric("Matches", len(results_df))
        st.subheader("📋 Matching Comments")
        for i, (_, row) in enumerate(results_df.iterrows()):
            st.markdown(f"<div class='comment-box'><strong>#{i+1}:</strong> {clean_text_for_display(row['content'], 300)}</div>", unsafe_allow_html=True)
            if i >= 20:
                break

def data_info_page(df, analyzer, video_ids):
    st.header("📊 Dataset Info")
    st.dataframe(df.describe(include='all'))

if __name__ == "__main__":
    main()
