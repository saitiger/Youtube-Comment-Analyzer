import os
import json
import re
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv
import requests

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)

print("DEBUG YOUTUBE_API_KEY =", os.getenv("YOUTUBE_API_KEY"))


class Config:
    """Configuration class with environment variables"""
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY") 
    CACHE_DIR = os.getenv("CACHE_DIR", "cache")
    DEFAULT_TOP_N = int(os.getenv("DEFAULT_TOP_N", "5"))
    MAX_COMMENTS_BATCH = int(os.getenv("MAX_COMMENTS_BATCH", "50"))

    @classmethod
    def validate_keys(cls):
        """Validate that required API keys are present and not placeholders"""
        missing_keys = []

        # OpenAI Key
        if not cls.OPENAI_API_KEY or cls.OPENAI_API_KEY.strip() == "":
            missing_keys.append("OPENAI_API_KEY")
        elif cls.OPENAI_API_KEY in ["your_openai_api_key_here", "sk-placeholder", "placeholder"]:
            missing_keys.append("OPENAI_API_KEY (placeholder detected)")
        elif not cls.OPENAI_API_KEY.startswith("sk-"):
            missing_keys.append("OPENAI_API_KEY (invalid format)")

        # YouTube Key
        if not cls.YOUTUBE_API_KEY or cls.YOUTUBE_API_KEY.strip() == "":
            missing_keys.append("YOUTUBE_API_KEY")
        elif cls.YOUTUBE_API_KEY in ["your_youtube_api_key_here", "placeholder", "AIza-placeholder"]:
            missing_keys.append("YOUTUBE_API_KEY (placeholder detected)")
        elif not cls.YOUTUBE_API_KEY.startswith("AIza"):
            missing_keys.append("YOUTUBE_API_KEY (invalid format)")

        return missing_keys

    @classmethod
    def get_status_info(cls):
        """Get detailed status information about API keys"""
        info = {}
        
        # OpenAI API Key status
        if not cls.OPENAI_API_KEY:
            info['openai'] = {'status': 'missing', 'message': 'Not configured'}
        elif cls.OPENAI_API_KEY in ["your_openai_api_key_here", "sk-placeholder", "placeholder"]:
            info['openai'] = {'status': 'placeholder', 'message': 'Placeholder value detected'}
        elif not cls.OPENAI_API_KEY.startswith("sk-"):
            info['openai'] = {'status': 'invalid', 'message': 'Invalid format (should start with sk-)'}
        else:
            info['openai'] = {'status': 'configured', 'message': f'Configured (ends with: ...{cls.OPENAI_API_KEY[-4:]})'}
        
        # YouTube API Key status
        if not cls.YOUTUBE_API_KEY:
            info['youtube'] = {'status': 'missing', 'message': 'Not configured'}
        elif cls.YOUTUBE_API_KEY in ["your_youtube_api_key_here", "placeholder", "AIza-placeholder"]:
            info['youtube'] = {'status': 'placeholder', 'message': 'Placeholder value detected'}
        elif not cls.YOUTUBE_API_KEY.startswith("AIza"):
            info['youtube'] = {'status': 'invalid', 'message': 'Invalid format (should start with AIza)'}
        else:
            info['youtube'] = {'status': 'configured', 'message': f'Configured (ends with: ...{cls.YOUTUBE_API_KEY[-4:]})'}
        
        return info


class CommentSearchSpec:
    """Data class for comment search specifications"""
    def __init__(self, question: str, keywords: List[str], search_type: str):
        self.question = question
        self.keywords = keywords
        self.search_type = search_type
    
    def to_dict(self):
        return {
            "question": self.question,
            "keywords": self.keywords,
            "search_type": self.search_type
        }


class YouTubeCommentAnalyzer:
    """Main analyzer class for YouTube comment analysis"""
    
    def __init__(self):
        # Static search specs that apply to all videos
        self.static_search_specs = [
            {
                "question": "What are users saying about the overall quality of the video?",
                "keywords": ["amazing", "terrible", "great video", "bad audio", "well made", "poor quality", 
                           "love this", "hate this", "awesome", "sucks", "excellent", "worst", "perfect", "awful"],
                "search_type": "feedback"
            },
            {
                "question": "Are users asking for more videos or follow-ups?",
                "keywords": ["make a video on", "do one about", "please cover", "can you also", "next video", 
                           "part 2", "follow up", "sequel", "more content", "continue this", "more like this"],
                "search_type": "suggestion"
            }
        ]
    
    def discover_videos(self, df: pd.DataFrame) -> List[str]:
        """Discover video IDs from the dataset"""
        # Videos are rows without author_id (video descriptions)
        video_ids = df[df['author_id'].isna()]['video_id'].dropna().unique().tolist()
        return video_ids
    
    def get_video_info(self, df: pd.DataFrame, video_id: str, openai_client=None) -> Dict:
        """
        Get video information: title from YouTube API, description from dataset,
        and (optionally) a summary/context using OpenAI if openai_client is provided.
        """
        # Get description from dataset
        video_row = df[(df['video_id'] == video_id) & (df['author_id'].isna())]
        description = ""
        if not video_row.empty and 'content' in video_row.columns:
            description = video_row['content'].iloc[0]

        # Get title from YouTube API
        api_key = Config.YOUTUBE_API_KEY
        title = f"Video {video_id}"
        if api_key:
            url = (
                f"https://www.googleapis.com/youtube/v3/videos"
                f"?part=snippet&id={video_id}&key={api_key}"
            )
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    items = data.get("items", [])
                    if items and "snippet" in items[0]:
                        title = items[0]["snippet"].get("title", title)
            except Exception:
                pass

        # Optionally summarize description using OpenAI
        summary = ""
        if openai_client and description:
            prompt = (
                "You are an assistant that summarizes YouTube video descriptions to help understand the context of the video.\n\n"
                f"Here's the video description:\n{description}\n\n"
                "What is this video about? Give a 2-3 sentence summary of the key topic and tone."
            )
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an assistant that helps understand YouTube video context."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                )
                summary = response.choices[0].message.content.strip()
            except Exception as e:
                summary = f"[Error summarizing video description: {e}]"

        return {"title": title, "description": description, "summary": summary}
    
    def is_valid_comment(self, comment: str) -> bool:
        """Check if comment is valid for analysis"""
        if not isinstance(comment, str) or len(comment.strip()) <= 3:
            return False
        return bool(re.search(r'\w', comment))
    
    def get_valid_comments(self, df: pd.DataFrame, video_id: str) -> List[str]:
        """Get valid comments for a video"""
        # Get comments (exclude video descriptions)
        comments = df[(df['video_id'] == video_id) & (df['author_id'].notna())]['content'].dropna().tolist()
        return [c for c in comments if self.is_valid_comment(c)]
    
    def execute_search_spec(self, comments: List[str], search_spec: Dict) -> List[str]:
        """Execute search specification on comments"""
        matching_comments = []
        keyword_pattern = "|".join([re.escape(k) for k in search_spec['keywords']])
        
        for comment in comments:
            if re.search(keyword_pattern, comment, re.IGNORECASE):
                matching_comments.append(comment)
        
        return matching_comments
    
    def calculate_sentiment_score(self, comments: List[str]) -> float:
        """Simple sentiment calculation based on positive/negative keywords"""
        if not comments:
            return 0.5
        
        positive_words = ['good', 'great', 'amazing', 'awesome', 'love', 'excellent', 'perfect', 
                         'best', 'wonderful', 'fantastic', 'brilliant', 'outstanding', 'superb']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'sucks', 
                         'boring', 'annoying', 'stupid', 'useless', 'disappointing', 'waste']
        
        positive_count = 0
        negative_count = 0
        
        for comment in comments:
            comment_lower = comment.lower()
            for word in positive_words:
                positive_count += comment_lower.count(word)
            for word in negative_words:
                negative_count += comment_lower.count(word)
        
        total_sentiment = positive_count + negative_count
        if total_sentiment == 0:
            return 0.5
        
        return positive_count / total_sentiment
    
    def extract_top_topics(self, comments: List[str], n: int = 5) -> List[Tuple[str, int]]:
        """Extract top topics using simple word frequency"""
        if not comments:
            return []
        
        # Combine all comments
        text = " ".join(comments).lower()
        
        # Remove common words and extract meaningful words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 
            'did', 'will', 'would', 'could', 'should', 'this', 'that', 'these', 'those', 
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 
            'my', 'your', 'his', 'her', 'its', 'our', 'their', 'a', 'an', 'can', 'cant', 
            'dont', 'isnt', 'wasnt', 'arent', 'werent', 'like', 'just', 'really', 'very',
            'also', 'get', 'got', 'going', 'go', 'see', 'know', 'think', 'want', 'need'
        }
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
        filtered_words = [word for word in words if word not in stop_words]
        
        word_counts = Counter(filtered_words)
        return word_counts.most_common(n)
    
    def extract_top_questions(self, comments: List[str], n: int = 5) -> List[str]:
        """Extract top questions from comments"""
        questions = []
        
        for comment in comments:
            # Find sentences with question marks
            potential_questions = re.findall(r'[^.!?]*\?', comment)
            for q in potential_questions:
                q = q.strip()
                if len(q) > 10 and len(q.split()) > 2:  # Filter short questions
                    # Clean up the question
                    q = re.sub(r'^[^a-zA-Z]*', '', q)  # Remove leading non-letters
                    if q:
                        questions.append(q)
        
        # Count and return most common questions
        question_counts = Counter(questions)
        return [q for q, count in question_counts.most_common(n)]
    
    def search_comments_by_keywords(self, df: pd.DataFrame, video_id: str, keywords: List[str]) -> pd.DataFrame:
        """Search comments by custom keywords"""
        video_comments = df[(df['video_id'] == video_id) & (df['author_id'].notna())].copy()
        
        if not keywords:
            return video_comments
        
        keyword_pattern = "|".join([re.escape(k.strip()) for k in keywords if k.strip()])
        if not keyword_pattern:
            return video_comments
        
        mask = video_comments['content'].str.contains(keyword_pattern, case=False, na=False)
        return video_comments[mask]


def create_sentiment_chart(sentiment_score: float) -> go.Figure:
    """Create a sentiment gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = sentiment_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Overall Sentiment Score"},
        delta = {'reference': 0.5, 'position': "top"},
        gauge = {
            'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 0.3], 'color': "lightcoral"},
                {'range': [0.3, 0.7], 'color': "lightyellow"},
                {'range': [0.7, 1], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.9
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        font={'color': "darkblue", 'family': "Arial"}
    )
    return fig


def create_sentiment_distribution_chart(sentiment_score: float) -> go.Figure:
    """Create a pie chart showing sentiment distribution"""
    # Estimate distribution based on overall score
    if sentiment_score > 0.7:
        positive, neutral, negative = 70, 25, 5
    elif sentiment_score > 0.5:
        positive = int(sentiment_score * 100)
        negative = int((1 - sentiment_score) * 50)
        neutral = 100 - positive - negative
    elif sentiment_score > 0.3:
        negative = int((1 - sentiment_score) * 100)
        positive = int(sentiment_score * 50)
        neutral = 100 - positive - negative
    else:
        negative, neutral, positive = 70, 25, 5
    
    fig = go.Figure(data=[go.Pie(
        labels=['Positive', 'Neutral', 'Negative'],
        values=[positive, neutral, negative],
        hole=.3,
        marker_colors=['lightgreen', 'lightyellow', 'lightcoral']
    )])
    
    fig.update_layout(
        title="Sentiment Distribution",
        height=300,
        showlegend=True
    )
    return fig


def get_comment_stats(comments: List[str]) -> Dict:
    """Get comprehensive statistics about comments"""
    if not comments:
        return {}
    
    lengths = [len(c) for c in comments]
    word_counts = [len(c.split()) for c in comments]
    
    return {
        'total_comments': len(comments),
        'avg_length': np.mean(lengths),
        'median_length': np.median(lengths),
        'avg_words': np.mean(word_counts),
        'median_words': np.median(word_counts),
        'shortest_comment': min(lengths),
        'longest_comment': max(lengths),
        'questions_count': sum(1 for c in comments if '?' in c),
        'exclamations_count': sum(1 for c in comments if '!' in c)
    }


def format_number(num: float, precision: int = 1) -> str:
    """Format numbers for display"""
    if num >= 1000000:
        return f"{num/1000000:.{precision}f}M"
    elif num >= 1000:
        return f"{num/1000:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"


def clean_text_for_display(text: str, max_length: int = 200) -> str:
    """Clean and truncate text for display"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Truncate if necessary
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    return text


def export_analysis_results(video_id: str, analysis_data: Dict) -> str:
    """Export analysis results to JSON string"""
    export_data = {
        'video_id': video_id,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'analysis': analysis_data
    }
    return json.dumps(export_data, indent=2, ensure_ascii=False)


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate uploaded dataframe structure"""
    errors = []
    
    # Check required columns (either video_id OR url must be present)
    has_video_id = 'video_id' in df.columns and not df['video_id'].isna().all()
    has_url = 'url' in df.columns and not df['url'].isna().all()
    
    if not has_video_id and not has_url:
        errors.append("Missing required column: either 'video_id' or 'url' must be present")
    
    # Check for content column
    if 'content' not in df.columns:
        errors.append("Missing required column: 'content'")
    
    # Check data types
    if 'video_id' in df.columns and df['video_id'].dtype not in ['object', 'string']:
        errors.append("video_id column should contain text/string values")
    
    if 'url' in df.columns and df['url'].dtype not in ['object', 'string']:
        errors.append("url column should contain text/string values")
    
    if 'content' in df.columns and df['content'].dtype not in ['object', 'string']:
        errors.append("content column should contain text/string values")
    
    # Check for empty dataframe
    if len(df) == 0:
        errors.append("Dataframe is empty")
    
    # If has_url, check if URLs look like YouTube URLs
    if has_url and 'url' in df.columns:
        youtube_urls = df['url'].dropna().str.contains(r'(youtube\.com|youtu\.be)', case=False, na=False)
        if not youtube_urls.any():
            errors.append("No YouTube URLs found in 'url' column")
    
    return len(errors) == 0, errors