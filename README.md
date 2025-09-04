# YouTube Comment Analyzer

A comprehensive Streamlit web application for analyzing YouTube comment data with advanced search specifications, sentiment analysis, and topic extraction.

## Features

### 🎯 Complete Video Analysis
- **Automatic Video Discovery**: Identifies videos from uploaded datasets
- **Sentiment Analysis**: 0-1 scale sentiment scoring with visual gauge
- **Topic Extraction**: Top 5 most discussed topics with frequency analysis
- **Question Mining**: Extracts top 5 questions from comment sections
- **Static Search Specs**: Pre-built search patterns for feedback and suggestions
- **Interactive Visualizations**: Charts and graphs using Plotly

### 🔍 Custom Keyword Search
- **Flexible Keyword Input**: Support for comma-separated or line-separated keywords
- **Real-time Search**: Filter comments by custom keywords
- **Export Results**: Download search results as CSV
- **Advanced Filtering**: Case-insensitive pattern matching

### 📊 Advanced Analytics
- **Comment Quality Metrics**: Average length, validity checks
- **Search Specification Engine**: Execute predefined and custom search patterns
- **Visual Sentiment Dashboard**: Interactive gauge charts
- **Topic Frequency Analysis**: Bar charts with most discussed topics

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
```bash
git clone <repository-url>
cd "Streamlit App"
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure API Keys**
   - Copy `.env.example` to `.env`
   - Add your API keys to the `.env` file:
```bash
cp .env.example .env
```

5. **Edit the .env file**
```
OPENAI_API_KEY=your_openai_api_key_here
YOUTUBE_API_KEY=your_youtube_api_key_here
```

## Getting API Key

### YouTube Data API Key
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable YouTube Data API v3
4. Create credentials (API Key)
5. Copy the key to your `.env` file

## Usage

### Running the Application

```bash
streamlit run main.py
```

### Using the Application

1. **Upload Data**: Use the sidebar to upload your CSV file
2. **Choose Analysis Type**:
   - **Video Analysis**: Complete analysis with all metrics
   - **Keyword Search**: Custom keyword-based comment filtering

#### Video Analysis
1. Select a video ID from the dropdown
2. Click "Run Analysis"
3. View comprehensive results including:
   - Basic metrics and sentiment score
   - Static search specification results
   - Top topics and questions
   - Comments

#### Keyword Search  
1. Select a video ID
2. Enter keywords (comma-separated or line-separated)
3. Click "Search Comments"
4. View and download filtered results

## Project Structure

```
youtube-comment-analyzer/
├── main.py                    # Main Streamlit application
├── requirements.txt           # Python dependencies
├── .env                       # Your API keys (create this)
├── .gitignore
├── utils.py                                  
├── README.md                  
└── cache/                     # Cached data directory (auto-created)
```

## Static Search Specifications

The application includes two pre-built search specifications:

1. **Feedback Analysis**
   - **Question**: "What are users saying about the overall quality of the video?"
   - **Keywords**: amazing, terrible, great video, bad audio, well made, poor quality, etc.
   - **Type**: feedback

2. **Suggestion Analysis**
   - **Question**: "Are users asking for more videos or follow-ups?"
   - **Keywords**: make a video on, do one about, please cover, next video, part 2, etc.
   - **Type**: suggestion

## Technical Details

### Core Components

- **YouTubeCommentAnalyzer**: Main analysis engine
- **Static Search Specs**: Pre-defined search patterns
- **Sentiment Analysis**: Keyword-based sentiment scoring
- **Topic Extraction**: Word frequency analysis with stop-word filtering
- **Question Mining**: Pattern matching for interrogative sentences

### Dependencies

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualizations
- **OpenAI**: AI-powered analysis (optional)
- **Google API Client**: YouTube API integration (optional)

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure `.env` file exists and contains valid API keys
   - Check API key permissions and quotas

2. **File Upload Issues**
   - Verify CSV format matches expected structure
   - Check for encoding issues (use UTF-8)

3. **No Videos Found**
   - Ensure dataset contains rows with null `author_id` (video descriptions)
   - Verify `video_id` column exists and contains valid IDs

4. **Performance Issues**
   - Large datasets may take time to process
   - Consider filtering data before upload for better performance
