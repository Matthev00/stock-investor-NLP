from src.utils import dumps

from crewai.tools import tool

from src.services.reddit.reddit_sentiment import RedditSentimentAnalyser
from src.experiments import tool_capture

analyser = RedditSentimentAnalyser()


@tool
def analyse_reddit(subreddits: list, stock: str, post_limit=50, days=30) -> str:
    """
    Analyses Reddit sentiment for a given stock across multiple subreddits.

    Args:
        subreddits (list): A list of subreddit names (e.g., ['stocks', 'investing']).
        stock (str): The stock ticker or keyword to search for (e.g., 'AAPL').
        post_limit (int): Maximum number of posts to analyse per subreddit.
        days (int): Time window (in days) to look back for Reddit posts.

    Returns:
        str: A JSON string containing the count of 'positive', 'neutral', and 'negative' sentiment results.
    """
    sentiments = analyser.analyse(subreddits, stock, post_limit, days)
    json_str = dumps(sentiments, indent=2)
    tool_capture.record("reddit_sentiment", json_str)
    return json_str
