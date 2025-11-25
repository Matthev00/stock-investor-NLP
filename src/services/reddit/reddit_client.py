import os
from datetime import datetime, timedelta, timezone

import praw
from dotenv import load_dotenv

load_dotenv()


class MockPost:
    def __init__(self, title, selftext):
        self.title = title
        self.selftext = selftext


class RedditClient:
    """
    A class to interact with Reddit API using PRAW (Python Reddit API Wrapper).

    Attributes:
        reddit (praw.Reddit): An instance of the Reddit API client.
    """

    def __init__(self):
        # self.reddit = praw.Reddit(
        #     client_id=os.getenv("REDDIT_CLIENT_ID"),
        #     client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        #     user_agent=os.getenv("REDDIT_USER_AGENT"),
        # )
        pass

    def get_posts(self, subreddit: str, query: str, post_limit: int = 50, days: int = 30) -> list:
        """
        Fetches posts from a specific subreddit based on a query.
        Args:
            subreddit (str): The name of the subreddit to search in.
            query (str): The search query (e.g., stock symbol).
            post_limit (int): Maximum number of posts to fetch.
            days (int): Time window (in days) to look back for posts.
        Returns:
            list: A list of Reddit posts matching the query.
        """
        # sub = self.reddit.subreddit(subreddit)
        # start_date = datetime.now(timezone.utc) - timedelta(days=days)
        post = MockPost("Będzie dobrze", "Będzie dobrze")
        posts = [post]

        # for post in sub.search(query, sort="new", time_filter="all", limit=post_limit):
        #     post_date = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
        #     if post_date >= start_date:
        #         posts.append(post)
        return posts
