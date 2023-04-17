"""Twitter document loader."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

if TYPE_CHECKING:
    import tweepy
    from tweepy import OAuthHandler


def _dependable_tweepy_import() -> tweepy:
    try:
        import tweepy
    except ImportError:
        raise ValueError(
            "requests package not found, please install it with " "`pip install tweepy`"
        )
    return tweepy


class TwitterTweetLoader(BaseLoader):
    """Twitter tweets loader.
    Read tweets of user twitter handle.

    First you need to go to
    ` https://developer.twitter.com/en/docs/twitter-api
    /getting-started/getting-access-to-the-twitter-api `
    to get your token. And create a v2 version of the app.
    """

    def __init__(
        self,
        auth_handler: OAuthHandler,
        twitter_users: Sequence[str],
        number_tweets: Optional[int] = 100,
    ):
        self.auth = auth_handler

        self.twitter_users = twitter_users
        self.number_tweets = number_tweets

    def load(self) -> List[Document]:
        """Load tweets."""
        tweepy = _dependable_tweepy_import()
        api = tweepy.API(self.auth, parser=tweepy.parsers.JSONParser())

        results = []
        for username in self.twitter_users:
            tweets = api.user_timeline(screen_name=username, count=self.number_tweets)
            response = self._format_tweets(tweets)
            user = api.get_user(screen_name=username)
            results.append(Document(page_content=response, metadata=user))
        return results

    def _format_tweets(self, tweets: List[Dict[str, Any]]) -> str:
        """Format tweets into a string."""
        response = ""
        for tweet in tweets:
            response += f"Created At:{tweet['created_at']},Content:{tweet['text']}\n"
        return response

    @classmethod
    def from_secrets(
        cls,
        access_token: str,
        access_token_secret: str,
        consumer_key: str,
        consumer_secret: str,
        twitter_users: Sequence[str],
        number_tweets: Optional[int] = 100,
    ) -> "TwitterTweetLoader":
        tweepy = _dependable_tweepy_import()
        auth = tweepy.OAuthHandler(
            access_token=access_token,
            access_token_secret=access_token_secret,
            consumer_key=consumer_key,
            consumer_secret=consumer_secret,
        )
        return cls(
            auth_handler=auth,
            twitter_users=twitter_users,
            number_tweets=number_tweets,
        )
