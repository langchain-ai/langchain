from typing import List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


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
        access_token: str,
        access_token_secret: str,
        consumer_key: str,
        consumer_secret: str,
        twitter_users: List[str],
        number_tweets: Optional[int] = 100,
    ):
        self.access_token = access_token
        self.access_token_secret = access_token_secret
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret

        self.twitter_users = twitter_users
        self.number_tweets = number_tweets

    def load(self) -> List[Document]:
        try:
            import tweepy
            import tweepy.parsers
        except ImportError:
            raise ValueError(
                "requests package not found, please install it with "
                "`pip install tweepy`"
            )

        auth = tweepy.OAuthHandler(
            access_token=self.access_token,
            access_token_secret=self.access_token_secret,
            consumer_key=self.consumer_key,
            consumer_secret=self.consumer_secret,
        )
        api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())

        results = []
        for username in self.twitter_users:
            tweets = api.user_timeline(screen_name=username, count=self.number_tweets)

            response = ""
            for tweet in tweets:
                response += (
                    f"Created At:{tweet['created_at']}," f"Content:{tweet['text']}\n"
                )

            user = api.get_user(screen_name=username)
            results.append(Document(page_content=response, metadata=user))
        return results
