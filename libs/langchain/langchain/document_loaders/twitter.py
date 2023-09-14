from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Union

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

if TYPE_CHECKING:
    import tweepy
    from tweepy import OAuth2BearerHandler, OAuthHandler


def _dependable_tweepy_import() -> tweepy:
    try:
        import tweepy
    except ImportError:
        raise ImportError(
            "tweepy package not found, please install it with `pip install tweepy`"
        )
    return tweepy


class TwitterTweetLoader(BaseLoader):
    """Load `Twitter` tweets.

    Read tweets of the user's Twitter handle.

    First you need to go to
    `https://developer.twitter.com/en/docs/twitter-api
    /getting-started/getting-access-to-the-twitter-api`
    to get your token. And create a v2 version of the app.
    """

    def __init__(
        self,
        auth_handler: Union[OAuthHandler, OAuth2BearerHandler],
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

        results: List[Document] = []
        for username in self.twitter_users:
            tweets = api.user_timeline(screen_name=username, count=self.number_tweets)
            user = api.get_user(screen_name=username)
            docs = self._format_tweets(tweets, user)
            results.extend(docs)
        return results

    def _format_tweets(
        self, tweets: List[Dict[str, Any]], user_info: dict
    ) -> Iterable[Document]:
        """Format tweets into a string."""
        for tweet in tweets:
            metadata = {
                "created_at": tweet["created_at"],
                "user_info": user_info,
            }
            yield Document(
                page_content=tweet["text"],
                metadata=metadata,
            )

    @classmethod
    def from_bearer_token(
        cls,
        oauth2_bearer_token: str,
        twitter_users: Sequence[str],
        number_tweets: Optional[int] = 100,
    ) -> TwitterTweetLoader:
        """Create a TwitterTweetLoader from OAuth2 bearer token."""
        tweepy = _dependable_tweepy_import()
        auth = tweepy.OAuth2BearerHandler(oauth2_bearer_token)
        return cls(
            auth_handler=auth,
            twitter_users=twitter_users,
            number_tweets=number_tweets,
        )

    @classmethod
    def from_secrets(
        cls,
        access_token: str,
        access_token_secret: str,
        consumer_key: str,
        consumer_secret: str,
        twitter_users: Sequence[str],
        number_tweets: Optional[int] = 100,
    ) -> TwitterTweetLoader:
        """Create a TwitterTweetLoader from access tokens and secrets."""
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
