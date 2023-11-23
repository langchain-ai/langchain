from typing import Any, Dict, List, Optional
import requests
import praw
from langchain.pydantic_v1 import BaseModel, Extra, root_validator
from langchain.utils import get_from_dict_or_env

class RedditSearchAPIWrapper(BaseModel):

    reddit_client_id: str
    reddit_client_secret: str
    reddit_user_agent: str

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        reddit_client_id = get_from_dict_or_env(
            values, "reddit_client_id", "REDDIT_CLIENT_ID"
        )
        values["reddit_client_id"] = reddit_client_id

        reddit_client_secret = get_from_dict_or_env(
            values, "reddit_client_secret", "REDDIT_CLIENT_SECRET"
        )
        values["reddit_client_secret"] = reddit_client_secret

        reddit_user_agent = get_from_dict_or_env(
            values, "reddit_user_agent", "REDDIT_USER_AGENT"
        )
        values["reddit_user_agent"] = reddit_user_agent

        try:
            import praw
        except ImportError:
            raise ImportError(
                "praw package not found, please install it with pip install praw"
            )
        return values

    def run(self, 
        query: str, 
        sort: str, 
        time_filter: str,
        subreddit: str,
        limit: int) -> str:
        results: List[Dict] = self.results(query=query, sort=sort, time_filter=time_filter, subreddit=subreddit, limit=limit)
        output: List[str] = []
        for r in results:
            p = f"{r['post_author']} posted '{r['post_title']}' in {r['post_subreddit']}:\n" \
                f"{r['post_text']}\n" \
                f"({r['post_url']})\n" \
                f"Score: {r['post_score']}\n"
            output.append(p)
        return '\n'.join(output)

    def results(
        self,
        query: str,
        sort: str, 
        time_filter: str,
        subreddit: str,
        limit: int
    ) -> List[Dict]:
        reddit = praw.Reddit(
            client_id=self.reddit_client_id,
            client_secret=self.reddit_client_secret,
            user_agent=self.reddit_user_agent,
        )
        subredditObject = reddit.subreddit(subreddit) 
        search_results = subredditObject.search(query=query, sort=sort, time_filter=time_filter, limit=limit) 
        search_results = [r for r in search_results]
        results_object = []
        for submission in search_results:
            results_object.append({
                'post_subreddit': submission.subreddit_name_prefixed,
                'post_category': submission.category,
                'post_title': submission.title,
                'post_text': submission.selftext,
                'post_score': submission.score,
                'post_id': submission.id,
                'post_url': submission.url,
                'post_author': submission.author,
            })
        return results_object