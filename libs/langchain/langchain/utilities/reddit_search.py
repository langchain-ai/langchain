from typing import Any, Dict, List, Optional
import requests

from langchain.pydantic_v1 import BaseModel, Extra, root_validator
from langchain.utils import get_from_dict_or_env

class RedditSearchAPIWrapper(BaseModel):

    client_id: str
    client_secret: str
    user_agent: str

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        reddit_client_id = get_from_dict_or_env(
            values, "reddit_client_id", "REDDIT_CLIENT_ID"
        )
        values["client_id"] = reddit_client_id

        reddit_client_secret = get_from_dict_or_env(
            values, "reddit_client_secret", "REDDIT_CLIENT_SECRET"
        )
        values["client_secret"] = reddit_client_secret

        reddit_user_agent = get_from_dict_or_env(
            values, "reddit_user_agent", "REDDIT_USER_AGENT"
        )
        values["user_agent"] = reddit_user_agent

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
        results: List[Dict] = self.results(query=query, sort=sort, time_filter=time_filter, subreddit=subreddit)
        output: List[str] = [f"{r['author']} posted: {r['text']}" for r in results]
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
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent,
        )
        subredditObject = reddit.subreddit(subreddit) 
        results = subredditObject.search(query=query, sort=sort, time_filter=time_filter, limit=limit) 
        results = []
        attributes = ['title', 'score', 'id', 'url', 'author']
        for submission in subredditObject:
            results.append({
                'title': submission.title, 
                'author': submission.author, 
                'text': submission.selftext,
                'url': submission.url,
            })
            # results.append({att: submission[att] for att in attributes})
        return results

        # # TODO: clean up parameters.
        # url: str = f'https://api.twitter.com/1.1/search/tweets.json?q={query}&count=1'
        # # url: str = 'https://api.twitter.com/2/tweets?ids=1261326399320715264,1278347468690915330'
        # print(self.twitter_access_token)
        # response = requests.get(url, 
        #     headers={
        #         'Authorization': f'Bearer {self.twitter_access_token}'
        #     })
        # results = response.json()
        # print('status code', response.status_code)
        # print('status text', response.text)
        # statuses = results['data']
        # search_results: List[str] = []
        # print()
        # for status in statuses:
        #     search_results.append({
        #         'username': status['author_id'],
        #         'created_at': status['created_at'],
        #         'text': status['text']
        #         # 'username': status['user']['name'],
        #     })
        # return '\n'.join(search_results)