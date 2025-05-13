from .tools import (
    DesearchTool,
    BasicWebSearchTool,
    BasicTwitterSearchTool,
    FetchTweetsByUrlsTool,
    FetchTweetsByIdTool,
    FetchLatestTweetsTool,
    FetchTweetsAndRepliesByUserTool,
    FetchRepliesByPostTool,
    FetchRetweetsByPostTool,
    FetchTwitterUserTool,
)

# Group only 3 search tools into a list
search_tools = [
    DesearchTool(),
    BasicWebSearchTool(),
    BasicTwitterSearchTool(),
]

# Group Twitter-related tools into a separate list
twitter_tools = [
    BasicTwitterSearchTool(),
    FetchTweetsByUrlsTool(),
    FetchTweetsByIdTool(),
    FetchLatestTweetsTool(),
    FetchTweetsAndRepliesByUserTool(),
    FetchRepliesByPostTool(),
    FetchRetweetsByPostTool(),
    FetchTwitterUserTool(),
]
