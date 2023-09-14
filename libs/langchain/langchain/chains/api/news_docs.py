# flake8: noqa
NEWS_DOCS = """API documentation:
Endpoint: https://newsapi.org
Top headlines /v2/top-headlines

This endpoint provides live top and breaking headlines for a country, specific category in a country, single source, or multiple sources. You can also search with keywords. Articles are sorted by the earliest date published first.

This endpoint is great for retrieving headlines for use with news tickers or similar.
Request parameters

    country | The 2-letter ISO 3166-1 code of the country you want to get headlines for. Possible options: ae ar at au be bg br ca ch cn co cu cz de eg fr gb gr hk hu id ie il in it jp kr lt lv ma mx my ng nl no nz ph pl pt ro rs ru sa se sg si sk th tr tw ua us ve za. Note: you can't mix this param with the sources param.
    category | The category you want to get headlines for. Possible options: business entertainment general health science sports technology. Note: you can't mix this param with the sources param.
    sources | A comma-seperated string of identifiers for the news sources or blogs you want headlines from. Use the /top-headlines/sources endpoint to locate these programmatically or look at the sources index. Note: you can't mix this param with the country or category params.
    q | Keywords or a phrase to search for.
    pageSize | int | The number of results to return per page (request). 20 is the default, 100 is the maximum.
    page | int | Use this to page through the results if the total results found is greater than the page size.

Response object
    status | string | If the request was successful or not. Options: ok, error. In the case of error a code and message property will be populated.
    totalResults | int | The total number of results available for your request.
    articles | array[article] | The results of the request.
    source | object | The identifier id and a display name name for the source this article came from.
    author | string | The author of the article
    title | string | The headline or title of the article.
    description | string | A description or snippet from the article.
    url | string | The direct URL to the article.
    urlToImage | string | The URL to a relevant image for the article.
    publishedAt | string | The date and time that the article was published, in UTC (+000)
    content | string | The unformatted content of the article, where available. This is truncated to 200 chars.

Use page size: 2
"""
