from langchain_community.tools.naver_search.tool import NaverSearchResults

tool = NaverSearchResults()
result = tool.invoke({'query': '삼성전자 관련 뉴스'}) 
print(result)