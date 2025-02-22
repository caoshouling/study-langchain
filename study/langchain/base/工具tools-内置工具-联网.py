# pip install google-search-results
from langchain_community.utilities import SerpAPIWrapper
import os
# os.environ["SERPAPI_API_KEY"] =

search = SerpAPIWrapper()
print(search.run("Obama's first name?"))
