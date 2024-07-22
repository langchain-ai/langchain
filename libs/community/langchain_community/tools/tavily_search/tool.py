"""Tool for the Tavily search API."""

from typing import Dict, List, Optional, Type, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool

from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper


class TavilyInput(BaseModel):
    """Input for the Tavily tool."""

    query: str = Field(description="search query to look up")


class TavilySearchResults(BaseTool):
    """Tool that queries the Tavily Search API and gets back json.

    Setup:
        Install ``langchain-openai`` and ``tavily-python``, and set environment variable ``TAVILY_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-openai
            export TAVILY_API_KEY="your-api-key"

    Instantiate:

        .. code-block:: python

            from langchain_community.tools.tavily_search import TavilySearchResults

            tool = TavilySearchResults(
                # max_results= 5
                # search_depth = "advanced"
                # include_domains = []
                # exclude_domains = []
                # include_answer = False
                # include_raw_content = False
                # include_images = False
            )

    Invoke:

        .. code-block:: python

            tool = TavilySearchResults(max_results=3)
            tool.invoke("What is the weather?")

        .. code-block:: python

            [{'url': 'https://www.weatherapi.com/',
            'content': "{'location': {'name': 'Current', 'region': 'Harbour Island', 'country': 'Bahamas', 'lat': 25.43, 'lon': -76.78, 'tz_id': 'America/Nassau', 'localtime_epoch': 1718077801, 'localtime': '2024-06-10 23:50'}, 'current': {'last_updated_epoch': 1718077500, 'last_updated': '2024-06-10 23:45', 'temp_c': 27.9, 'temp_f': 82.1, 'is_day': 0, 'condition': {'text': 'Patchy rain nearby', 'icon': '//cdn.weatherapi.com/weather/64x64/night/176.png', 'code': 1063}, 'wind_mph': 14.5, 'wind_kph': 23.4, 'wind_degree': 161, 'wind_dir': 'SSE', 'pressure_mb': 1014.0, 'pressure_in': 29.94, 'precip_mm': 0.01, 'precip_in': 0.0, 'humidity': 88, 'cloud': 74, 'feelslike_c': 33.1, 'feelslike_f': 91.5, 'windchill_c': 27.9, 'windchill_f': 82.1, 'heatindex_c': 33.1, 'heatindex_f': 91.5, 'dewpoint_c': 25.6, 'dewpoint_f': 78.1, 'vis_km': 10.0, 'vis_miles': 6.0, 'uv': 1.0, 'gust_mph': 20.4, 'gust_kph': 32.9}}"},
            {'url': 'https://www.localconditions.com/weather-ninnescah-kansas/67069/',
            'content': 'The following chart reports what the hourly Ninnescah, KS temperature has been today, from 12:56 AM to 3:56 AM Tue, May 21st 2024. The lowest temperature reading has been 73.04 degrees fahrenheit at 3:56 AM, while the highest temperature is 75.92 degrees fahrenheit at 12:56 AM. Ninnescah KS detailed current weather report for 67069 in Kansas.'},
            {'url': 'https://www.weather.gov/forecastmaps/',
            'content': 'Short Range Forecasts. Short range forecast products depicting pressure patterns, circulation centers and fronts, and types and extent of precipitation. 12 Hour | 24 Hour | 36 Hour | 48 Hour.'}]

        When converting ``TavilySearchResults`` to a tool, you may want to not return all of the content resulting from ``invoke``. You can select what parts of the response to keep depending on your use case.

    Invoke with tool call:

        .. code-block:: python

            tool = TavilySearchResults(max_results=3)
            tool_msg = tool.invoke({"args": {'query': 'how to cook a steak?'}, "type": "tool_call", "id": "foo", "name": "tavily"})
            print(tool_msg.content,"\n",tool_msg.artifact)

        .. code-block:: python

            'Pan-Seared Steaks\nPan-searing is the best way to cook a steak, and it's also the easiest!\nIngredients\nInstructions\nPair with\nNutrition Information\nPowered by\nThis website is written and produced for informational purposes only. When I do this again I will do for 5 minutes but will turn off the heat on my cast Iron frying pan 2 minutes before and add butter and rosemary and garlic to get the steak more to our liking.\n I got a ribeye steak, heated the pan to the top heat and did everything like you mentioned, but after three minutes the steak was burned, on the other side the same happened. After doing some more research, I find you have to bring the steak to room temperature before you cook it and yiu have to snip the fat around the edges to keep it from curling. 22 Quick and Easy Recipes in 30 Minutes (or less) + 5 Chef Secrets To Make You A Better Cook!\nFind a Recipe\nHow To Cook Steak On The Stovetop\nThis post may contain affiliate links.'

            {'query': 'how to cook a steak?',
            'follow_up_questions': None,
            'answer': None,
            'images': [],
            'results': [{'title': 'How To Cook Steak On The Stovetop - Once Upon a Chef',
            'url': 'https://www.onceuponachef.com/recipes/how-to-cook-steak-on-the-stovetop.html',
            'content': 'Pan-Seared Steaks\nPan-searing is the best way to cook a steak, and it's also the easiest!\nIngredients\nInstructions\nPair with\nNutrition Information\nPowered by\nThis website is written and produced for informational purposes only. When I do this again I will do for 5 minutes but will turn off the heat on my cast Iron frying pan 2 minutes before and add butter and rosemary and garlic to get the steak more to our liking.\n I got a ribeye steak, heated the pan to the top heat and did everything like you mentioned, but after three minutes the steak was burned, on the other side the same happened. After doing some more research, I find you have to bring the steak to room temperature before you cook it and yiu have to snip the fat around the edges to keep it from curling. 22 Quick and Easy Recipes in 30 Minutes (or less) + 5 Chef Secrets To Make You A Better Cook!\nFind a Recipe\nHow To Cook Steak On The Stovetop\nThis post may contain affiliate links.',
            'score': 0.98514,
            'raw_content': "22 Quick and Easy Recipes in 30 Minutes (or less) + 5 Chef Secrets To Make You A Better Cook!\nFind a Recipe\nHow To Cook Steak On The Stovetop\nThis post may contain affiliate links. Read my full disclosure policy.\nPan-searing is the best way to cook a steak, and it's also the easiest!\nI love the kind of dinner you can create without relying on a recipe. Truth be told, good cooking is more about mastering techniques than following recipes, and the best dishes are often the simplest to whip up. A perfectly cooked steak is a prime example. With just a handful of ingredients and a single pan, you can prepare a steak that rivals one you'd enjoy at a high-end steakhouse.\nThe secret lies in mastering the art of pan-searing. This classic technique involves cooking the surface of your food undisturbed in a piping hot pan until a crisp, golden-brown, and flavorful crust forms. It's the key to building flavor and texture in a dish, while also preventing sticking and giving your meal a restaurant-quality appearance. Pan-searing is hands-down the best way to cook a steak (it works wonders for salmon and scallops, too), and it also happens to be the easiest.\nWhat you'll need to Cook Steak on The Stovetop\nWhen it comes to beef, the best candidates for pan-searing are boneless, quick-cooking cuts between one and one-and-a-half inches thick, such as NY Strip, rib eye or filet mignon. (For larger or slow-cooking cuts, like beef tenderloin with red wine sauce or beef stew with carrots and potatoes, pan-searing is usually the first step, and then you finish the cooking in the oven.)\nHow to cook steak On The Stovetop\nTo begin, pat the steak dry with paper towels. (Any moisture on the exterior of the steak must first evaporate before the meat begins to brown.)\nSeason the steaks generously on both sides with salt and pepper; the seasoning will stick to the surface and help create a delicious crust.\nTurn on your exhaust fan and heat a heavy pan over medium-high heat until it's VERY hot. The best pans for pan-searing are stainless steel or cast-iron since they can withstand high temperatures.\nAdd the oil to the pan. You'll know it's hot enough when it begins to shimmer and move fluidly around the pan.\nCarefully set the steak in the pan, releasing it away from you so the oil doesn't splatter in your direction. It should sizzle. (Use a pan that is large enough that it's not such a tight fit or the pan will cool down and your food will steam instead of sear.)\nLeave it alone! Avoid the temptation to peek or fiddle or flip repeatedly. The steaks need a few minutes undisturbed to develop a brown crust. (Don't worry about sticking; the steaks will release easily when they are ready to flip.)\nFlip the steaks when they release easily and the bottom is a deep-brown color (usually about 3 minutes).\nContinue to cook the steaks for another 3 to 4 minutes on the bottom side for rare or medium-rare.\nDuring the last minute of cooking, add 1 tablespoon of butter and a few sprigs of fresh thyme to the pan with the steaks (this is optional but delicious).\nIf you are serving the steaks unsliced, transfer them to plates and serve hot. If you plan to slice the steaks, transfer them to a cutting board and let rest, covered with aluminum foil, for 5 to 10 minutes; then slice thinly against the grain. (Resting allows the juices to redistribute from the outside of the steaks; if you slice them too soon, the juices will pour out of them.)\nVideo Tutorial\nYou May Also Like\nDid you make this recipe?\nI'd love to know how it turned out! Please let me know by leaving a review below. Or snap a photo and share it on Instagram; be sure to tag me @onceuponachef.\nPan-Seared Steaks\nPan-searing is the best way to cook a steak, and it's also the easiest!\nIngredients\nInstructions\nPair with\nNutrition Information\nPowered by\nThis website is written and produced for informational purposes only. I am not a certified nutritionist and the nutritional data on this site has not been evaluated or approved by a nutritionist or the Food and Drug Administration. Nutritional information is offered as a courtesy and should not be construed as a guarantee. The data is calculated through an online nutritional calculator, Edamam.com. Although I do my best to provide accurate nutritional information, these figures should be considered estimates only. Varying factors such as product types or brands purchased, natural fluctuations in fresh produce, and the way ingredients are processed change the effective nutritional information in any given recipe. Furthermore, different online calculators provide different results depending on their own nutrition fact sources and algorithms. To obtain the most accurate nutritional information in a given recipe, you should calculate the nutritional information with the actual ingredients used in your recipe, using your preferred nutrition calculator.\nSee more recipes:\nComments\nMy husband and I made this using a 1lb Rib Eye at room temp. ( My cast iron pan was a bit too hot so I adjusted it. It was perfectly cooked after following your directions. We are so grateful to you.\nPS, we made the Pumpkin bread last week and it was so delicious tasting a hint of cloves.\nI followed the recipe exactly. It turned out burned on the outside, cold and raw on the inside, grease splattered everywhere, and the butter burned as soon as it hit the pan. Someone's got some explaining to do.\nAfter doing some more research, I find you have to bring the steak to room temperature before you cook it and yiu have to snip the fat around the edges to keep it from curling. Still cooked too hot tho.\nThe timing was completely incorrect. I got a ribeye steak, heated the pan to the top heat and did everything like you mentioned, but after three minutes the steak was burned, on the other side the same happened. My whole house got filled with smoke. Steak turned black.\nMine nearly burned… on the outside… inside was raw.. almost freaking purple with no heat in the middle.\nJen,\nDefinitely deliciously simple steak. I made a NY Strip steak. I added vadallia onions and portobello mushrooms. How do you tenderize a steak? Please add some variations.\nGlad you liked it! I wouldn't really recommend tenderizing a steak; rather, I'd just buy a tender cut like a filet.\nI found a good recipe to make something interesting for dinner.Thank you.\nThis is THE BEST stovetop recipe I have ever served\nPerfect!!!!!\nDelicious!! I never thought I could make steak taste like this without a grill.\nLOVED it!!\nHello all,\nI'll make this short and say this article was a life saver! I needed an easy and fast way to cook steaks and this was the best thing ever! I cooked a NY steak and it cooked it to a perfect and taste medium! If there are bad reviews, they clearly did not read or follow the article. Pat dry, medium/high heat, drop steak once it is hot enough, cook for desired time. I did 4 min, flip and another 4 min. It was a 1lb steak and I did add butter, no herbs. Extremely tasty and will make again! Again, this was cooked to a perfect medium.\nYum! Thanks for this recipe.\nJenn, thanks so very much for your meticulous instructions on searing a steak. I cook rarely and haven't cooked a steak in at least 10 years, so had zero recollection how. I bought a beautiful, thick, rib-eye and followed your instructions to the letter. Perfection! I enjoyed the steak very much! Thanks again.\nHi Jenn, Thank you so much for sharing this technique. It turned my steak into steakhouse quality, just by following your instructions. I used T-bone with the bone cut out (for my dog) and they were less than 1 inch thick so were a little overdone for my taste, but that's my fault for trying this with thinner steak. Even so, the flavor was excellent! To think all these years I'v been marinating, rubbing, grilling my steaks when the best and simplest and best way was to just pull out my cast iron skillet. Can't wait to get some thick ribeye!\nThanks for teaching me how to cook a delicious steak.\nJenn…\nYou are my Cooking Genie!!! Been cooking steaks for years and this simple but amazing recipe is genius! Tried it today in my mother's cast iron pan. Turned out just PERFECT!! You are a gift to the cooking world!!\nThank you!\nCarol\nSeems to me that a lot of the complaints about ‘too rare', ‘too done', etc. could easily be solved by using an instant read meat thermometer. That way you can more accurately get the meat cooked to your liking.\nThanks Very Much!! Never in my 66 years cooked steaks any other than on a grill. Been too hot in Arkansas lately, so we tried your instructions …….They Turned Out Great!!! THANKS!!!\nwhat about reverse searing if you want the inside to be more done than medium rare\nSure, Terry, that should work. 🙂\nQuestions Jenn: to dry the steaks out, does it work to put the steaks in the frig uncovered, and for how long? And: I have an electric stove, will they sear as well as a gas range?\nHi Pamela, I don't find it necessary to have the steaks sit in the fridge to dry, but it's certainly OK to do for up to a few hours. And an electric stove should work just as well as gas; the key is just to have the pan very hot before you put the steaks in. Enjoy!\nWe love garlic on our steaks. Would you advise fresh garlic or granulated & how & when would you advise adding? Thanks in advance Jenn. You are the best out of all the recipe sites. You are so kind & seem to respond to questions all the time & needless to say you are making us all better cooks! 😊\nHi Dana, thanks for your very kind words – so glad you like the recipes! For the garlic, I would go with fresh and add them along with the butter. It will help to infuse the butter with a subtle garlic flavor. Just keep your eye on them as you don't want them to burn. Enjoy!\nLoved it, first time I pan seared steaks…came out great! Will be doing again.\nWe've made this recipe a few times. It comes out perfect every time.\nAbsolutely delicious…and oh so easy.\nDefinitely don't recommend following these directions. You'll end up with a rare steak, even following the time recommended for medium.\nThe technique is solid,did you let your steaks come to room temperature before cooking? Usually 30-45 minutes.Hope this helps and happy cooking.\nHANNAH: Any steak (your choice!) can be made this way. Just remember that Bone-In or boneless will have different cook times. As per your comment though, are your steaks thin or thick cut? Oven temps. vary by oven (+/-) of set temp.. May need to cook longer to get a Medium. Be Patient.\nI have followed this recipe several times since first seeing it in 2020. Every time I have made this using strip steak or ribeye it has been fantastic! We just had strip steaks tonight for Valentines Day and they were as always terrific. Thank you!\nWill this work for skirt steak too?\nHi P, This method could be used, but depending on how well-done you like your skirt steak (and how large it is), you may need to finish cooking it in the oven.\nThese days I often buy steak when it's on sale and will cook 2 of a 3 pack and freeze the 3rd. I have often heard steak from frozen is not as good. Do you think this method will work as well from frozen (thawed) as from fresh?\nSure, Colleen, it should be fine with thawed steak. Hope you enjoy!\nI made this steak today, it was great and I followed the recipe to the letter. I did make one change In the last minute I used 2 tablespoons of butter with garlic and rosemary. The flavour was there and the crust was perfect. I cooked for the 4 Minutes as we like our steaks medium. When I do this again I will do for 5 minutes but will turn off the heat on my cast Iron frying pan 2 minutes before and add butter and rosemary and garlic to get the steak more to our liking.\nYummy!\nJust cooked two filets according to this fantastic recipe, and they were medium-rare and beautifully browned–much more tasty than done on the grill and perhaps less mess. Also love all your salmon recipes, which use a similar method. Ignoring the urge to fiddle with the beef or fish is key for a perfectly done dinner. Yours is my go-to website for delicious recipes; photos and step-by-step instructions help to make each recipe foolproof. Thanks, Jenn!\nBest steak I've ever made\nHello Chef,\nThank you for those quick tips! My rib eye steaks came out delicious… I am serving it with broccoli and cheese, and some Spanish rice…\nHi my name is Rob. I have always been looking for a way to sear and cook steaks on the stove but Oh my god this worked so well it was better than the grill. I'm wondering if I can use the same method to make Hamburgrrs ? I also took the juices with the oil and made a dipping sauce. I can't wait to try again.\nGlad you enjoyed, Robert! It will work with burgers, too.\nI'm on a low carb diet with a low grocery budget, a combination that rendered my take on this recipe to be done with top round instead of a strip or filet. I have to say, adjusting a minute less per side for the thinner cut, the method described here was absolutely perfect. I have found the one I'm sticking with after some less than stellar results from recipes on other sites. Turned out flavorful, perfectly medium-rare, and-and this was a nice bonus for going with a cut of round-delightfully tender. Thank you so much for the great information!\nThis steak is on my “next week” docket, although obsessing and looking forward begins today 🙂 Would be interested in hearing if you pan sear steaks directly from the fridge, or if you let them come to room temperature first.\nHi Jeff, you definitely can take them out about 30 minutes ahead of time. (I usually forget though and they still turn out well.) 🙂\nCan confirm – this is the *best* way to make steaks. Simple, straightforward, and flawless each time. You can also include slightly more involved rubs, but don't go too crazy and be sure to be careful in what you choose. The very high heat can scald some herbs, so make sure you choose a more heat-tolerant rub or seasoning. The herb butter process here is truly *chefs kiss*\nA great recipe and the only way we prepare steaks! A suggestion for those concerned about the smoke and potential splatter indoors is to heat your skillet and perform the entire operation on your outdoor grill. Works just as well without chancing the smoke alarms!\nAdd a Comment Cancel reply\nYour comment *\nRate the recipe: 5 stars means you loved it, 1 star means you really disliked it\nYour name *\nYour email (will not be published) *\nSave my name, email, and website in this browser for the next time I comment.\nΔ\nThis site uses Akismet to reduce spam. Learn how your comment data is processed.\nWelcome - I'm Jenn Segal - Classically Trained Chef, Cookbook Author & Busy Mom\nOnce upon a time, I went to culinary school and worked in fancy restaurants. Now, I'm cooking for my family and sharing all my tested & perfected recipes with you here! Read more…\nFall Recipes\nPopular Recipes\n22 Quick and Easy Recipes in 30 Minutes (or less)\nPlus 5 Chef Secrets To Make You A Better Cook!\nCooking Question?\nΔ\nAs Featured On\nBuy Now\nBuy Now"}],
            'response_time': 0.9}

    """  # noqa: E501

    name: str = "tavily_search_results_json"
    description: str = (
        "A search engine optimized for comprehensive, accurate, and trusted results. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query."
    )
    api_wrapper: TavilySearchAPIWrapper = Field(default_factory=TavilySearchAPIWrapper)  # type: ignore[arg-type]
    max_results: int = 5
    """Max search results to return, default is 5"""
    search_depth: str = "advanced"
    '''The depth of the search. It can be "basic" or "advanced"'''
    include_domains: List[str] = []
    """A list of domains to specifically include in the search results. Default is None, which includes all domains."""  # noqa: E501
    exclude_domains: List[str] = []
    """A list of domains to specifically exclude from the search results. Default is None, which doesn't exclude any domains."""  # noqa: E501
    include_answer: bool = False
    """Include a short answer to original query in the search results. Default is False."""  # noqa: E501
    include_raw_content: bool = False
    """Include cleaned and parsed HTML of each site search results. Default is False."""
    include_images: bool = False
    """Include a list of query related images in the response. Default is False."""
    args_schema: Type[BaseModel] = TavilyInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Union[List[Dict], str]:
        """Use the tool."""
        try:
            raw_result = self.api_wrapper.raw_results(
                query,
                self.max_results,
                self.search_depth,
                self.include_domains,
                self.exclude_domains,
                self.include_answer,
                self.include_raw_content,
                self.include_images,
            )
        except Exception as e:
            return repr(e)
        content = self.api_wrapper.clean_results(raw_result["results"])
        return content, raw_result  # type: ignore

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Union[List[Dict], str]:
        """Use the tool asynchronously."""
        try:
            raw_result = await self.api_wrapper.raw_results_async(
                query,
                self.max_results,
                self.search_depth,
                self.include_domains,
                self.exclude_domains,
                self.include_answer,
                self.include_raw_content,
                self.include_images,
            )
        except Exception as e:
            return repr(e)
        content = self.api_wrapper.clean_results(raw_result["results"])
        return content, raw_result  # type: ignore


class TavilyAnswer(BaseTool):
    """Tool that queries the Tavily Search API and gets back an answer."""

    name: str = "tavily_answer"
    description: str = (
        "A search engine optimized for comprehensive, accurate, and trusted results. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query. "
        "This returns only the answer - not the original source data."
    )
    api_wrapper: TavilySearchAPIWrapper = Field(default_factory=TavilySearchAPIWrapper)  # type: ignore[arg-type]
    args_schema: Type[BaseModel] = TavilyInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Union[List[Dict], str]:
        """Use the tool."""
        try:
            return self.api_wrapper.raw_results(
                query,
                max_results=5,
                include_answer=True,
                search_depth="basic",
            )["answer"]
        except Exception as e:
            return repr(e)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Union[List[Dict], str]:
        """Use the tool asynchronously."""
        try:
            result = await self.api_wrapper.raw_results_async(
                query,
                max_results=5,
                include_answer=True,
                search_depth="basic",
            )
            return result["answer"]
        except Exception as e:
            return repr(e)
