# flake8: noqa
NASA_SEARCH_PROMPT = """
    This tool is a wrapper around NASA's search API, useful when you need to search through NASA's Image and Video Library. 
    The input to this tool is a query specified by the user, and will be passed into NASA's `search` function.
    
    At least one parameter must be provided.

    There are optional parameters that can be passed by the user based on their query
    specifications. Each item in this list contains pound sign (#) separated values, the first value is the parameter name, 
    the second value is the datatype and the third value is the description: {{

        - q#string#Free text search terms to compare to all indexed metadata.
        - center#string#NASA center which published the media.
        - description#string#Terms to search for in “Description” fields.
        - description_508#string#Terms to search for in “508 Description” fields.
        - keywords #string#Terms to search for in “Keywords” fields. Separate multiple values with commas.
        - location #string#Terms to search for in “Location” fields.
        - media_type#string#Media types to restrict the search to. Available types: [“image”,“video”, “audio”]. Separate multiple values with commas.
        - nasa_id #string#The media asset’s NASA ID.
        - page#integer#Page number, starting at 1, of results to get.-
        - page_size#integer#Number of results per page. Default: 100.
        - photographer#string#The primary photographer’s name.
        - secondary_creator#string#A secondary photographer/videographer’s name.
        - title #string#Terms to search for in “Title” fields.
        - year_start#string#The start year for results. Format: YYYY.
        - year_end #string#The end year for results. Format: YYYY.

    }}
    
    Below are several task descriptions along with their respective input examples.
    Task: get the 2nd page of image and video content starting from the year 2002 to 2010
    Example Input: {{"year_start":  "2002", "year_end":  "2010", "page": 2}}
    
    Task: get the image and video content of saturn photographed by John Appleseed
    Example Input: {{"q": "saturn", "photographer": "John Appleseed"}}
    
    Task: search for Meteor Showers with description "Search Description" with media type image
    Example Input: {{"q": "Meteor Shower", "description": "Search Description", "media_type": "image"}}
    
    Task: get the image and video content from year 2008 to 2010 from Kennedy Center
    Example Input: {{"year_start":  "2002", "year_end":  "2010", "location": "Kennedy Center}}
    """


NASA_MANIFEST_PROMPT = """
    This tool is a wrapper around NASA's media asset manifest API, useful when you need to retrieve a media 
    asset's manifest. The input to this tool should include a string representing a NASA ID for a media asset that the user is trying to get the media asset manifest data for. The NASA ID will be passed as a string into NASA's `get_media_metadata_manifest` function.

    The following list are some examples of NASA IDs for a media asset that you can use to better extract the NASA ID from the input string to the tool.
    - GSFC_20171102_Archive_e000579
    - Launch-Sound_Delta-PAM-Random-Commentary
    - iss066m260341519_Expedition_66_Education_Inflight_with_Random_Lake_School_District_220203
    - 6973610
    - GRC-2020-CM-0167.4
    - Expedition_55_Inflight_Japan_VIP_Event_May_31_2018_659970
    - NASA 60th_SEAL_SLIVER_150DPI
"""

NASA_METADATA_PROMPT = """
    This tool is a wrapper around NASA's media asset metadata location API, useful when you need to retrieve the media asset's metadata. The input to this tool should include a string representing a NASA ID for a media asset that the user is trying to get the media asset metadata location for. The NASA ID will be passed as a string into NASA's `get_media_metadata_manifest` function.

    The following list are some examples of NASA IDs for a media asset that you can use to better extract the NASA ID from the input string to the tool.
    - GSFC_20171102_Archive_e000579
    - Launch-Sound_Delta-PAM-Random-Commentary
    - iss066m260341519_Expedition_66_Education_Inflight_with_Random_Lake_School_District_220203
    - 6973610
    - GRC-2020-CM-0167.4
    - Expedition_55_Inflight_Japan_VIP_Event_May_31_2018_659970
    - NASA 60th_SEAL_SLIVER_150DPI
"""

NASA_CAPTIONS_PROMPT = """
    This tool is a wrapper around NASA's video assests caption location API, useful when you need 
    to retrieve the location of the captions of a specific video. The input to this tool should include a string representing a NASA ID for a video media asset that the user is trying to get the get the location of the captions for. The NASA ID will be passed as a string into NASA's `get_media_metadata_manifest` function.

    The following list are some examples of NASA IDs for a video asset that you can use to better extract the NASA ID from the input string to the tool.
    - 2017-08-09 - Video File RS-25 Engine Test
    - 20180415-TESS_Social_Briefing
    - 201_TakingWildOutOfWildfire
    - 2022-H1_V_EuropaClipper-4
    - 2022_0429_Recientemente
"""
