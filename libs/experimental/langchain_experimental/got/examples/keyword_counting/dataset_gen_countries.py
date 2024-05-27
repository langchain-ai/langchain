# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Ales Kubicek

import csv
from typing import List, Tuple
from graph_of_thoughts import controller


def find_country_indices(text: str, country: str) -> List[Tuple[int, str]]:
    """
    Finds the indices of the occurences of a given country in the input text.

    :param text: Input text.
    :type text: str
    :param country: Country to search for.
    :type country: str
    :return: List of tuples, where each tuple consists of index and country.
    :rtype: List[Tuple[int, str]]
    """

    indices = []
    index = text.find(country)
    while index != -1:
        indices.append(index)
        index = text.find(country, index + 1)
    return [(index, country) for index in indices]


primary_countries = [
    "Afghanistan",
    "Argentina",
    "Australia",
    "Brazil",
    "Canada",
    "China",
    "Colombia",
    "Cuba",
    "Egypt",
    "France",
    "Germany",
    "Greece",
    "India",
    "Indonesia",
    "Iran",
    "Iraq",
    "Ireland",
    "Israel",
    "Italy",
    "Japan",
    "Kenya",
    "Mexico",
    "Netherlands",
    "New Zealand",
    "Nigeria",
    "North Korea",
    "Pakistan",
    "Peru",
    "Philippines",
    "Poland",
    "Portugal",
    "Russia",
    "Saudi Arabia",
    "South Africa",
    "South Korea",
    "Spain",
    "Sweden",
    "Switzerland",
    "Thailand",
    "Turkey",
    "Ukraine",
    "United Arab Emirates",
    "United Kingdom",
    "United States",
    "Venezuela",
    "Vietnam",
    "Yemen",
    "Zimbabwe",
    "Belgium",
    "Norway",
]
primary_adjectives = [
    "Afghan",
    "Argentine ",
    "Argentinean",
    "Australian",
    "Brazilian",
    "Canadian",
    "Chinese",
    "Colombian",
    "Cuban",
    "Egyptian",
    "French",
    "German",
    "Greek",
    "Indian",
    "Indonesian",
    "Iranian",
    "Iraqi",
    "Irish",
    "Israeli",
    "Italian",
    "Japanese",
    "Kenyan",
    "Mexican",
    "Dutch",
    "New Zealander ",
    "Kiwi",
    "Nigerian",
    "North Korean",
    "Pakistani",
    "Peruvian",
    "Filipino",
    "Philippine",
    "Polish",
    "Portuguese",
    "Russian",
    "Saudi ",
    "Saudi Arabian",
    "South African",
    "South Korean",
    "Spanish",
    "Swedish",
    "Swiss",
    "Thai",
    "Turkish",
    "Ukrainian",
    "United Arab Emirates",
    "Emirati",
    "British",
    "American",
    "Venezuelan",
    "Vietnamese",
    "Yemeni",
    "Zimbabwean",
    "Belgian",
    "Norwegian",
]
rest_countries = [
    "Albania",
    "Algeria",
    "Andorra",
    "Angola",
    "Antigua and Barbuda",
    "Armenia",
    "Austria",
    "Azerbaijan",
    "The Bahamas",
    "Bahrain",
    "Bangladesh",
    "Barbados",
    "Belarus",
    "Belize",
    "Benin",
    "Bhutan",
    "Bolivia",
    "Bosnia and Herzegovina",
    "Botswana",
    "Brunei",
    "Bulgaria",
    "Burkina Faso",
    "Burundi",
    "Cabo Verde",
    "Cambodia",
    "Cameroon",
    "Central African Republic",
    "Chad",
    "Chile",
    "Comoros",
    "Congo",
    "Costa Rica",
    "Côte d’Ivoire",
    "Croatia",
    "Cyprus",
    "Czech Republic",
    "Czechia",
    "Denmark",
    "Djibouti",
    "Dominica",
    "Dominican Republic",
    "East Timor",
    "Timor-Leste",
    "Ecuador",
    "El Salvador",
    "Equatorial Guinea",
    "Eritrea",
    "Estonia",
    "Eswatini",
    "Ethiopia",
    "Fiji",
    "Finland",
    "Gabon",
    "The Gambia",
    "Georgia",
    "Ghana",
    "Grenada",
    "Guatemala",
    "Guinea",
    "Guinea-Bissau",
    "Guyana",
    "Haiti",
    "Honduras",
    "Hungary",
    "Iceland",
    "Jamaica",
    "Jordan",
    "Kazakhstan",
    "Kiribati",
    "Kosovo",
    "Kuwait",
    "Kyrgyzstan",
    "Laos",
    "Latvia",
    "Lebanon",
    "Lesotho",
    "Liberia",
    "Libya",
    "Liechtenstein",
    "Lithuania",
    "Luxembourg",
    "Madagascar",
    "Malawi",
    "Malaysia",
    "Maldives",
    "Mali",
    "Malta",
    "Marshall Islands",
    "Mauritania",
    "Mauritius",
    "Micronesia",
    "Moldova",
    "Monaco",
    "Mongolia",
    "Montenegro",
    "Morocco",
    "Mozambique",
    "Myanmar",
    "Burma",
    "Namibia",
    "Nauru",
    "Nepal",
    "Nicaragua",
    "Niger",
    "North Macedonia",
    "Oman",
    "Palau",
    "Panama",
    "Papua New Guinea",
    "Paraguay",
    "Qatar",
    "Romania",
    "Rwanda",
    "Saint Kitts and Nevis",
    "Saint Lucia",
    "Saint Vincent and the Grenadines",
    "Samoa",
    "San Marino",
    "Sao Tome and Principe",
    "Senegal",
    "Serbia",
    "Seychelles",
    "Sierra Leone",
    "Singapore",
    "Slovakia",
    "Slovenia",
    "Solomon Islands",
    "Somalia",
    "Sri Lanka",
    "Sudan",
    "Suriname",
    "Syria",
    "Taiwan",
    "Tajikistan",
    "Tanzania",
    "Togo",
    "Tonga",
    "Trinidad and Tobago",
    "Tunisia",
    "Turkmenistan",
    "Tuvalu",
    "Uganda",
    "Uruguay",
    "Uzbekistan",
    "Vanuatu",
    "Vatican City",
    "Zambia",
]
rest_adjectives = [
    "Albanian",
    "Algerian",
    "Andorran",
    "Angolan",
    "Antiguan and Barbudan",
    "Armenian",
    "Austrian",
    "Azerbaijani",
    "Bahamian",
    "Bahraini",
    "Bangladeshi",
    "Barbadian",
    "Belarusian",
    "Belizean",
    "Beninese",
    "Bhutanese",
    "Bolivian",
    "Bosnian and Herzegovinian",
    "Botswanan",
    "Bruneian",
    "Bulgarian",
    "Burkinabè",
    "Burundian",
    "Cape Verdean",
    "Cambodian",
    "Cameroonian",
    "Central African",
    "Chadian",
    "Chilean",
    "Comorian",
    "Congolese",
    "Costa Rican",
    "Ivorian",
    "Croatian",
    "Cypriot",
    "Czech",
    "Czech",
    "Danish",
    "Djiboutian",
    "Dominican",
    "Dominican",
    "East Timorese",
    "Timorese",
    "Ecuadorian",
    "Salvadoran",
    "Equatorial Guinean",
    "Eritrean",
    "Estonian",
    "Swazi",
    "Ethiopian",
    "Fijian",
    "Finnish",
    "Gabonese",
    "Gambian",
    "Georgian",
    "Ghanaian",
    "Grenadian",
    "Guatemalan",
    "Guinean",
    "Bissau-Guinean",
    "Guyanese",
    "Haitian",
    "Honduran",
    "Hungarian",
    "Icelandic",
    "Jamaican",
    "Jordanian",
    "Kazakh",
    "I-Kiribati",
    "Kosovar",
    "Kuwaiti",
    "Kyrgyz",
    "Laotian",
    "Latvian",
    "Lebanese",
    "Basotho",
    "Liberian",
    "Libyan",
    "Liechtensteiner",
    "Lithuanian",
    "Luxembourger",
    "Malagasy",
    "Malawian",
    "Malaysian",
    "Maldivian",
    "Malian",
    "Maltese",
    "Marshallese",
    "Mauritanian",
    "Mauritian",
    "Micronesian",
    "Moldovan",
    "Monégasque",
    "Mongolian",
    "Montenegrin",
    "Moroccan",
    "Mozambican",
    "Myanmarese",
    "Burmese",
    "Namibian",
    "Nauruan",
    "Nepali",
    "Nicaraguan",
    "Nigerien",
    "Macedonian",
    "Omani",
    "Palauan",
    "Panamanian",
    "Papua New Guinean",
    "Paraguayan",
    "Qatari",
    "Romanian",
    "Rwandan",
    "Kittitian",
    "Nevisian",
    "Saint Lucian",
    "Vincentian",
    "Samoan",
    "Sammarinese",
    "Santomean",
    "Senegalese",
    "Serbian",
    "Seychellois",
    "Sierra Leonean",
    "Singaporean",
    "Slovak",
    "Slovenian",
    "Solomon Islander",
    "Somali",
    "Sri Lankan",
    "Sudanese",
    "Surinamese",
    "Syrian",
    "Taiwanese",
    "Tajik",
    "Tanzanian",
    "Togolese",
    "Tongan",
    "Trinidadian ",
    "Tobagonian",
    "Tunisian",
    "Turkmen",
    "Tuvaluan",
    "Ugandan",
    "Uruguayan",
    "Uzbek",
    "Ni-Vanuatu",
    "Vatican",
    "Zambian",
]

lm = controller.ChatGPT(
    "../../graph_of_thoughts/controller/config.json", model_name="chatgpt4"
)

prompt = """<Instruction> Generate a continuous passage (single paragraph) of 16 sentences following the provided restrictions precisely. </Instruction>

<Restrictions>
The following restrictions must apply to the generated text:
1. Single continuous passage of exactly 16 sentences without any paragraphs (line breaks).
2. Countries appearing in the passage must be only from the provided list. No other countries can be mentioned.
3. When a country is mentioned in the passage, it must be mentioned multiple times consecutively in the same or following sentences.
4. Passage should be creative and coherent.
5. Using adjectives of a country is NOT allowed (e.g., "Colombian coffee" should be "coffee from Colombia" instead)
</Restrictions>

<Example>
List of countries: [Afghanistan, Argentina, Australia, Brazil, Canada, China, Colombia, Cuba, Egypt, France, Germany, Greece, India, Indonesia, Iran, Iraq, Ireland, Israel, Italy, Japan, Kenya, Mexico, Netherlands, New Zealand, Nigeria, North Korea, Pakistan, Peru, Philippines, Poland, Portugal, Russia, Saudi Arabia, South Africa, South Korea, Spain, Sweden, Switzerland, Thailand, Turkey, Ukraine, United Arab Emirates, United Kingdom, United States, Venezuela, Vietnam, Yemen, Zimbabwe, Belgium, Norway]
Passage:
While exploring the ancient ruins in Greece, Sam discovered manuscripts that hinted at the hidden treasures of Egypt. It seemed these treasures were once stolen from Egypt by rogue merchants and secretly moved to Greece, only to be buried under layers of time. Intrigued, he shared the findings with his friend Maya from India, who was an expert in decoding ancient languages. She pointed out that there was a similar legend in India about treasures from China that had somehow ended up in the southern parts of India, possibly through trade or conquest. She also recounted tales from China that spoke of incredible artifacts from Indonesia, suggesting a rich tapestry of cultural exchanges throughout history. Their conversation took an interesting turn when Sam mentioned a book he'd read about the mysterious connections between Argentina and Brazil. The book detailed how both Argentina and Brazil, despite their differences, shared tales of lost civilizations and forgotten cities deep within their jungles. Maya excitedly mentioned that she'd been to the Philippines and had heard local legends about ancient ties with Indonesia and how traders from the Philippines would journey to Indonesia in search of spices and other goods. Thinking of spices, Sam fondly recalled his trip to Spain, where he had learned about the country's historical links with Portugal. Spain and Portugal, both maritime giants of their time, had extensively explored unknown lands and established trade routes. Maya, remembering her travels, said that she had been to Belgium once and was fascinated by its connections with the Netherlands. Both Belgium and the Netherlands, she explained, had rich histories of art, trade, and diplomacy that intertwined them for centuries. They both sat back, marveling at the interconnectedness of the world and how countries from Greece to the Netherlands shared tales of adventure, discovery, and mystery.
</Example>

List of countries: [Afghanistan, Argentina, Australia, Brazil, Canada, China, Colombia, Cuba, Egypt, France, Germany, Greece, India, Indonesia, Iran, Iraq, Ireland, Israel, Italy, Japan, Kenya, Mexico, Netherlands, New Zealand, Nigeria, North Korea, Pakistan, Peru, Philippines, Poland, Portugal, Russia, Saudi Arabia, South Africa, South Korea, Spain, Sweden, Switzerland, Thailand, Turkey, Ukraine, United Arab Emirates, United Kingdom, United States, Venezuela, Vietnam, Yemen, Zimbabwe, Belgium, Norway]
Passage:
"""

num_samples = 100
sample_id = 0
result = [["ID", "Text", "Countries", "Sentences", "Characters"]]

"""
Generate passages of text that contain country names to be used as input for the
keyword counting.

Input(x)  : Number of samples
Output(y) : Passages written to a file in the CSV format.
            File contains the sample ID, the passage, the countries the passage
            contains, the sentences of the passages, number of characters of the
            passage.
"""

# For x batches of y responses
for _ in range(num_samples):
    response = lm.query(prompt, 1)
    texts = lm.get_response_texts(response)
    for text in texts:
        # Clean paragraphs - single long passage
        text = text.strip().replace("\n", "")

        # Get all occurrences of all primary permissible countries
        occurrences = []
        for country in [country for country in primary_countries if country in text]:
            occurrences.extend(find_country_indices(text, country))
        # Order exactly how they appear in the text
        ordered_occurrences = [country[1] for country in sorted(occurrences)]

        # Check invalid countries and adjectives
        invalid_primary_adjective = [
            adjective for adjective in primary_adjectives if adjective in text
        ]
        invalid_rest_country = [
            country for country in rest_countries if country in text
        ]
        invalid_rest_adjective = [
            adjective for adjective in rest_adjectives if adjective in text
        ]
        invalid_count = (
            len(invalid_primary_adjective)
            + len(invalid_rest_country)
            + len(invalid_rest_adjective)
        )

        if invalid_count > 0:
            print(
                f"Invalid countries or adjectives present: {invalid_primary_adjective}, {invalid_rest_country}, {invalid_rest_adjective}"
            )
            continue

        result.append(
            [
                sample_id,
                text,
                "[{0}]".format(", ".join(map(str, ordered_occurrences))),
                len(text.split(".")) - 1,
                len(text),
            ]
        )
        sample_id += 1

# Writing to csv file
with open("countries_script.csv", "w") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(result)
