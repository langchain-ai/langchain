Changelog
=========
8.1.0 - 2023-06-22
--------------------
Add support for the `Transfer Search API <https://developers.amadeus.com/self-service/category/cars-and-transfers/api-doc/transfer-search/api-reference>`_

Add support for the `Transfer Booking API <https://developers.amadeus.com/self-service/category/cars-and-transfers/api-doc/transfer-booking/api-reference>`_

Add support for the `Transfer Management API <https://developers.amadeus.com/self-service/category/cars-and-transfers/api-doc/transfer-management/api-reference>`_

Big thanks to `Siddhartha Dutta <https://github.com/siddydutta>`_ for his contribution to the above implementations!

8.0.0 - 2023-01-30
--------------------
Decommissioned Travel Restrictions API v1

Decommissioned Hotel Search API v2

Upgraded Python v3.8+ support and old dependencies

Upgraded testing by using `pytest` and `mock`

Fixed #175 Replace type() with instance()

Fixed #177  Update the default value {} as an argument

Minor updates in How to Release and running test in contribution guide

7.1.0 - 2022-11-04
--------------------
Add support for `Travel Restrictions v2 API <https://developers.amadeus.com/self-service/category/covid-19-and-travel-safety/api-doc/travel-restrictions/api-reference>`_

Bug fix in pagination

Add SonarCloud support

7.0.0 - 2022-07-20
--------------------
Add support for `Trip Parser v3 API <https://developers.amadeus.com/self-service/category/trip/api-doc/trip-parser/api-reference>`_ and remove Trip Parser v2

Add support for `City Search API <https://developers.amadeus.com/self-service/category/trip/api-doc/city-search/api-reference>`_

Add support for `Airline Routes API <https://developers.amadeus.com/self-service/category/air/api-doc/airline-routes/api-reference>`_ and `Hotel Name Autocomplete API <https://developers.amadeus.com/self-service/category/hotel/api-doc/hotel-name-autocomplete/api-reference>`_. Big thanks to `Siddhartha Dutta <https://github.com/siddydutta>`_ for his contribution! 

Implement the coverage report generation at CI time

6.0.1 - 2022-05-23
--------------------
Removing all references to the unused media namespace

6.0.0 - 2022-05-23
--------------------
Add support for `Hotel List API <https://developers.amadeus.com/self-service/category/hotel/api-doc/hotel-list/api-reference>`_

Add support for `Hotel Search v3 <https://developers.amadeus.com/self-service/category/hotel/api-doc/hotel-search/api-reference>`_

Add support for the X-HTTP-Method-Override header

Remove the AI-Generated Photos API

5.3.1 - 2022-02-24
--------------------
Update release workflow

5.3.0 - 2022-02-24
--------------------
Add support for `Travel Restrictions API <https://developers.amadeus.com/self-service/category/covid-19-and-travel-safety/api-doc/travel-restrictions/api-reference>`_
Add support for `Airport Routes API <https://developers.amadeus.com/self-service/category/air/api-doc/airport-routes/api-reference>`_

5.2.0 - 2021-11-29
--------------------
Migrate CI/CD to GitHub Actions

5.1.0 - 2021-05-12
--------------------
Add support for the `Flight Availabilities Search API <https://developers.amadeus.com/self-service/category/air/api-doc/flight-availabilities-search/api-reference>`_

Add support for the `Branded Fares Upsell API <https://developers.amadeus.com/self-service/category/air/api-doc/branded-fares-upsell/api-reference>`_

5.0.0 - 2021-02-02
--------------------
Remove support for Python 2. The SDK requires Python 3.4+

Fix unwanted exception on DELETE method of Flight Order Management API

4.5.0 - 2020-11-05
--------------------
Add support for the `Flight Price Analysis API <https://developers.amadeus.com/self-service/category/air/api-doc/flight-price-analysis/api-reference>`_

4.4.0 - 2020-10-09
--------------------
Add support for the `Tours and Activities API <https://developers.amadeus.com/self-service/category/destination-content/api-doc/tours-and-activities/api-reference>`_

4.3.0 - 2020-09-10
--------------------
Add support for the `On-Demand Flight Status API <https://developers.amadeus.com/self-service/category/air/api-doc/on-demand-flight-status/api-reference>`_

4.2.0 - 2020-08-05
--------------------
Add support for the `Travel Recommendations API <https://developers.amadeus.com/self-service/category/trip/api-doc/travel-recommendations>`_

Moved the code examples directory to a dedicated `code examples repository <https://github.com/amadeus4dev/amadeus-code-examples>`_

4.1.0 - 2020-06-11
--------------------
Add support for the `Safe Place API <https://developers.amadeus.com/self-service/category/destination-content/api-doc/safe-place-api>`_

4.0.0 - 2020-04-27
--------------------
Add support for the `Flight Choice Prediction v2 <https://developers.amadeus.com/self-service/category/air/api-doc/flight-choice-prediction>`_

The input of Flight Choice Prediction v2 is the result of Flight Offers Search API - in v1 the input was the result of Flight Low-Fare Search

Add support for the Retrieve (3rd) endpoint of `Points Of Interest API <https://developers.amadeus.com/self-service/category/destination-content/api-doc/points-of-interest>`_

Remove support for Flight Choice Prediction v1

Remove support for Flight Low-Fare Search: decommission on May 28, 2020 and mandatory migration to Flight Offers Search

Remove support for Most Searched Destinations

Add Trip Parser, Flight Create Orders and Flight Order Management executable examples

3.5.0 - 2020-02-13
--------------------
Add support for the `SeatMap Display <https://developers.amadeus.com/self-service/category/air/api-doc/seatmap-display>`_

SeatMap Display API allows you to get information to display airplane cabin plan from a Flight Offer in order for the traveler to be able to choose his seat during the flight booking flow thanks to POST method. In addition GET method allows you to display airplane cabin plan from an existing Flight Order.

3.4.0 - 2020-01-28
--------------------
Add support for the `Hotel Booking <https://developers.amadeus.com/self-service/category/hotel/api-doc/hotel-booking>`_

The Amadeus Hotel Booking API lets you complete bookings at over 150,000 hotels and accommodations around the world. To complete bookings, you must first use the Amadeus Hotel Search API to search for hotel deals, select the desired offer and confirm the final price and availability. You can then use the Hotel Booking API to complete the reservation by providing an offer id, guest information and payment information.

Add support for the `Flight Order Management <https://developers.amadeus.com/self-service/category/air/api-doc/flight-order-management>`_

The Flight Order Management API lets you consult bookings created through the Flight Create Orders API. Using the booking ID generated by Flight Create Orders, Flight Order Management returns the last-updated version of the booking record with any post-booking modifications including but not limited to ticket information, form of payment or other remarks.

Add support for the `Flight Create Orders <https://developers.amadeus.com/self-service/category/air/api-doc/flight-create-orders>`_

The Flight Create Order API is a flight booking API that lets you perform the final booking for a desired flight and ancillary products (additional bags, extra legroom, etc.). The API returns a unique ID for the flight order and reservation details. This API is used to perform the final booking on confirmed fares returned by the Flight Offers Price API.

Add support for the `Flight Offers Price <https://developers.amadeus.com/self-service/category/air/api-doc/flight-offers-price>`_

The Flight Offers Price API confirms the flight price (including taxes and fees) and availability for a given flight returned by the Flight Offers Search API. The API also returns pricing for ancillary products (additional bags, extra legroom, etc.) and the payment information details needed for booking.

Add support for the `Flight Offers Search <https://developers.amadeus.com/self-service/category/air/api-doc/flight-offers-search>`_

The Flight Offers Search API is a flight search API that returns cheap flights between two airports for a given number of passengers and for a given date or date range. The API returns airline name, price and fare details, as well as additional information like baggage allowance, prices for additional baggage and departure terminal.

Add support for the `Trip Parser <https://developers.amadeus.com/self-service/category/trip/api-doc/trip-parser>`_

The Trip Parser API parses information from various booking confirmation emails and returns a standardized, structured travel itinerary. The API can extract relevant information from a wide variety of flight, hotel, rental car and rail providersâ€™ confirmation emails by first identifying the provider and then using a database of provider-specific email structures to determine which information to extract. The API then returns a link to the JSON structure of the itinerary.

Add self-containing executable examples for the existing supported endpoints.

3.3.0 - 2019-12-04
--------------------
Add support for the `AI-Generated Photos`

The AI-Generated Photos API returns a link to download a rendered image of a landscape. The image size is 512x512 pixels and the currently available image categories are BEACH and MOUNTAIN. The link to download the AI-generated picture is valid for 24 hours. This API is an experimental project created by the Amadeus AI Lab using the Nvidia StyleGAN framework. This API is free to use and we welcome any feedback you may have about improvements.

Add support for the `Flight Delay Prediction <https://developers.amadeus.com/self-service/category/air/api-doc/flight-delay-prediction>`_

The Flight Delay Prediction API returns the probability that a given flight will be delayed by four possible delay lengths: less than 30 minutes, 30-60 minutes, 60-120 minutes and over 120 minutes/cancellation. The API receives flight information and applies a machine-learning model trained with Amadeus historical data to determine the probability of flight delay.

Release of the `Airport On-Time Performance <https://developers.amadeus.com/self-service/category/air/api-doc/airport-on-time-performance>`_

The Airport On-Time Performance API returns the estimated percentage of on-time flight departures for a given airport and date. The API receives the 3-letter IATA airport code and departure date and applies a machine-learning model trained with Amadeus historical data to estimate the overall airport on-time performance. This API is in currently in beta and only returns accurate data for airports located in the U.S.

3.2.0 - 2019-11-07
--------------------
Add support for the `Trip Purpose Prediction API <https://developers.amadeus.com/self-service/category/trip/api-doc/trip-purpose-prediction>`_

The Trip Purpose Prediction API returns the probability of whether a round-trip flight itinerary is for business or leisure travel. The API takes flight dates, departure city and arrival city and then applies a machine-learning model trained with Amadeus historical data to determine the probability that the itinerary is for business or leisure travel. This API is useful for gaining insight and optimizing the search and shopping experience.

Add support for the `Hotel Ratings API <https://developers.amadeus.com/self-service/category/hotel/api-doc/hotel-ratings>`_

The Hotel Ratings API provides hotel ratings based on automated sentiment analysis algorithm applied on the online reviews. Apart from an overall rating for a hotel also provides ratings for different categories of each (e.g.: staff, pool, internet, location). This provides a key content information for decision making during a shopping experience being able to compare how good a hotel is compared to others, sort hotels by ratings, filter by categories or recommend a hotel based on the trip context.

Release of the `Flight Choice Prediction API <https://developers.amadeus.com/self-service/category/air/api-doc/flight-choice-prediction>`_

The Flight Choice Prediction API allows developers to forecast traveler choices in the context of search & shopping. Exposing machine learning & AI services for travel, this API consumes the output of the Flight Low-fare Search API and returns augmented content with probabilities of choices for each flight offers.

3.1.0 - 2019-03-25
--------------------
Release of the `Points Of Interest API <https://developers.amadeus.com/self-service/category/210/api-doc/55>`_

The Points Of Interest API, powered by AVUXI TopPlace, is a search API that returns a list of popular places for a particular location. The location can be defined as area bound by four coordinates or as a geographical coordinate with a radius. The popularity of a place or 'point of interest' is determined by AVUXI's proprietary algorithm that considers factors such as ratings, check-ins, category scores among other factors from a host of online media sources.


3.0.0 - 2019-01-22
--------------------
**  Hotel Search v2 has been deployed (Hotel Search v1 is now deprecated) **

** General **
- Remove support of Hotel Search v1
- URLs for all three endpoints have been simplified for ease-of-use and consistency
** Find Hotels - 1st endpoint **
- The parameter `hotels` has been renamed to `hotelIds`
** View Hotel Rooms - 2nd endpoint **
- Update from `amadeus.shopping.hotel('IALONCHO').hotel_offers.get` to `amadeus.shopping.hotel_offers_by_hotel.get(hotelId: 'IALONCHO')`
- Now get all images in ‘View Hotels Rooms’ endpoint using the view parameter as `FULL_ALL_IMAGES`
** View Room Details - 3rd endpoint **
- Updated from `amadeus.shopping.hotel('IALONCHO').offer('XXX').get` to `amadeus.shopping.hotel_offer('XXX').get`
- Image category added under Media in the response
- Hotel distance added in the response
- Response now refers to the common HotelOffer object model

2.0.1 - 2019-01-17
--------------------

Fix pagination URL encoding parameters

2.0.0 - 2018-10-14
--------------------

`Flight Most Searched Destinations <https://developers.amadeus.com/self-service/category/203/api-doc/6>`_: Redesign of the API - Split the previous endpoint in 2 endpoints:

- 1st endpoint to find the most searched destinations
- 2nd endpoint to have more data about a dedicated origin & destination

`Flight Most Booked Destinations <https://developers.amadeus.com/self-service/category/203/api-doc/27>`_:

- Rename origin to originCityCode

`Flight Most Traveled Destinations <https://developers.amadeus.com/self-service/category/203/api-doc/7>`_:

- Rename origin in originCityCode

`Flight Check-in Links <https://developers.amadeus.com/self-service/category/203/api-doc/8>`_:

- Rename airline to airlineCode

`Airport & City Search <https://developers.amadeus.com/self-service/category/203/api-doc/10>`_:

- Remove parameter onlyMajor

`Airport Nearest Relevant <https://developers.amadeus.com/self-service/category/203/api-doc/9>`_:

- Add radius as parameter

`Airline Code Lookup <https://developers.amadeus.com/self-service/category/203/api-doc/26>`_:

- Regroup parameters *IATACode* and *ICAOCode* under the same name *airlineCodes*

1.1.0 - 2018-08-01
--------------------

Release 1.1.0

1.0.0 - 2018-04-20
--------------------

Release 1.0.0

1.0.0b8 - 2018-04-19
--------------------

Update namespace for `air_traffic/traveled` path.

1.0.0b7 - 2018-04-09
--------------------

Fix an issue where UTF8 was not properly decoded.

1.0.0b6 - 2018-04-05
--------------------

Set logging to silent by default

1.0.0b5 - 2018-04-05
--------------------

Adds easier to read error messages

1.0.0b4 - 2018-04-04
--------------------

Bug fix for install from PyPi

1.0.0b3 - 2018-04-05
--------------------

-  Renamed back to “amadeus”

1.0.0b2 - 2018-04-05
--------------------

-  Updated README for PyPi

1.0.0b1 - 2018-04-05
--------------------

-  Initial Beta Release
