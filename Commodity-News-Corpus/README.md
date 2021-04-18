# Commodity-News-Corpus

This folder contains the files of the Commodity News Corpus, which is made up of a pair of files : a text file (.txt) containing the link to the news article and a annotation file (.ann) containing the annotation details. Pre-processing and post-processing codes are provided for the re-producibility of experiment results.

## Summary information
The dataset contains about 8,500 sentences, over 3,075 events, 21 Entity Types, 19 Event Types and 21 Argument Roles. More details can be found in subsections below.

### Entity Types
21 Entity types covering both named and nominal entities
  |      Entity Types     |                                  Examples                                   |
  |-----------------------|-----------------------------------------------------------------------------|
  | 1. Commodity          | *oil, crude oil, Brent, West Texas Intermediate (WTI), fuel, U.S Shale*     | 
  | 2. Country**          | *Libya, China, U.S., Venezuela, Greeze*                                     |
  | 3. Date**             | *1998, Wednesday, Jan. 30, the final quarter of 1991, the end of this year* |
  | 4. Duration**         | *two years, three-week, 5-1/2-year, multiyear, another six months*          |
  | 5. Economic Item      | *economy, economic growth, market, economic outlook, employment data*       |
  | 6. Financial attribute| *supply, demand, output, production, price, import, export*                 |
  | 7. Forecast target    | *forecast, target, estimate, projection, bets*                              |
  | 8. Group              | *global producers, oil producers, hedge funds, non-OECD, Gulf oil producers*|
  | 9. Location**         | *global, world, domestic, Middle East, Europe*                              |
  | 10. Money**           | *$60, USD 50*                                                               |
  | 11. Nationality**     | *Chinese, Russian, European, African*                                       |
  | 12. Number**          | (any numerical value that does not have a currency sign)                    |
  | 13. Organization**    | *OPEC, Organization of Petroleum Exporting Countries, EIA*                  |
  | 14. Other activities  | (free text)                                                                 |
  | 15. Percent**         | *25%, 1.4 percent*                                                          |
  | 16. Person**          | *Trump, Putin* (and other political figures)                                |
  | 17. Phenomenon        | (free text)                                                                 |
  | 18. Price Unit        | *$100-a-barrel, $40 per barrel, USD58 per barrel*                           |
  | 19. Production Unit   | *170,000 bpd, 400,000 barrels per day, 29 million barrels per day*          |
  | 20. Quantity          | *1.3500 million barrels, 1.8 million gallons, 18 million tonnes*            |
  | 21. State or province | *Washington, Moscow, Cushing, North America*                                |
  
### Event Types
19 event types
  1. **Geo-political News**
      - Civil unrest (*civil-unrest*):  Violence or turmoil within the oil producing country.
        * Example [1] *.....a fragile recovery in Libyan supply outweighed **fighting** in Iraq ......*
        * Example [2] *.......a backdrop of the worst **strife** in Iran this decade....*
      - Embargo (*Embargo* / *Prohibiting*): Trade or other commercial activity of the commodity is banned.
        * Example [1] *..... and **sanctions** against Iran.*
        * Example [2] *.....prepared to impose `` strong and swift '' economic **sanctions** on Venezuela.....*
      - Geo-political tension: Political tension between oil-producing nation with other nations. 
        * Example [1] *..... heightened **tensions** between the West and Russia.....* 
        * Example [2] *..... despite geopolitical **war** in Iraq , Libya and Ukraine.*
      - Trade tensions (*Trade-tensions*): Trade-related tension between oil-producing and oil-consuming nations. 
        * Example [1] *..... escalating global **trade wars**, especially between the US and China.*
        * Example [2] *.... showing that OPEC is not ready to end its **trade tensions**......*
      - Other forms of Crisis (*Crisis*): (a) A time of intense difficulty, such as other forms of unspecified crisis that do not fall into any of the above category and (b) Financial / Economic Crisis (which can be grouped under Macro-economic News)
        * Example for (a) **.... Ukraine declared an end to an oil **crisis** that has .........**
        * Example for (b) **....since the 2014/15 financial **crisis** as .......**
  2. **Macro-economic News**
      - Economy / GDP (*Grow-strong* / *Slow-weak*): Economic / GDP growth of a nation.
        * Example [1] *....... concerns over **slowing** global growth.....*
        * Example [2] *``Fear of domestic economic growth **contract** is afflicting .....* 
      - Employment (*Grow-strong* / *Slow-weak*): Status of US Employment Data, which is an indicator of economic situation. 
        * Example [1] *U.S. employment data **contrasts** with the euro zone.....*
        * Example [2] *as **strong** U.S. employment data.....*
      - Bearish technical view / outlook (*Negative-sentiment*): Bearish sentiment or outlook
        * Example [1] *But in a market **clouded by uncertainties**.....*
        * Example [2] *....supply concerns would ease even more ......*
  3. **Commodity Supply (includes exports)**
      - Oversupply (*Oversupply*): Situation where production goes into surplus.
        * Example [1] *..... the region **surplus** of supply.....*
        * Example [2] *....the market is still working off the **gluts** built up.....*
      - Shortage (*Shortage*): Situation where demand is more than supply.
        * Example [1] *.... increase a supply **shortage** from chaotic Libya....*
        * Example [2] *......and there is no **shortfall** in supply , the minister added.*
      - Supply increase (*Movement-up-gain*): Situation where supply increased.
        * Example [1] *....further **increases** in U.S. crude production.....*
        * Example [2] *The **rise** in production is definitely benefiting the United States....*
      - Supply increase (*Cause-movement-up-gain*): Deliberate action to increase supply.
        * Example [1] *The IEA **boosted** its estimate of production from ExxonMobil to 1.8 million bpd in July 4 holiday weekend.*
        * Example [2] *.....urged the kingdom to **ramp up** production.....*
      - Supply decrease (*Movement-down-loss*): Situation where supply decreased.
        * Example [1] *UAE 's production has almost **halved** in two years to 31.6 million bpd...*
        * Example [2] *...fears that global supplies will **drop** due to Washington 's sanctions on the OPEC member nation .* 
      - Supply decrease (*Cause-movement-down-loss*): Deliberate action to decrease supply. 
        * Example [1] *......by **slashing** production by almost three quarters in the 1980s....*
        * Example [2] *.....an announcement by Iran that it would **cut** its production last week.*
  4. **Commodity Demand (includes imports)**
      - Demand increase (*Movement-up-gain*): Situation where demand increased.
        * Example [1] *It expects consumption to **trend upward** by 1.05 million bpd , below 40,000 bpd from July .*
        * Example [2] *....as **more** seasonal demand kicks in due to colder weather.*
      - Demand decrease (*Movement-down-loss*): Situation where demand decreased.
        * Example [1] *....onto a market reeling from **falling** demand because of the virus outbreak.*
        * Example [2] *....when global demand growth for air conditioning **collapses** from its summer peak....*
  5. **Commodity Price Movement** (Commodity price here includes *spot price*, *futures* and *futures contract*.)
      - Price increase (*Movement-up-gain*): Situation where commodity price rises.
        * Example [1] **
        * Example [2] **
      - Price decrease (*Movement-down-loss*): Situation where commodity price drops.
        * Example [1] *The **drop** in oil prices to their lowest in two years.....7.*
        * Example [2] *Oil prices **declined** back the final quarter of 1991 to 87 cents....*
      - Price movement flat (*Movement-flat*): Situation where no or little change to commodity price.
        * Example [1] *Contango spread in Brent is **steady** at 15 cents per barrel.....*
        * Example [2] *U.S. crude is expected to **hold** around $ 105 per barrel , Spooner forecast .*
      - Price target /forecast increase (*Caused-movement-up-gain*): Commodity forecasted / target price is raised.
        * Example [1] *The IMF earlier said it **increased** its 2019 global economic growth forecast to 3.30%*
        * Example [2] *Tthe International Monetary Fund **doubled** its global growth forecast for 2013.....*
      - Price target /forecast decrease (*Caused-movement-down-loss*): Commodity forecasted / target price is lowered.
        * Example [1] *Germany 's Bundesbank this week **halved** its 2015 growth forecasts for Europe 's largest economy to 1 percent.*
        * Example [2] *OPEC also **lowered** forecast global demand for its crude oil....*
      - Price position (*Position-high*, *Position-low*): Describes the position of the current commodity price.
        * Example [1] *Oil price remained close to four-year **highs**....*
        * Example [2] *Oil slipped more than 20% to its **weakest level** in two years on 1980s.....*

There are 3,075 events and their distributions are as follows:
  |      Event Type                                   |        Ration        |     # Sentence      |
  |---------------------------------------------------|----------------------|---------------------|
  | 1. Cause-movement-down-loss                       |        14.9%         |        457          |
  | 2. Cause-movement-up-gain                         |           2%         |         63          |
  | 3. Civil-unrest                                   |         2.6%         |         79          |
  | 4. Crisis                                         |         1.2%         |         36          |
  | 5. Embargo                                        |         4.8%         |        148          |
  | 6. Geopolitical-tensions                          |           2%         |         63          |
  | 7. Grow-strong                                    |           6%         |        183          |
  | 8. Movement-down-loss                             |          24%         |        753          |
  | 9. Movement-flat                                  |         2.6%         |         80          |
  | 10. Movement-up-gain                              |          15%         |        461          |
  | 11. Negative-sentiment                            |        4.07%         |        125          |
  | 12. Oversupply                                    |         3.8%         |        116          |
  | 13. Position-high                                 |        3.06%         |         94          |
  | 14. Position-low                                  |        3.58%         |        110          |
  | 15. Prohibiting                                   |         0.9%         |         28          |
  | 16. Shortage                                      |           1%         |         31          |
  | 17. Situation-deteriorate                         |         1.1%         |         35          |
  | 18. Slow-weak                                     |        5.79%         |        178          |
  | 19. Trade-tensions                                |         1.7%         |         53          |

### Argument Roles
21 Argument roles - 

Please refer to the annotation guide for list of events and its corresponding list of argument roles.

For complete list of the above, please refer to **Event Annotation Guidelines.pdf**.

The diagram below shows the annotation details using the tool called Brat Annotation Tool.

![Annotation](brat_annotation.png)


## Preprocessing
In data pre-processing, the annotation information in Brat standoff format (.ann file) is combined with the text (.txt file) to produce a corresponding .json file as input to the event extraction model.
