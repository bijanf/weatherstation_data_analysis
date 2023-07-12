# Weatherstation data analysis
This is a Jupter Notbook which tries to explore the long-term climate change signal at one of the oldest weather stations of the world, i.e. Säkularstation Potsdam Telegraphenberg

## Getting started
You could easily clone the project and use the code and of-course add your own analysis. 

### Prerequisites 
The Jupyter notebook imports the following libraries : 
```Python
import pandas as pd
import matplotlib.pyplot as plt
from meteostat import Stations, Daily, Hourly
from datetime import datetime,date , timedelta
```
The list of the climate station could be found for Germany here : https://www.dwd.de/DE/leistungen/klimadatendeutschland/stationsuebersicht.html?nn=16102&lsbId=343278
### Installing
For installing the meteostat library you could use the pip commdad: 
```Bash
pip instal meteostat
```
## Authors

* **Bijan Fallah** - *Initial work* - [bijanf](https://github.com/bijanf)
