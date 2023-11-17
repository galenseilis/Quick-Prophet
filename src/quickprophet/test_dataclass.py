from dataclasses import dataclass

import pandas as pd
import numpy as np

import impute

@dataclass(order=True, frozen=True)
class InventoryItem:
    """Class for keeping track of an item in inventory."""
    name: str
    unit_price: float
    quantity_on_hand: int = 0


dates = pd.date_range('2018-11-23', '2018-12-01')

d = {'ds':dates,
     'y':np.random.random(size=dates.size),
     'forecast_group':[('a', InventoryItem('b', np.random.random(), 1)) for i in range(dates.size)]
     }

df = pd.DataFrame(d)
