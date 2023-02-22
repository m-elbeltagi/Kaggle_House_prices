## testing file for random stuff

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder



## after messing around with ordinal encoding for a bit, this works:

# ord_dict = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}


# ordinal_encoder = OrdinalEncoder()


# myarr = pd.DataFrame(['Ex', 'Fa', 'Gd', 'TA'], columns = ['quality'])

# print (myarr)

# myarr['quality'] = myarr.quality.map(ord_dict)

# print (myarr)


# print (ordinal_encoder.categories_)

############################################################################################



print (list({1, 5, 7, 9, 3}&{3}))