from sklearn import linear_model
from sklearn.metrics import r2_score
import pickle
import numpy as np
import pandas as pd

def calc_ledger(blotter, prices):

    pickle.dump(blotter, open("blotter.p", "wb"))
    pickle.dump(prices, open("prices.p", "wb"))

    # blotter = pickle.load(open("blotter.p", "rb"))


    # blotter = pd.DataFrame(blotter)



    return 'asdf'