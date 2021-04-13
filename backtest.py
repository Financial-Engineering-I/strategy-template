from utils import *
from backtest_utils import *
from ledger import *

def backtest(
        ivv_hist, bonds_hist, n, N, alpha, lot_size, start_date, end_date,
        starting_cash
):

    # Convert JSON data to dataframes
    ivv_hist   = quick_date_cols(pd.read_json(ivv_hist), ['Date'])
    bonds_hist = quick_date_cols(pd.read_json(bonds_hist), ['Date'])

    # Create the features data frame from the bond yields & IVV hist data
    # Create the response information: describes what would have happened in
    # the past if the strategy had been applied indiscriminately every day.
    features = calc_features(ivv_hist, bonds_hist, N)
    response = calc_long_response(features, ivv_hist, n, alpha)

    features_and_responses = pd.concat([features, response], axis=1)

    del features
    del response

    blotter = calc_blotter(
        features_and_responses, start_date, end_date, n, N, lot_size
    )

    calendar_ledger = calc_calendar_ledger(
        blotter, starting_cash, ivv_hist, start_date
    )

    trade_ledger = calc_trade_ledger(blotter, ivv_hist)

    return features_and_responses, blotter, calendar_ledger, trade_ledger
