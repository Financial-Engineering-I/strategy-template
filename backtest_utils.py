import pandas._libs.tslibs.nattype
from sklearn import linear_model
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from math import log, isnan
from statistics import stdev
from numpy import repeat
from strategy import *

def calc_features(ivv_hist, bonds_hist, n_vol):
    # Takes in:
    # ivv_hist, a pandas dataframe of OHLC data with a Date column
    # bonds_hist, a CMT rates dataframe of this form:
    #   https://www.treasury.gov/resource-center/data-chart-center
    #       /interest-rates/pages/textview.aspx?data=yield.
    # n_vol: number of trading days over which vol for IVV is to be calculated

    # This function is what we'll apply to every row in bonds_hist.
    def bonds_fun(yields_row):
        maturities = pd.DataFrame([1 / 12, 2 / 12, 3 / 12, 6 / 12, 1, 2])
        linreg_model = linear_model.LinearRegression()
        linreg_model.fit(maturities, yields_row[1:])
        modeled_bond_rates = linreg_model.predict(maturities)
        return [yields_row["Date"], linreg_model.coef_[0],
                linreg_model.intercept_,
                r2_score(yields_row[1:], modeled_bond_rates)]

    # apply bonds_fun to every row in bonds_hist to make the features dataframe.
    bonds_features = bonds_hist[
        ["Date", "1 mo", "2 mo", "3 mo", "6 mo", "1 yr", "2 yr"]
    ].apply(bonds_fun, axis=1, result_type='expand')
    bonds_features.columns = ["Date", "a", "b", "R2"]
    bonds_features['Date'] = pd.to_datetime(bonds_features['Date']).dt.date

    # Get available volatility of day-over-day log returns based on closing
    # prices for IVV using a window size of N days.
    ivv_features = []

    for dt in ivv_hist['Date'][n_vol:]:
        eod_close_prices = list(
            ivv_hist['Close'][ivv_hist['Date'] <= dt].tail(n_vol))
        vol = stdev([
            log(i / j) for i, j in zip(
                eod_close_prices[:n_vol - 1], eod_close_prices[1:]
            )
        ])
        vol_row = [dt, vol]
        ivv_features.append(vol_row)

    ivv_features = pd.DataFrame(ivv_features)
    ivv_features.columns = ["Date", "ivv_vol"]
    ivv_features['Date'] = pd.to_datetime(ivv_features['Date']).dt.date

    # here, I'm doing an inner merge on features from IVV and the bond rates,
    # storing the result in a dataframe called 'features'.
    # The reason is because federal and NYSE holidays are not exactly the same, so
    # there are some days on which the federal government reports bond features
    # but no IVV data exists, and vice versa.
    features = pd.merge(bonds_features, ivv_features, on='Date')

    # save the csv
    features.to_csv('app_data/features.csv', index=False)

    return features


def calc_long_response(features, ivv_hist, n, alpha):

    response = []

    for features_dt in features['Date']:
        # Get data for the next n days after response_date
        ohlc_data = ivv_hist[['Date', 'Open', 'High', 'Low', 'Close']][
            ivv_hist['Date'] > features_dt
            ].head(n)

        if len(ohlc_data) == 0:
            response_row = repeat(None, 8).tolist()
            response.append(response_row)
            continue

        entry_date   = ohlc_data['Date'].head(1).item()
        entry_price  = ohlc_data['Open'].head(1).item()
        target_price = entry_price * (1 + alpha)

        # Find the earliest date and price in ohlc_data on which the high was
        # higher than target_price (meaning that the SELL limit order filled), if
        # they exist. If not, then return the date and closing price of the last
        # row in ohlc_data.
        (exit_date, exit_price) = next(
            (tuple(y) for x, y in
             ohlc_data[['Date', 'High']].iterrows() if
             y[1] >= target_price),
            (ohlc_data['Date'].values[-1], ohlc_data['Close'].values[-1])
        )

        highest_price = max(ohlc_data['High'])
        highest_price_date = ohlc_data['Date'][
            ohlc_data['High'] == highest_price
            ].values[0]

        success = int(exit_price >= target_price)

        if len(ohlc_data) < n and success == 0:
            exit_date = exit_price = highest_price = highest_price_date = \
                success = None

        response_row = [
            entry_date, entry_price, target_price, exit_date, exit_price,
            highest_price, highest_price_date, success
        ]

        response.append(response_row)

    response = pd.DataFrame(response)
    response.columns = [
        "entry_date", "entry_price", "target_price", "exit_date", "exit_price",
        "highest_price", "highest_price_date", "success"
    ]
    response = response.round(2)

    response.to_csv('app_data/response.csv', index=False)

    return response

def calc_blotter(features_and_responses, start_date, end_date, n, N, lot_size):
    # Build the blotter.
    blotter = []
    trade_id = 0

    for trading_date in features_and_responses['Date'][
        (features_and_responses['Date'] >= pd.to_datetime(start_date)) & (
                features_and_responses['Date'] <= pd.to_datetime(end_date)
        )
    ]:

        trade_decision = strategy(
            'exit_date', 'success', features_and_responses,
            trading_date, N, n
        )

        if trade_decision == 1:
            right_answer = features_and_responses[
                features_and_responses['Date'] == trading_date
                ]

            # Create a market order to enter.
            if trading_date == features_and_responses['Date'].tail(1).item():
                order_status = 'PENDING'
                submitted = order_price = fill_price = filled_or_cancelled = None
            else:
                submitted = filled_or_cancelled = right_answer[
                    'entry_date'].item()
                order_price = fill_price = right_answer['entry_price'].item()
                order_status = 'FILLED'

            entry_trade_mkt = [
                trade_id, 'L', submitted, 'BUY', lot_size, 'IVV',
                order_price, 'MKT', order_status, fill_price,
                filled_or_cancelled
            ]

            blotter.append(entry_trade_mkt)

            # Create the limit order to exit position.

            # Either successful (success == 1), unsuccessful (0), or it's
            # unsuccessful so far but n days haven't passed yet (NaN).
            success = right_answer['success'].item()

            # If unknown, then the limit order is still open and hasn't filled.
            if isnan(success):
                if trading_date == features_and_responses['Date'].tail(
                        1).item():
                    order_status = 'PENDING'
                else:
                    order_status = 'OPEN'
                order_price = right_answer['target_price'].item()
                fill_price = filled_or_cancelled = None

            # If limit order failed after n days:
            if success == 0:
                order_status = 'CANCELLED'
                order_price = right_answer['target_price'].item()
                fill_price = None
                filled_or_cancelled = right_answer['exit_date'].item()
                # Don't forget the market order to close position:
                exit_trade_mkt = [
                    trade_id, 'L', submitted, 'SELL', lot_size, 'IVV',
                    right_answer['exit_price'].item(), 'MKT', 'FILLED',
                    right_answer['exit_price'].item(), filled_or_cancelled
                ]
                blotter.append(exit_trade_mkt)

            # If the trade was successful:
            if success == 1:
                order_status = 'FILLED'
                order_price = right_answer['target_price'].item()
                fill_price = right_answer['exit_price'].item()
                filled_or_cancelled = right_answer['exit_date'].item()

            exit_trade_lmt = [
                trade_id, 'L', submitted, 'SELL', lot_size, 'IVV',
                order_price, 'LIMIT', order_status, fill_price,
                filled_or_cancelled
            ]

            blotter.append(exit_trade_lmt)
            trade_id += 1

    blotter = pd.DataFrame(blotter)
    blotter.columns = [
        'ID', 'ls', 'submitted', 'action', 'size', 'symbol', 'price', 'type',
        'status', 'fill_price', 'filled_or_cancelled'
    ]
    blotter = blotter[
        (blotter['submitted'] >= pd.to_datetime(start_date)) & (
                blotter['submitted'] <= pd.to_datetime(end_date)
        )
    ]
    blotter = blotter.round(2)
    blotter.sort_values(by='ID', inplace=True, ascending=False)
    blotter.reset_index()

    blotter.to_csv('app_data/blotter.csv', index=False)

    return blotter
