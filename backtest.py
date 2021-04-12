from sklearn import linear_model
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd

def backtest(
        ivv_hist, bonds_hist, n, N, alpha, lot_size, start_date, end_date
):

    ivv_hist = pd.read_json(ivv_hist)
    bonds_hist = pd.read_json(bonds_hist)

    # Create the features data frame from the bond yields & IVV hist data

    # This function is what we'll apply to every row in bonds_hist.
    def bonds_fun(yields_row):
        maturities = pd.DataFrame([1 / 12, 2 / 12, 3 / 12, 6 / 12, 1, 2])
        linreg_model = linear_model.LinearRegression()
        linreg_model.fit(maturities, yields_row[1:])
        modeled_bond_rates = linreg_model.predict(maturities)
        return [yields_row["Date"].date(), linreg_model.coef_[0],
                linreg_model.intercept_,
                r2_score(yields_row[1:], modeled_bond_rates)]

    # apply bonds_fun to every row in bonds_hist to make the features dataframe.
    features = bonds_hist[
        ["Date", "1 mo", "2 mo", "3 mo", "6 mo", "1 yr", "2 yr"]
    ].apply(bonds_fun, axis=1,result_type='expand')
    features.columns = ["Date", "a", "b", "R2"]
    features['Date'] = pd.to_datetime(features['Date'])



    response = pd.DataFrame({'Date': ivv_hist['Date'], 'response': ""})

    for i in range(0, len(ivv_hist) - n + 1):
        response._set_value(
            i, 'response', ivv_hist['Open'][i]*(1+alpha) <= max(
                ivv_hist['High'][i:i+n]
            )
        )

    response = response[response['response']!=""]
    response['response'] = response['response'].astype(int)

    # here, I'm doing an inner merge on features and response and storing it
    # as dataframe x_y, which contains features & responses rows only for the
    # dates that both features & response have in common.
    # The reason is because the features dataframe derives from the bonds
    # data, and the response dataframe comes from IVV price history. However,
    # federal and NYSE holidays are not exactly the same, so there are some
    # days on which the federal government reports bond features but no
    # response exists, and vice versa.
    x_y = pd.merge(features, response, on = 'Date')

    entry_trades = []


    for trading_date in ivv_hist['Date'][
        ivv_hist['Date'] >= pd.to_datetime(start_date)
    ]:

        xy = x_y[x_y['Date'] <= trading_date]
        xy = xy.iloc[0:len(xy) - n]
        xy = xy.tail(N)

        if not any(trading_date == x_y['Date']):
            continue

        if sum(xy['response']) == 0 or sum(xy['response']) == N:
            trade_decision = bool(sum(xy['response']))
        else:
            logisticRegr = linear_model.LogisticRegression()
            logisticRegr.fit(
                np.float64(xy[["a", "b", "R2"]]),
                np.float64(xy["response"])
            )
            trade_decision = logisticRegr.predict(
                np.float64(
                    features[["a", "b", "R2"]][
                        features['Date'] == trading_date.strftime("%Y-%m-%d")
                        ]
                )
            )

            if trade_decision:
                fill_price = ivv_hist['Open'][ivv_hist['Date'] == trading_date]
                entry_trade = [
                    trading_date.date(), "BUY", "IVV", lot_size,
                    float(fill_price), "MARKET", "FILLED"
                    ]
                entry_trades.append(entry_trade)

    entry_trades = pd.DataFrame(
        entry_trades,
        columns=["Date", "Action", "Symbol", "Size", "Price", "Type", "Status"]
    )

    entry_trades['Cycle'] = "open"
    entry_trades['ID'] = entry_trades.index

    exit_trades = []

    for entry_trade in entry_trades.iterrows():

        prices_window = ivv_hist[ivv_hist['Date'] >= pd.to_datetime(
                    entry_trade[1]['Date']
                )].head(n)

        close_dt = prices_window['Date'].iloc[n-1].date()
        action = 'SELL'
        symbol = 'IVV'
        size = lot_size
        price = float(prices_window['Close'].tail(1))
        type = 'MARKET'
        status = 'FILLED'
        cycle = 'close'
        id = entry_trade[1]['ID']

        for trding_day in prices_window.iterrows():
            if trding_day[1]['High'] >= entry_trade[1]['Price']*(1+alpha):
                close_dt = trding_day[1]['Date'].date()
                price = float(entry_trade[1]['Price']*(1+alpha))
                type = 'LIMIT'
                break

        exit_trade = [
            close_dt, action, symbol, size, price, type, status, cycle, id
        ]
        exit_trades.append(exit_trade)

    exit_trades = pd.DataFrame(
        exit_trades,
        columns=[
            "Date", "Action", "Symbol", "Size", "Price", "Type", "Status",
            "Cycle", "ID"
        ]
    )

    blotter = pd.concat([entry_trades, exit_trades], axis=0)
    blotter = blotter.round(2)
    blotter.sort_values(by='Date', inplace=True)

    return blotter
