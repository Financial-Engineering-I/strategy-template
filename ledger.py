import pandas as pd
from math import log

def calc_calendar_ledger(blotter, starting_cash, ivv_hist, start_date):

    calendar_ledger = []
    cash = starting_cash
    position = 0

    for ivv_row in ivv_hist[
        ivv_hist['Date'] >= pd.to_datetime(start_date)
    ].iterrows():
        ivv_row = ivv_row[1]
        trading_date = ivv_row['Date']
        ivv_close = ivv_row['Close']
        trades = blotter[
            (blotter['filled_or_cancelled'] == trading_date) & (
                    blotter['status'] == 'FILLED'
            )]

        if len(trades) > 0:
            position = position + sum(
                trades['size'][trades['action'] == 'BUY']
            ) - sum(
                trades['size'][trades['action'] == 'SELL']
            )
            cash = cash - sum(
                trades['size'][trades['action'] == 'BUY'] *
                trades['fill_price'][
                    trades['action'] == 'BUY'
                    ]
            ) + sum(
                trades['size'][trades['action'] == 'SELL'] *
                trades['fill_price'][
                    trades['action'] == 'SELL'
                    ]
            )
            stock_value = position * ivv_close
            total_value = cash + stock_value
        else:
            stock_value = position * ivv_close
            total_value = cash + stock_value

        ledger_row = [
            trading_date, position, ivv_close, cash, stock_value, total_value
        ]
        calendar_ledger.append(ledger_row)

    calendar_ledger = pd.DataFrame(calendar_ledger)
    calendar_ledger.columns = [
        'Date', 'position', 'ivv_close', 'cash', 'stock_value', 'total_value'
    ]
    calendar_ledger.round(2)

    calendar_ledger.to_csv('app_data/calendar_ledger.csv', index=False)

    return calendar_ledger

def calc_trade_ledger(blotter, ivv_hist):

    trade_ledger = []

    for trade in blotter['ID'].unique():

        round_trip_trade = blotter[
            (blotter['ID'] == trade) & (blotter['status'] == 'FILLED')
            ]

        if len(round_trip_trade) < 2:
            continue

        print(round_trip_trade)

        trade_id = round_trip_trade['ID'].unique().item()

        date_opened = min(round_trip_trade['filled_or_cancelled'])
        date_closed = max(round_trip_trade['filled_or_cancelled'])

        ivv_df = ivv_hist[
            (ivv_hist['Date'] <= date_closed) & \
            (ivv_hist['Date'] >= date_opened)
        ]

        print(ivv_df)

        buy_price = round_trip_trade['fill_price'][
            round_trip_trade['action'] == 'BUY'
            ].item()
        sell_price = round_trip_trade['fill_price'][
            round_trip_trade['action'] == 'SELL'
            ].item()

        ivv_price_enter = ivv_df['Open'][
            ivv_df['Date'] == date_opened
        ].item()
        ivv_price_exit  = ivv_df['Close'][
            ivv_df['Date'] == date_closed
        ].item()

        trade_rtn = log(sell_price / buy_price)
        ivv_rtn = log(ivv_price_exit / ivv_price_enter)

        trading_days_open = len(ivv_df)

        trade_rtn_per_trading_day = trade_rtn / trading_days_open
        benchmark_rtn_per_trading_day = ivv_rtn / trading_days_open

        trade_ledger_row = [
            trade_id, date_opened, date_closed, trading_days_open, buy_price,
            sell_price, ivv_price_enter, ivv_price_exit, trade_rtn, ivv_rtn,
            trade_rtn_per_trading_day, benchmark_rtn_per_trading_day
        ]

        trade_ledger.append(trade_ledger_row)

    trade_ledger = pd.DataFrame(trade_ledger)
    trade_ledger.columns = [
        'trade_id', 'open_dt', 'close_dt', 'trading_days_open', 'buy_price',
        'sell_price', 'benchmark_buy_price', 'benchmark_sell_price',
        'trade_rtn', 'benchmark_rtn', 'trade_rtn_per_trading_day',
        'benchmark_rtn_per_trading_day'
    ]

    trade_ledger.to_csv('app_data/trade_ledger.csv', index=False)

    return trade_ledger
