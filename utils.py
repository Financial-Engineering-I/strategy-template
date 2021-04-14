# For utilities (helper functions)
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import date, timedelta
from math import ceil

def quick_date_cols(df, colnames=['Date']):
    for col in colnames:
        df[col] = pd.to_datetime(df[col]).dt.date
    return df

def date_to_str(date_obj, format = "%Y-%m-%d"):
    return date.strftime(pd.to_datetime(date_obj).date(), format)

def to_years(x):
    str_split = x.lower().split()
    if len(str_split) == 2:
        if str_split[1] == 'mo':
            return int(str_split[0]) / 12
        if str_split[1] == 'yr':
            return int(str_split[0])

def fetch_usdt_rates(YYYY):
    # Requests the USDT's daily yield data for a given year. Results are
    #   returned as a DataFrame object with the 'Date' column formatted as a
    #   pandas datetime type.

    URL = 'https://www.treasury.gov/resource-center/data-chart-center/' + \
          'interest-rates/pages/TextView.aspx?data=yieldYear&year=' + str(YYYY)

    cmt_rates_page = requests.get(URL)

    soup = BeautifulSoup(cmt_rates_page.content, 'html.parser')

    table_html = soup.findAll('table', {'class': 't-chart'})

    df = pd.read_html(str(table_html))[0]
    df.Date = pd.to_datetime(df.Date)

    return df

def usdt_cmt_rates(startDate, N, n, endDate):

    startDate = '2015-01-01'
    endDate   = '2021-04-14'
    N         = 10
    n         = 5

    old_bonds_data = quick_date_cols(pd.read_csv('usdt_data/cmt_rates.csv'))

    # Need to query enough days to run the backtest on every date in the
    # range start_date to end_date, so make startDate a little earlier in time
    startDate = pd.to_datetime(startDate).date() - timedelta(
        days=ceil((N + n) * (365 / 252))
    )
    endDate   = pd.to_datetime(endDate).date()

    if max(old_bonds_data.Date) >= endDate and \
            min(old_bonds_data.Date) <= startDate:
        return old_bonds_data[
            (old_bonds_data.Date >= startDate) & (
                    old_bonds_data.Date <= endDate
            )]

    data_years = list(
        range(startDate.year, endDate.year + 1, 1)
    )

    bonds_data = fetch_usdt_rates(data_years[0])

    if len(data_years) > 1:
        for year in data_years[1:]:
            bonds_data = pd.concat([bonds_data, fetch_usdt_rates(year)],
                                   axis=0, ignore_index=True)

    bonds_data['Date'] = pd.to_datetime(bonds_data['Date']).dt.date
    bonds_data.to_csv('usdt_data/cmt_rates.csv', index=False)

    # How to filter a dataframe for rows that you want
    bonds_data = bonds_data[
        (bonds_data.Date >= pd.to_datetime(startDate)) & (
                bonds_data.Date <= pd.to_datetime(endDate)
        )]







def Y_m_d_to_unix_str(ymd_str):
    return str(int(time.mktime(pd.to_datetime(ymd_str).date().timetuple())))

def fetch_GSPC_data(start_date, end_date):
    # Requests the USDT's daily yield data for a given year. Results are
    #   returned as a DataFrame object with the 'Date' column formatted as a
    #   pandas datetime type.

    URL = 'https://finance.yahoo.com/quote/%5EGSPC/history?' + \
            'period1=' + Y_m_d_to_unix_str(start_date) + \
            '&period2=' + Y_m_d_to_unix_str(end_date) + \
            '&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true'

    gspc_page = requests.get(URL)

    soup = BeautifulSoup(gspc_page.content, 'html.parser')

    table_html = soup.findAll('table', {'data-test': 'historical-prices'})


    df = pd.read_html(str(table_html))[0]

    df.drop(df.tail(1).index, inplace=True)

    # see formats here: https://www.w3schools.com/python/python_datetime.asp
    df.Date = pd.to_datetime(df.Date)

    return df

