import numpy as np
import pandas as pd
import datetime as dt
import utility.tax_library as tx
from utility.PricesClass import Prices
from utility.tax_log import log


def soglia(balances: pd.DataFrame, prices: Prices, year_sel=None) -> pd.DataFrame:
    """"Funzione per calcolare il valore delle posizioni con il prezzo al primo gennaio"""

    balances_soglia = balances.copy()

    for anno in np.unique([k.year for k in balances.index]):
        balance_anno = balances[balances.index >= dt.date(anno, 1, 1)]
        balance_anno = balance_anno[balance_anno.index <= dt.date(anno, 12, 31)]
        balance_anno.columns = [x.upper() for x in balance_anno.columns]
        balances.columns = [x.upper() for x in balances.columns]
        for coin in balances.columns:
            if coin.upper() not in (prices.prices['EUR'].keys()):
                prices.convert_prices('EUR')
            if prices.prices['EUR'][coin] is None:
                conv = [0]
            else:
                conv = [x[1] for x in prices.prices['EUR'][coin] if x[0] == dt.date(anno, 1, 1)]
            if len(conv) == 0:
                conv = [prices.prices['EUR'][coin][0][1]]
            balance_anno[coin] = [x * conv[0] for x in list(balance_anno[coin])]
        balances_soglia = pd.concat([balances_soglia, balance_anno])

    if year_sel is not None:
        temp_df = balances_soglia[balances_soglia.index >= dt.date(year_sel, 1, 1)]
        temp_df = temp_df[temp_df.index <= dt.date(year_sel, 12, 31)]
        return temp_df
    else:
        return balances_soglia


def balances_fiat(balances: pd.DataFrame, prices: Prices, currency='eur', year_sel=None):
    balances_in = balances.copy()
    prices_df = prices.to_pd_dataframe(currency)
    prices_df = prices_df[prices_df.index >= balances_in.index[0]]
    for coin in list(balances_in.columns):
        if coin not in prices_df.columns:
            prices_df[coin] = 0
        out_ser = balances_in[coin] * prices_df[coin]
        out_ser.dropna(inplace=True)
        balances_in[coin] = out_ser

    if year_sel is not None:
        temp_df = balances_in[balances_in.index >= dt.date(year_sel, 1, 1)]
        temp_df = temp_df[temp_df.index <= dt.date(year_sel, 12, 31)]
        return temp_df
    else:
        return balances_in


def prepare_df(df_in: pd.DataFrame, year_sel=None, cummulative=True):
    # Il df dev'essere un dataframe con index orario senza NaN
    df = df_in.copy()
    df.index = [k.date() for k in df.index]
    temp_df = df.groupby(df.index).sum()

    fill_na_index = pd.date_range(dt.date(min(df.index).year, 1, 1), dt.date.today() - dt.timedelta(days=1))
    fill_na_index = [tx.str_to_datetime(a.date().isoformat()).date() for a in fill_na_index]
    fill_na = pd.DataFrame(index=fill_na_index, data=np.zeros([len(fill_na_index), 1]))
    temp_df = temp_df.join(fill_na, how='outer')
    temp_df.drop([0], axis=1, inplace=True)
    temp_df.fillna(0, inplace=True)
    if cummulative:
        temp_df = temp_df.cumsum()
    if year_sel is not None:
        temp_df = temp_df[temp_df.index >= dt.date(year_sel, 1, 1)]
        temp_df = temp_df[temp_df.index <= dt.date(year_sel, 12, 31)]

    temp_df[temp_df < 10 ** -9] = 0
    return temp_df


def income(transactions: pd.DataFrame, type_out='fiat', cummulative=True, year_sel=None, name=None):
    # Obtain daily income (earn products/supercharge) in cryptocurrency of native fiat
    rendita = transactions[transactions['Tag'].isin(['Reward', 'Interest'])].copy()
    if rendita.shape[0] == 0:
        if type_out == 'fiat':
            log.info(f'No income for {name}')
        return pd.DataFrame()
    temp_df_fiat = pd.DataFrame()
    temp_df_token = pd.DataFrame()
    for index, tok in enumerate(np.unique(rendita['Coin'])):
        temp_df = rendita[rendita['Coin'] == tok]
        if tok == '':
            continue
        if index == 0:
            temp_df_token = pd.DataFrame(temp_df['Amount'])
            temp_df_fiat = pd.DataFrame(temp_df['Fiat Price'])
            temp_df_fiat.columns = temp_df_token.columns = [tok]
        else:
            colnames = list(temp_df_fiat.columns)
            colnames.append(tok)
            temp_df_token = temp_df_token.join(pd.DataFrame(temp_df['Amount']), how='outer')
            temp_df_fiat = temp_df_fiat.join(pd.DataFrame(temp_df['Fiat Price']), how='outer')
            temp_df_fiat.columns = temp_df_token.columns = colnames

    temp_df_fiat.fillna(0, inplace=True)
    temp_df_token.fillna(0, inplace=True)
    if year_sel is not None:
        temp_df_fiat[temp_df_fiat.index < dt.datetime(year_sel, 1, 1, 0, 0, 0)] = 0
        temp_df_token[temp_df_token.index < dt.datetime(year_sel, 1, 1, 0, 0, 0)] = 0
    if type_out == 'fiat':
        return prepare_df(temp_df_fiat, year_sel, cummulative)
    else:
        return prepare_df(temp_df_token, year_sel, cummulative)


def balances(transactions: pd.DataFrame, cummulative=True, year_sel=None):
    # Obtain daily balances in native cryptocurrency
    from_df = transactions[['Coin', 'Amount']].copy()
    to_df = transactions[['To Coin', 'To Amount']].copy()
    to_df.columns = ['Coin', 'Amount']

    fees = transactions[['Fee Currency', 'Fee']].copy()
    fees.columns = ['Coin', 'Amount']

    balance_df = pd.concat([from_df, to_df, fees], axis=0)
    balance_df = balance_df[balance_df['Coin'] != 'EUR']
    balance_df = balance_df[balance_df != '']

    balance_df.dropna(inplace=True)
    currencies = list(np.unique(balance_df['Coin']))
    temp_df = pd.DataFrame()
    for index, currency in enumerate(currencies):
        if index == '':
            continue
        if index == 0:
            temp_df = pd.DataFrame(balance_df.loc[balance_df['Coin'] == currency, 'Amount'])
            temp_df.index = [t + dt.timedelta(milliseconds=100) for t in temp_df.index]
            temp_df.sort_index(inplace=True)
            temp_df.columns = [currency]
        else:
            colnames = list(temp_df.columns)
            colnames.append(currency)
            temp_df.index = [t + dt.timedelta(milliseconds=100) for t in temp_df.index]
            temp_df = temp_df.join(pd.DataFrame(balance_df.loc[balance_df['Coin'] == currency, 'Amount']), how='outer')
            temp_df.columns = colnames

    return prepare_df(temp_df, year_sel, cummulative)
