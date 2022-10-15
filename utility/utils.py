import numpy as np
import pandas as pd
import datetime as dt
import utility.tax_library as tx
from utility.PricesClass import Prices
from utility.tax_log import log

fiat_list = ["AUD", "BRL", "EUR", "GBP", "GHS", "HKD", "KES", "KZT", "NGN", "NOK", "PHP", "PEN", "RUB",
             "TRY", "UGX",
             "UAH", ""]


def soglia(balances_in: pd.DataFrame, prices: Prices, year_sel=None) -> pd.DataFrame:
    """"Funzione per calcolare il valore delle posizioni con il prezzo al primo gennaio"""

    balances_soglia = balances_in.copy()
    temp_df = pd.DataFrame()
    for anno in np.unique([k.year for k in balances_in.index]):
        balance_anno = balances_in[balances_in.index >= dt.date(anno, 1, 1)]
        balance_anno = balance_anno[balance_anno.index <= dt.date(anno, 12, 31)]
        balance_anno.columns = [x.upper() for x in balance_anno.columns]
        balances_in.columns = [x.upper() for x in balances_in.columns]
        for coin in balances_in.columns:
            if coin.upper() not in (prices.prices['EUR'].keys()):
                prices.convert_prices('EUR')
            if prices.prices['EUR'][coin] is None:
                conv = [0]
            else:
                conv = [x[1] for x in prices.prices['EUR'][coin] if x[0] == dt.date(anno, 1, 1)]
            if len(conv) == 0:
                conv = [prices.prices['EUR'][coin][0][1]]
            balance_anno[coin] = [x * conv[0] for x in list(balance_anno[coin])]
        temp_df = pd.concat([temp_df, balance_anno])
    return temp_df

def balances_fiat(balances: pd.DataFrame, prices: Prices, currency='eur', year_sel=None):
    balances_in = balances.copy()
    balances_in.columns = [x.upper() for x in balances_in.columns]
    prices_df = prices.to_pd_dataframe(currency)
    prices_df = prices_df[prices_df.index >= balances_in.index[0]]
    for coin in list(balances_in.columns):
        if coin not in prices_df.columns:
            prices_df[coin] = 0
        out_ser = balances_in[coin] * prices_df[coin]
        out_ser.dropna(inplace=True)
        balances_in[coin] = out_ser

    if year_sel is not None:
        temp_df = balances_in[balances_in.index >= dt.date(year_sel, 1, 1)].copy()
        temp_df = temp_df[temp_df.index <= dt.date(year_sel, 12, 31)].copy()
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
    temp_df = temp_df.loc[:, ~temp_df.columns.isin(list(temp_df.sum(axis=0)[temp_df.sum(axis=0) == 0].index))]

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


def evm_liquidity_swap_calculation(outdf: pd.DataFrame, address: str):
    """The input df should be the DF after downloaoding erc20 transactions"""
    outdf['isLiquidity'] = [1 if 'LP' in x else 0 for x in list(outdf['Coin'])]
    outdf.sort_index(inplace=True)

    duplicated_index = np.unique(outdf.index[outdf.index.duplicated(keep='first')])
    liquidity_provider = {}
    for index in duplicated_index:
        temp = outdf.loc[[index], :].copy()
        if address in list(temp['From']) and address in list(temp['To']) and \
                temp.loc[temp['isLiquidity'] == 1, 'Amount'].shape[0] == 0 \
                and temp[temp['Amount'] == 0].shape[0] == 0:  # Swap
            temp['From'] = temp['To'] = 'Swap'
            temp['To Coin'] = temp['Coin'][1]
            temp['To Amount'] = temp['Amount'][1]
            outdf.drop([pd.to_datetime(index)], axis=0, inplace=True)
            outdf = pd.concat([outdf, temp.iloc[[0], :]], axis=0)
        elif temp['isLiquidity'].sum() > 0:  # Liquidity pool
            if list(temp.loc[temp['isLiquidity'] == 1, 'Amount'])[0] > 0 and \
                    address not in list(temp.loc[temp['isLiquidity'] != 1, 'To']) and \
                    sum(list(temp.loc[temp['isLiquidity'] == 0, 'Amount'] / abs(temp.loc[temp[
                                                                                             'isLiquidity'] == 0, 'Amount']))) < 0:  # I am adding liquidity, ignore this movement
                temp1 = temp.loc[temp['isLiquidity'] == 0, ['Coin', 'Amount']]
                temp['From'] = temp['To'] = 'Adding Liquidity'
                for coin_l in temp1['Coin']:
                    liquidity_provider[coin_l] = abs(list(temp1.loc[temp1['Coin'] == coin_l, 'Amount'])[0])
                temp = temp.iloc[[0], :]
                temp['Amount'] *= 0
                outdf.drop([pd.to_datetime(index)], axis=0, inplace=True)
                outdf = pd.concat([outdf, temp], axis=0)
            elif list(temp.loc[temp['isLiquidity'] == 1, 'Amount'])[0] < 0 and address not in list(temp['To']) \
                    and temp[temp['Amount'] < 0].shape[0] == 1:  # I am putting LP in LP farm, ignore this
                temp['From'] = temp['To'] = 'Adding LP token in farm'
                temp = temp.iloc[[0], :]
                temp['Amount'] *= 0
                outdf.drop([pd.to_datetime(index)], axis=0, inplace=True)
                outdf = pd.concat([outdf, temp], axis=0)
            elif list(temp.loc[temp['isLiquidity'] == 1, 'Amount'])[0] > 0 and address in list(
                    temp['To']):  # I am removing LP in LP farm + rewards (if any)
                temp = temp.loc[temp['isLiquidity'] != 1, :]
                temp = temp.loc[temp['Amount'] != 0, :]
                temp.index = [k + pd.Timedelta(seconds=i) for i, k in enumerate(temp.index)]
                temp['Fee'] /= temp.shape[0]
                temp['Tag'] = 'Reward'
                temp['To Coin'] = ''
                outdf.drop([pd.to_datetime(index)], axis=0, inplace=True)
                outdf = pd.concat([outdf, temp], axis=0)
            elif list(temp.loc[temp['isLiquidity'] == 1, 'Amount'])[0] < 0 and address in list(
                    temp['To']):  # I am removing Liquidity
                temp = temp.loc[temp['isLiquidity'] != 1, :]
                temp = temp.loc[temp['Amount'] != 0, :]
                temp['From'] = temp['To'] = 'Removing Liquidity'
                for coin_l in list(temp['Coin']):
                    temp.loc[temp['Coin'] == coin_l, 'Amount'] = list(temp.loc[temp['Coin'] == coin_l, 'Amount'])[0] - \
                                                                 liquidity_provider[coin_l]

                    liquidity_provider.pop(coin_l)
                temp.index = [k + pd.Timedelta(seconds=i) for i, k in enumerate(temp.index)]
                temp['Fee'] /= temp.shape[0]
                temp['Tag'] = 'Reward'
                temp['To Coin'] = ''
                outdf.drop([pd.to_datetime(index)], axis=0, inplace=True)
                outdf = pd.concat([outdf, temp], axis=0)
    outdf.drop(['isLiquidity'], axis=1, inplace=True)
    return outdf
