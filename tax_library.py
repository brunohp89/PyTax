import copy
import os
import pandas as pd
import numpy as np
import datetime as dt
import pickle as pk

import requests


def str_to_datetime(date: str):
    try:
        if len(date) > 11:
            new_date = dt.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        else:
            new_date = dt.datetime.strptime(date, '%Y-%m-%d')

        return new_date

    except ValueError:
        print("Invalid format. Allowed formats are: YYYY-MM-DD and YYYY-MM-DD HH:MM:SS")


def get_currency_balance_cdc(currency: str, crypto_transactions: pd.DataFrame):
    crypto_transactions.index = [str_to_datetime(day[0:10]).date() for day in
                                 crypto_transactions['Timestamp (UTC)']]

    currency_df = crypto_transactions[
        np.logical_or(crypto_transactions['Currency'] == currency,
                      crypto_transactions['To Currency'] == currency)]
    currency_dates = currency_df.index
    currency_df.index = [k for k in range(currency_df.shape[0])]

    pd.options.mode.chained_assignment = None
    for i in range(currency_df.shape[0]):
        if not pd.isna(currency_df["To Amount"].tolist()[i]) and currency_df["To Currency"].tolist()[i] == currency:
            currency_df.loc[i, "Amount"] = currency_df["To Amount"].tolist()[i]

    currency_amount = currency_df['Amount'].tolist()
    currency_amount.reverse()
    currency_amount = np.array(currency_amount).cumsum()
    currency_amount = list(currency_amount)
    currency_amount.reverse()

    currency_series = pd.DataFrame(index=currency_dates, data=currency_amount)
    currency_series['Number'] = [k for k in range(currency_df.shape[0])]
    todrop = []
    for date in currency_series.index:
        if not (isinstance(currency_series.loc[date, 'Number'], float) or isinstance(
                currency_series.loc[date, 'Number'], np.int64)):
            todrop.extend(
                currency_series.loc[date, 'Number'].tolist()[1:len(currency_series.loc[date, 'Number'].tolist())])
    todrop = np.unique(todrop)

    currency_series = currency_series[~currency_series['Number'].isin(todrop)]
    currency_series.drop(['Number'], axis=1, inplace=True)
    currency_series.columns = ['Amount']

    return currency_series


def get_new_history_cdc(new_transactions_file):
    crypto_transactions_df = pd.read_csv(new_transactions_file[0])
    new_df = pd.DataFrame()
    # 'first_use_cdc.pickle' contiene il file raw delle transazioni
    if 'first_use_cdc.pickle' not in os.listdir():
        with open('first_use_cdc.pickle', 'wb') as handle:
            pk.dump(crypto_transactions_df, handle, protocol=pk.HIGHEST_PROTOCOL)
    else:
        with open('first_use_cdc.pickle', 'rb') as handle:
            crypto_transactions_df_old = pk.load(handle)

        crypto_transactions_df_old.index = [str_to_datetime(j).date() for j in
                                            crypto_transactions_df_old['Timestamp (UTC)']]
        crypto_transactions_df.index = [str_to_datetime(j).date() for j in crypto_transactions_df['Timestamp (UTC)']]
        max_date = max(crypto_transactions_df_old.index) - dt.timedelta(days=1)

        new_df = crypto_transactions_df[crypto_transactions_df.index >= max_date]
        old_df = crypto_transactions_df_old[crypto_transactions_df_old.index < max_date]
        crypto_transactions_df = new_df.append(old_df)
        crypto_transactions_df.index = [i for i in range(crypto_transactions_df.shape[0])]

        with open('first_use_cdc.pickle', 'wb') as handle:
            pk.dump(crypto_transactions_df, handle, protocol=pk.HIGHEST_PROTOCOL)

    crypto_transactions_df = crypto_transactions_df[
        ~crypto_transactions_df['Transaction Description'].isin(
            ["Crypto Earn Deposit", "Crypto Earn Withdrawal"])]

    crypto_transactions_df.loc[
        crypto_transactions_df['Transaction Description'] == 'Recurring Buy', 'Amount'] *= -1
    crypto_transactions_df.loc[
        crypto_transactions_df['Transaction Description'] == 'Supercharger Deposit (via app)', 'Amount'] = 0
    crypto_transactions_df.loc[
        crypto_transactions_df['Transaction Description'] == 'Supercharger Withdrawal (via app)', 'Amount'] = 0
    crypto_transactions_df.loc[crypto_transactions_df['Transaction Description'] == 'CRO Stake', 'Amount'] = 0

    # Getting rewards from crypto earn and supercharger
    interests = crypto_transactions_df[crypto_transactions_df['Transaction Description'] == "Crypto Earn"]
    interests = interests.append(
        crypto_transactions_df[crypto_transactions_df['Transaction Description'] == "Supercharger Reward"])

    interests_df_crypto_final = pd.DataFrame()
    interests_df_eur_final = pd.DataFrame()
    for tok in np.unique(interests['Currency']):
        interests_df_crypto = pd.DataFrame()
        interests_df_eur = pd.DataFrame()
        temp_df = interests[interests['Currency'] == tok]
        interests_df_crypto.index = interests_df_eur.index = [str_to_datetime(k).date() for k in
                                                              temp_df['Timestamp (UTC)']]
        interests_df_crypto[tok] = temp_df['Amount'].tolist()
        interests_df_eur[tok] = temp_df['Native Amount'].tolist()
        interests_df_eur = interests_df_eur.groupby(interests_df_eur.index).sum()
        interests_df_crypto = interests_df_crypto.groupby(interests_df_crypto.index).sum()
        interests_df_eur_final = interests_df_eur_final.join(interests_df_eur, lsuffix=f'L-{tok}-', how='outer')
        interests_df_crypto_final = interests_df_crypto_final.join(interests_df_crypto, lsuffix=f'L-{tok}-',
                                                                   how='outer')
    interests_df_eur_final.fillna(0, inplace=True)
    interests_df_crypto_final.fillna(0, inplace=True)

    # Getting rewards from card cashback
    cashback = crypto_transactions_df[
        crypto_transactions_df['Transaction Description'].isin(["Card Cashback", "Card Cashback Reversal"])]
    cashback = cashback.append(
        crypto_transactions_df[crypto_transactions_df['Transaction Description'].str.contains("Card Rebate")])

    cashback_df_crypto_final = pd.DataFrame()
    cashback_df_eur_final = pd.DataFrame()
    cashback_df_crypto_final.index = cashback_df_eur_final.index = [str_to_datetime(k).date() for k in
                                                                    cashback['Timestamp (UTC)']]
    cashback_df_crypto_final['CRO'] = cashback['Amount'].tolist()
    cashback_df_eur_final['CRO'] = cashback['Native Amount'].tolist()
    cashback_df_eur_final = cashback_df_eur_final.groupby(cashback_df_eur_final.index).sum()
    cashback_df_crypto_final = cashback_df_crypto_final.groupby(cashback_df_crypto_final.index).sum()

    # Getting other transactions
    currencies = crypto_transactions_df['Currency'].tolist()
    currencies2 = [k for k in crypto_transactions_df['To Currency'].tolist() if not pd.isna(k)]
    currencies.extend(currencies2)
    currencies = [k for k in np.unique(currencies) if k != "EUR"]

    output_df = pd.DataFrame()
    for currency in currencies:
        temp = get_currency_balance_cdc(currency, crypto_transactions_df)
        if output_df.shape[0] == 0:
            output_df = temp
        else:
            output_df = output_df.join(temp, lsuffix=f'-{currency}', how='outer')

    output_df.columns = currencies
    output_df.iloc[0, :] = output_df.iloc[0, :].fillna(0)
    output_df.ffill(inplace=True)

    if new_df.shape[0] == 0 and 'cdc_prices.pickle' in os.listdir():
        with open('cdc_prices.pickle', 'rb') as handle:
            cdc_prices = pk.load(handle)
    else:
        cdc_prices = dict()
        api_key = '7Gulw489-Wyntjwa1ZPk'
        response = requests.get(
            f'https://fxmarketapi.com/apitimeseries?currency=EURUSD&start_date=2020-01-01&interval=daily&api_key={api_key}')
        time = [str_to_datetime(list(response.json().get('price').keys())[i]) for i in
                range(len(list(response.json().get('price').keys())))]
        open_price = [response.json().get('price')[list(response.json().get('price').keys())[i]]['EURUSD']['open'] for i
                      in range(len(list(response.json().get('price').keys())))]
        pd_curr = pd.DataFrame(index=time, data=open_price).join(
            pd.DataFrame(index=pd.date_range(time[0], time[-1]), data=[0] * len(pd.date_range(time[0], time[-1]))),
            lsuffix="L", how='outer')
        time = [k for k in pd_curr.index]
        pd_curr.ffill(inplace=True)
        open_price = pd_curr.iloc[:, 0].tolist()

        cdc_prices['EUR'] = [[x, y] for x, y in zip(time, open_price)]

        for coin in output_df.columns:
            print(f'Getting price for {coin} - Crypto.com')
            if coin in ['USDT', 'USDC', 'BUSD']:
                cdc_prices[coin] = [1] * 365
            else:
                temp_price = requests.get(
                    f"https://api.crypto.com/v2/public/get-candlestick?instrument_name={coin}_USDT&timeframe=1D")
                time = [dt.datetime.fromtimestamp(int(k.get('t')) / 1000).date() for k in
                        temp_price.json().get('result').get('data')]
                open_price = [k.get('o') for k in temp_price.json().get('result').get('data')]
                cdc_prices[coin] = [[x, y] for x, y in zip(time, open_price)]
        with open('cdc_prices.pickle', 'wb') as handle:
            pk.dump(cdc_prices, handle, protocol=pk.HIGHEST_PROTOCOL)

    index_temp = pd.date_range(output_df.index[0], output_df.index[-1])
    temp_df_1 = pd.DataFrame(index=index_temp, data=[np.nan] * len(index_temp), columns=['TEMP'])
    output_df = output_df.join(temp_df_1, how='outer')
    output_df.drop(['TEMP'], axis=1, inplace=True)
    output_df.iloc[0, :].fillna(0, inplace=True)
    output_df.ffill(inplace=True)

    output_df_eur = copy.deepcopy(output_df)

    for date_loop in output_df_eur.index:
        for coin in output_df_eur.columns:
            conversion = [k[1] for k in cdc_prices["EUR"] if k[0].date() == date_loop][0]
            if coin in ['USDC', 'USDT', 'BUSD']:
                price = 1 / conversion
            else:
                coin_dates = [k[0] for k in cdc_prices[coin]]
                if date_loop in coin_dates:
                    price = [k[1] for k in cdc_prices[coin] if k[0] == date_loop][0]
                else:
                    price = cdc_prices[coin][0][1]
                price *= conversion
            output_df_eur.loc[date_loop, coin] *= price

    vOut = dict()
    vOut['cashback-EUR'] = cashback_df_eur_final
    vOut['cashback-Crypto'] = cashback_df_crypto_final
    vOut['interest-EUR'] = interests_df_eur_final
    vOut['interest-Crypto'] = interests_df_crypto_final
    vOut['balances'] = output_df
    vOut['balances-EUR'] = output_df_eur
    return vOut


def to_binance_date_format(date_to_convert):
    month_out = None
    if date_to_convert.month == 1:
        month_out = 'Jan'
    elif date_to_convert.month == 2:
        month_out = 'Feb'
    elif date_to_convert.month == 3:
        month_out = 'Mar'
    elif date_to_convert.month == 4:
        month_out = 'Apr'
    elif date_to_convert.month == 5:
        month_out = 'May'
    elif date_to_convert.month == 6:
        month_out = 'Jun'
    elif date_to_convert.month == 7:
        month_out = 'Jul'
    elif date_to_convert.month == 8:
        month_out = 'Aug'
    elif date_to_convert.month == 9:
        month_out = 'Sep'
    elif date_to_convert.month == 10:
        month_out = 'Oct'
    elif date_to_convert.month == 11:
        month_out = 'Nov'
    elif date_to_convert.month == 12:
        month_out = 'Dec'
    return f'{date_to_convert.day} {month_out}, {date_to_convert.year}'


def get_token_prices(tokens: list, contracts: list, networks: list, timeframe: int, currency='usd'):
    tokens_tickers = [o.upper() for o in tokens]

    if 'coingeckolist.pickle' not in os.listdir():
        coingecko_coins_list = requests.get('https://api.coingecko.com/api/v3/coins/list')
        coingecko_coins_list = coingecko_coins_list.json()
        with open('coingeckolist.pickle', 'wb') as gecko_list:
            pk.dump(coingecko_coins_list, gecko_list, protocol=pk.HIGHEST_PROTOCOL)
    else:
        if dt.datetime.now().timestamp() - os.path.getmtime('coingeckolist.pickle') >= 604800:
            coingecko_coins_list = requests.get('https://api.coingecko.com/api/v3/coins/list')
            coingecko_coins_list = coingecko_coins_list.json()
            with open('coingeckolist.pickle', 'wb') as gecko_list:
                pk.dump(coingecko_coins_list, gecko_list, protocol=pk.HIGHEST_PROTOCOL)
        else:
            with open('coingeckolist.pickle', 'rb') as gecko_list:
                coingecko_coins_list = pk.load(gecko_list)
    tokens_prices = []

    for coin, contract, network in zip(tokens_tickers, contracts, networks):
        print(f'Getting prices for {coin} - CoinGecko')
        if contract == 0:
            tok1 = [g.get('id') for g in coingecko_coins_list if g.get('symbol').upper() == coin]
            tokens_prices.append(requests.get(f'http://api.coingecko.com/api/v3/coins/{tok1[0]}'
                                              f'/market_chart?vs_currency={currency}&days={timeframe}'))

        else:
            temp_contract = requests.get(f'https://api.coingecko.com/api/v3/coins/{network}/contract/'
                                         f'{contract}')
            if temp_contract.status_code != 200:
                tokens_prices.append(0)
                print(f'ERROR GETTING PRICE: {temp_contract.json().get("error")} --> {coin}')
            else:
                temp1 = requests.get(f'https://api.coingecko.com/api/v3/coins/{temp_contract.json().get("id")}'
                                     f'/market_chart?vs_currency={currency}&days={timeframe}')
                if temp1.status_code != 200:
                    tokens_prices.append(0)
                    print(f'ERROR GETTING PRICE: {temp1.reason} --> {coin}')
                else:
                    tokens_prices.append(temp1)

    vout = dict()
    for price, coin in zip(tokens_prices, tokens_tickers):
        if price == 0:
            vout[coin] = 1
        else:
            resplist = price.json()['prices']
            temp_list = [[dt.date.fromtimestamp(r[0] / 1000), r[1]] for r in resplist]
            vout[coin] = temp_list
    return vout


def uphold_date_to_date(date):
    month_out = None
    if date[0:3] == 'Jan':
        month_out = 1
    elif date[0:3] == 'Feb':
        month_out = 2
    elif date[0:3] == 'Mar':
        month_out = 3
    elif date[0:3] == 'Apr':
        month_out = 4
    elif date[0:3] == 'May':
        month_out = 5
    elif date[0:3] == 'Jun':
        month_out = 6
    elif date[0:3] == 'Jul':
        month_out = 7
    elif date[0:3] == 'Aug':
        month_out = 8
    elif date[0:3] == 'Sep':
        month_out = 9
    elif date[0:3] == 'Oct':
        month_out = 10
    elif date[0:3] == 'Nov':
        month_out = 11
    elif date[0:3] == 'Dec':
        month_out = 12
    return dt.date(int(date[7:12]),month_out,int(date[4:6]))
