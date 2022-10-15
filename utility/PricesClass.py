import time
import numpy as np
import pandas as pd
import requests
import pickle as pk
import yfinance as yf
import datetime as dt
from utility.tax_log import log
import os

log.info('Prices object - updated on 15/10/2022')


class Prices:
    def __init__(self):
        if 'prices.pickle' not in os.listdir():
            self.prices = dict()
            self.prices['USD'] = dict()
        else:
            with open(f'prices.pickle', 'rb') as prices_list:
                self.prices = pk.load(prices_list)

        if f'exchange_rates.pickle' not in os.listdir():
            self.exchange_rates = dict()
        else:
            with open(f'exchange_rates.pickle', 'rb') as prices_list:
                self.exchange_rates = pk.load(prices_list)

        if 'coingeckolist.pickle' not in os.listdir():
            self.coingecko_coins_list = requests.get('https://api.coingecko.com/api/v3/coins/list')
            self.coingecko_coins_list = self.coingecko_coins_list.json()
            with open('coingeckolist.pickle', 'wb') as gecko_list:
                pk.dump(self.coingecko_coins_list, gecko_list, protocol=pk.HIGHEST_PROTOCOL)
        else:
            if dt.datetime.now().timestamp() - os.path.getmtime('coingeckolist.pickle') >= 604800:
                self.coingecko_coins_list = requests.get('https://api.coingecko.com/api/v3/coins/list')
                self.coingecko_coins_list = self.coingecko_coins_list.json()
                with open('coingeckolist.pickle', 'wb') as gecko_list:
                    pk.dump(self.coingecko_coins_list, gecko_list, protocol=pk.HIGHEST_PROTOCOL)
            else:
                with open('coingeckolist.pickle', 'rb') as gecko_list:
                    self.coingecko_coins_list = pk.load(gecko_list)

    def write_prices(self):
        with open('prices.pickle', 'wb') as prices_list:
            pk.dump(self.prices, prices_list, protocol=pk.HIGHEST_PROTOCOL)

    def coingecko(self, coin, _write_output=True):
        log.info(f'Getting prices for {coin} - CoinGecko')
        if coin.lower() == 'btt':
            token_loop = ['bittorrent-old']
        elif coin.lower() == 'squid':
            token_loop = ['squid-game']
        elif coin.lower() == 'beth':
            token_loop = ['eth']
        elif coin.lower() == 'cro':
            token_loop = ['crypto-com-chain']
        elif coin.lower() == 'cgld':
            token_loop = ['celo']
        elif coin.lower() == 'cake':
            token_loop = ['pancakeswap-token']
        elif coin.lower() == 'city':
            token_loop = ['manchester-city-fan-token']
        elif coin.lower() == 'lunc':
            token_loop = ['terra-luna']
        elif coin.lower() == 'luna' or coin.lower() == 'luna2':
            token_loop = ['terra-luna-2']
        elif coin.lower() == 'grt':
            token_loop = ['the-graph']
        elif coin.lower() == 'shib':
            token_loop = ['shiba-inu']
        elif coin.lower() == 'mmo':
            token_loop = ['mad-meerkat-optimizer']
        elif coin.lower() == 'sand':
            token_loop = ['the-sandbox']
        else:
            token_loop = [x['id'] for x in self.coingecko_coins_list if x['symbol'].upper() == coin.upper()]
        if len(token_loop) == 0:
            log.info(f'Could not find token {coin} in coingecko symbols, defaulting to None')
            vout = None
        else:
            try:
                if "-" in token_loop[-1]:
                    price_response = requests.get(f'http://api.coingecko.com/api/v3/coins/{token_loop[0]}'
                                                  f'/market_chart?vs_currency=usd&days=max&interval=daily')
                    if price_response.status_code == 429:
                        log.info('Too many requests, waiting 120 seconds before retry')
                        time.sleep(120)
                        price_response = requests.get(f'http://api.coingecko.com/api/v3/coins/{token_loop[0]}'
                                                      f'/market_chart?vs_currency=usd&days=max&interval=daily')
                else:
                    price_response = requests.get(f'http://api.coingecko.com/api/v3/coins/{token_loop[-1]}'
                                                  f'/market_chart?vs_currency=usd&days=max&interval=daily')
                    if price_response.status_code == 429:
                        log.info('Too many requests, waiting 120 seconds before retry')
                        time.sleep(120)
                        price_response = requests.get(f'http://api.coingecko.com/api/v3/coins/{token_loop[0]}'
                                                      f'/market_chart?vs_currency=usd&days=max&interval=daily')
            except:
                price_response = requests.get(f'http://api.coingecko.com/api/v3/coins/{token_loop[0]}'
                                              f'/market_chart?vs_currency=usd&days=max&interval=daily')
                if price_response.status_code == 429:
                    log.info('Too many requests, waiting 90 seconds before retry')
                    time.sleep(90)
                    price_response = requests.get(f'http://api.coingecko.com/api/v3/coins/{token_loop[0]}'
                                                  f'/market_chart?vs_currency=usd&days=max&interval=daily')

            if price_response.status_code != 200:
                log.info(
                    f'Could not download prices for {coin} with coingecko symbols {", ".join(token_loop)}, '
                    f'defaulting to None, code: {price_response.status_code}')
                vout = None
            else:
                out_list = price_response.json()['prices']
                vout = [(dt.date.fromtimestamp(x[0] / 1000), x[1]) for x in out_list]

        if vout is not None:
            vout = [x for x in vout if x[0] != dt.datetime.today().date()]
        if coin.upper() in list(self.prices['USD'].keys()):
            if self.prices['USD'][coin.upper()] is None:
                if vout is None:
                    self.prices['USD'][coin.upper()] = None
                else:
                    self.prices['USD'][coin.upper()] = vout
            else:
                if vout is not None:
                    self.prices['USD'][coin.upper()].extend(vout)
                    self.prices['USD'][coin.upper()] = list(set(self.prices['USD'][coin.upper()]))
                    self.prices['USD'][coin.upper()].sort()
        else:
            self.prices['USD'][coin.upper()] = vout

        if _write_output:
            self.write_prices()

    def coingecko_batch(self, token_list):
        for token in token_list:
            self.coingecko(token, False)
        self.write_prices()

    def to_pd_series(self, token, currency='USD'):
        currency = currency.upper()
        if token not in self.prices[currency].keys():
            log.info(f'{token} not in prices, please use coingecko method first')
            return None
        else:
            if self.prices[currency][token] is None:
                return None
            else:
                index_token = [x[0] for x in self.prices[currency][token]]
                values_token = [x[1] for x in self.prices[currency][token]]
                vout = pd.Series(data=values_token, index=index_token)
                vout = vout.sort_index()
                return vout

    def to_pd_dataframe(self, currency='USD'):
        currency = currency.upper()
        vout = 0
        colnames = []
        for i, key in enumerate(list(self.prices[currency].keys())):
            if self.prices[currency][key] is None:
                vout[key] = 0
            if i == 0:
                vout = pd.DataFrame(self.to_pd_series(key, currency=currency))
                colnames.append(key)
            else:
                ser_temp = pd.DataFrame(self.to_pd_series(key, currency=currency))
                vout = vout.join(ser_temp, lsuffix=f'L{str(i)}-', how='outer')
                colnames.append(key)
        first_row = [0 if pd.isna(x) else x for x in vout.iloc[0, :].tolist()]
        vout.iloc[0, :] = first_row
        vout.ffill(inplace=True)
        vout.columns = colnames

        vout = vout[~vout.index.duplicated(keep='first')]

        return vout

    def get_exchange_rates(self, currency):
        exc_rate = yf.Ticker(f'{currency.upper()}=X')
        exc_rate_history = exc_rate.history(period='max')

        if '+' in str(exc_rate_history.index[0]):
            exc_rate_history.index = [str(x).split("+")[0] for x in exc_rate_history.index]

        exc_rate_index = [dt.datetime.strptime(str(exc_rate_history.iloc[x, :].name), "%Y-%m-%d %H:%M:%S").date()
                          for x in range(exc_rate_history.shape[0])]
        exc_rate_vals = [float(exc_rate_history.loc[exc_rate_history.index[x], 'Close']) for x in
                         range(exc_rate_history.shape[0])]

        exc_rate_df = pd.DataFrame()
        exc_rate_df['vals'] = exc_rate_vals
        exc_rate_df.index = exc_rate_index

        placeholder_index = [(exc_rate_df.index[0] + dt.timedelta(days=x)) for x in
                             range(1, (exc_rate_df.index[-1] - exc_rate_df.index[0]).days)]
        placeholder_df = pd.DataFrame()
        placeholder_df.index = placeholder_index
        placeholder_df['vals'] = np.NaN

        joined_df = placeholder_df.join(exc_rate_df, lsuffix='L-', how='outer')
        joined_df.drop([joined_df.columns[0]], axis=1, inplace=True)
        joined_df.ffill(inplace=True)

        exc_rate_list = [(joined_df.index[x], joined_df['vals'][x]) for x in
                         range(joined_df.shape[0])]

        if exc_rate_list[-1][0] == (dt.date.today() - dt.timedelta(days=2)):
            exc_rate_list.append(((dt.date.today() - dt.timedelta(days=1)), exc_rate_list[-1][1]))

        self.exchange_rates[currency.upper()] = exc_rate_list
        with open('exchange_rates.pickle', 'wb') as exchange:
            pk.dump(self.exchange_rates, exchange, protocol=pk.HIGHEST_PROTOCOL)

    def convert_prices(self, currency, tokens=None, _write_output=True):
        currency = currency.upper()
        if tokens is not None:
            tokens = [x.upper() for x in tokens]
        else:
            tokens = [k for k, x in self.prices['USD'].items()]

        if currency not in self.exchange_rates.keys():
            self.prices[currency] = {}
            self.get_exchange_rates(currency)
        else:
            if max(self.exchange_rates[currency])[0] < (dt.date.today() - dt.timedelta(days=1)):
                self.get_exchange_rates(currency)

        if currency in self.prices.keys():
            temp_dict = {k: self.prices[currency][k] for k in tokens if k in list(self.prices[currency].keys())}
            last_update = [max(y) for x, y in temp_dict.items() if y is not None]

            is_updated = [True if last_update[k][0] >= dt.datetime.today().date() - dt.timedelta(days=1) else False for
                          k in
                          range(len(last_update))]

            is_in_price = [True if k in list(self.prices[currency].keys()) else False for k in tokens]
        else:
            is_updated = is_in_price = [False]
        if not all(is_updated) or not all(is_in_price):
            for index, values in self.prices['USD'].items():
                if index.upper() not in tokens:
                    continue
                else:
                    if self.prices['USD'][index] is None or index in ['AUD', 'BRL', 'EUR', 'GBP', 'GHS', 'HKD', 'KES',
                                                                      'KZT', 'NGN', 'NOK', 'PHP', 'PEN', 'RUB',
                                                                      'TRY', 'UGX', 'KRW', 'SEK', 'JPY', 'DKK', 'IDR',
                                                                      'SDR', 'INR', 'THB', 'MNT', 'UAH', '']:
                        self.prices[currency.upper()][index] = None
                        continue
                    conversion_list = []
                    log.info(f'Converting {index.upper()}')
                    for index_list, val_loop in enumerate(values):
                        conversion = [float(i[1]) for i in self.exchange_rates[currency.upper()] if i[0] == val_loop[0]]
                        conversion = 1 / conversion[0]
                        conversion_list.append(conversion)

                    self.prices[currency.upper()][index] = [(x[0], x[1] / y) for x, y in zip(self.prices['USD'][index],
                                                                                             conversion_list)]
        if _write_output:
            self.write_prices()


def update_prices(price_object, tokens, forceUpdate=False):
    if len(price_object.prices['USD']) == 0:
        price_object.coingecko_batch(tokens)

    temp_dict = {k: price_object.prices['USD'][k] for k in tokens if k in list(price_object.prices['USD'].keys())}
    last_update = [max(y) for x, y in temp_dict.items() if y is not None]

    is_updated = [True if last_update[k][0] >= dt.datetime.today().date() - dt.timedelta(days=1) else False for k in
                  range(len(last_update))]

    is_in_price = [True if k in list(price_object.prices['USD'].keys()) else False for k in tokens]

    is_not_in_key = [True if k not in list(price_object.prices['USD'].keys()) else False for k in tokens]
    if not forceUpdate:
        if not all(is_updated):
            which_not = [tokens[y] for y in range(len(is_updated)) if not is_updated[y]]
            price_object.coingecko_batch(which_not)
            return True
        elif not all(is_in_price):
            which_not = [tokens[y] for y in range(len(is_in_price)) if not is_in_price[y]]
            price_object.coingecko_batch(which_not)
            return True
        elif all(is_not_in_key):
            which_not = [tokens[y] for y in range(len(is_not_in_key)) if is_not_in_key[y]]
            price_object.coingecko_batch(which_not)
            return True
        else:
            return False
    else:
        tokens = list(price_object.prices['USD'].keys())
        price_object.coingecko_batch(tokens)
        return True
