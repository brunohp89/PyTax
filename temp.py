import copy
import shutil
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import datetime as dt
import requests
import tax_library as tx
import pickle as pk
import binance.exceptions
from binance.client import Client

# PER CRYPTO AL PRIMO UTILIZZO PRENDERE TUTTO LO STORICO DALL'INIZIO DELL'ACCOUNT
# PER AGGIORNAMENTI ESTRARRE LO STORICO INIZIANDO ALMENO DA DATA - 2

new_transactions_file = [file_os for file_os in os.listdir() if "crypto_transactions_record" in file_os]
history_file = [file_os for file_os in os.listdir() if file_os == "cryptodotcom.pickle"]

if len(new_transactions_file) == 0 and len(history_file) == 0:
    exit("ERROR: Not history nor transaction file found for crypto.com app. " \
         "Note that the file name should follow the nomenlaclature: " \
         "crypto_transactions_record_XXXXXXXX_XXXXXXXX.csv for transaction file and " \
         "cryptohistory.csv for the history file")
if len(new_transactions_file) != 0:
    if len(history_file) == 0:
        cryptodotcom_dict = tx.get_new_history_cdc(new_transactions_file)
        with open('cryptodotcom.pickle', 'wb') as handle:
            pk.dump(cryptodotcom_dict, handle, protocol=pk.HIGHEST_PROTOCOL)
    else:
        last_update_str = history_file[0].split("\\")[-1].replace("cryptohistory_", "").replace(".csv", "")

        with open('cryptodotcom.pickle', 'rb') as handle:
            cryptodotcom_dict = pk.load(handle)

        cashback_eur_history = cryptodotcom_dict['cashback-EUR']
        cashback_eur_history.index = cashback_eur_history.iloc[:, 0]
        cashback_eur_history.drop([cashback_eur_history.columns[0]], inplace=True, axis=1)

        cashback_crypto_history = cryptodotcom_dict['cashback-Crypto']
        cashback_crypto_history.index = cashback_crypto_history.iloc[:, 0]
        cashback_crypto_history.drop([cashback_crypto_history.columns[0]], inplace=True, axis=1)

        interest_eur_history = cryptodotcom_dict['interest-EUR']
        interest_eur_history.index = interest_eur_history.iloc[:, 0]
        interest_eur_history.drop([interest_eur_history.columns[0]], inplace=True, axis=1)

        interest_crypto_history = cryptodotcom_dict['interest-Crypto']
        interest_crypto_history.index = interest_crypto_history.iloc[:, 0]
        interest_crypto_history.drop([interest_crypto_history.columns[0]], inplace=True, axis=1)

        balance_history = cryptodotcom_dict['balances']
        balance_history.index = balance_history.iloc[:, 0]
        balance_history.drop([balance_history.columns[0]], inplace=True, axis=1)

        balance_history_EUR = cryptodotcom_dict['balances-EUR']
        balance_history_EUR.index = balance_history_EUR.iloc[:, 0]
        balance_history_EUR.drop([balance_history_EUR.columns[0]], inplace=True, axis=1)

        new_history = tx.get_new_history_cdc(new_transactions_file)

        cryptodotcom_dict['balances'] = new_history['balances']
        cryptodotcom_dict['balances-EUR'] = new_history['balances-EUR']
        cryptodotcom_dict['interest-Crypto'] = new_history['interest-Crypto']
        cryptodotcom_dict['interest-EUR'] = new_history['interest-EUR']
        cryptodotcom_dict['cashback-Crypto'] = new_history['cashback-Crypto']
        cryptodotcom_dict['cashback-EUR'] = new_history['cashback-EUR']

        with open('cryptodotcom.pickle', 'wb') as handle:
            pk.dump(cryptodotcom_dict, handle, protocol=pk.HIGHEST_PROTOCOL)
    shutil.move(new_transactions_file[0], "cryptodotcom_history")
else:
    last_update_str = history_file[0].split("\\")[-1].replace("cryptohistory_", "").replace(".csv", "")

    with open('cryptodotcom.pickle', 'rb') as handle:
        cryptodotcom_dict = pk.load(handle)



# BINANCE
binance_token = "dWmsjaJlm3eESfF90k704cyqN3BTaaLFsmXK4NA3y5QPI0SUk4W3RjPDImbIKMqd"
binance_secret = "3drhdpfQQVegQxYLlehmlytvdXO8Mxt1PNugK1QmoLzCX5IRjZt7tc5pXOIm7rwj"
binance_client = Client(binance_token, binance_secret)

new_binance_files = [file_os for file_os in os.listdir() if "part-" in file_os]

if len(new_binance_files) > 0:
    binance_transactions_df_new = binance_transactions_df = new_df = pd.DataFrame()
    for file in new_binance_files:
        binance_transactions_df_new = binance_transactions_df_new.append(pd.read_csv(file))
        binance_transactions_df_new.drop_duplicates(inplace=True)
        binance_transactions_df_new.index = [tx.str_to_datetime(x) for x in binance_transactions_df_new["UTC_Time"]]
        binance_transactions_df_new.sort_index(inplace=True)
        binance_transactions_df = copy.deepcopy(binance_transactions_df_new)
        shutil.move(file, "binance_history")

    if 'first_use_binance.pickle' not in os.listdir():
        with open('first_use_binance.pickle', 'wb') as handle:
            pk.dump(binance_transactions_df_new, handle, protocol=pk.HIGHEST_PROTOCOL)
    else:
        with open('first_use_binance.pickle', 'rb') as handle:
            binance_transactions_df_old = pk.load(handle)

        max_date = max(binance_transactions_df_old.index) - dt.timedelta(days=1)

        new_df = binance_transactions_df_new[binance_transactions_df_new.index >= max_date]
        old_df = binance_transactions_df_old[binance_transactions_df_old.index < max_date]
        binance_transactions_df = new_df.append(old_df)
        binance_transactions_df.sort_index(inplace=True)

    to_exclude = ['ETH 2.0 Staking',
                  'Launchpad subscribe',
                  'POS savings purchase',
                  'POS savings redemption',
                  'Savings Principal redemption',
                  'Savings purchase']
    binance_transactions_df = binance_transactions_df[~binance_transactions_df['Operation'].isin(to_exclude)]
    binance_transactions_df.loc[binance_transactions_df['Coin'] == 'BETH', 'Coin'] = 'ETH'

    with open('first_use_binance.pickle', 'wb') as handle:
        pk.dump(binance_transactions_df, handle, protocol=pk.HIGHEST_PROTOCOL)

    price_dict = dict()

    if new_df.shape[0] == 0 and 'binance_prices.pickle' in os.listdir():
        with open('binance_prices.pickle', 'rb') as handle:
            price_dict = pk.load(handle)
    else:
        temp_price = binance_client.get_historical_klines("EURBUSD", binance_client.KLINE_INTERVAL_1DAY,
                                                          tx.to_binance_date_format(dt.date(2021, 1, 1)))
        temp_price = [[dt.datetime.fromtimestamp(i[0] / 1000).date(), i[1]] for i in temp_price]
        price_dict['EUR'] = temp_price
        for coin in np.unique(binance_transactions_df['Coin']):
            print(f'Getting price for {coin} - Binance')
            if coin not in list(price_dict.keys()) and coin not in ['BUSD', 'USDC', 'USDT']:
                try:
                    temp_price = binance_client.get_historical_klines(f"{coin}BUSD",
                                                                      binance_client.KLINE_INTERVAL_1DAY,
                                                                      tx.to_binance_date_format(dt.date(2021, 1, 1)))
                    temp_price = [[dt.datetime.fromtimestamp(i[0] / 1000).date(), i[1]] for i in temp_price]
                    price_dict[coin] = temp_price
                except binance.exceptions.BinanceAPIException:  # Catching binance-python exception APIerror
                    print(f'{coin} could not be found, defaulting price to zero')
                    price_dict[coin] = 0
        with open('binance_prices.pickle', 'wb') as handle:
            pk.dump(price_dict, handle, protocol=pk.HIGHEST_PROTOCOL)

    # Getting interests from savings, launchpad and launchpool, eth staking 2.0
    interest_cols = ['ETH 2.0 Staking Rewards', 'Launchpad token distribution', 'Launchpool Interest',
                     'POS savings interest', 'Savings Interest', 'Super BNB Mining']
    interest_df = binance_transactions_df[binance_transactions_df['Operation'].isin(interest_cols)]
    interest_df.index = [x.date() for x in interest_df.index]
    interest_df = interest_df.drop(['User_ID', 'UTC_Time', 'Account', 'Operation', 'Remark'], axis=1)

    binance_interest_nat = pd.DataFrame()
    binance_interest_eur = pd.DataFrame()

    for coin in np.unique(interest_df['Coin']):
        temp_df = interest_df[interest_df['Coin'] == coin]
        temp_df = temp_df.groupby(temp_df.index).sum()
        binance_interest_nat = binance_interest_nat.join(temp_df, lsuffix=f'L-{coin}-', how='outer')
        for index, date_loop in enumerate(temp_df.index):
            # index,date_loop = next(enumerate(binance_interest_nat.index))
            conversion = [float(i[1]) for i in price_dict['EUR'] if i[0] == date_loop]
            conversion = 1 / conversion[0]
            if coin in ['BUSD', 'USDC', 'USDT']:
                price = conversion
            else:
                coin_price = price_dict[coin]
                if coin_price == 0:
                    price = 0
                else:
                    price = [float(i[1]) for i in coin_price if i[0] == date_loop]
                    if len(price) == 0:
                        price = float(coin_price[0][1]) * conversion
                    else:
                        price = price[0] * conversion
            temp_df.iloc[index, 0] *= price
        binance_interest_eur = binance_interest_eur.join(temp_df, lsuffix=f'L-{coin}-', how='outer')

    binance_interest_eur.fillna(0, inplace=True)
    binance_interest_nat.fillna(0, inplace=True)
    binance_interest_eur.columns = np.unique(interest_df['Coin'])
    binance_interest_nat.columns = np.unique(interest_df['Coin'])
    binance_transactions_df.index = [x.date() for x in binance_transactions_df.index]

    binance_transactions_df = binance_transactions_df.drop(['User_ID', 'UTC_Time', 'Account', 'Operation', 'Remark'],
                                                           axis=1)

    total_binance_transactions_nat = pd.DataFrame()
    total_binance_transactions_eur = pd.DataFrame()
    for coin in np.unique(binance_transactions_df['Coin']):
        temp = binance_transactions_df.loc[binance_transactions_df['Coin'] == coin, 'Change']
        temp = temp.groupby(temp.index).sum()
        total_binance_transactions_nat = total_binance_transactions_nat.join(temp, lsuffix=f'L-{coin}', how='outer')

    total_binance_transactions_nat = total_binance_transactions_nat.groupby(total_binance_transactions_nat.index).sum()
    total_binance_transactions_nat.columns = np.unique(binance_transactions_df['Coin'])
    total_binance_transactions_nat = total_binance_transactions_nat.cumsum()
    total_binance_transactions_nat[total_binance_transactions_nat < 0] = 0

    index_temp = pd.date_range(total_binance_transactions_nat.index[0], total_binance_transactions_nat.index[-1])
    temp_df_1 = pd.DataFrame(index=index_temp, data=[np.nan]*len(index_temp), columns=['TEMP'])
    total_binance_transactions_nat = total_binance_transactions_nat.join(temp_df_1, how='outer')
    total_binance_transactions_nat.drop(['TEMP'], axis=1, inplace=True)
    total_binance_transactions_nat.iloc[0,:].fillna(0, inplace=True)
    total_binance_transactions_nat.ffill(inplace=True)

    total_binance_transactions_eur = copy.deepcopy(total_binance_transactions_nat)

    for index, date_loop in enumerate(total_binance_transactions_eur.index):
        conversion = [float(i[1]) for i in price_dict['EUR'] if i[0] == date_loop]
        conversion = 1 / conversion[0]
        for colindex, coin in enumerate(total_binance_transactions_eur.columns):
            # colindex, coin = next(enumerate(total_binance_transactions_eur.columns))
            # index,date_loop = next(enumerate(total_binance_transactions_eur.index))
            if coin in ['BUSD', 'USDC', 'USDT']:
                price = conversion
            else:
                coin_price = price_dict[coin]
                if isinstance(coin_price, int):
                    price = 0
                else:
                    price = [float(i[1]) for i in coin_price if i[0] == date_loop]
                    if len(price) == 0:
                        price = float(coin_price[0][1]) * conversion
                    else:
                        price = price[0] * conversion
            total_binance_transactions_eur.iloc[index, colindex] *= price

    binance_dict = dict()
    binance_dict['balances'] = total_binance_transactions_nat
    binance_dict['balances-EUR'] = total_binance_transactions_eur
    binance_dict['interest-Crypto'] = binance_interest_nat
    binance_dict['interest-EUR'] = binance_interest_eur

    with open('binance.pickle', 'wb') as handle:
        pk.dump(binance_dict, handle, protocol=pk.HIGHEST_PROTOCOL)
else:
    with open('binance.pickle', 'rb') as handle:
        binance_dict = pk.load(handle)
#
# #TOTALE
#
# a = pd.DataFrame(total_binance_transactions_eur.sum(axis=1))
# b=pd.DataFrame(balance_history_EUR.sum(axis=1))
#
# b.index = [tx.str_to_datetime(k) for k in b.index]
# c=a.join(b, lsuffix='d', how='outer')
# #c.fillna(0,inplace=True)
# c.iloc[0,:].fillna(0,inplace=True)
# c.ffill(inplace=True)
# h = c.sum(axis=1)
# import  matplotlib.pyplot as plt
# plt.plot(h)

# INDIRIZZI ETHEREUM
addresses = ["0xF5a75B5aA78EC478D84FA16F39492E8E80748fD4","0xb746b672FCBaeA66804F4782C963FeD03caD6959"]

for address in addresses:

    ethscan = "GFID2HN2QCS6UR4K1CX13F946P2V1S7Q7X"

    response_main = requests.get(
        f'https://api.etherscan.io/api?module=account&action=txlist&address={address}&startblock=0&endblock=9999999999999999999&sort=asc&apikey={ethscan}')
    response_internal = requests.get(
        f'https://api.etherscan.io/api?module=account&action=txlistinternal&address={address}&startblock=0&endblock=9999999999999999999&sort=asc&apikey={ethscan}')
    response_erc20 = requests.get(
        f'https://api.etherscan.io/api?module=account&action=tokentx&address={address}&startblock=0&endblock=999999999999&sort=asc&apikey={ethscan}')

    # Get Total ETH transactions
    total_transactions = response_main.json()['result']
    total_transactions = [k for k in total_transactions if k['isError'] == '0']
    response_main.json()['result'] = total_transactions
    direction = [1 if k['to'].upper() == address.upper() else -1 for k in response_main.json()['result']]
    balance_eth = [int(k['value']) / 10 ** 18 for k in response_main.json()['result']]
    balance_eth = [balance_eth[i] * direction[i] for i in range(len(direction))]
    gas = [-1 * int(k['gasUsed']) * int(k['gasPrice']) / 10 ** 18 if int(k['value']) == 0 else 0 for k in
           response_main.json()['result']]
    eth_index = [dt.datetime.fromtimestamp(int(k['timeStamp'])) for k in response_main.json()['result']]
    balance_eth = [x + y for x, y in zip(balance_eth, gas)]
    eth_df = pd.DataFrame(index=eth_index, data=balance_eth)
    contract_list = [0]
    tokens_list = ['ETH']

    if response_internal.json()['message'] == 'OK':
        # Get total ETH internal transaction
        internal_transactions = response_internal.json()['result']
        internal_transactions = [k for k in internal_transactions if k['isError'] == '0']
        response_internal.json()['result'] = internal_transactions
        direction_internal = [1 if k['to'].upper() == address.upper() else -1 for k in response_internal.json()['result']]
        balance_eth_internal = [int(k['value']) / 10 ** 18 for k in response_internal.json()['result']]
        balance_eth_internal = [balance_eth_internal[i] * direction_internal[i] for i in range(len(direction_internal))]
        # gas_internal = [-1*int(k['gas'])*int(k['gasPrice'])/10**18 for k in response_internal.json()['result']]
        eth_index_internal = [dt.datetime.fromtimestamp(int(k['timeStamp'])) for k in response_internal.json()['result']]
        internal_df = pd.DataFrame(index=eth_index_internal, data=balance_eth_internal)
        eth_df = eth_df.join(internal_df, lsuffix=" Internal ", how='outer')
        eth_df = pd.DataFrame(eth_df.sum(axis=1))

    eth_df.columns = ['ETH']

    if response_erc20.json()['message'] == 'OK':
        tokens_erc20 = [k['tokenSymbol'].upper() for k in response_erc20.json()['result']]
        gas_erc20 = [0 if k['to'].upper() == address.upper() else -1 * int(k['gasUsed']) * int(k['gasPrice']) / 10 ** 18 for
                     k in response_erc20.json()['result']]
        erc20_index = [dt.datetime.fromtimestamp(int(k['timeStamp'])) for k in response_erc20.json()['result']]
        eth_df = eth_df.join(pd.DataFrame(index=erc20_index, data=gas_erc20), lsuffix=" ERC ", how='outer')
        eth_df = pd.DataFrame(eth_df.sum(axis=1))
        eth_df.columns = ['ETH']
        for token_loop in list(np.unique(tokens_erc20)):
            tokens_list.append(token_loop)
            temp_contract = [k['contractAddress'] for k in response_erc20.json()['result'] if
                             k['tokenSymbol'] == token_loop]
            contract_list.append(temp_contract[0])
            direction_erc20 = [1 if k['to'].upper() == address.upper() else -1 for k in
                               response_erc20.json()['result'] if
                             k['tokenSymbol'] == token_loop]
            erc20_index = [dt.datetime.fromtimestamp(int(k['timeStamp'])) for k in response_erc20.json()['result'] if
                           k['tokenSymbol'] == token_loop]
            balance_erc20 = [int(k['value']) / 10 ** int(k['tokenDecimal']) for k in response_erc20.json()['result'] if
                             k['tokenSymbol'] == token_loop]
            balance_erc20 = [x * y for x, y in zip(balance_erc20, direction_erc20)]
            temp_df = pd.DataFrame(index=erc20_index, data=balance_erc20)
            temp_df.index = [k.date() for k in temp_df.index]
            # METTERE L'ORARIO DELL'INDICE A 0 PER EVITARE DUPLOCAZIONI
            eth_df = eth_df.join(temp_df, lsuffix=f' {token_loop} ', how='outer')
        colnames = ['ETH']
        colnames.extend(list(np.unique(tokens_erc20)))
        eth_df.columns = colnames

    eth_df.fillna(0, inplace=True)
    eth_df.index = [k.date() for k in eth_df.index]
    eth_df = eth_df.groupby(eth_df.index).sum()
    eth_df = eth_df.cumsum(axis=0)

    fill_na_index = pd.date_range(eth_df.index[-1], dt.date.today())
    fill_na_index = [tx.str_to_datetime(a.date().isoformat()).date() for a in fill_na_index]
    fill_na = pd.DataFrame(index=fill_na_index, data=np.zeros([len(fill_na_index), eth_df.shape[1]]))
    fill_na[fill_na == 0] = np.nan
    fill_na.columns = eth_df.columns

    eth_df = eth_df.append(fill_na)
    eth_df.ffill(inplace=True)

    network_list = ['ethereum'] * len(contract_list)

    index_temp = pd.date_range(eth_df.index[0], eth_df.index[-1])
    temp_df_1 = pd.DataFrame(index=index_temp, data=[np.nan] * len(index_temp), columns=['TEMP'])
    eth_df = eth_df.join(temp_df_1, how='outer')
    eth_df.drop(['TEMP'], axis=1, inplace=True)
    eth_df.iloc[0, :].fillna(0, inplace=True)
    eth_df.ffill(inplace=True)
    eth_df = eth_df[~eth_df.index.duplicated(keep='first')]

    # import matplotlib.pyplot as plt
    # plt.plot(eth_df['ETH'])
    timeframe = (eth_df.index[-1].date() - eth_df.index[0].date()).days + 1
    prices_dict = tx.get_token_prices(tokens=list(eth_df.columns), contracts=contract_list, networks=network_list, timeframe=timeframe, currency='eur')

    eth_df_EUR = copy.deepcopy(eth_df)
    eth_df_EUR.index = eth_df.index =  [dt.date.fromisoformat(k.isoformat()[0:10]) for k in eth_df_EUR.index]
    for coin in eth_df_EUR.columns:
        for date_loop in eth_df_EUR.index:
            price = [i[1] for i in prices_dict[coin] if i[0] == date_loop]
            eth_df_EUR.loc[date_loop,coin] *= price[0]

    exec(f'eth{address[3:10]}=dict()')
    exec(f'eth{address[3:10]}["balance"]=eth_df')
    exec(f'eth{address[3:10]}["balance-EUR"]=eth_df_EUR')
    file_1 = f'eth{address[3:10]}'
    exec(f'with open("{file_1}.pickle", "wb") as handle: '
         f'pk.dump({exec(file_1)}, handle, protocol=pk.HIGHEST_PROTOCOL)')

# tokens=list(eth_df.columns)
# contracts= contract_list
# networks= network_list
# currency='usd'

# COINBASE - STORICO scaricato dall'API
from coinbase.wallet.client import Client as CoinbaseClient

client = CoinbaseClient("g5LIAFMd6yvOxSKl","5Cf0ymwH2Tr3rjuhaVclnbRoC4vK7BhX")
account = client.get_accounts()
accounts_id = [acc['id'] for acc in account['data']]

coinbase_df = pd.DataFrame()
coinbase_interests_and_earn_nat = pd.DataFrame()
coinbase_interests_and_earn_eur = pd.DataFrame()
for id in accounts_id:
    transactions = client.get_transactions(id)
    transactions = transactions['data']
    if len(transactions) != 0:
        amount = [float(transactions[i]['amount']['amount']) for i in range(len(transactions))]
        token = [transactions[i]['amount']['currency'] for i in range(len(transactions))]
        eur = [float(transactions[i]['native_amount']['amount']) for i in range(len(transactions))]
        isearn = [True if 'Coinbase Earn' in transactions[i]['details']['subtitle'] else False for i in
                  range(len(transactions))]
        date_time = [tx.str_to_datetime(transactions[i]['created_at'][0:-1].replace('T', ' '))
                     for i in range(len(transactions))]
        temp = pd.DataFrame(index=date_time, data=amount)
        temp.columns = [token[0]]
        coinbase_df = coinbase_df.join(temp, how='outer')#, lsuffix=' L ')

        amount1 = [eur[i] for i in range(len(eur)) if isearn[i]]
        if len(amount1) > 0 :
            date_time1 = [date_time[i] for i in range(len(eur)) if isearn[i]]
            temp = pd.DataFrame(index=date_time1, data=amount1)
            temp.columns = [token[0]]
            coinbase_interests_and_earn_eur = coinbase_interests_and_earn_eur.join(temp, how='outer')

            amount1 = [amount[i] for i in range(len(amount)) if isearn[i]]
            temp = pd.DataFrame(index=date_time1, data=amount1)
            temp.columns = [token[0]]
            coinbase_interests_and_earn_nat = coinbase_interests_and_earn_nat.join(temp, how='outer')


if coinbase_interests_and_earn_nat.shape[0] > 0:
    coinbase_interests_and_earn_eur.fillna(0, inplace=True)
    coinbase_interests_and_earn_nat.fillna(0, inplace=True)
    coinbase_interests_and_earn_nat.groupby(coinbase_interests_and_earn_nat.index).sum()
    coinbase_interests_and_earn_eur.groupby(coinbase_interests_and_earn_eur.index).sum()

index_temp = pd.date_range(coinbase_df.index[0], dt.date.today())
temp_df_1 = pd.DataFrame(index=index_temp, data=[np.nan] * len(index_temp), columns=['TEMP'])
coinbase_df = coinbase_df.join(temp_df_1, how='outer')
coinbase_df.drop(['TEMP'], axis=1, inplace=True)
coinbase_df.fillna(0, inplace=True)
coinbase_df.index = [dt.date.fromisoformat(k.isoformat()[0:10]) for k in coinbase_df.index]
coinbase_df=coinbase_df.groupby(coinbase_df.index).sum()

coinbase_df = coinbase_df.cumsum(axis=0)

from_date = coinbase_df.index[0].strftime('%Y-%m-%dT00:00:00')
to_date = coinbase_df.index[-1].strftime('%Y-%m-%dT00:00:00')

coinbase_prices = dict()
for coin in coinbase_df.columns:
    temp = client.get_historic_prices(currency_pair=f'{coin}-USD',period="year")
    coinbase_prices[coin] = [[tx.str_to_datetime(temp['prices'][i]['time'].replace("T"," ").replace("Z","")).date(),
                              float(temp['prices'][i]['price'])] for i in range(len(temp['prices']))]

temp = client.get_historic_prices(currency_pair=f'USDT-EUR',period="year")
coinbase_prices['EUR'] = [[tx.str_to_datetime(temp['prices'][i]['time'].replace("T"," ").replace("Z","")).date(),
                              float(temp['prices'][i]['price'])] for i in range(len(temp['prices']))]

coinbase_df_EUR = copy.deepcopy(coinbase_df)
coinbase_df_EUR.index = coinbase_df.index
for coin in coinbase_df_EUR.columns:
    for date_loop in coinbase_df_EUR.index:
        price = [i[1] for i in coinbase_prices[coin] if i[0] == date_loop]
        if len(price) == 0:
            coinbase_df_EUR.loc[date_loop,coin] = 0.
        else:
            conversion = [i[1] for i in coinbase_prices['EUR'] if i[0] == date_loop]
            loop_control = 0
            while len(conversion) == 0 and loop_control < 7:
                loop_control += 1
                conversion = [i[1] for i in coinbase_prices['EUR'] if i[0] == date_loop - dt.timedelta(days=1)]
            coinbase_df_EUR.loc[date_loop,coin] *= price[0]*conversion[0]
#
# import requests
# from bs4 import BeautifulSoup
#
# page = requests.get('https://crypto.com/exchange/document/fees-limits')  # Getting page HTML through request
# soup = BeautifulSoup(page.content, 'html.parser')


# UPHOLD

new_transactions_file_uphold = [file_os for file_os in os.listdir() if "-transactions.csv" in file_os]
if len(new_transactions_file_uphold) > 0:
    uphold_df = pd.DataFrame()
    new_uphold_df = pd.read_csv(new_transactions_file_uphold[0])
    for i in range(new_uphold_df.shape[0]):
        if new_uphold_df.loc[i,'Destination'] != 'uphold':
            new_uphold_df.loc[i, 'Destination Amount'] *= -1
            if not pd.isna(new_uphold_df.loc[i,'Fee Amount']):
                new_uphold_df.loc[i, 'Destination Amount'] += new_uphold_df.loc[i,'Fee Amount']*-1
        if new_uphold_df.loc[i,'Origin Currency'] == new_uphold_df.loc[i,'Destination Currency']:
            new_uphold_df.loc[i, 'Origin Amount'] = 0
        else:
            new_uphold_df.loc[i, 'Origin Amount'] *= -1
        new_uphold_df.loc[i,'Date'] = tx.uphold_date_to_date(new_uphold_df.loc[i,'Date'][4:24])
    serie_1 = new_uphold_df[['Date','Destination Currency','Destination Amount']]
    serie_1.columns = ['Date','Currency','Amount']
    serie_2 = new_uphold_df[['Date', 'Origin Currency', 'Origin Amount']]
    serie_2.columns = ['Date', 'Currency', 'Amount']
    uphold_transactions = serie_1.append(serie_2)
    uphold_transactions.index = uphold_transactions['Date']
    uphold_transactions.drop(['Date'],axis=1,inplace=True)
    for token in np.unique(uphold_transactions['Currency']):
        temp = uphold_transactions[uphold_transactions['Currency'] == token]
        temp.drop(['Currency'],axis=1,inplace=True)
        temp.columns = [token]
        temp = temp.groupby(temp.index).sum()
        uphold_df = uphold_df.join(temp, how='outer')
    uphold_df.fillna(0,inplace=True)
    uphold_df = uphold_df.cumsum(axis=0)
    uphold_df.drop(['EUR'], inplace=True, axis=1)