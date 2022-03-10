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
from forex_python.converter import get_rate

anno_fiscale = 2021
soglia_totale = pd.DataFrame()
interessi_totali = pd.DataFrame()
giacenza_media = pd.DataFrame()
carta = pd.DataFrame()

print(f'Last updated: 10/03/2022')

# CARTA CDC
new_transactions_file = [file_os for file_os in os.listdir() if "card_transactions_record" in file_os]

if len(new_transactions_file) > 0:
    print("Getting data from CDC Card. It may take a while.")
    if 'cryptodotcom_card.pickle' in os.listdir():
        with open('cryptodotcom_card.pickle', 'rb') as handle:
            old_card_transactions = pk.load(handle)
        old_card_transactions.index = [tx.str_to_datetime(x) for x in old_card_transactions.iloc[:,0]]
        card_transactions = pd.read_csv(new_transactions_file[0])
        for i in range(card_transactions.shape[0]):
            if card_transactions.loc[i, "Native Currency"] != "EUR":
                conv = get_rate(card_transactions.loc[i, "Native Currency"], "EUR", tx.str_to_datetime(card_transactions.iloc[i, 0]).date())
                card_transactions.loc[i, "Native Amount"] *= float(conv)
        card_transactions.index = [tx.str_to_datetime(k) for k in card_transactions.iloc[:, 0]]
        card_transactions.sort_index(inplace=True)
        old_card_transactions = old_card_transactions.loc[old_card_transactions.index < card_transactions.index[0],].copy()
        new_old_card = pd.concat([old_card_transactions, card_transactions])
        new_old_card.sort_index(inplace=True)
        new_old_card.drop_duplicates(subset=['Timestamp (UTC)'], inplace=True)

        with open('cryptodotcom_card.pickle', 'wb') as handle:
            pk.dump(new_old_card, handle, protocol=pk.HIGHEST_PROTOCOL)
        card_transactions = new_old_card.copy()

    else:
        card_transactions = pd.read_csv(new_transactions_file[0])
        for i in range(card_transactions.shape[0]):
            if card_transactions.loc[i, "Native Currency"] != "EUR":
                conv = get_rate(card_transactions.loc[i, "Native Currency"], "EUR", tx.str_to_datetime(card_transactions.iloc[i, 0]).date())
                card_transactions.loc[i, "Native Amount"] /= float(conv)
        card_transactions.sort_index(inplace=True)
        card_transactions.drop_duplicates(subset=['Timestamp (UTC)'], inplace=True)

        with open('cryptodotcom_card.pickle', 'wb') as handle:
            pk.dump(card_transactions, handle, protocol=pk.HIGHEST_PROTOCOL)
    shutil.move(new_transactions_file[0], "cryptodotcom_history")
else:
    print("Getting data from CDC Card. It may take a while.")
    with open('cryptodotcom_card.pickle', 'rb') as handle:
        card_transactions = pk.load(handle)


card_transactions.index = [tx.str_to_datetime(x) for x in card_transactions.iloc[:,0]]
card_transactions = card_transactions[['Transaction Description', 'Native Amount']]
card_transactions.index = [k.date() for k in card_transactions.index]
card_transactions = pd.DataFrame(card_transactions.groupby(card_transactions.index).sum())
card_transactions_giacenza = pd.DataFrame(card_transactions['Native Amount'].cumsum())
card_transactions_giacenza['Anno'] = [k.year for k in card_transactions_giacenza.index]
card_transactions_giacenza.drop(['Anno'], axis=1, inplace=True)


carta = card_transactions_giacenza.loc[card_transactions_giacenza['Anno'] == anno_fiscale].copy()
carta.drop(['Anno'], axis=1, inplace=True)

giacenza_media.loc[0, 'Carta Crypto.com'] = carta.sum(axis=1)[carta.sum(axis=1) != 0].sum(axis=0) / \
                                          carta.sum(axis=1)[carta.sum(axis=1) != 0].shape[0]


# PER CRYPTO AL PRIMO UTILIZZO PRENDERE TUTTO LO STORICO DALL'INIZIO DELL'ACCOUNT
# PER AGGIORNAMENTI ESTRARRE LO STORICO INIZIANDO ALMENO DA DATA - 2
# PER ORA SOLTANTO UN FILE CON NUOVE TRANSAZIONI PUO ESSERE GESTITO ALLA VOLTA

new_transactions_file = [file_os for file_os in os.listdir() if "crypto_transactions_record" in file_os]
history_file = [file_os for file_os in os.listdir() if file_os == "cryptodotcom.pickle"]

print("CRYPTO.COM APP")
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

        cryptodotcom_dict['soglia'] = new_history['soglia']
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

cryptodotcom_dict['cashback-EUR']['Anno'] = [k.year for k in cryptodotcom_dict['cashback-EUR'].index]
cashback_totale = cryptodotcom_dict['cashback-EUR'][cryptodotcom_dict['cashback-EUR']['Anno'] == anno_fiscale].copy()
cashback_totale.drop(['Anno'], axis=1, inplace=True)

soglia_cdc = cryptodotcom_dict['soglia'].loc[cryptodotcom_dict['soglia']['Anno'] == anno_fiscale, :].copy()
soglia_cdc.drop(['Anno'], axis=1, inplace=True)
soglia_totale['Crypto.com App'] = soglia_cdc.sum(axis=1)

cryptodotcom_dict['interest-EUR']['Anno'] = [k.year for k in cryptodotcom_dict['interest-EUR'].index]
interesse_cdc = cryptodotcom_dict['interest-EUR'].loc[cryptodotcom_dict['interest-EUR']['Anno'] == anno_fiscale,
                :].copy()
interesse_cdc.drop(['Anno'], axis=1, inplace=True)
interessi_totali.loc[0, 'Crypto.com App'] = interesse_cdc.sum(axis=1).sum(axis=0)

cryptodotcom_dict['balances-EUR']['Anno'] = [k.year for k in cryptodotcom_dict['balances-EUR'].index]
giacenza_cdc = cryptodotcom_dict['balances-EUR'].loc[cryptodotcom_dict['balances-EUR']['Anno'] == anno_fiscale,
               :].copy()
giacenza_cdc.drop(['Anno'], axis=1, inplace=True)
giacenza_media.loc[0, 'Crypto.com App'] = giacenza_cdc.sum(axis=1)[giacenza_cdc.sum(axis=1) != 0].sum(axis=0) / \
                                          giacenza_cdc.sum(axis=1)[giacenza_cdc.sum(axis=1) != 0].shape[0]
print("CRYPTO.COM APP DONE")

# BINANCE
print("BINANCE")
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
        binance_transactions_df = pd.concat([new_df, old_df])
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

    start_date = dt.date(2021, 1, 1)
    if 'binance_prices.pickle' in os.listdir():
        with open('binance_prices.pickle', 'rb') as handle:
            price_dict = pk.load(handle)
        lastup = []
        for key in list(price_dict.keys()):
            if not isinstance(price_dict[key], int):
                lastup.append(price_dict[key][-1][0])
        if isinstance(lastup[0], list) or isinstance(lastup[1], list) or isinstance(lastup[2], list):
            lastup = [k if isinstance(k, dt.date) else k[0] for k in lastup]
        start_date = min(lastup)
    if binance_transactions_df.index[-1] > pd.Timestamp(start_date) or 'binance_prices.pickle' not in os.listdir():
        # price_dict = dict()
        temp_price = binance_client.get_historical_klines("EURBUSD", binance_client.KLINE_INTERVAL_1DAY,
                                                          tx.to_binance_date_format(start_date))
        temp_price = [[dt.datetime.fromtimestamp(i[0] / 1000).date(), i[1]] for i in temp_price]
        if 'EUR' not in list(price_dict.keys()):
            price_dict['EUR'] = temp_price
        else:
            price_dict['EUR'].append(temp_price)
        for coin in np.unique(binance_transactions_df['Coin']):
            print(f'Getting price for {coin} - Binance')
            if coin not in ['BUSD', 'USDC', 'USDT', 'UST']:
                try:
                    temp_price = binance_client.get_historical_klines(f"{coin}BUSD",
                                                                      binance_client.KLINE_INTERVAL_1DAY,
                                                                      start_str=tx.to_binance_date_format(start_date))
                    temp_price = [[dt.datetime.fromtimestamp(i[0] / 1000).date(), i[1]] for i in temp_price]
                    if coin not in list(price_dict.keys()):
                        price_dict[coin] = temp_price
                    else:
                        price_dict[coin].append(temp_price)
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

    index_temp = pd.date_range(dt.date(total_binance_transactions_nat.index[0].year, 1, 1),
                               total_binance_transactions_nat.index[-1])
    temp_df_1 = pd.DataFrame(index=index_temp, data=[np.nan] * len(index_temp), columns=['TEMP'])
    total_binance_transactions_nat = total_binance_transactions_nat.join(temp_df_1, how='outer')
    total_binance_transactions_nat.drop(['TEMP'], axis=1, inplace=True)
    total_binance_transactions_nat.iloc[0, :].fillna(0, inplace=True)
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

    output_df_soglia = copy.deepcopy(total_binance_transactions_nat)
    coin_names = output_df_soglia.columns
    output_df_soglia['Anno'] = [k.year for k in total_binance_transactions_nat.index]
    for y in np.unique(output_df_soglia['Anno']):
        for coin in coin_names:
            if coin in ['USDC', 'USDT', 'BUSD', 'UST']:
                coin_price = [x[1] for x in price_dict['EUR'] if x[0] == pd.Timestamp(dt.date(y, 1, 1))]
            else:
                if price_dict[coin] == 0:
                    coin_price = [0]
                else:
                    coin_price = [x[1] for x in price_dict[coin] if x[0] == dt.date(y, 1, 1)]
            if len(coin_price) == 0:
                coin_price = [price_dict[coin][0][1]]
            output_df_soglia.loc[output_df_soglia['Anno'] == y, coin] *= float(coin_price[0])

    binance_dict = dict()
    output_df_soglia.index = [k.date() for k in output_df_soglia.index]
    binance_dict['soglia'] = output_df_soglia
    binance_dict['balances'] = total_binance_transactions_nat
    binance_dict['balances-EUR'] = total_binance_transactions_eur
    binance_dict['interest-Crypto'] = binance_interest_nat
    binance_dict['interest-EUR'] = binance_interest_eur

    with open('binance.pickle', 'wb') as handle:
        pk.dump(binance_dict, handle, protocol=pk.HIGHEST_PROTOCOL)
else:
    with open('binance.pickle', 'rb') as handle:
        binance_dict = pk.load(handle)

soglia_binance = binance_dict['soglia'].loc[binance_dict['soglia']['Anno'] == anno_fiscale, :].copy()
soglia_binance.drop(['Anno'], axis=1, inplace=True)
soglia_totale['Binance'] = soglia_binance.sum(axis=1)

binance_dict['interest-EUR']['Anno'] = [k.year for k in binance_dict['interest-EUR'].index]
interesse_binance = binance_dict['interest-EUR'].loc[binance_dict['interest-EUR']['Anno'] == anno_fiscale, :].copy()
interesse_binance.drop(['Anno'], axis=1, inplace=True)
interessi_totali.loc[0, 'Binance'] = interesse_binance.sum(axis=1).sum(axis=0)

binance_dict['balances-EUR']['Anno'] = [k.year for k in binance_dict['balances-EUR'].index]
giacenza_binance = binance_dict['balances-EUR'].loc[binance_dict['balances-EUR']['Anno'] == anno_fiscale, :].copy()
giacenza_binance.drop(['Anno'], axis=1, inplace=True)
giacenza_media.loc[0, 'Binance'] = giacenza_binance.sum(axis=1)[giacenza_binance.sum(axis=1) != 0].sum(axis=0) / \
                                   giacenza_binance.sum(axis=1)[giacenza_binance.sum(axis=1) != 0].shape[0]
print("BINANCE DONE")

# CARDANO
print("CARDANO NON CUSTODIAL")
ada_addresses = pd.read_csv("wallets.csv")
ada_addresses = ada_addresses['ADA'].copy()
ada_addresses.dropna(inplace=True)
ada_addresses = list(ada_addresses)

first_delegate_day = [dt.datetime(2022, 2, 5, 14, 30, 53)]  # 14,30,53)]
intrawallet_df = pd.DataFrame()
if "intrawallet_transactions_ada.csv" in os.listdir():
    with open("intrawallet_transactions_ada.csv", "r") as handle:
        intrawallet_transactions = pd.read_csv(handle)
    intra_index = []
    [exec("intra_index.append(dt.datetime(" + ",".join(list(intrawallet_transactions.iloc[i, :].astype(str))) + "))")
     for i in range(intrawallet_transactions.shape[0])]
    intrawallet_df = pd.DataFrame([-0.172761] * len(intra_index), index=intra_index)
out = pd.DataFrame()
for address in ada_addresses:

    req_response = requests.get(f"https://api.blockchair.com/cardano/raw/address/{address}")
    req_response = req_response.json()
    tx_list = req_response['data'][address]['address']['caTxList']

    tx_list_in = []
    tx_list_out = []
    fees_tx = []
    for txl in tx_list:
        list_temp = [(dt.datetime.fromtimestamp(txl['ctbTimeIssued']),
                      int(txl['ctbOutputs'][i]['ctaAmount']['getCoin']) / 10 ** 6) for i in
                     range(len(txl['ctbOutputs'])) if
                     txl['ctbOutputs'][i]['ctaAddress'] == address]
        tx_list_out.extend(list_temp)
        list_temp = [(dt.datetime.fromtimestamp(txl['ctbTimeIssued']),
                      -int(txl['ctbInputs'][i]['ctaAmount']['getCoin']) / 10 ** 6) for i in range(len(txl['ctbInputs']))
                     if
                     txl['ctbInputs'][i]['ctaAddress'] == address]
        tx_list_in.extend(list_temp)
        fees_tx.append((dt.datetime.fromtimestamp(txl['ctbTimeIssued']), -int(txl['ctbFees']['getCoin']) / 10 ** 6))

    tx_list_in = pd.DataFrame(tx_list_in)
    if tx_list_in.shape[0] != 0:
        tx_series_in = pd.DataFrame(tx_list_in.iloc[:, 1].tolist(), index=tx_list_in.iloc[:, 0].tolist(),
                                    columns=["IN"])

    tx_list_out = pd.DataFrame(tx_list_out)
    if tx_list_out.shape[0] != 0:
        tx_series_out = pd.DataFrame(tx_list_out.iloc[:, 1].tolist(), index=tx_list_out.iloc[:, 0].tolist(),
                                     columns=["OUT"])

    if "tx_series_in" in globals().keys() and "tx_series_out" in globals().keys():
        tx_inout = tx_series_in.join(tx_series_out, lsuffix="l", how='outer')
    elif "tx_series_in" not in globals().keys() and "tx_series_out" in globals().keys():
        tx_series_in = copy.deepcopy(tx_series_out)
        tx_inout = tx_series_in.join(tx_series_out, lsuffix="l", how='outer')
        tx_inout['OUTl'] = 0
        tx_inout.columns = ["IN", "OUT"]
    elif "tx_series_in" in globals().keys() and "tx_series_out" not in globals().keys():
        tx_series_out = copy.deepcopy(tx_series_in)
        tx_inout = tx_series_in.join(tx_series_out, lsuffix="l", how='outer')
        tx_inout['INl'] = 0
        tx_inout.columns = ["IN", "OUT"]

    fees_tx = pd.DataFrame(fees_tx)
    tx_series_fee = pd.DataFrame(fees_tx.iloc[:, 1].tolist(), index=fees_tx.iloc[:, 0].tolist(), columns=["FEES"])
    tx_series_fee = tx_series_fee.groupby(tx_series_fee.index).sum()

    tx_inout_fees = tx_inout.join(tx_series_fee, lsuffix="l", how='outer')
    tx_inout_fees.fillna(0, inplace=True)

    # for i in range(tx_inout_fees.shape[0]):
    #     if tx_inout_fees.iloc[i,0] == -tx_inout_fees.iloc[i,1]:
    #         tx_inout_fees.iloc[i, 1] = -2

    del (tx_series_in, tx_series_out)

    out = pd.concat([out, tx_inout_fees])

if first_delegate_day is not None:
    for delegate in first_delegate_day:
        out.loc[delegate, "IN"] = 2

out = pd.DataFrame(out[['IN', 'OUT']].sum(axis=1)).copy()
if intrawallet_df.shape[0] > 0:
    out = out.join(intrawallet_df, lsuffix='l', how='outer')
    out.fillna(0, inplace=True)
    out = pd.DataFrame(out.sum(axis=1))

out.index = [k.date() for k in out.index]
out = out.groupby(out.index).sum()
out = out.cumsum()

index_temp = pd.date_range(dt.date(out.index[0].year, 1, 1), dt.datetime.today().date())
temp_df_1 = pd.DataFrame(index=index_temp, data=[np.nan] * len(index_temp), columns=['TEMP'])
out = out.join(temp_df_1, lsuffix='l', how='outer')
out.drop(['TEMP'], axis=1, inplace=True)
out.columns = ['ADA']
out.iloc[0, :].fillna(0, inplace=True)
out.ffill(inplace=True)
cardano_ADA = copy.deepcopy(out)
cardano_ADA[cardano_ADA < 0] = 0

if 'cardano_price.pickle' in os.listdir():
    with open('cardano_price.pickle', 'rb') as handle:
        prices_dict = pk.load(handle)
    timeframe = (dt.datetime.today().date() - prices_dict['ADA'][-1][0]).days + 1
    if timeframe > 30:
        timeframe = 'max'
    prices_new = tx.get_token_prices(tokens=['ADA'], contracts=[0], networks=['cardano'], timeframe=timeframe,
                                     currency='eur')
    prices_dict['ADA'].extend(prices_new['ADA'])
    with open('cardano_price.pickle', 'wb') as handle:
        pk.dump(prices_dict, handle, protocol=pk.HIGHEST_PROTOCOL)
else:
    timeframe = (dt.datetime.today().date() - tx.str_to_datetime(str(cardano_ADA['ADA'].index[0])).date()).days + 1
    if timeframe > 365:
        timeframe = 'max'
    else:
        timeframe = 365
    prices_dict = tx.get_token_prices(tokens=['ADA'], contracts=[0], networks=['cardano'], timeframe=timeframe,
                                      currency='eur')

    if prices_dict['ADA'][0][0] > tx.str_to_datetime(str(cardano_ADA['ADA'].index[0])).date():
        print('Importing ADA price day by day to cover the whole period, it may take a long time')
        import time

        date_range = pd.date_range(tx.str_to_datetime(str(cardano_ADA['ADA'].index[0])).date(),
                                   prices_dict['ADA'][0][0])
        new_prices = []
        counter = len(date_range)
        for date_loop in date_range:
            i_date = tx.datetime_to_str(tx.str_to_datetime(str(date_loop)), False)
            i_date = f'{i_date[8:10]}-{i_date[5:7]}-{i_date[0:4]}'
            temp = requests.get(f'https://api.coingecko.com/api/v3/coins/cardano/history?date={i_date}')
            if temp.status_code == 429:
                print('Too many requests for server, sleeping for 70 seconds')
                time.sleep(70)
                print('Retrying')
                temp = requests.get(f'https://api.coingecko.com/api/v3/coins/cardano/history?date={i_date}')
            temp = temp.json()
            temp = temp['market_data']['current_price']['eur']
            new_prices.append([tx.str_to_datetime(str(date_loop)).date(), float(temp)])
            counter -= 1
            if counter % 10 == 0:
                print(f'{counter} days left')
            new_prices.extend(prices_dict['ADA'])
        prices_dict['ADA'] = new_prices
        with open('cardano_price.pickle', 'wb') as handle:
            pk.dump(prices_dict, handle, protocol=pk.HIGHEST_PROTOCOL)

cardano_eur = copy.deepcopy(cardano_ADA)
cardano_eur.index = cardano_eur.index = [dt.date.fromisoformat(k.isoformat()[0:10]) for k in cardano_eur.index]

for date_loop in cardano_eur.index:
    price = [float(i[1]) for i in prices_dict['ADA'] if i[0] == date_loop]
    if len(price) == 0:
        price = 0
        print(f'Price for date {tx.datetime_to_str(date_loop, False)} not found, defaulting to zero')
    cardano_eur.loc[date_loop, 'ADA'] *= price[0]

cardano_ADA_soglia = copy.deepcopy(cardano_ADA)
cardano_ADA_soglia['Anno'] = [k.year for k in cardano_ADA.index]
for y in np.unique(cardano_ADA_soglia['Anno']):
    first_of_year = requests.get(f'https://api.coingecko.com/api/v3/coins/cardano/history?date=01-01-{y}')
    first_of_year = first_of_year.json()
    first_of_year_price = first_of_year['market_data']['current_price']['eur']
    cardano_ADA_soglia.loc[cardano_ADA_soglia['Anno'] == y, 'ADA'] *= float(first_of_year_price)

soglia_cardano = cardano_ADA_soglia.loc[cardano_ADA_soglia['Anno'] == anno_fiscale, :].copy()
soglia_cardano.drop(['Anno'], axis=1, inplace=True)
soglia_totale['Cardano Non-Custodial'] = soglia_cardano

cardano_eur['Anno'] = [k.year for k in cardano_eur.index]
giacenza_cardano = cardano_eur.loc[cardano_eur['Anno'] == anno_fiscale, :].copy()
giacenza_cardano.drop(['Anno'], axis=1, inplace=True)
giacenza_media.loc[0, 'Cardano Non-Custodial'] = giacenza_cardano.sum(axis=1)[giacenza_cardano.sum(axis=1) != 0].sum(
    axis=0) / giacenza_cardano.sum(axis=1)[giacenza_cardano.sum(axis=1) != 0].shape[0]
print("CARDANO NON CUSTODIAL DONE")
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
print("ETHEREUM NON CUSTODIAL")
addresses = pd.read_csv("wallets.csv")
addresses = addresses['ETH'].copy()
addresses.dropna(inplace=True)
addresses = list(addresses)

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
        direction_internal = [1 if k['to'].upper() == address.upper() else -1 for k in
                              response_internal.json()['result']]
        balance_eth_internal = [int(k['value']) / 10 ** 18 for k in response_internal.json()['result']]
        balance_eth_internal = [balance_eth_internal[i] * direction_internal[i] for i in range(len(direction_internal))]
        # gas_internal = [-1*int(k['gas'])*int(k['gasPrice'])/10**18 for k in response_internal.json()['result']]
        eth_index_internal = [dt.datetime.fromtimestamp(int(k['timeStamp'])) for k in
                              response_internal.json()['result']]
        internal_df = pd.DataFrame(index=eth_index_internal, data=balance_eth_internal)
        eth_df = eth_df.join(internal_df, lsuffix=" Internal ", how='outer')
        eth_df = pd.DataFrame(eth_df.sum(axis=1))

    eth_df.columns = ['ETH']

    if response_erc20.json()['message'] == 'OK':
        tokens_erc20 = [k['tokenSymbol'].upper() for k in response_erc20.json()['result']]
        gas_erc20 = [0 if k['to'].upper() == address.upper() else -1 * int(k['gasUsed']) * int(k['gasPrice']) / 10 ** 18
                     for
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

    eth_df = pd.concat([eth_df, fill_na])
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
    if f'{address[1:10]}_price.pickle' in os.listdir():
        with open(f'{address[1:10]}_price.pickle', 'rb') as handle:
            prices_dict = pk.load(handle)
        lastup = []
        for key in list(prices_dict.keys()):
            if not isinstance(prices_dict[key], int):
                lastup.append(prices_dict[key][-1][0])
        start_date = min(lastup)
        if start_date < dt.datetime.today().date():
            timeframe = (dt.datetime.today().date() - start_date).days + 1
            if timeframe > 365:
                timeframe = 'max'
            else:
                timeframe = 365
            new_prices = tx.get_token_prices(tokens=list(eth_df.columns), contracts=contract_list,
                                             networks=network_list,
                                             timeframe=timeframe, currency='eur')
            for key in list(new_prices.keys()):
                if key not in list(prices_dict.keys()):
                    prices_dict[key] = new_prices[key]
                else:
                    prices_dict[key].extend(new_prices[key])
            with open(f'{address[1:10]}_price.pickle', 'wb') as handle:
                pk.dump(prices_dict, handle, protocol=pk.HIGHEST_PROTOCOL)
    else:
        timeframe = (eth_df.index[-1].date() - eth_df.index[0].date()).days + 1
        if timeframe > 30:
            timeframe = 'max'
        prices_dict = tx.get_token_prices(tokens=list(eth_df.columns), contracts=contract_list, networks=network_list,
                                          timeframe=timeframe, currency='eur')
        with open(f'{address[1:10]}_price.pickle', 'wb') as handle:
            pk.dump(prices_dict, handle, protocol=pk.HIGHEST_PROTOCOL)
    print("CORREGGERE GESTIONE PREZZI STABLECOIN IN ETHEREUM")
    eth_df_EUR = copy.deepcopy(eth_df)
    eth_df_EUR.index = eth_df.index = [dt.date.fromisoformat(k.isoformat()[0:10]) for k in eth_df_EUR.index]
    for coin in eth_df_EUR.columns:
        for date_loop in eth_df_EUR.index:
            price = [i[1] for i in prices_dict[coin] if i[0] == date_loop]
            eth_df_EUR.loc[date_loop, coin] *= price[0]

    eth_df_soglia = copy.deepcopy(eth_df)
    coin_names = eth_df_soglia.columns
    eth_df_soglia['Anno'] = [k.year for k in eth_df_soglia.index]
    for y in np.unique(eth_df_soglia['Anno']):
        for coin in coin_names:
            if coin in ['USDC', 'USDT', 'BUSD', 'UST']:
                coin_price = [x[1] for x in prices_dict['EUR'] if x[0] == dt.date(y, 1, 1)]
            else:
                if prices_dict[coin] == 0:
                    coin_price = [0]
                else:
                    coin_price = [x[1] for x in prices_dict[coin] if x[0] == dt.date(y, 1, 1)]
            if len(coin_price) == 0:
                coin_price = [prices_dict[coin][0][1]]
            eth_df_soglia.loc[eth_df_soglia['Anno'] == y, coin] *= float(coin_price[0])

    index_temp = pd.date_range(dt.date(eth_df.index[0].year, 1, 1), dt.datetime.today().date())
    temp_df_1 = pd.DataFrame(index=index_temp, data=[np.nan] * len(index_temp), columns=['TEMP'])

    soglia_eth = eth_df_soglia.loc[eth_df_soglia['Anno'] == anno_fiscale, :].copy()
    eth_df_soglia.drop(['Anno'], axis=1, inplace=True)

    eth_df = temp_df_1.join(eth_df)
    eth_df.drop(['TEMP'], axis=1, inplace=True)
    eth_df.fillna(0, inplace=True)

    eth_df_soglia = temp_df_1.join(eth_df_soglia)
    eth_df_soglia.drop(['TEMP'], axis=1, inplace=True)
    eth_df_soglia.fillna(0, inplace=True)

    eth_df_EUR = temp_df_1.join(eth_df_EUR)
    eth_df_EUR.drop(['TEMP'], axis=1, inplace=True)
    eth_df_EUR.fillna(0, inplace=True)

    soglia_totale[f'ETH Non-Custodial {address[3:10]}'] = eth_df_soglia.sum(axis=1)

    eth_df_EUR['Anno'] = [k.year for k in eth_df_EUR.index]
    giacenza_eth = eth_df_EUR.loc[eth_df_EUR['Anno'] == anno_fiscale, :].copy()
    giacenza_eth.drop(['Anno'], axis=1, inplace=True)
    giacenza_media.loc[0, f'ETH Non-Custodial {address[3:10]}'] = giacenza_eth.sum(axis=1)[
                                                                      giacenza_eth.sum(axis=1) != 0].sum(axis=0) / \
                                                                  giacenza_eth.sum(axis=1)[
                                                                      giacenza_eth.sum(axis=1) != 0].shape[0]

    exec(f'eth{address[3:10]}=dict()')
    exec(f'eth{address[3:10]}["balance"]=eth_df')
    exec(f'eth{address[3:10]}["balance-EUR"]=eth_df_EUR')
    exec(f'eth{address[3:10]}["soglia"]=eth_df_soglia')
    file_1 = f'eth{address[3:10]}'
    exec(f'with open("{file_1}.pickle", "wb") as handle: '
         f'pk.dump({exec(file_1)}, handle, protocol=pk.HIGHEST_PROTOCOL)')
print("ETHEREUM NON CUSTODIAL DONE")
# tokens=list(eth_df.columns)
# contracts= contract_list
# networks= network_list
# currency='usd'

# COINBASE - STORICO scaricato dall'API
print("COINBASE")
from coinbase.wallet.client import Client as CoinbaseClient

client = CoinbaseClient("g5LIAFMd6yvOxSKl", "5Cf0ymwH2Tr3rjuhaVclnbRoC4vK7BhX")
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
        if token[0] == 'CGLD':
            token[0] = 'CELO'
        temp.columns = [token[0]]
        coinbase_df = coinbase_df.join(temp, how='outer')  # , lsuffix=' L ')

        amount1 = [eur[i] for i in range(len(eur)) if isearn[i]]
        if len(amount1) > 0:
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

index_temp = pd.date_range(dt.date(coinbase_df.index[0].year, 1, 1), dt.date.today())
temp_df_1 = pd.DataFrame(index=index_temp, data=[np.nan] * len(index_temp), columns=['TEMP'])
coinbase_df = coinbase_df.join(temp_df_1, how='outer')
coinbase_df.drop(['TEMP'], axis=1, inplace=True)
coinbase_df.fillna(0, inplace=True)
coinbase_df.index = [dt.date.fromisoformat(k.isoformat()[0:10]) for k in coinbase_df.index]
coinbase_df = coinbase_df.groupby(coinbase_df.index).sum()

coinbase_df = coinbase_df.cumsum(axis=0)

from_date = coinbase_df.index[0].strftime('%Y-%m-%dT00:00:00')
to_date = coinbase_df.index[-1].strftime('%Y-%m-%dT00:00:00')

if 'coinbase_price.pickle' in os.listdir():
    with open('coinbase_price.pickle', 'rb') as handle:
        coinbase_prices = pk.load(handle)
    lastup = []
    for key in list(coinbase_prices.keys()):
        if isinstance(coinbase_prices[key], int):
            continue
        lastup.append(coinbase_prices[key][-1][0])
    start_date = min(lastup)
    if start_date < dt.datetime.today().date():
        timeframe = (dt.datetime.today().date() - start_date).days
        if timeframe > 365:
            timeframe = 'max'
        else:
            timeframe = 365
        c_list = [0] * len(coinbase_df.columns)
        n_list = c_list
        temp = tx.get_token_prices(coinbase_df.columns, c_list, n_list, timeframe, 'eur')
        for key2 in list(temp.keys()):
            if isinstance(temp[key2], int):
                continue
            if key2 not in list(coinbase_prices.keys()):
                coinbase_prices[key2] = temp[key2]
            else:
                temp_loop = [x for x in temp[key2] if x[0] >= start_date]
                coinbase_prices[key2].extend(temp_loop)
        with open('coinbase_price.pickle', 'wb') as handle:
            pk.dump(coinbase_prices, handle, protocol=pk.HIGHEST_PROTOCOL)
else:
    c_list = [0] * len(coinbase_df.columns)
    n_list = c_list
    coinbase_prices = tx.get_token_prices(coinbase_df.columns, c_list, n_list, 'max', 'eur')
    with open('coinbase_price.pickle', 'wb') as handle:
        pk.dump(coinbase_prices, handle, protocol=pk.HIGHEST_PROTOCOL)

coinbase_df_EUR = copy.deepcopy(coinbase_df)
coinbase_df_EUR.index = coinbase_df.index
for coin in coinbase_df_EUR.columns:
    print(f'Preparing EUR dataframe for Coinbase for coin {coin}')
    for date_loop in coinbase_df_EUR.index:
        if isinstance(coinbase_prices[coin], int):
            price = [0]
        else:
            price = [i[1] for i in coinbase_prices[coin] if i[0] == tx.str_to_datetime(str(date_loop)).date()]
        if len(price) == 0:
            loop_control = 0
            dateloop2 = copy.deepcopy(date_loop)
            while len(price) == 0 and loop_control < 50:
                loop_control += 1
                price = [i[1] for i in coinbase_prices[coin] if
                         i[0] == tx.str_to_datetime(str(dateloop2)).date() - dt.timedelta(days=1)]
                dateloop2 -= dt.timedelta(days=1)
            if loop_control > 0:
                print(
                    f'Coinbase - could not find price for date {tx.datetime_to_str(date_loop, False)} for coin {coin} could not be found using date {tx.datetime_to_str(dateloop2, False)}')
            if len(price) == 0:
                price = [0]
        coinbase_df_EUR.loc[date_loop, coin] *= price[0]

coinbase_df_soglia = copy.deepcopy(coinbase_df)
coin_names = coinbase_df_soglia.columns
coinbase_df_soglia['Anno'] = [k.year for k in coinbase_df_soglia.index]
for y in np.unique(coinbase_df_soglia['Anno']):
    for coin in coin_names:
        if coinbase_prices[coin] == 0 or coinbase_prices[coin] == 1:
            coin_price = [0]
        else:
            coin_price = [x[1] for x in coinbase_prices[coin] if x[0] == dt.date(y, 1, 1)]
            if len(coin_price) == 0:
                coin_price = [coinbase_prices[coin][0][1]]
        coinbase_df_soglia.loc[coinbase_df_soglia['Anno'] == y, coin] *= float(coin_price[0])

coinbase_df_soglia[coinbase_df_soglia < 0] = 0

temp_soglia = coinbase_df_soglia[coinbase_df_soglia['Anno'] == anno_fiscale].copy()
temp_soglia.drop(['Anno'], axis=1, inplace=True)
soglia_totale["Coinbase"] = temp_soglia.sum(axis=1)

coinbase_interests_and_earn_eur['Anno'] = [k.year for k in coinbase_interests_and_earn_eur.index]
temp_interest = coinbase_interests_and_earn_eur[coinbase_interests_and_earn_eur['Anno'] == anno_fiscale].copy()
temp_interest.drop(['Anno'], axis=1, inplace=True)
interessi_totali.loc[0, "Coinbase"] = temp_interest.sum(axis=1).sum(axis=0)

giacenza_coinbase = copy.deepcopy(coinbase_df_EUR)
giacenza_coinbase['Anno'] = [k.year for k in giacenza_coinbase.index]
giacenza_coinbase = giacenza_coinbase.loc[giacenza_coinbase['Anno'] == anno_fiscale, :].copy()
giacenza_coinbase.drop(['Anno'], axis=1, inplace=True)
giacenza_media.loc[0, 'Coinbase'] = giacenza_coinbase.sum(axis=1)[giacenza_coinbase.sum(axis=1) != 0].sum(axis=0) / \
                                    giacenza_coinbase.sum(axis=1)[giacenza_coinbase.sum(axis=1) != 0].shape[0]
print("COINBASE DONE")
# UPHOLD
print("UPHOLD")
if 'uphold.pickle' in os.listdir():
    with open('uphold.pickle', 'rb') as handle:
        uphold_dict = pk.load(handle)
else:
    uphold_dict = dict()
new_transactions_file_uphold = [file_os for file_os in os.listdir() if "-transactions.csv" in file_os]
if len(new_transactions_file_uphold) > 0:
    uphold_df = pd.DataFrame()
    new_uphold_df = pd.read_csv(new_transactions_file_uphold[0])
    for i in range(new_uphold_df.shape[0]):
        if new_uphold_df.loc[i, 'Destination'] != 'uphold':
            new_uphold_df.loc[i, 'Destination Amount'] *= -1
            if not pd.isna(new_uphold_df.loc[i, 'Fee Amount']):
                new_uphold_df.loc[i, 'Destination Amount'] += new_uphold_df.loc[i, 'Fee Amount'] * -1
        if new_uphold_df.loc[i, 'Origin Currency'] == new_uphold_df.loc[i, 'Destination Currency']:
            new_uphold_df.loc[i, 'Origin Amount'] = 0
        else:
            new_uphold_df.loc[i, 'Origin Amount'] *= -1
        new_uphold_df.loc[i, 'Date'] = tx.uphold_date_to_date(new_uphold_df.loc[i, 'Date'][4:24])
    serie_1 = new_uphold_df[['Date', 'Destination Currency', 'Destination Amount']]
    serie_1.columns = ['Date', 'Currency', 'Amount']
    serie_2 = new_uphold_df[['Date', 'Origin Currency', 'Origin Amount']]
    serie_2.columns = ['Date', 'Currency', 'Amount']
    uphold_transactions = pd.concat([serie_1, serie_2])
    uphold_transactions.index = uphold_transactions['Date']
    uphold_transactions.drop(['Date'], axis=1, inplace=True)
    for token in np.unique(uphold_transactions['Currency']):
        temp = uphold_transactions[uphold_transactions['Currency'] == token].copy()
        temp.drop(['Currency'], axis=1, inplace=True)
        temp.columns = [token]
        temp = temp.groupby(temp.index).sum()
        uphold_df = uphold_df.join(temp, how='outer')
    uphold_df.fillna(0, inplace=True)
    uphold_df = uphold_df.cumsum(axis=0)
    uphold_df.drop(['EUR'], inplace=True, axis=1)

    c_list = [0] * len(uphold_df.columns)
    n_list = c_list
    tokens_prices = tx.get_token_prices(uphold_df.columns, c_list, n_list, 'max', 'eur')

    index_temp = pd.date_range(dt.date(uphold_df.index[0].year, 1, 1), uphold_df.index[-1])
    temp_df_1 = pd.DataFrame(index=index_temp, data=[np.nan] * len(index_temp), columns=['TEMP'])
    uphold_df = uphold_df.join(temp_df_1, how='outer')
    uphold_df.drop(['TEMP'], axis=1, inplace=True)

    uphold_df.iloc[0, :].fillna(0, inplace=True)
    uphold_df.ffill(inplace=True)
    uphold_df.index = [k.date() for k in uphold_df.index]

    uphold_df_EUR = copy.deepcopy(uphold_df)
    for coin in uphold_df_EUR.columns:
        print(f'Preparing EUR dataframe for Uphold for coin {coin}')
        for date_loop in uphold_df_EUR.index:
            price = [i[1] for i in tokens_prices[coin] if i[0] == tx.str_to_datetime(str(date_loop)).date()]
            if len(price) == 0:
                loop_control = 0
                dateloop2 = copy.deepcopy(date_loop)
                while len(price) == 0 and loop_control < 50:
                    loop_control += 1
                    price = [i[1] for i in tokens_prices[coin] if
                             i[0] == tx.str_to_datetime(str(dateloop2)).date() - dt.timedelta(days=1)]
                    dateloop2 -= dt.timedelta(days=1)
                if loop_control > 0:
                    print(
                        f'Uphold - could not find price for date {tx.str_to_datetime(date_loop)} for coin {coin} could not be found using date {tx.str_to_datetime(dateloop2)}')
                if len(price) == 0:
                    price = [0]
            uphold_df_EUR.loc[date_loop, coin] *= price[0]

    interests_uphold_crypto = pd.DataFrame(
        uphold_transactions.loc[uphold_transactions['Currency'] == 'BAT', 'Amount'].copy())

    interests_uphold_EUR = copy.deepcopy(interests_uphold_crypto)
    print(f'Preparing EUR dataframe for Uphold for BAT prizes')
    if 'BAT' not in list(tokens_prices.keys()):
        tokens_prices['BAT'] = tx.get_token_prices(['BAT'], [0], [0], 'max', 'eur')
    for date_loop in interests_uphold_EUR.index:
        price = [i[1] for i in tokens_prices['BAT'] if i[0] == tx.str_to_datetime(str(date_loop)).date()]
        if len(price) == 0:
            loop_control = 0
            dateloop2 = copy.deepcopy(date_loop)
            while len(price) == 0 and loop_control < 50:
                loop_control += 1
                price = [i[1] for i in tokens_prices[coin] if
                         i[0] == tx.str_to_datetime(str(dateloop2)).date() - dt.timedelta(days=1)]
                dateloop2 -= dt.timedelta(days=1)
            if loop_control > 0:
                print(
                    f'uphold - could not find price for date {tx.str_to_datetime(date_loop)} for coin {coin} could not be found using date {tx.str_to_datetime(dateloop2)}')
            if len(price) == 0:
                price = [0]
        interests_uphold_EUR.loc[date_loop, 'Amount'] *= price[0]

    uphold_df_soglia = copy.deepcopy(uphold_df)
    coin_names = uphold_df_soglia.columns
    uphold_df_soglia['Anno'] = [k.year for k in uphold_df.index]

    for y in np.unique(uphold_df_soglia['Anno']):
        for coin in coin_names:
            if tokens_prices[coin] == 0:
                print(f'Uphold - could not find price for coin {coin} for 1/1/{y} setting to zero')
                coin_price = [0]
            else:
                coin_price = [x[1] for x in tokens_prices[coin] if x[0] == dt.date(y, 1, 1)]
            if len(coin_price) == 0:
                coin_price = [tokens_prices[coin][0][1]]
            uphold_df_soglia.loc[uphold_df_soglia['Anno'] == y, coin] *= float(coin_price[0])

    uphold_dict = dict()

    uphold_dict['soglia'] = uphold_df_soglia
    uphold_dict['balances'] = uphold_df
    uphold_dict['balances-EUR'] = uphold_df_EUR
    uphold_dict['interest-Crypto'] = interests_uphold_crypto
    uphold_dict['interest-EUR'] = interests_uphold_EUR

    with open('uphold.pickle', 'wb') as handle:
        pk.dump(uphold_dict, handle, protocol=pk.HIGHEST_PROTOCOL)

    shutil.move(new_transactions_file_uphold[0], "uphold_history")

soglia_uphold = uphold_dict['soglia'].loc[uphold_dict['soglia']['Anno'] == anno_fiscale, :].copy()
soglia_uphold.drop(['Anno'], axis=1, inplace=True)
soglia_totale['Uphold'] = soglia_uphold.sum(axis=1)

uphold_dict['interest-EUR']['Anno'] = [k.year for k in uphold_dict['interest-EUR'].index]
interesse_uphold = uphold_dict['interest-EUR'].loc[uphold_dict['interest-EUR']['Anno'] == anno_fiscale, :].copy()
interesse_uphold.drop(['Anno'], axis=1, inplace=True)
interessi_totali.loc[0, 'Uphold'] = interesse_uphold.sum(axis=1).sum(axis=0)

uphold_dict['balances-EUR']['Anno'] = [k.year for k in uphold_dict['balances-EUR'].index]
giacenza_uphold = uphold_dict['balances-EUR'].loc[uphold_dict['balances-EUR']['Anno'] == anno_fiscale, :].copy()
giacenza_uphold.drop(['Anno'], axis=1, inplace=True)
giacenza_media.loc[0, 'Uphold'] = giacenza_uphold.sum(axis=1)[giacenza_uphold.sum(axis=1) != 0].sum(axis=0) / \
                                  giacenza_uphold.sum(axis=1)[giacenza_uphold.sum(axis=1) != 0].shape[0]
print("UPHOLD DONE")


def get_bsc_bep20_balance(bscaddress, bsctoken, locname):
    bscaddress = ["0xF15965AEBA71E4A9D2ED0D1aB568a6A3334F8e1F"]
    addr_bc = "bnb1pgl4cp4v6433w5rfwezagzzvs9wdv3pu7h7tl8"
    bsctoken = "AD3549Z3D6T3J24SPSY3SZRMD9CUAANS91"

    # FOR ADDRESSES STARTING WITH BNB (BINANCE CHAIN)
    output_bnb = pd.DataFrame()
    num_of_weeks = (dt.datetime.today() - dt.datetime(dt.datetime.today().year, 1, 1)).days
    num_of_weeks = int(num_of_weeks / 7)
    num_of_days = round((num_of_weeks / 7 % 1) * 7)

    week_days = [7] * 52 * 2
    week_days_this_year = [7] * num_of_weeks
    days_this_week = [num_of_days]
    week_days.extend(week_days_this_year)
    week_days.extend(days_this_week)

    st_d = int(dt.datetime(dt.datetime.today().year - 2, 1, 1).timestamp()) * 1000
    for index_loop, j in enumerate(week_days):
        if index % 20 == 0:
            print(f'Getting {index_loop + 1} week of BNB transactions out of {len(week_days)}')
        et_d = st_d + (j * 86400000)
        urls = f'https://api.binance.org/bc/api/v1/txs?address=bnb1pgl4cp4v6433w5rfwezagzzvs9wdv3pu7h7tl8&startTime={st_d}&endTime={et_d}'
        response_bnb = requests.get(urls)
        response_bnb = response_bnb.json()
        st_d = et_d
        output_bnb = pd.concat([output_bnb, pd.DataFrame(response_bnb['txs'])])
    print('Done')

    output_bnb['blockTime'] = output_bnb['blockTime'].map(lambda x: dt.datetime.fromtimestamp(x / 1000))
    output_bnb.loc[output_bnb['type'] == "CROSS_TRANSFER_OUT", 'amount'] = 0

    output_bnb.loc[output_bnb['fromAddr'] == addr_bc, 'amount'] *= -1

    url = f'https://api.bscscan.com/api?module=account&action=txlist&address={bscaddress}&startblock=0&endblock=99999999999&page=1&offset=1000&sort=asc&apikey={bsctoken}'
    response = requests.get(url)
    normal_transactions = pd.DataFrame(response.json().get('result'))
    normal_transactions['timeStamp'] = normal_transactions['timeStamp'].map(lambda x: dt.datetime.fromtimestamp(int(x)))
    normal_transactions['from'] = normal_transactions['from'].map(lambda x: x.lower())
    normal_transactions['to'] = normal_transactions['to'].map(lambda x: x.lower())
    normal_transactions['value'] = [-int(normal_transactions.loc[i, 'value']) / 10 ** 18 if normal_transactions.loc[
                                                                                                i, 'from'] == bscaddress.lower() else int(
        normal_transactions.loc[i, 'value']) / 10 ** 18 for i in range(normal_transactions.shape[0])]
    normal_transactions['gas'] = [
        -(int(normal_transactions.loc[i, 'gas']) * int(normal_transactions.loc[i, 'gasPrice'])) / 10 ** 18 for i in
        range(normal_transactions.shape[0])]
    normal_transactions = normal_transactions.loc[:, ['timeStamp', 'value', 'gas']]

    # INTERNAL
    url = f'https://api.bscscan.com/api?module=account&action=txlistinternal&address={bscaddress}&startblock=0&endblock=99999999999&page=1&offset=10&sort=asc&apikey={bsctoken}'
    response = requests.get(url)
    internal_transactions = pd.DataFrame(response.json().get('result'))
    internal_transactions['timeStamp'] = internal_transactions['timeStamp'].map(
        lambda x: dt.datetime.fromtimestamp(int(x)))

    # Get current BEP20
    url = f'https://api.bscscan.com/api?module=account&action=tokentx&address={bscaddress}&startblock=0&endblock=99999999999&page=1&offset=10&sort=asc&apikey={bsctoken}'
    response = requests.get(url)
    bep20_transactions = pd.DataFrame(response.json().get('result'))
    bep20_transactions['timeStamp'] = bep20_transactions['timeStamp'].map(lambda x: dt.datetime.fromtimestamp(int(x)))

    bep20 = requests.get(f'https://api.bscscan.com/api?module=account&action=tokentx&address={bscaddress}&'
                         f'startblock=0&endblock=999999999999999&sort=asc&apikey={bsctoken}')

    bep20 = bep20.json()

    [total_tokens.append(s.get('tokenSymbol')) for s in bep20.get('result')]
    [contracts_list.append(s.get('contractAddress')) for s in bep20.get('result')]
    [token_decimals.append(10 ** int(s.get('tokenDecimal'))) for s in bep20.get('result')]

    vout = pd.DataFrame({'Tokens': total_tokens, 'Contract': contracts_list, 'Dec': token_decimals})
    vout.drop_duplicates(inplace=True)
    vout.reset_index(inplace=True, drop=True)
    vout['Amount'] = 0

    for line_index in range(vout.shape[0]):
        if vout.loc[line_index, 'Tokens'] == 'BNB':
            vout.loc[line_index, 'Amount'] = bnb_balance
        else:
            tokenbalance = requests.get(f'https://api.bscscan.com/api?module=account&action=tokenbalance'
                                        f'&contractaddress={vout.loc[line_index, "Contract"]}&address={bscaddress}&'
                                        f'tag=latest&apikey={bsctoken}')
            vout.loc[line_index, 'Amount'] = int(tokenbalance.json().get('result')) / vout.loc[line_index, 'Dec']

    duplicated_tokens = [(x, k) for x, k in enumerate(list(vout['Tokens'])) if list(vout['Tokens']).count(k) > 1]
    if len(duplicated_tokens) > 0:
        for ind, name in duplicated_tokens:
            vout.loc[ind, 'Tokens'] = name + "-" + vout.loc[ind, 'Contract'][-5:len(vout.loc[ind, 'Contract'])]

    vout.index = [i.upper() for i in vout['Tokens']]
    vout.drop(['Dec', 'Tokens'], axis=1, inplace=True)
    vout = vout[vout['Amount'] > 0]
    vout = ignore_tokens(vout, locname)
    vout['Network'] = 'binance-smart-chain'
    return vout
