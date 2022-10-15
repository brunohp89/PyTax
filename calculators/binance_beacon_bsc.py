import os
import numpy as np
from utility.PricesClass import Prices, update_prices
from utility import tax_library as tx
import datetime as dt
import pandas as pd
import requests
import utility.utils as ut
from utility.tax_log import log
import json

log.info('BSC calculator - updated on 15/10/2022')
bsc_prices = Prices()
scam_tokens = ["0xf3822314b333cbd7a36753b77589afbe095df1ba",
               "0x0df62d2cd80591798721ddc93001afe868c367ff",
               "0xb0557906c617f0048a700758606f64b33d0c41a6",
               "0xb8a9704d48c3e3817cc17bc6d350b00d7caaecf6",
               "0x5558447b06867ffebd87dd63426d61c868c45904",
               "0xd22202d23fe7de9e3dbe11a2a88f42f4cb9507cf",
               "0xab57aef3601cad382aa499a6ae2018a69aad9cf0",
               "0x5190b01965b6e3d786706fd4a999978626c19880",
               "0x569b2cf0b745ef7fad04e8ae226251814b3395f9",
               "0x8ee3e98dcced9f5d3df5287272f0b2d301d97c57",
               "0xbc6675de91e3da8eac51293ecb87c359019621cf",
               "0x64f2c2aa04755507a2ecd22ceb8c475b7a750a3a",
               "0x9028418bbf045fcfe85a3d44ab8054712d98872b",
               "0x4a5fad6631fd3df66f23519608185cb96e9a687d",
               "0x0b7dc561777842d55163e0f48886295aad1359b9"]

with open(os.getcwd() + '\\.json') as creds:
    api_key = json.load(creds)['BSCScanToken']

if api_key == '':
    raise PermissionError('No API KEY for BSC Scan found in .json')


def get_transactions_df(address, api_key, beacon_address=None):
    address = address.lower()
    if beacon_address is not None:
        beacon = tx.get_bnb(beacon_address)

    url = f'https://api.bscscan.com/api?module=account&action=txlist&address={address}&startblock=0&endblock=99999999999&page=1&offset=1000&sort=asc&apikey={api_key}'
    response = requests.get(url)

    normal_transactions = pd.DataFrame(response.json().get('result'))
    normal_transactions = normal_transactions[normal_transactions['isError'] != 1].copy()
    normal_transactions.reset_index(inplace=True, drop=True)

    normal_transactions['isScam'] = [1 if k in scam_tokens else 0 for k in normal_transactions['from']]
    normal_transactions = normal_transactions[normal_transactions['isScam'] == 0]

    normal_transactions['from'] = normal_transactions['from'].map(lambda x: x.lower())
    normal_transactions['to'] = normal_transactions['to'].map(lambda x: x.lower())
    normal_transactions['value'] = [-int(normal_transactions.loc[i, 'value']) / 10 ** 18 if normal_transactions.loc[
                                                                                                i, 'from'] == address.lower() else int(
        normal_transactions.loc[i, 'value']) / 10 ** 18 for i in range(normal_transactions.shape[0])]
    normal_transactions['gas'] = [
        -(int(normal_transactions.loc[i, 'gasUsed']) * int(normal_transactions.loc[i, 'gasPrice'])) / 10 ** 18 for i
        in
        range(normal_transactions.shape[0])]
    normal_transactions.index = normal_transactions['timeStamp'].map(
        lambda x: dt.datetime.fromtimestamp(int(x)))

    normal_transactions.rename(columns={'from': 'From', 'to': 'To', 'value': 'Amount', 'gas': 'Fee'}, inplace=True)
    normal_transactions.drop(
        ['blockHash', 'blockNumber', 'hash', 'nonce', 'transactionIndex', 'gasPrice', 'isError', 'txreceipt_status',
         'contractAddress', 'isScam', 'cumulativeGasUsed', 'gasUsed', 'confirmations', 'timeStamp', 'input'], axis=1,
        inplace=True)

    normal_transactions['Fiat Price'] = 0
    normal_transactions['Fiat'] = 'EUR'
    normal_transactions['Fee Currency'] = 'BNB'
    normal_transactions['To Coin'] = ''
    normal_transactions['Coin'] = 'BNB'
    normal_transactions['To Amount'] = ''
    normal_transactions['Tag'] = 'Movement'

    normal_transactions = normal_transactions.reindex(
        columns=['From', 'To', 'Coin', 'Amount', 'To Coin', 'To Amount', 'Fiat Price',
                 'Fiat', 'Fee', 'Fee Currency', 'Tag'])

    if beacon_address is not None:
        bnb = pd.concat([beacon, normal_transactions], axis=0)
    else:
        bnb = normal_transactions.copy()
    bnb.sort_index(inplace=True)

    # INTERNAL
    url = f'https://api.bscscan.com/api?module=account&action=txlistinternal&address={address}&startblock=0&endblock=99999999999&page=1&offset=10&sort=asc&apikey=AD3549Z3D6T3J24SPSY3SZRMD9CUAANS91'
    response_internal = requests.get(url)
    internal_transactions = pd.DataFrame(response_internal.json().get('result'))

    if internal_transactions.shape[0] > 0:
        internal_transactions.index = internal_transactions['timeStamp'].map(
            lambda x: dt.datetime.fromtimestamp(int(x)))

        normal_transactions_bis = pd.DataFrame(response.json().get('result'))
        normal_transactions_bis.index = normal_transactions_bis['timeStamp'].map(
            lambda x: dt.datetime.fromtimestamp(int(x)))

        internal_transactions = internal_transactions.join(normal_transactions_bis['gasPrice'], how='left')
        internal_transactions.bfill(inplace=True)

        internal_transactions = internal_transactions[internal_transactions['isError'] != 1].copy()

        internal_transactions['isScam'] = [1 if k in scam_tokens else 0 for k in internal_transactions['from']]
        internal_transactions = internal_transactions[internal_transactions['isScam'] == 0]
        internal_transactions.reset_index(inplace=True, drop=True)

        internal_transactions.reset_index(inplace=True, drop=True)
        internal_transactions['from'] = internal_transactions['from'].map(lambda x: x.lower())
        internal_transactions['to'] = internal_transactions['to'].map(lambda x: x.lower())
        internal_transactions['value'] = [
            -int(internal_transactions.loc[i, 'value']) / 10 ** 18 if internal_transactions.loc[
                                                                          i, 'from'] == address.lower() else int(
                internal_transactions.loc[i, 'value']) / 10 ** 18 for i in range(internal_transactions.shape[0])]
        internal_transactions['gas'] = [
            -(int(internal_transactions.loc[i, 'gas']) * int(internal_transactions.loc[i, 'gasPrice'])) / 10 ** 18 for i
            in
            range(internal_transactions.shape[0])]
        internal_transactions.index = internal_transactions['timeStamp'].map(
            lambda x: dt.datetime.fromtimestamp(int(x)))

        internal_transactions.rename(columns={'from': 'From', 'to': 'To', 'value': 'Amount', 'gas': 'Fee'},
                                     inplace=True)
        internal_transactions.drop(['blockNumber', 'hash', 'gasPrice', 'isError', 'traceId',
                                    'gasUsed', 'timeStamp', 'input', 'contractAddress', 'isScam', 'type', 'errCode'],
                                   axis=1,
                                   inplace=True)

        internal_transactions['Fiat Price'] = 0
        internal_transactions['Fiat'] = 'EUR'
        internal_transactions['Fee Currency'] = 'BNB'
        internal_transactions['To Coin'] = ''
        internal_transactions['Coin'] = 'BNB'
        internal_transactions['To Amount'] = ''
        internal_transactions['Tag'] = 'Movement'

        internal_transactions = internal_transactions.reindex(
            columns=['From', 'To', 'Coin', 'Amount', 'To Coin', 'To Amount', 'Fiat Price',
                     'Fiat', 'Fee', 'Fee Currency', 'Tag'])

        bnb = pd.concat([bnb, internal_transactions], axis=0)
    bnb.sort_index(inplace=True)

    bnb.loc[bnb['To'] == "0x0000000000000000000000000000000000001004", 'Amount'] *= 0

    # Get BEP20 tokens
    url = f'https://api.bscscan.com/api?module=account&action=tokentx&address={address}&startblock=0&endblock=999999999999999999&sort=asc&apikey=AD3549Z3D6T3J24SPSY3SZRMD9CUAANS91'
    response = requests.get(url)
    bep20_transactions = pd.DataFrame(response.json().get('result'))
    if bep20_transactions.shape[0] > 0:
        bep20_transactions['from'] = bep20_transactions['from'].map(lambda x: x.lower())
        bep20_transactions['to'] = bep20_transactions['to'].map(lambda x: x.lower())

        bep20_transactions['isScam'] = [1 if k in scam_tokens else 0 for k in bep20_transactions['from']]
        bep20_transactions = bep20_transactions[bep20_transactions['isScam'] == 0]
        bep20_transactions.reset_index(inplace=True, drop=True)

        bep20_transactions['value'] = [int(s) / 10 ** int(x) for s, x in
                                       zip(bep20_transactions['value'], bep20_transactions['tokenDecimal'])]
        bep20_transactions['gas'] = [
            -(int(bep20_transactions.loc[i, 'gasUsed']) * int(bep20_transactions.loc[i, 'gasPrice'])) / 10 ** 18 for i
            in
            range(bep20_transactions.shape[0])]

        bep20_transactions.rename(columns={'from': 'From', 'to': 'To', 'value': 'Amount', 'gas': 'Fee', 'tokenSymbol':'Coin'}, inplace=True)
        bep20_transactions.index = bep20_transactions['timeStamp'].map(lambda x: dt.datetime.fromtimestamp(int(x)))
        bep20_transactions.drop(
            ['blockNumber', 'timeStamp', 'tokenDecimal', 'nonce', 'blockHash', 'transactionIndex', 'gasPrice',
             'contractAddress', 'cumulativeGasUsed', 'gasUsed', 'confirmations', 'hash', 'isScam', 'input',
             'tokenName'],
            axis=1,
            inplace=True)

        bep20_transactions.loc[bep20_transactions['From'] == address.lower(), 'Amount'] *= -1

        bep20_transactions.loc[bep20_transactions['To'] == address, 'Fee'] = 0

        bep20_transactions['Fiat Price'] = 0
        bep20_transactions['Fiat'] = 'EUR'
        bep20_transactions['Fee Currency'] = 'BNB'
        bep20_transactions['To Amount'] = ''
        bep20_transactions['To Coin'] = ''
        bep20_transactions['Tag'] = 'Movement'

        #temp = bep20_transactions[['tokenSymbol', 'Amount', 'Fee']]
       # temp.columns = ['TOKEN', 'TOKEN_AMOUNT', 'FEE_TOKEN']
        bnb_bep20 = pd.concat([bnb,bep20_transactions])

        bnb_bep20 = ut.evm_liquidity_swap_calculation(bnb_bep20, address)
        bnb_bep20['Coin'] = [x.replace('S*','') for x in bnb_bep20['Coin']]

        bnb_bep20.sort_index(inplace=True)
        bnb_bep20['Fee'].fillna('', inplace=True)
        bnb_bep20['Fiat Price'].fillna(0, inplace=True)
        bnb_bep20['Fee Currency'].fillna('', inplace=True)
        bnb_bep20['Fiat'].fillna('EUR', inplace=True)
        bnb_bep20['From'].fillna('', inplace=True)
        bnb_bep20['To'].fillna('', inplace=True)
        bnb_bep20['Tag'].fillna('Movement', inplace=True)

        bnb_bep20['From'] = bnb_bep20['From'].map(lambda x: x.lower())
        bnb_bep20['To'] = bnb_bep20['To'].map(lambda x: x.lower())

    else:
        bnb_bep20 = bnb.copy()

    global bsc_prices

    # Gestione interessi in STG (la transazione di rimozione della liquidita e claim è fatta in modo particolare)
    if bnb_bep20[bnb_bep20['Coin'] == 'STG'].shape[0] > 0:
        duplicated_index = np.unique(bnb_bep20[bnb_bep20['Coin'] == 'STG'].index[
                                         bnb_bep20[bnb_bep20['Coin'] == 'STG'].index.duplicated(keep='first')])
        if len(duplicated_index) > 0:
            for index in duplicated_index:
                temp_list = []
                for i in range(bnb_bep20[bnb_bep20['Coin'] == 'STG'].loc[index, :].shape[0]):
                    if list(bnb_bep20[bnb_bep20['Coin'] == 'STG'].loc[index, 'Tag'])[i] == 'Movement':
                        temp_list.append(0)
                    else:
                        temp_list.append(list(bnb_bep20[bnb_bep20['Coin'] == 'STG'].loc[index, 'Amount'])[i])
                bnb_bep20.loc[
                    np.logical_and(bnb_bep20['Coin'] == 'STG', bnb_bep20.index == index), 'Amount'] = temp_list

    tokens = bnb_bep20['Coin'].tolist()
    tokens.extend(bnb_bep20['To Coin'].tolist())
    tokens = [x.upper() for x in list(set(tokens)) if
              x not in ut.fiat_list]

    was_updated = update_prices(bsc_prices, tokens)

    if 'EUR' not in list(bsc_prices.exchange_rates.keys()) or was_updated:
        bsc_prices.convert_prices('EUR', tokens)

    bnb_bep20.sort_index(inplace=True)
    price = bsc_prices.to_pd_dataframe('EUR')
    price = price[~price.index.duplicated(keep='first')]
    for tok in tokens:
        temp_df = bnb_bep20[bnb_bep20['Coin'] == tok].copy()
        temp_df.index = [k.date() for k in temp_df.index]

        temp_df = temp_df.join(pd.DataFrame(price[tok]))
        temp_df['Fiat Price'] = temp_df['Amount'] * temp_df[tok]
        temp_df.loc[temp_df['To Coin'] == 'EUR', 'Fiat Price'] = temp_df.loc[
            temp_df['To Coin'] == 'EUR', 'To Amount']
        temp_df.index = bnb_bep20[bnb_bep20['Coin'] == tok].index
        bnb_bep20.loc[bnb_bep20['Coin'] == tok, 'Fiat Price'] = temp_df['Fiat Price']

    if beacon_address is None:
        bnb_bep20.loc[bnb_bep20['To'] == address, 'Fee'] = 0

    bnb_bep20['Tag Account'] = f'BSC - {address[0:7]}'

    return bnb_bep20


def calculate_all(address, beacon_address=None, year_sel=None, name='BSC_NC', api_key=api_key):
    transactions = get_transactions_df(address=address, beacon_address=beacon_address, api_key=api_key)
    balances_in = ut.balances(transactions=transactions, year_sel=year_sel)
    balaces_fiat_in = ut.balances_fiat(balances=balances_in, prices=bsc_prices, year_sel=year_sel)
    soglia_in = ut.soglia(balances=balances_in, prices=bsc_prices, year_sel=year_sel)
    income_in = ut.income(transactions=transactions, type_out='crypto', name=name, year_sel=year_sel)
    income_in_fiat = ut.income(transactions=transactions, name=name, year_sel=year_sel)
    vout = {"transactions": transactions, "transactions_raw": transactions, "balances": balances_in,
            "balaces_fiat": balaces_fiat_in, "soglia": soglia_in,
            "income": income_in, "income_fiat": income_in_fiat}
    return vout
