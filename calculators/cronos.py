import numpy as np
from utility.PricesClass import Prices, update_prices
from utility import tax_library as tx
import datetime as dt
import pandas as pd
import requests
import os
import utility.utils as ut
from utility.tax_log import log

cro_prices = Prices()
log.info('Cronos calculator - updated on 15/10/2022')

first = True
scamtokens = ['0x5c7f8a570d578ed84e63fdfa7b1ee72deae1ae23']


# The transactions on Crypto.org chain have to be extracted manually, refer to the example file
def get_crypto_dot_org_transactions():
    cronos_files = [os.getcwd() + '\\cryptodotorg' + '\\' + x for x in os.listdir(os.getcwd() + '\\cryptodotorg')]
    if len(cronos_files) == 0:
        log.info('No files for crypto.org found')
        return None
    else:
        df_list = []
        for filename in cronos_files:
            df_loop = pd.read_csv(filename, index_col=None, header=0)
            df_list.append(df_loop)
        final_df = pd.concat(df_list, axis=0, ignore_index=True)
        final_df.index = [tx.str_to_datetime(x.replace(" UTC", "")) + dt.timedelta(hours=1) for x in
                          list(final_df['Timestamp'])]

        final_df.drop(['Timestamp'], axis=1, inplace=True)

        final_df['Fiat Price'] = 0
        final_df['Fiat'] = 'EUR'
        final_df['Fee Currency'] = 'CRO'
        final_df['To Coin'] = ''
        final_df['Coin'] = 'CRO'
        final_df['To Amount'] = ''
        final_df['Tag'] = 'Movement'
        global first
        if first:
            log.info(
                f'Crypto.org transactions up to {max(cronos_files).split("_")[0].replace(".csv", "").split("cryptodotorg")[1]}')
            first = False

    return final_df


def get_transactions_df(address, beacon_address=None):
    address = address.lower()
    if beacon_address is not None:
        beacon = get_crypto_dot_org_transactions()
        beacon['Fee'] *= -1
        beacon['To'] = beacon['To'].map(lambda x: x.lower())
        beacon['From'] = beacon['From'].map(lambda x: x.lower())
        temp = pd.DataFrame(beacon[beacon['To'] == address])
        temp.index = [k + dt.timedelta(seconds=1) for k in temp.index]
        beacon.loc[beacon['From'] == beacon_address, 'Amount'] *= -1
        beacon = pd.concat([beacon, temp], axis=0)
        beacon.sort_index(inplace=True)

    url = f'https://api.cronoscan.com/api?module=account&action=txlist&address={address}&startblock=1&endblock=999999999999&sort=asc&apikey=PQGZXHK6QJDCW1KCKSN7TR76UUBVRV9A23'
    response = requests.get(url)

    normal_transactions = pd.DataFrame(response.json().get('result'))
    normal_transactions = normal_transactions[normal_transactions['isError'] != 1].copy()
    normal_transactions.reset_index(inplace=True, drop=True)

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

    normal_transactions = normal_transactions[~normal_transactions['hash'].isin(list(normal_transactions.loc[
                                                                                         np.logical_and(
                                                                                             normal_transactions[
                                                                                                 'value'] == 0,
                                                                                             normal_transactions[
                                                                                                 'functionName'].str.contains(
                                                                                                 'transfer')), 'hash']))]

    normal_transactions.rename(columns={'from': 'From', 'to': 'To', 'value': 'Amount', 'gas': 'Fee'}, inplace=True)
    normal_transactions.drop(
        ['blockHash', 'blockNumber', 'hash', 'nonce', 'transactionIndex', 'gasPrice', 'isError', 'txreceipt_status',
         'contractAddress', 'cumulativeGasUsed', 'gasUsed', 'confirmations', 'timeStamp', 'input'], axis=1,
        inplace=True)

    normal_transactions['Fiat Price'] = 0
    normal_transactions['Fiat'] = 'EUR'
    normal_transactions['Fee Currency'] = 'CRO'
    normal_transactions['To Coin'] = ''
    normal_transactions['Coin'] = 'CRO'
    normal_transactions['To Amount'] = ''
    normal_transactions['Tag'] = 'Movement'

    normal_transactions = normal_transactions.reindex(
        columns=['From', 'To', 'Coin', 'Amount', 'To Coin', 'To Amount', 'Fiat Price',
                 'Fiat', 'Fee', 'Fee Currency', 'Tag'])

    normal_transactions.sort_index(inplace=True)

    outdf = normal_transactions.copy()

    # INTERNAL
    url = f'https://api.cronoscan.com/api?module=account&action=txlistinternal&address={address}&startblock=0&endblock=999999999999&sort=asc&apikey=PQGZXHK6QJDCW1KCKSN7TR76UUBVRV9A23'
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
                                    'gasUsed', 'timeStamp', 'input', 'contractAddress', 'type', 'errCode'],
                                   axis=1,
                                   inplace=True)

        internal_transactions['Fiat Price'] = 0
        internal_transactions['Fiat'] = 'EUR'
        internal_transactions['Fee Currency'] = 'CRO'
        internal_transactions['To Coin'] = ''
        internal_transactions['Coin'] = 'CRO'
        internal_transactions['To Amount'] = ''
        internal_transactions['Tag'] = 'Movement'

        internal_transactions = internal_transactions.reindex(
            columns=['From', 'To', 'Coin', 'Amount', 'To Coin', 'To Amount', 'Fiat Price',
                     'Fiat', 'Fee', 'Fee Currency', 'Tag'])

        outdf = pd.concat([outdf, internal_transactions], axis=0)
    outdf.sort_index(inplace=True)

    if beacon_address is not None:
        outdf = pd.concat([outdf, beacon], axis=0)
        outdf.sort_index(inplace=True)
    # Get CRC20 tokens
    url = f'https://api.cronoscan.com/api?module=account&action=tokentx&address={address}&startblock=0&endblock=999999999999&sort=asc&apikey=PQGZXHK6QJDCW1KCKSN7TR76UUBVRV9A23'
    response = requests.get(url)
    erc20_transactions = pd.DataFrame(response.json().get('result'))
    if erc20_transactions.shape[0] > 0:
        erc20_transactions['from'] = erc20_transactions['from'].map(lambda x: x.lower())
        erc20_transactions['to'] = erc20_transactions['to'].map(lambda x: x.lower())

        erc20_transactions.reset_index(inplace=True, drop=True)

        erc20_transactions['value'] = [int(s) / 10 ** int(x) for s, x in
                                       zip(erc20_transactions['value'], erc20_transactions['tokenDecimal'])]
        erc20_transactions['gas'] = [
            -(int(erc20_transactions.loc[i, 'gasUsed']) * int(erc20_transactions.loc[i, 'gasPrice'])) / 10 ** 18 for i
            in
            range(erc20_transactions.shape[0])]

        erc20_transactions.rename(
            columns={'from': 'From', 'to': 'To', 'value': 'Amount', 'gas': 'Fee', 'tokenSymbol': 'Coin'}, inplace=True)
        erc20_transactions.index = erc20_transactions['timeStamp'].map(lambda x: dt.datetime.fromtimestamp(int(x)))
        erc20_transactions.drop(
            ['blockNumber', 'timeStamp', 'tokenDecimal', 'nonce', 'blockHash', 'transactionIndex', 'gasPrice',
             'contractAddress', 'cumulativeGasUsed', 'gasUsed', 'confirmations', 'hash', 'input', 'tokenName'],
            axis=1,
            inplace=True)

        erc20_transactions.loc[erc20_transactions['From'] == address.lower(), 'Amount'] *= -1

        erc20_transactions['Fiat Price'] = 0
        erc20_transactions['Fiat'] = 'EUR'
        erc20_transactions['Fee Currency'] = 'CRO'
        erc20_transactions['To Amount'] = ''
        erc20_transactions['Tag'] = 'Movement'

        erc20_transactions = erc20_transactions[~erc20_transactions['From'].isin(scamtokens)]

        outdf = pd.concat([outdf, erc20_transactions], axis=0)
        outdf.loc[outdf['From'] == '0xcAa8c10B81DDD462AFf6bA33aC8242255504B3Db'.lower(), 'Tag'] = 'Reward'

        outdf = ut.evm_liquidity_swap_calculation(outdf, address)

        outdf.sort_index(inplace=True)
        outdf['Amount'].fillna('', inplace=True)
        outdf['Coin'].fillna('', inplace=True)
        outdf['To Amount'].fillna('', inplace=True)
        outdf['To Coin'].fillna('', inplace=True)
        outdf['Fee'].fillna('', inplace=True)
        outdf['Fiat Price'].fillna(0, inplace=True)
        outdf['Fee Currency'].fillna('', inplace=True)
        outdf['Fiat'].fillna('EUR', inplace=True)
        outdf['From'].fillna('', inplace=True)
        outdf['To'].fillna('', inplace=True)
        outdf['Tag'].fillna('Movement', inplace=True)

        outdf['From'] = outdf['From'].map(lambda x: x.lower())
        outdf['To'] = outdf['To'].map(lambda x: x.lower())

    global cro_prices

    tokens = outdf['Coin'].tolist()
    tokens.extend(outdf['To Coin'].tolist())
    tokens = [x.upper() for x in list(set(tokens)) if x not in ut.fiat_list]

    was_updated = update_prices(cro_prices, tokens)

    if 'EUR' not in list(cro_prices.exchange_rates.keys()) or was_updated:
        cro_prices.convert_prices('EUR', tokens)

    outdf.sort_index(inplace=True)
    price = cro_prices.to_pd_dataframe('EUR')
    price = price[~price.index.duplicated(keep='first')]
    for tok in tokens:
        temp_df = outdf[outdf['Coin'] == tok].copy()
        temp_df.index = [k.date() for k in temp_df.index]

        temp_df = temp_df.join(pd.DataFrame(price[tok]))
        temp_df['Fiat Price'] = temp_df['Amount'] * temp_df[tok]
        temp_df.loc[temp_df['To Coin'] == 'EUR', 'Fiat Price'] = temp_df.loc[
            temp_df['To Coin'] == 'EUR', 'To Amount']
        temp_df.index = outdf[outdf['Coin'] == tok].index
        outdf.loc[outdf['Coin'] == tok, 'Fiat Price'] = temp_df['Fiat Price']

    outdf.loc[outdf['To'] == address, 'Fee'] = 0

    outdf['Tag Account'] = f'Cronos - {address[0:7]}'
    return outdf


def calculate_all(address: str, beacon_address=None, year_sel=None, name='Cronos'):
    transactions = get_transactions_df(address=address, beacon_address=beacon_address)
    balances_in = ut.balances(transactions=transactions, year_sel=year_sel)
    balaces_fiat_in = ut.balances_fiat(balances=balances_in, prices=cro_prices, year_sel=year_sel)
    soglia_in = ut.soglia(balances=balances_in, prices=cro_prices, year_sel=year_sel)
    income_in = ut.income(transactions=transactions, type_out='crypto', name=name, year_sel=year_sel)
    income_in_fiat = ut.income(transactions=transactions, name=name, year_sel=year_sel)
    vout = {"transactions": transactions, "transactions_raw": transactions, "balances": balances_in,
            "balaces_fiat": balaces_fiat_in, "soglia": soglia_in,
            "income": income_in, "income_fiat": income_in_fiat}
    return vout
