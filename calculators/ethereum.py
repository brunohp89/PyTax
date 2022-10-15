import os
import json
from utility.PricesClass import Prices, update_prices
import utility.utils as ut
from utility.tax_log import log
import datetime as dt
import pandas as pd
import requests

log.info('Ethereum calculator - updated on 15/10/2022')

scam_tokens = ['0x1883a07c429e84aca23b041c357e1d21a2b645f3']
eth_prices = Prices()
with open(os.getcwd() + '\\.json') as creds:
    api_key = json.load(creds)['ETHScanToken']

if api_key == '':
    raise PermissionError('No API KEY for ETH Scan found in .json')


def get_transactions_df(address, apikey=api_key):
    address = address.lower()
    url = f'https://api.etherscan.io/api?module=account&action=txlist&address={address}&startblock=0&endblock=9999999999999999999&sort=asc&apikey={apikey}'
    response = requests.get(url)

    normal_transactions = pd.DataFrame(response.json().get('result'))
    normal_transactions = normal_transactions[normal_transactions['isError'] != 1].copy()
    normal_transactions.reset_index(inplace=True, drop=True)

    normal_transactions['isScam'] = [1 if k in scam_tokens else 0 for k in normal_transactions['contractAddress']]
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
         'contractAddress', 'cumulativeGasUsed', 'gasUsed', 'confirmations', 'timeStamp', 'input', 'isScam'], axis=1,
        inplace=True)

    normal_transactions['Fiat Price'] = 0
    normal_transactions['Fiat'] = 'EUR'
    normal_transactions['Fee Currency'] = 'ETH'
    normal_transactions['To Coin'] = ''
    normal_transactions['Coin'] = 'ETH'
    normal_transactions['To Amount'] = ''
    normal_transactions['Tag'] = 'Movement'

    normal_transactions = normal_transactions.reindex(
        columns=['From', 'To', 'Coin', 'Amount', 'To Coin', 'To Amount', 'Fiat Price',
                 'Fiat', 'Fee', 'Fee Currency', 'Tag'])

    normal_transactions.sort_index(inplace=True)
    outdf = normal_transactions.copy()

    # INTERNAL
    url = f'https://api.etherscan.io/api?module=account&action=txlistinternal&address={address}&startblock=0&endblock=9999999999999999999&sort=asc&apikey=GFID2HN2QCS6UR4K1CX13F946P2V1S7Q7X'
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
        internal_transactions['Fee Currency'] = 'ETH'
        internal_transactions['To Coin'] = ''
        internal_transactions['Coin'] = 'ETH'
        internal_transactions['To Amount'] = ''
        internal_transactions['Tag'] = 'Movement'

        internal_transactions = internal_transactions.reindex(
            columns=['From', 'To', 'Coin', 'Amount', 'To Coin', 'To Amount', 'Fiat Price',
                     'Fiat', 'Fee', 'Fee Currency', 'Tag'])

        outdf = pd.concat([outdf, internal_transactions], axis=0)
    outdf.sort_index(inplace=True)

    # Get ERC20 tokens
    url = f'https://api.etherscan.io/api?module=account&action=tokentx&address={address}&startblock=0&endblock=999999999999&sort=asc&apikey=GFID2HN2QCS6UR4K1CX13F946P2V1S7Q7X'
    response = requests.get(url)
    erc20_transactions = pd.DataFrame(response.json().get('result'))
    if erc20_transactions.shape[0] > 0:
        erc20_transactions['from'] = erc20_transactions['from'].map(lambda x: x.lower())
        erc20_transactions['to'] = erc20_transactions['to'].map(lambda x: x.lower())

        erc20_transactions['isScam'] = [1 if k in scam_tokens else 0 for k in erc20_transactions['contractAddress']]
        erc20_transactions = erc20_transactions[erc20_transactions['isScam'] == 0]

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
             'contractAddress', 'cumulativeGasUsed', 'gasUsed', 'confirmations', 'hash', 'input', 'tokenName',
             'isScam'],
            axis=1,
            inplace=True)

        erc20_transactions.loc[erc20_transactions['From'] == address.lower(), 'Amount'] *= -1

        erc20_transactions['Fiat Price'] = 0
        erc20_transactions['Fiat'] = 'EUR'
        erc20_transactions['Fee Currency'] = 'ETH'
        erc20_transactions['To Amount'] = ''
        erc20_transactions['Tag'] = 'Movement'
        erc20_transactions['To Coin'] = ''

        if erc20_transactions[erc20_transactions.duplicated(keep=False)].shape[0] > 0:
            temp_df = erc20_transactions[erc20_transactions.duplicated(keep='first')].copy()
            erc20_transactions.drop(temp_df.index, axis=0, inplace=True)
            temp_df.index = [x + pd.Timedelta(seconds=i) for i, x in enumerate(temp_df.index)]
            erc20_transactions = pd.concat([temp_df, erc20_transactions], axis=0)
        outdf = pd.concat([outdf, erc20_transactions])
        outdf.sort_index(inplace=True)
        outdf = ut.evm_liquidity_swap_calculation(outdf, address)

    global eth_prices

    tokens = outdf['Coin'].tolist()
    tokens.extend(outdf['To Coin'].tolist())
    tokens = [x.upper() for x in list(set(tokens)) if x not in ut.fiat_list]

    was_updated = update_prices(eth_prices, tokens)

    if 'EUR' not in list(eth_prices.exchange_rates.keys()) or was_updated:
        eth_prices.convert_prices('EUR', tokens)

    outdf.sort_index(inplace=True)
    price = eth_prices.to_pd_dataframe('EUR')
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
    outdf['Tag Account'] = f'ETH - {address[0:7]}'
    return outdf


def calculate_all(address: str, year_sel=None, name='Ethereum'):
    transactions = get_transactions_df(address=address)
    balances_in = ut.balances(transactions=transactions, year_sel=year_sel)
    balaces_fiat_in = ut.balances_fiat(balances=balances_in, prices=eth_prices, year_sel=year_sel)
    soglia_in = ut.soglia(balances_in=balances_in, prices=eth_prices, year_sel=year_sel)
    income_in = ut.income(transactions=transactions, type_out='crypto', name=name, year_sel=year_sel)
    income_in_fiat = ut.income(transactions=transactions, name=name, year_sel=year_sel)
    vout = {"transactions": transactions, "transactions_raw": transactions, "balances": balances_in,
            "balaces_fiat": balaces_fiat_in, "soglia": soglia_in,
            "income": income_in, "income_fiat": income_in_fiat}
    return vout
