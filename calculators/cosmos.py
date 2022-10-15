import numpy as np
from utility.PricesClass import Prices, update_prices
from utility import tax_library as tx
from utility.tax_log import log
import utility.utils as ut
import datetime as dt
import pandas as pd
import requests

log.info('Cosmos calculator - updated on 15/10/2022')
cosmos_prices = Prices()


def get_transactions_df(address):
    address = address.lower()
    url = f'https://api.cosmoscan.net/transactions?address={address}'
    response = requests.get(url)

    transactions = pd.DataFrame(response.json()['items'])
    trxresp = transactions['hash'].apply(lambda x: requests.get(f'https://api.cosmoscan.net/transaction/{x}'))
    trxresp = trxresp.apply(lambda x: pd.DataFrame(x.json()))

    normal_transactions = pd.concat(list(trxresp), axis=0, ignore_index=True)
    final_df = []
    for trx in range(normal_transactions.shape[0]):
        if normal_transactions['messages'][trx]['type'] == 'Delegate':
            final_df.append(pd.DataFrame([[dt.datetime.fromtimestamp(int(normal_transactions['created_at'][trx])),
                                           normal_transactions['messages'][trx]['body']['delegator_address'],
                                           normal_transactions['messages'][trx]['body']['validator_address'], 0,
                                           normal_transactions['messages'][trx]['body']['amount']['denom'][1:].upper(),
                                           -float(normal_transactions['fee'][trx]), 'ATOM', 'Delegate', 'EUR', 0,
                                           'Cosmos']]))
        elif normal_transactions['messages'][trx]['type'] == 'Send':
            final_df.append(pd.DataFrame(
                [[dt.datetime.fromtimestamp(int(normal_transactions['created_at'][trx])),
                  normal_transactions['messages'][trx]['body']['from_address'],
                  normal_transactions['messages'][trx]['body']['to_address'],
                  int(normal_transactions['messages'][trx]['body']['amount'][0]['amount']) / 10 ** 6,
                  normal_transactions['messages'][trx]['body']['amount'][0]['denom'][1:].upper(),
                  -float(normal_transactions['fee'][trx]), 'ATOM', 'Movement', 'EUR', 0, 'Cosmos']]))
    final_df = pd.concat(final_df, ignore_index=True, axis=0)

    final_df.columns = ['Timestamp', 'From', 'To', 'Amount', 'Coin', 'Fee', 'Fee Currency', 'Tag', 'Fiat', 'Fiat Price',
                        'Tag Account']
    final_df.index = final_df['Timestamp']
    final_df.drop(['Timestamp'], inplace=True, axis=1)

    final_df['To Coin'] = ''
    final_df['To Amount'] = ''

    final_df = final_df.reindex(
        columns=['From', 'To', 'Coin', 'Amount', 'To Coin', 'To Amount', 'Fiat Price',
                 'Fiat', 'Fee', 'Fee Currency', 'Tag'])

    final_df.sort_index(inplace=True)
    outdf = final_df.copy()
    outdf.loc[outdf['From'] == address, 'Amount'] *= -1
    global cosmos_prices

    tokens = outdf['Coin'].tolist()
    tokens.extend(outdf['To Coin'].tolist())
    tokens = [x.upper() for x in list(set(tokens)) if x not in ut.fiat_list]

    was_updated = update_prices(cosmos_prices, tokens)

    if 'EUR' not in list(cosmos_prices.exchange_rates.keys()) or was_updated:
        cosmos_prices.convert_prices('EUR', tokens)

    outdf.sort_index(inplace=True)
    price = cosmos_prices.to_pd_dataframe('EUR')
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
    outdf['Tag Account'] = f'Cosmos - {address[0:7]}'

    return outdf


def calculate_all(address: str, year_sel=None, name='Cosmos'):
    transactions = get_transactions_df(address=address)
    balances_in = ut.balances(transactions=transactions, year_sel=year_sel)
    balaces_fiat_in = ut.balances_fiat(balances=balances_in, prices=cosmos_prices, year_sel=year_sel)
    soglia_in = ut.soglia(balances_in=balances_in, prices=cosmos_prices, year_sel=year_sel)
    income_in = ut.income(transactions=transactions, type_out='crypto', name=name, year_sel=year_sel)
    income_in_fiat = ut.income(transactions=transactions, name=name, year_sel=year_sel)
    vout = {"transactions": transactions, "transactions_raw": transactions, "balances": balances_in,
            "balaces_fiat": balaces_fiat_in, "soglia": soglia_in,
            "income": income_in, "income_fiat": income_in_fiat}
    return vout
