import numpy as np
import pandas as pd
from utility.tax_log import log
import utility.utils as ut
import datetime as dt
import requests
from utility.PricesClass import Prices, update_prices

log.info('Cardano calculator - updated on 15/10/2022')

ada_prices = Prices()


def get_transactions_df(address):
    req_response = requests.get(f"https://api.blockchair.com/cardano/raw/address/{address}")
    req_response = req_response.json()
    tx_list = req_response['data'][address]['address']['caTxList']

    output_df = pd.DataFrame()
    if len(tx_list) == 0:
        return output_df
    for transaction in tx_list:
        transaction_time = dt.datetime.fromtimestamp(transaction['ctbTimeIssued'])

        output = pd.DataFrame(transaction['ctbOutputs'])
        output['ctaAmount'] = [int(x['getCoin']) / 10 ** 6 for x in output['ctaAmount']]

        inputs = pd.DataFrame(transaction['ctbInputs'])
        inputs['ctaAmount'] = [int(x['getCoin']) / 10 ** 6 for x in inputs['ctaAmount']]

        final_df = output.join(inputs, how='outer', lsuffix='-Output')
        final_df.rename(columns={'ctaAddress': 'From', 'ctaAddress-Output': 'To',
                                 'ctaAmount': 'Amount', 'ctaAmount-Output': 'To Amount'}, inplace=True)

        final_df['Fee'] = final_df['Amount'].sum() - final_df['To Amount'].sum()
        final_df['Fee Currency'] = 'ADA'

        final_df = final_df[np.logical_or(final_df['From'] == address, final_df['To'] == address)]
        final_df.index = [transaction_time]

        if output_df.shape[0] == 0:
            output_df = final_df.copy()
        else:
            output_df = pd.concat([output_df, final_df], axis=0)

    output_df.loc[output_df['From'] == address, 'Amount'] *= -1
    output_df.loc[output_df['From'] == address, 'To Amount'] *= 0
    output_df.loc[output_df['To'] == address, 'Amount'] *= 0

    output_df['To Coin'] = 'ADA'
    output_df['Coin'] = 'ADA'
    output_df['Fee'] *= -1
    output_df['Fiat Price'] = 0
    output_df['Fiat'] = 'EUR'

    output_df.drop(['ctaTxHash-Output', 'ctaTxIndex-Output', 'ctaTxHash', 'ctaTxIndex'], axis=1, inplace=True)

    output_df.fillna(0, inplace=True)
    global ada_prices

    was_updated = update_prices(ada_prices, list(set(output_df['Coin'])))

    if 'EUR' not in list(ada_prices.exchange_rates.keys()) or was_updated:
        ada_prices.get_exchange_rates('EUR')
        ada_prices.convert_prices('EUR', list(set(output_df['Coin'])))

    price = ada_prices.to_pd_dataframe('EUR')
    price = price[~price.index.duplicated(keep='first')]
    for tok in set(output_df['Coin']):
        temp_df = output_df[output_df['Coin'] == tok].copy()
        temp_df.index = [k.date() for k in temp_df.index]
        temp_df = temp_df.join(pd.DataFrame(price[tok]))
        temp_df['Fiat Price'] = temp_df['Amount'] * temp_df[tok]
        temp_df.loc[temp_df['To Coin'] == 'EUR', 'Fiat Price'] = temp_df.loc[
            temp_df['To Coin'] == 'EUR', 'To Amount']
        temp_df.index = output_df[output_df['Coin'] == tok].index
        output_df.loc[output_df['Coin'] == tok, 'Fiat Price'] = temp_df['Fiat Price']

    output_df['Tag'] = 'Movement'
    output_df = output_df.reindex(
        columns=['From', 'To', 'Coin', 'Amount', 'To Coin', 'To Amount', 'Fiat Price',
                 'Fiat', 'Fee', 'Fee Currency', 'Tag'])
    output_df.sort_index(inplace=True)

    output_df['Tag Account'] = f'Cardano - {address[0:7]}'

    return output_df


def get_transactions_df_batch(address_list):
    final_df = pd.DataFrame()

    for address in address_list:
        if final_df.shape[0] == 0:
            final_df = get_transactions_df(address)
        else:
            final_df = pd.concat([final_df, get_transactions_df(address)])

    return final_df


def get_staking(stake_key):
    output_df = pd.DataFrame()
    headers = {'project_id': 'mainnetmcMyUEdLQFlMvOZQRwaahWZhFzc9A4mL'}
    response = requests.get(f'https://cardano-mainnet.blockfrost.io/api/v0/accounts/{stake_key}/withdrawals',
                            headers=headers)
    withdrawals = response.json()
    hashes = [k['tx_hash'] for k in withdrawals]
    amounts = [int(k['amount']) / 10 ** 6 for k in withdrawals]
    fees = []
    times = []

    for hash_in in hashes:
        response = requests.get(f'https://cardano-mainnet.blockfrost.io/api/v0/txs/{hash_in}', headers=headers)
        content = response.json()
        fees.append(int(content['fees']) / 10 ** 6)
        times.append(dt.datetime.fromtimestamp(content['block_time']))

    output_df['Amount'] = amounts
    output_df['Fee'] = fees
    output_df['To'] = ''
    output_df['To Amount'] = 0
    output_df['From'] = ''
    output_df['Fee Currency'] = 'ADA'
    output_df['To Coin'] = ''
    output_df['Coin'] = 'ADA'
    output_df['Fiat Price'] = 0
    output_df['Fiat'] = 'EUR'
    output_df['Tag'] = 'Reward'
    output_df.index = times
    output_df = output_df.reindex(
        columns=['From', 'To', 'Coin', 'Amount', 'To Coin', 'To Amount', 'Fiat Price',
                 'Fiat', 'Fee', 'Fee Currency', 'Tag'])

    output_df.sort_index(inplace=True)

    if output_df.shape[0] == 0:
        return output_df

    output_df.sort_index(inplace=True)
    price = ada_prices.to_pd_dataframe('EUR')
    price = price[~price.index.duplicated(keep='first')]
    for tok in set(output_df['Coin']):
        temp_df = output_df[output_df['Coin'] == tok].copy()
        temp_df.index = [k.date() for k in temp_df.index]
        temp_df = temp_df.join(pd.DataFrame(price[tok]))
        temp_df['Fiat Price'] = temp_df['Amount'] * temp_df[tok]
        temp_df.loc[temp_df['To Coin'] == 'EUR', 'Fiat Price'] = temp_df.loc[
            temp_df['To Coin'] == 'EUR', 'To Amount']
        temp_df.index = output_df[output_df['Coin'] == tok].index
        output_df.loc[output_df['Coin'] == tok, 'Fiat Price'] = temp_df['Fiat Price']

    return output_df


def get_stake_df_batch(stake_list):
    final_df = pd.DataFrame()

    for address in stake_list:
        if final_df.shape[0] == 0:
            final_df = get_staking(address)
        else:
            final_df = pd.concat([final_df, get_staking(address)])

    return final_df


def get_transactions_and_staking(address_list, stake_list=None):
    transactions = get_transactions_df_batch(address_list)
    if stake_list is not None:
        stakes = get_stake_df_batch(stake_list)
        final_df = pd.concat([transactions, stakes])
    else:
        final_df = transactions.copy()
    final_df.sort_index(inplace=True)
    return final_df


def calculate_all(address_list: list[str], stake_list=None, year_sel=None, name='ADA_NC'):
    transactions = get_transactions_and_staking(address_list=address_list, stake_list=stake_list)
    balances_in = ut.balances(transactions=transactions, year_sel=year_sel)
    balaces_fiat_in = ut.balances_fiat(balances=balances_in, prices=ada_prices, year_sel=year_sel)
    soglia_in = ut.soglia(balances=balances_in, prices=ada_prices, year_sel=year_sel)
    income_in = ut.income(transactions=transactions, type_out='crypto', name=name, year_sel=year_sel)
    income_in_fiat = ut.income(transactions=transactions, name=name, year_sel=year_sel)
    vout = {"transactions": transactions, "transactions_raw": transactions, "balances": balances_in,
            "balaces_fiat": balaces_fiat_in, "soglia": soglia_in,
            "income": income_in, "income_fiat": income_in_fiat}
    return vout
