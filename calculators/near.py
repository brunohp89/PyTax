import numpy as np
from utility.PricesClass import Prices, update_prices
from utility import tax_library as tx
import datetime as dt
import pandas as pd
import psycopg2
from utility.tax_log import log
import utility.utils as ut

log.info('Near calculator - updated on 15/10/2022')

scam_tokens = []

conn = psycopg2.connect(host="104.199.89.51", database="mainnet_explorer", user="public_readonly",
                        password="nearprotocol")
cur = conn.cursor()

near_prices = Prices()


def create_pandas_table(sql_query, database=conn):
    table = pd.read_sql_query(sql_query, database)
    return table


def get_transactions_df(addresses_list):
    addlist = addresses_list
    addresses = "','".join(addlist)
    addresses = f"'{addresses}'"

    query_normal = f"""
SELECT block_timestamp as "timestamp", signer_account_id as "From", receiver_account_id as "To", action_kind as "kind", "args"
FROM TRANSACTIONS inner join transaction_actions ac on transactions.transaction_hash = ac.transaction_hash
WHERE signer_account_id in ({addresses})
	OR receiver_account_id in ({addresses})
	AND action_kind = 'TRANSFER'"""
    normal_transactions = create_pandas_table(query_normal)

    normal_transactions = normal_transactions[normal_transactions['kind'] == 'TRANSFER']
    normal_transactions['kind'] = 'Movement'

    normal_transactions['From'] = normal_transactions['From'].map(lambda x: x.lower())
    normal_transactions['To'] = normal_transactions['To'].map(lambda x: x.lower())

    normal_transactions.index = normal_transactions['timestamp'].map(
        lambda x: dt.datetime.fromtimestamp(int(x) / 10 ** 9))

    normal_transactions['args'] = normal_transactions['args'].map(lambda x: int(x['deposit']) / 10 ** 24)

    normal_transactions.rename(columns={'args': 'Amount', 'kind': 'Tag'}, inplace=True)
    normal_transactions.drop(
        ['timestamp'], axis=1,
        inplace=True)

    normal_transactions['Fiat Price'] = 0
    normal_transactions['Fiat'] = 'EUR'
    normal_transactions['Fee Currency'] = 'NEAR'
    normal_transactions['Fee'] = 0
    normal_transactions['To Coin'] = ''
    normal_transactions['Coin'] = 'NEAR'
    normal_transactions['To Amount'] = ''

    normal_transactions = normal_transactions.reindex(
        columns=['From', 'To', 'Coin', 'Amount', 'To Coin', 'To Amount', 'Fiat Price',
                 'Fiat', 'Fee', 'Fee Currency', 'Tag'])

    normal_transactions.loc[
        np.logical_and(normal_transactions['From'].isin(addlist),
                       ~normal_transactions['To'].isin(addlist)), 'Amount'] *= -1

    # Get NEP20 tokens
    query_nep20 = f"""SELECT INCLUDED_IN_BLOCK_TIMESTAMP AS "timestamp",
	AMOUNT,
	EVENT_KIND,
	TOKEN_NEW_OWNER_ACCOUNT_ID AS "To",
	TOKEN_OLD_OWNER_ACCOUNT_ID AS "From",
	RECEIPT_CONVERSION_GAS_BURNT AS "gas",
	upper(substring(emitted_by_contract_account_id,7)) as "coin"
FROM ASSETS__FUNGIBLE_TOKEN_EVENTS AC
INNER JOIN RECEIPTS RP ON AC.EMITTED_FOR_RECEIPT_ID = RP.RECEIPT_ID
INNER JOIN TRANSACTIONS TRX ON TRX.TRANSACTION_HASH = RP.ORIGINATED_FROM_TRANSACTION_HASH
WHERE TOKEN_NEW_OWNER_ACCOUNT_ID in ({addresses})
	OR TOKEN_OLD_OWNER_ACCOUNT_ID in ({addresses})"""
    nep20_transactions = create_pandas_table(query_nep20)

    if nep20_transactions.shape[0] > 0:
        nep20_transactions['From'] = nep20_transactions['From'].map(lambda x: x.lower())
        nep20_transactions['To'] = nep20_transactions['To'].map(lambda x: x.lower())

        nep20_transactions.loc[nep20_transactions[
                                   'To'] == 'deposits.grow.sweat'.lower(), 'amount'] = 0  # Sweat staking

        nep20_transactions['amount'] = [int(s) / 10 ** 18 for s in nep20_transactions['amount']]
        nep20_transactions['gas'] = [int(s) / 10 ** 18 for s in nep20_transactions['gas']]

        nep20_transactions.loc[np.logical_and(nep20_transactions['To'].isin(addlist),
                                              nep20_transactions['From'].isin(addlist),
                                              nep20_transactions['event_kind'] == 'TRANSFER'), 'amount'] = 0

        nep20_transactions.rename(columns={'from': 'From', 'to': 'To', 'amount': 'Amount', 'gas': 'Fee', 'coin': 'Coin',
                                           'event_kind': 'Tag'}, inplace=True)
        nep20_transactions.index = nep20_transactions['timestamp'].map(
            lambda x: dt.datetime.fromtimestamp(int(x) / 10 ** 9))
        nep20_transactions.drop(['timestamp'], axis=1, inplace=True)

        nep20_transactions.loc[
            np.logical_and(nep20_transactions['From'].isin(addlist), ~nep20_transactions['To'].isin(addlist),
                           nep20_transactions['Tag'] == 'TRANSFER'), 'Amount'] *= -1

        nep20_transactions['Fiat Price'] = 0
        nep20_transactions['Fiat'] = 'EUR'
        nep20_transactions['Fee Currency'] = 'NEAR'
        nep20_transactions['To Amount'] = ''
        nep20_transactions['To Coin'] = ''
        nep20_transactions.loc[nep20_transactions['Tag'] == 'TRANSFER', 'Tag'] = 'Movement'
        nep20_transactions.loc[nep20_transactions['Tag'] == 'MINT', 'Tag'] = 'Reward'
        nep20_transactions.loc[np.logical_and(nep20_transactions['Tag'] == 'Movement', nep20_transactions[
            'From'] == 'tge-lockup.sweat'), 'Tag'] = 'Reward'  # SWEAT airdrops

        nep20_transactions = nep20_transactions.reindex(
            columns=['From', 'To', 'Coin', 'Amount', 'To Coin', 'To Amount', 'Fiat Price',
                     'Fiat', 'Fee', 'Fee Currency', 'Tag'])

        outdf = pd.concat([nep20_transactions, normal_transactions])
    else:
        outdf = normal_transactions.copy()

    outdf.sort_index(inplace=True)
    global near_prices

    tokens = outdf['Coin'].tolist()
    tokens.extend(outdf['To Coin'].tolist())
    tokens = [x.upper() for x in list(set(tokens)) if x not in ut.fiat_list]

    was_updated = update_prices(near_prices, tokens)

    if 'EUR' not in list(near_prices.exchange_rates.keys()) or was_updated:
        near_prices.convert_prices('EUR', tokens)

    outdf.sort_index(inplace=True)
    price = near_prices.to_pd_dataframe('EUR')
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

    outdf.loc[outdf['To'].isin(addlist), 'Fee'] = 0
    outdf['Tag Account'] = f'Near - {addresses[0:7]}'
    return outdf


def calculate_all(address_list: list[str], year_sel=None, name='NEAR'):
    transactions = get_transactions_df(addresses_list=address_list)
    balances_in = ut.balances(transactions=transactions, year_sel=year_sel)
    balaces_fiat_in = ut.balances_fiat(balances=balances_in, prices=near_prices, year_sel=year_sel)
    soglia_in = ut.soglia(balances_in=balances_in, prices=near_prices, year_sel=year_sel)
    income_in = ut.income(transactions=transactions, type_out='crypto', name=name, year_sel=year_sel)
    income_in_fiat = ut.income(transactions=transactions, name=name, year_sel=year_sel)
    vout = {"transactions": transactions, "transactions_raw": transactions, "balances": balances_in,
            "balaces_fiat": balaces_fiat_in, "soglia": soglia_in,
            "income": income_in, "income_fiat": income_in_fiat}
    return vout
