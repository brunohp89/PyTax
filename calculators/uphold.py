import os
import numpy as np
import pandas as pd
from utility import tax_library as tx
from utility.PricesClass import Prices, update_prices
from utility.tax_log import log
import utility.utils as ut

log.info('Uphold calculator - updated on 15/10/2022')

first_use = True
up_prices = Prices()


def get_transactions_df(raw=False):
    transactions_uphold = [os.getcwd() + '\\uphold' + '\\' + x for x in os.listdir(os.getcwd() + '\\uphold')]
    if len(transactions_uphold) == 0:
        log.info('No files for uphold found')
        return None
    else:
        df_list = []
        for filename in transactions_uphold:
            df_loop = pd.read_csv(filename, index_col=None, header=0)
            df_list.append(df_loop)
        final_df = pd.concat(df_list, axis=0, ignore_index=True)
        final_df.drop_duplicates(subset=['Date', 'Destination Amount', 'Destination Currency'], inplace=True)
        final_df.reset_index(inplace=True)
        if raw:
            return final_df
        for i in range(final_df.shape[0]):
            if final_df.loc[i, 'Destination'] != 'uphold':
                final_df.loc[i, 'Destination Amount'] *= -1
                if not pd.isna(final_df.loc[i, 'Fee Amount']):
                    final_df.loc[i, 'Destination Amount'] += final_df.loc[i, 'Fee Amount'] * -1
            if final_df.loc[i, 'Origin Currency'] == final_df.loc[i, 'Destination Currency']:
                final_df.loc[i, 'Origin Amount'] = 0
            else:
                final_df.loc[i, 'Origin Amount'] *= -1

        final_df.index = [tx.uphold_date_to_datetime(final_df.loc[i, 'Date']) for i in range(final_df.shape[0])]

        global first_use
        if first_use:
            log.info(f'Uphold transactions last updated on {max(final_df.index).date()}')
            first_use = False

        final_df.sort_index(inplace=True)
        final_df['From'] = ''
        final_df['To'] = ''
        final_df['Fiat Price'] = 0
        final_df['Fiat'] = 'EUR'

        final_df.rename(columns={'Origin Currency': 'Coin', 'Origin Amount': 'Amount',
                                 'Destination Currency': 'To Coin', 'Destination Amount': 'To Amount',
                                 'Fee Amount': 'Fee', 'Type': 'Tag'}, inplace=True)

        final_df.loc[final_df['Tag'] == 'out', 'Amount'] = final_df.loc[final_df['Tag'] == 'out', 'To Amount']
        final_df.loc[final_df['Tag'] == 'out', 'To Amount'] = ''
        final_df.loc[final_df['Tag'] == 'out', 'To Coin'] = ''

        final_df.drop(['Date', 'Destination', 'Id', 'Origin', 'Status'], axis=1, inplace=True)
        final_df['Fee'].fillna('', inplace=True)
        final_df['Fee Currency'].fillna('', inplace=True)
        final_df.fillna(0, inplace=True)

        tokens = final_df['Coin'].tolist()
        tokens.extend(final_df['To Coin'].tolist())
        tokens = [x.upper() for x in list(set(tokens)) if
                  x not in ut.fiat_list]
        global up_prices

        update_prices(up_prices, tokens)
        up_prices.get_exchange_rates('EUR')
        up_prices.convert_prices('EUR', tokens)

        final_df.loc[np.logical_and(final_df['Coin'] == 'BAT', final_df['Amount'] == 0), 'Amount'] = \
            final_df.loc[np.logical_and(final_df['Coin'] == 'BAT', final_df['Amount'] == 0), 'To Amount']

        final_df.sort_index(inplace=True)
        price = up_prices.to_pd_dataframe('EUR')
        price = price[~price.index.duplicated(keep='first')]
        for tok in set(final_df['Coin']):
            if tok == 'EUR':
                temp_df = final_df[final_df['Coin'] == tok].copy()
                temp_df.loc[temp_df['Coin'] == 'EUR', 'Fiat Price'] = temp_df.loc[temp_df['Coin'] == 'EUR', 'Amount']
                final_df.loc[final_df['Coin'] == tok, 'Fiat Price'] = temp_df['Fiat Price']
                continue
            temp_df = final_df[final_df['Coin'] == tok].copy()
            temp_df.index = [k.date() for k in temp_df.index]
            temp_df = temp_df.join(pd.DataFrame(price[tok]))
            temp_df['Fiat Price'] = temp_df['Amount'] * temp_df[tok]
            temp_df.loc[temp_df['To Coin'] == 'EUR', 'Fiat Price'] = temp_df.loc[
                temp_df['To Coin'] == 'EUR', 'To Amount']
            temp_df.index = final_df[final_df['Coin'] == tok].index
            final_df.loc[final_df['Coin'] == tok, 'Fiat Price'] = temp_df['Fiat Price']

        final_df = final_df.reindex(columns=['From', 'To', 'Coin', 'Amount', 'To Coin', 'To Amount', 'Fiat Price',
                                             'Fiat', 'Fee', 'Fee Currency', 'Tag'])

        final_df['Tag'] = 'Movement'
        final_df.loc[np.logical_and(final_df['Coin'] == 'BAT', final_df['Amount'] > 0), 'Tag'] = 'Reward'
        final_df.loc[np.logical_and(final_df['Coin'] == 'BAT', final_df['Amount'] > 0), 'Amount'] = 0
        final_df.loc[np.logical_and(final_df['From'] == '', final_df['Amount'] < 0), 'From'] = 'Uphold'

        final_df['Tag Account'] = 'Uphold'
        return final_df


def calculate_all(year_sel=None, name='Uphold'):
    transactions = get_transactions_df(False)
    transactions_raw = get_transactions_df(True)
    balances_in = ut.balances(transactions=transactions, year_sel=year_sel)
    balaces_fiat_in = ut.balances_fiat(balances=balances_in, prices=up_prices, year_sel=year_sel)
    soglia_in = ut.soglia(balances_in=balances_in, prices=up_prices, year_sel=year_sel)
    income_in = ut.income(transactions=transactions, type_out='crypto', name=name, year_sel=year_sel)
    income_in_fiat = ut.income(transactions=transactions, name=name, year_sel=year_sel)
    vout = {"transactions": transactions, "transactions_raw": transactions_raw, "balances": balances_in,
            "balaces_fiat": balaces_fiat_in, "soglia": soglia_in,
            "income": income_in, "income_fiat": income_in_fiat}
    return vout
