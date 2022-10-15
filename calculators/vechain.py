# Download transactions using this website https://explore.vechain.org/download?address=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

import os
import datetime as dt
import numpy as np
import pandas as pd
from utility.tax_log import log
from utility import tax_library as tx
from utility.PricesClass import Prices, update_prices
import utility.utils as ut

log.info('VeChain calculator - updated on 15/10/2022')

first = True

vet_prices = Prices()


def get_transactions_df(raw=False):
    vechain_files = [os.getcwd() + '\\VeChain' + '\\' + x for x in os.listdir(os.getcwd() + '\\VeChain')]
    if len(vechain_files) == 0:
        log.info('No files for VeChain found')
        return None
    else:
        df_list = []
        for filename in vechain_files:
            df_loop = pd.read_csv(filename, index_col=None, header=0)
            df_list.append(df_loop)
        final_df = pd.concat(df_list, axis=0, ignore_index=True)
        final_df.index = [tx.str_to_datetime(x) for x in list(final_df['Date(GMT)'])]

        global first
        if first:
            log.info(f'VeChain transactions up to {str(max(final_df.index))[0:10]}')
            first = False

        final_df.drop_duplicates(inplace=True, subset=['Txid'])
        final_df.sort_index(inplace=True)

        final_df['Fee'] = ''
        final_df['Fee Currency'] = 'VTHO'
        final_df['To Coin'] = ''
        final_df['To Amount'] = ''
        final_df['Fiat'] = 'EUR'
        final_df['Fiat Price'] = 0

        final_df.rename(columns={'Remark': 'Tag', 'Sender': 'From', 'Recipient': 'To', 'Token': 'Coin'}, inplace=True)

        final_df['Tag'] = 'Movement'

        final_df = final_df.reindex(
            columns=['From', 'To', 'Coin', 'Amount', 'To Coin', 'To Amount', 'Fiat Price',
                     'Fiat', 'Fee', 'Fee Currency', 'Tag'])

        vtho_transactions = pd.date_range(min(final_df.index), dt.datetime.now() - dt.timedelta(days=1), freq='d')
        vtho_df = pd.DataFrame(index=vtho_transactions, data=np.zeros([len(vtho_transactions), final_df.shape[1]]),
                               columns=final_df.columns)
        vtho_df.index = [tx.str_to_datetime(str(x)) for x in vtho_df.index]
        vtho_date = [x.date() for x in vtho_df.index]

        vet_bal = ut.balances(final_df)

        vtho_df['Amount'] = list(vet_bal.loc[vtho_date, 'VET'] * 0.000432)
        vtho_df['Fee'] = ''
        vtho_df['To'] = ''
        vtho_df['From'] = ''
        vtho_df['Fee Currency'] = 'VTHO'
        vtho_df['Coin'] = 'VTHO'
        vtho_df['To Coin'] = ''
        vtho_df['To Amount'] = ''
        vtho_df['Fiat'] = 'EUR'
        vtho_df['Fiat Price'] = 0
        vtho_df['Tag'] = 'Reward'

        final_df = pd.concat([final_df, vtho_df], axis=0)

        if raw:
            return final_df

        global vet_prices

        tokens = final_df['Coin'].tolist()
        tokens.extend(final_df['To Coin'].tolist())
        tokens = [x.upper() for x in list(set(tokens)) if x not in ut.fiat_list]

        was_updated = update_prices(vet_prices, tokens)

        if 'EUR' not in list(vet_prices.exchange_rates.keys()) or was_updated:
            vet_prices.convert_prices('EUR', tokens)

        final_df.sort_index(inplace=True)
        price = vet_prices.to_pd_dataframe('EUR')
        price = price[~price.index.duplicated(keep='first')]
        for tok in tokens:
            temp_df = final_df[final_df['Coin'] == tok].copy()
            temp_df.index = [k.date() for k in temp_df.index]

            temp_df = temp_df.join(pd.DataFrame(price[tok]))
            temp_df['Fiat Price'] = temp_df['Amount'] * temp_df[tok]
            temp_df.loc[temp_df['To Coin'] == 'EUR', 'Fiat Price'] = temp_df.loc[
                temp_df['To Coin'] == 'EUR', 'To Amount']
            temp_df.index = final_df[final_df['Coin'] == tok].index
            final_df.loc[final_df['Coin'] == tok, 'Fiat Price'] = temp_df['Fiat Price']
        final_df['Tag Account'] = 'VEChain'
        return final_df


def calculate_all(year_sel=None, name='VeChain'):
    transactions = get_transactions_df(False)
    transactions_raw = get_transactions_df(True)
    balances_in = ut.balances(transactions=transactions, year_sel=year_sel)
    balaces_fiat_in = ut.balances_fiat(balances=balances_in, prices=vet_prices, year_sel=year_sel)
    soglia_in = ut.soglia(balances_in=balances_in, prices=vet_prices, year_sel=year_sel)
    income_in = ut.income(transactions=transactions, type_out='crypto', name=name, year_sel=year_sel)
    income_in_fiat = ut.income(transactions=transactions, name=name, year_sel=year_sel)
    vout = {"transactions": transactions, "transactions_raw": transactions_raw, "balances": balances_in,
            "balaces_fiat": balaces_fiat_in, "soglia": soglia_in,
            "income": income_in, "income_fiat": income_in_fiat}
    return vout
