import os
import datetime as dt
import numpy as np
import pandas as pd
from utility import tax_library as tx
from utility.PricesClass import Prices, update_prices
from utility.tax_log import log
import utility.utils as ut

log.info('Crypto.com calculator - updated on 15/10/2022')
# Aggiunte transazioni finte di -34.92794 USDC e -0.00335 per appianare delle discrepanze non individuate dal file/codice
first_use = True
cdc_prices = Prices()


def get_transactions_df(raw=False):
    cdc_files = [os.getcwd() + '\\crypto.com' + '\\' + x for x in os.listdir(os.getcwd() + '\\crypto.com')]
    if len(cdc_files) == 0:
        log.info('No files for crypto.com found')
        return None
    else:
        df_list = []
        for filename in cdc_files:
            df_loop = pd.read_csv(filename, index_col=None, header=0)
            df_list.append(df_loop)
        final_df = pd.concat(df_list, axis=0, ignore_index=True)
        final_df.index = [tx.str_to_datetime(j) for j in final_df['Timestamp (UTC)']]
        final_df.loc[final_df['Currency'] == 'LUNA', 'Currency'] = 'LUNC'
        final_df.loc[final_df['Currency'] == 'LUNA2', 'Currency'] = 'LUNC'
        final_df.loc[final_df['To Currency'] == 'LUNA', 'To Currency'] = 'LUNC'
        final_df.loc[final_df['To Currency'] == 'LUNA2', 'Currency'] = 'LUNC'

        global first_use
        if first_use:
            log.info(f'Crypto.com transactions last updated on {max(final_df.index).date()}')
            first_use = False

        final_df.drop_duplicates(inplace=True,
                                 subset=['Timestamp (UTC)', 'Amount', 'Transaction Description', 'Currency'])
        final_df.sort_index(inplace=True)
        if raw:
            return final_df
        final_df['From'] = ''
        final_df['To'] = ''
        final_df['Fee'] = 0
        final_df['Fee Currency'] = ''
        final_df.rename(columns={'Transaction Kind': 'Tag', 'Currency': 'Coin', 'To Currency': 'To Coin',
                                 'Native Amount': 'Fiat Price', 'Native Currency': 'Fiat'}, inplace=True)

        final_df.loc[
            final_df['Transaction Description'].str.contains('Recurring'), 'Transaction Description'] = 'Recurring'
        final_df.loc[final_df['Transaction Description'].str.contains('Buy'), 'To Coin'] = 'EUR'
        final_df.loc[final_df['Transaction Description'].str.contains('Buy'), 'To Amount'] = final_df.loc[
            final_df['Transaction Description'].str.contains('Buy'), 'Amount']

        final_df = final_df[~final_df['Transaction Description'].isin(['Crypto Earn Deposit', 'Crypto Earn Allocation',
                                                                       'Crypto Earn Withdrawal', 'CRO Stake',
                                                                       'CRO Unstake',
                                                                       'Supercharger Deposit (via app)',
                                                                       'Supercharger Stake (via app)',
                                                                       'Supercharger Withdrawal (via app)'])]

        final_df.loc[final_df['Transaction Description'] == 'Recurring', 'Amount'] *= -1

        final_df.loc[final_df['Tag'].str.contains("cashback|Rebate"), 'Tag'] = 'Cashback'
        final_df.loc[final_df['Transaction Description'].str.contains("Rebate"), 'Tag'] = 'Cashback'
        final_df.loc[final_df['Tag'].str.contains("earn|supercharger"), 'Tag'] = 'Interest'
        final_df.loc[final_df['Transaction Description'].str.contains("Reward"), 'Tag'] = 'Reward'
        final_df.loc[~pd.isna(final_df['To Amount']), 'Tag'] = 'Trade'
        final_df.loc[~final_df['Tag'].isin(['Cashback', 'Interest', 'Reward', 'Trade']), 'Tag'] = 'Movement'

        final_df.drop(['Timestamp (UTC)', 'Transaction Description', 'Native Amount (in USD)'], axis=1, inplace=True)

        final_df = final_df.reindex(columns=['From', 'To', 'Coin', 'Amount', 'To Coin', 'To Amount', 'Fiat Price',
                                             'Fiat', 'Fee', 'Fee Currency', 'Tag'])

        final_df.loc[np.logical_and(final_df['From'] == '', final_df['Amount'] < 0), 'From'] = 'Crypto.com'
        final_df['Tag Account'] = 'Crypto.com'

        final_df['To Coin'].fillna('', inplace=True)
        final_df['To Amount'].fillna('', inplace=True)

        tokens = final_df['Coin'].tolist()
        tokens.extend(final_df['To Coin'].tolist())
        tokens = [x.upper() for x in list(set(tokens)) if x not in ut.fiat_list]

        global cdc_prices
        was_updated = update_prices(cdc_prices, tokens)

        if 'EUR' not in list(cdc_prices.exchange_rates.keys()) or was_updated:
            cdc_prices.convert_prices('EUR', tokens)

        return final_df


def calculate_all(year_sel=None, name='Crypto.com'):
    transactions = get_transactions_df(False)
    transactions_raw = get_transactions_df(True)
    balances_in = ut.balances(transactions=transactions, year_sel=year_sel)
    balaces_fiat_in = ut.balances_fiat(balances=balances_in, prices=cdc_prices, year_sel=year_sel)
    soglia_in = ut.soglia(balances_in=balances_in, prices=cdc_prices, year_sel=year_sel)
    income_in = ut.income(transactions=transactions, type_out='crypto', name=name, year_sel=year_sel)
    income_in_fiat = ut.income(transactions=transactions, name=name, year_sel=year_sel)
    vout = {"transactions": transactions, "transactions_raw": transactions_raw, "balances": balances_in,
            "balaces_fiat": balaces_fiat_in, "soglia": soglia_in,
            "income": income_in, "income_fiat": income_in_fiat}
    return vout
