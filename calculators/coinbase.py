import os

import numpy as np

from utility.utils import log
from utility.PricesClass import Prices, update_prices
import pandas as pd
from utility import utils as ut, tax_library as tx

log.info('Coinbase calculator - updated on 15/10/2022')

first_use = True
coinbase_prices = Prices()


def get_transactions_df(raw=False):
    coinbase_files = [os.getcwd() + '\\coinbase' + '\\' + x for x in os.listdir(os.getcwd() + '\\coinbase')]
    if len(coinbase_files) == 0:
        log.info('No files for coinbase found')
        return None
    else:
        df_list = []
        for filename in coinbase_files:
            df_loop = pd.read_csv(filename, index_col=None, header=0)
            df_list.append(df_loop)
        final_df = pd.concat(df_list, axis=0, ignore_index=True)
        final_df = final_df[final_df['Transaction Type'] != 'Learning Reward']

        final_df.index = [tx.str_to_datetime(j.replace('T', ' ').replace('Z', '')) for j in
                          final_df['Timestamp']]

        final_df.sort_index(inplace=True)
        final_df = final_df.drop_duplicates(
            subset=['Timestamp', 'Transaction Type', 'Asset', 'Quantity Transacted', 'Fees', 'Subtotal']).copy()
        if raw:
            return final_df

        global first_use
        if first_use:
            log.info(f'Coinbase transactions last updated on {max(final_df.index).date()}')
            first_use = False

        final_df['From'] = ''
        final_df['To'] = ''
        final_df['Fee Currency'] = 'EUR'
        final_df['To Coin'] = ''
        final_df['To Amount'] = ''

        final_df.rename(columns={'Transaction Type': 'Tag', 'Asset': 'Coin', 'Quantity Transacted': 'Amount',
                                 'Spot Price at Transaction': 'Fiat Price', 'Spot Price Currency': 'Fiat',
                                 'Fees': 'Fee'}, inplace=True)

        final_df.loc[final_df['Tag'] == 'Convert', 'Amount'] *= -1
        final_df.loc[final_df['Tag'] == 'Send', 'Amount'] *= -1
        final_df.loc[final_df['Tag'] == 'Sell', 'Amount'] *= -1

        final_df.loc[final_df['Tag'] == 'Convert', 'To Coin'] = [i.split(" ")[-1] for i in list(
            final_df.loc[final_df['Tag'] == 'Convert', 'Notes'])]
        final_df.loc[final_df['Tag'] == 'Convert', 'To Amount'] = [
            float(i.split(" ")[-2].replace(".", "").replace(",", ".")) for i in
            list(final_df.loc[
                     final_df['Tag'] == 'Convert', 'Notes'])]

        final_df.loc[np.logical_or(final_df['Tag'] == 'Sell', final_df['Tag'] == 'Buy'), 'To Coin'] = [i.split(" ")[-1]
                                                                                                       for i in
                                                                                                       list(
                                                                                                           final_df.loc[
                                                                                                               np.logical_or(
                                                                                                                   final_df[
                                                                                                                       'Tag'] == 'Sell',
                                                                                                                   final_df[
                                                                                                                       'Tag'] == 'Buy'), 'Notes'])]
        final_df.loc[final_df['Tag'] == 'Sell', 'To Amount'] = [-float(i.split(" ")[-3].replace(",", ".")) for i in
                                                                list(final_df.loc[final_df['Tag'] == 'Sell', 'Notes'])]

        final_df.loc[final_df['Tag'] == 'Buy', 'To Amount'] = [float(i.split(" ")[-3].replace(",", ".")) for i in
                                                               list(final_df.loc[final_df['Tag'] == 'Buy', 'Notes'])]

        final_df.loc[final_df['Tag'] == 'Send', 'To'] = [i.split(" ")[-1] for i in
                                                         list(final_df.loc[final_df['Tag'] == 'Send', 'Notes'])]
        final_df.loc[final_df['Tag'] == 'Receive', 'From'] = [i.split(" ")[-1] for i in
                                                              list(final_df.loc[final_df['Tag'] == 'Receive', 'Notes'])]

        final_df.fillna(0, inplace=True)
        final_df.loc[np.logical_or(final_df['Tag'] == 'Coinbase Earn',
                                   final_df['From'] == 'COIN'), 'Tag'] = 'Reward'
        final_df.loc[final_df['Tag'] != 'Reward', 'Tag'] = 'Movement'

        final_df.drop(['Timestamp', 'Subtotal', 'Total (inclusive of fees)', 'Notes'], axis=1,
                      inplace=True)

        final_df = final_df.reindex(
            columns=['From', 'To', 'Coin', 'Amount', 'To Coin', 'To Amount', 'Fiat Price',
                     'Fiat', 'Fee', 'Fee Currency', 'Tag'])

        final_df.loc[np.logical_and(final_df['From'] == '', final_df['Amount'] < 0), 'From'] = 'Coinbase'
        final_df['Fiat Price'] = final_df['Fiat Price'] * final_df['Amount']
        final_df['Tag Account'] = 'Coinbase'

        tokens = final_df['Coin'].tolist()
        tokens.extend(final_df['To Coin'].tolist())
        tokens = [x.upper() for x in list(set(tokens)) if
                  x not in ut.fiat_list]

        global coinbase_prices
        if first_use:
            was_updated = update_prices(coinbase_prices, tokens)

            if 'EUR' not in list(coinbase_prices.exchange_rates.keys()) or was_updated:
                coinbase_prices.convert_prices('EUR', tokens)
        return final_df


def calculate_all(year_sel=None, name='Coinbase'):
    transactions = get_transactions_df(False)
    transactions_raw = get_transactions_df(True)
    balances_in = ut.balances(transactions=transactions, year_sel=year_sel)
    balaces_fiat_in = ut.balances_fiat(balances=balances_in, prices=coinbase_prices, year_sel=year_sel)
    soglia_in = ut.soglia(balances=balances_in, prices=coinbase_prices, year_sel=year_sel)
    income_in = ut.income(transactions=transactions, type_out='crypto', name=name, year_sel=year_sel)
    income_in_fiat = ut.income(transactions=transactions, name=name, year_sel=year_sel)
    vout = {"transactions": transactions, "transactions_raw": transactions_raw, "balances": balances_in,
            "balaces_fiat": balaces_fiat_in, "soglia": soglia_in,
            "income": income_in, "income_fiat": income_in_fiat}
    return vout
