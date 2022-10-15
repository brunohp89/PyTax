import os
from utility.PricesClass import Prices, update_prices
import numpy as np
import pandas as pd
import utility.tax_library as tx
import datetime as dt
import utility.utils as ut
from utility.utils import log

log.info('Binance calculator - updated on 15/10/2022')

first_use = True
bin_prices = Prices()


# ACQUISTI FATTI CON LA CARTA DI CREDITO DEVONO ESSERE INSERITE MANUALMENTE NELLO STORICO


def auto_invest(binance_auto):
    df_list = []
    for filename in binance_auto:
        df_loop = pd.read_csv(filename, index_col=None, header=0)
        df_list.append(df_loop)
    final_df = pd.concat(df_list, axis=0, ignore_index=True)

    final_df['Auto-Invest Date(UTC)'] = [
        dt.datetime.strptime(j, '%Y-%m-%d %H:%M:%S') + dt.timedelta(milliseconds=x + 10)
        for x, j in enumerate(list(final_df['Auto-Invest Date(UTC)']))]

    final_df.drop_duplicates(inplace=True,
                             subset=['Auto-Invest Date(UTC)', 'Holding Coin', 'Amount per period', 'Units'])

    final_df = final_df[final_df['Status'] == 'Success']

    final_df['Trading Fee'] = final_df['Trading Fee'].map(lambda x: float(x.split(' ')[0]))

    final_df['Coin1'] = final_df['Amount per period'].map(lambda x: x.split(' ')[1])
    final_df['Change1'] = final_df['Amount per period'].map(lambda x: float(x.split(' ')[0]))
    final_df['Change1'] += final_df['Trading Fee']
    final_df['Change1'] *= -1

    final_df['Coin2'] = final_df['Units'].map(lambda x: x.split(' ')[1])
    final_df['Change2'] = final_df['Units'].map(lambda x: float(x.split(' ')[0]))

    final_df.drop(['Holding Coin', 'Amount per period', 'Units', 'From', 'Status'], axis=1, inplace=True)

    final_df['User_ID'] = 99999
    final_df['Account'] = 'Recurring'
    final_df['Operation'] = 'Buy'

    df1 = final_df[['User_ID', 'Auto-Invest Date(UTC)', 'Account', 'Operation', 'Coin1', 'Change1']].copy()
    df2 = final_df[['User_ID', 'Auto-Invest Date(UTC)', 'Account', 'Operation', 'Coin2', 'Change2']].copy()

    df1.rename(columns={'Auto-Invest Date(UTC)': 'UTC_Time', 'Coin1': 'Coin', 'Change1': 'Change'}, inplace=True)
    df2.rename(columns={'Auto-Invest Date(UTC)': 'UTC_Time', 'Coin2': 'Coin', 'Change2': 'Change'}, inplace=True)

    vout = pd.concat([df1, df2])
    vout['Remark'] = ""
    vout.index = vout['UTC_Time']

    return vout


def get_transactions_df(raw=False):
    binance_files = [os.getcwd() + '\\binance' + '\\' + x for x in os.listdir(os.getcwd() + '\\binance') if
                     'automatico' not in x]
    binance_auto = [os.getcwd() + '\\binance' + '\\' + x for x in os.listdir(os.getcwd() + '\\binance') if
                    'automatico' in x]

    if len(binance_files) == 0:
        log.info('No files for binance found')
        return None
    else:
        df_list = []
        for filename in binance_files:
            df_loop = pd.read_csv(filename, index_col=None, header=0)
            df_list.append(df_loop)
        final_df = pd.concat(df_list, axis=0, ignore_index=True)
        final_df.index = [tx.str_to_datetime(j) for j in final_df['UTC_Time']]

        final_df.drop_duplicates(inplace=True,
                                 subset=['UTC_Time', 'Operation', 'Coin', 'Change'])

        if len(binance_auto) > 0:
            recurring_df = auto_invest(binance_auto)
            final_df = pd.concat([final_df, recurring_df])

        final_df.sort_index(inplace=True)
        final_df.loc[
            np.logical_and(final_df.index < pd.to_datetime('2022-08-01'), final_df['Coin'] == 'LUNA'), 'Coin'] = 'LUNC'
        if raw:
            return final_df

        global first_use
        if first_use:
            log.info(f'Binance transactions last updated on {max(final_df.index).date()}')

        to_exclude = ['ETH 2.0 Staking',
                      'Launchpad subscribe',
                      'POS savings purchase',
                      'POS savings redemption',
                      'Savings Principal redemption',
                      'Savings purchase']

        final_df = final_df[~final_df['Operation'].isin(to_exclude)].copy()

        final_df.loc[final_df['Coin'] == 'BETH', 'Coin'] = 'ETH'
        final_df.loc[final_df['Coin'] == 'LDBNB', 'Coin'] = 'BNB'

        final_df['From'] = ''
        final_df['To'] = ''
        final_df['Fee'] = 0
        final_df['Fee Currency'] = ''
        final_df['To Coin'] = ''
        final_df['To Amount'] = ''
        final_df.rename(columns={'Change': 'Amount'}, inplace=True)
        final_df.drop(['User_ID', 'Account', 'UTC_Time', 'Remark'], axis=1, inplace=True)

        final_df_new = final_df[~final_df.index.duplicated(keep='first')].copy()

        for ind in np.unique(final_df.index[final_df.index.duplicated(keep='first')]):
            temp = pd.DataFrame(final_df.loc[[ind], :])
            if len(temp.loc[temp['Amount'] < 0, 'Amount']) != 0:
                coins = temp.loc[np.logical_and(temp['Amount'] < 0, temp['Operation'] != 'Fee'), 'Coin'].tolist()
                if len(coins) > len(np.unique(coins)):
                    to_concat = temp.iloc[[0], :].copy()
                    to_concat['Coin'] = \
                        temp.loc[np.logical_and(temp['Amount'] < 0, temp['Operation'] != 'Fee'), 'Coin'].tolist()[0]
                    to_concat['To Coin'] = temp.loc[temp['Amount'] > 0, 'Coin'].tolist()[0]
                    to_concat['Operation'] = temp.loc[temp['Amount'] > 0, 'Operation'].tolist()[0]
                    to_concat['To Amount'] = sum(
                        temp.loc[np.logical_and(temp['Amount'] > 0, temp['Operation'] != 'Fee'), 'Amount'].tolist())
                    to_concat['Amount'] = sum(
                        temp.loc[np.logical_and(temp['Amount'] < 0, temp['Operation'] != 'Fee'), 'Amount'].tolist())

                    fee = temp.loc[np.logical_and(temp['Amount'] < 0, temp['Operation'] == 'Fee'), 'Amount'].tolist()
                    if len(fee) > 0:
                        to_concat['Fee'] = sum(fee)
                        to_concat['Fee Currency'] = \
                            temp.loc[np.logical_and(temp['Amount'] < 0, temp['Operation'] == 'Fee'), 'Coin'].tolist()[0]
                    final_df_new.loc[[ind], :] = to_concat
                    continue

                to_concat = temp.iloc[[k for k in range(len(np.unique(
                    temp.loc[np.logical_and(temp['Amount'] < 0, temp['Operation'] != 'Fee'), 'Coin'].tolist())))],
                            :].copy()

                to_concat['Coin'] = \
                    temp.loc[np.logical_and(temp['Amount'] < 0, temp['Operation'] != 'Fee'), 'Coin'].tolist()

                if len(np.unique(temp.loc[temp['Amount'] > 0, 'Coin'].tolist())) == 1:
                    to_concat['To Coin'] = temp.loc[temp['Amount'] > 0, 'Coin'].tolist()[0]
                    to_concat['Operation'] = temp.loc[temp['Amount'] > 0, 'Operation'].tolist()[0]
                    to_concat['To Amount'] = sum(
                        temp.loc[np.logical_and(temp['Amount'] > 0, temp['Operation'] != 'Fee'), 'Amount'].tolist())
                    to_concat['Amount'] = sum(
                        temp.loc[np.logical_and(temp['Amount'] < 0, temp['Operation'] != 'Fee'), 'Amount'].tolist())
                else:
                    to_concat['To Coin'] = temp.loc[temp['Amount'] > 0, 'Coin'].tolist()
                    to_concat['Operation'] = temp.loc[temp['Amount'] > 0, 'Operation'].tolist()
                    to_concat['To Amount'] = temp.loc[
                        np.logical_and(temp['Amount'] > 0, temp['Operation'] != 'Fee'), 'Amount'].tolist()
                    to_concat['Amount'] = temp.loc[
                        np.logical_and(temp['Amount'] < 0, temp['Operation'] != 'Fee'), 'Amount'].tolist()

                fee = temp.loc[np.logical_and(temp['Amount'] < 0, temp['Operation'] == 'Fee'), 'Amount'].tolist()
                if len(fee) > 0:
                    to_concat['Fee'] = sum(fee)
                    to_concat['Fee Currency'] = \
                        temp.loc[np.logical_and(temp['Amount'] < 0, temp['Operation'] == 'Fee'), 'Coin'].tolist()[0]

                if np.unique(to_concat['Operation'])[0] == 'Small assets exchange BNB':
                    if to_concat.shape[0] > 1:
                        to_concat.iloc[1:, -1] = 0
                    if len(fee) == 0:
                        to_concat.iloc[1:, 5] = -to_concat.iloc[0, -1] * 0.02

                if to_concat.shape[0] > 1:
                    final_df_new.loc[[ind], :] = to_concat.iloc[[0], :]
                    for j in range(1, to_concat.shape[0]):
                        to_concat.iloc[[j], :].index = to_concat.iloc[[j], :].index + dt.timedelta(milliseconds=j)
                        final_df_new = pd.concat([final_df_new, to_concat.iloc[[j], :]], axis=0)
                else:
                    final_df_new.loc[[ind], :] = to_concat

        final_df_new['Tag'] = ''
        final_df_new['Operation'] = final_df_new['Operation'].map(lambda x: x.lower())
        final_df_new.loc[final_df_new['Operation'].str.contains("distribution|interest"), 'Tag'] = 'Interest'
        final_df_new.loc[final_df_new['Operation'].str.contains("mining"), 'Tag'] = 'Reward'
        final_df_new.loc[final_df_new['Tag'] == '', 'Tag'] = 'Movement'

        tokens = final_df_new['Coin'].tolist()
        tokens.extend(final_df_new['To Coin'].tolist())
        tokens = [x.upper() for x in list(set(tokens)) if
                  x not in ["AUD", "BRL", "EUR", "GBP", "GHS", "HKD", "KES", "KZT", "NGN", "NOK", "PHP", "PEN", "RUB",
                            "TRY", "UGX",
                            "UAH", ""]]

        global bin_prices

        if first_use:
            was_updated = update_prices(bin_prices, tokens)

            if 'EUR' not in list(bin_prices.exchange_rates.keys()) or was_updated:
                bin_prices.get_exchange_rates('EUR')
                bin_prices.convert_prices('EUR', tokens)

        final_df_new['Fiat Price'] = 0
        final_df_new['Fiat'] = 'EUR'

        final_df_new.sort_index(inplace=True)
        price = bin_prices.to_pd_dataframe('EUR')
        price = price[~price.index.duplicated(keep='first')]
        for tok in set(final_df_new['Coin']):
            if tok == 'EUR':
                temp_df = final_df_new[final_df_new['Coin'] == tok].copy()
                temp_df.loc[temp_df['Coin'] == 'EUR', 'Fiat Price'] = temp_df.loc[temp_df['Coin'] == 'EUR', 'Amount']
                final_df_new.loc[final_df_new['Coin'] == tok, 'Fiat Price'] = temp_df['Fiat Price']
                continue
            temp_df = final_df_new[final_df_new['Coin'] == tok].copy()
            temp_df.index = [k.date() for k in temp_df.index]
            temp_df = temp_df.join(pd.DataFrame(price[tok]))
            temp_df['Fiat Price'] = temp_df['Amount'] * temp_df[tok]
            temp_df.loc[temp_df['To Coin'] == 'EUR', 'Fiat Price'] = temp_df.loc[
                temp_df['To Coin'] == 'EUR', 'To Amount']
            temp_df.index = final_df_new[final_df_new['Coin'] == tok].index
            final_df_new.loc[final_df_new['Coin'] == tok, 'Fiat Price'] = temp_df['Fiat Price']

        final_df_new = final_df_new.reindex(
            columns=['From', 'To', 'Coin', 'Amount', 'To Coin', 'To Amount', 'Fiat Price',
                     'Fiat', 'Fee', 'Fee Currency', 'Tag'])
        final_df_new.loc[np.logical_and(final_df_new['From'] == '', final_df_new['Amount'] < 0), 'From'] = 'Binance'
        final_df_new['Tag Account'] = 'Binance'

        if first_use:
            first_use = False
        return final_df_new


def calculate_all(year_sel=None, name='Binance'):
    transactions = get_transactions_df(False)
    transactions_raw = get_transactions_df(True)
    balances_in = ut.balances(transactions=transactions, year_sel=year_sel)
    balaces_fiat_in = ut.balances_fiat(balances=balances_in, prices=bin_prices, year_sel=year_sel)
    soglia_in = ut.soglia(balances=balances_in, prices=bin_prices, year_sel=year_sel)
    income_in = ut.income(transactions=transactions, type_out='crypto', name=name, year_sel=year_sel)
    income_in_fiat = ut.income(transactions=transactions, name=name, year_sel=year_sel)
    vout = {"transactions": transactions, "transactions_raw": transactions_raw, "balances": balances_in,
            "balaces_fiat": balaces_fiat_in, "soglia": soglia_in,
            "income": income_in, "income_fiat": income_in_fiat}
    return vout