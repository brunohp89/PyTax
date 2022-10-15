from utility.PricesClass import Prices, update_prices
import pandas as pd
import requests
from utility import utils as ut, tax_library as tx
from utility.utils import log

log.info('Bitcoin calculator - updated on 15/10/2022')

btc_prices = None


def get_transactions_df(address_list):
    vout = pd.DataFrame()
    for address in address_list:
        url = f'https://api.blockchair.com/bitcoin/dashboards/address/{address}?transaction_details=true'
        response = requests.get(url)
        if vout.shape[0] == 0:
            vout = pd.DataFrame(response.json()['data'][address]['transactions'])
        else:
            vout = pd.concat([pd.DataFrame(response.json()['data'][address]['transactions']), vout], axis=0)

    vout.index = vout['time'].map(lambda x: tx.str_to_datetime(x))

    vout['balance_change'] /= 10 ** 8

    vout.rename(columns={'balance_change': 'Amount'}, inplace=True)
    vout.drop(['block_id', 'hash', 'time'], axis=1, inplace=True)

    vout['Fee'] = 0
    vout['Fee Currency'] = 'BTC'
    vout['Fiat Price'] = 0
    vout['Fiat'] = 'EUR'
    vout['Fee Currency'] = 'BTC'
    vout['To Coin'] = ''
    vout['Coin'] = 'BTC'
    vout['To Amount'] = ''
    vout['Tag'] = 'Movement'
    vout['From'] = ''
    vout['To'] = ','.join(address_list)

    vout = vout.reindex(
        columns=['From', 'To', 'Coin', 'Amount', 'To Coin', 'To Amount', 'Fiat Price',
                 'Fiat', 'Fee', 'Fee Currency', 'Tag'])

    vout.sort_index(inplace=True)
    vout['Amount'].fillna(0, inplace=True)
    vout['From'].fillna('', inplace=True)
    vout['To'].fillna('', inplace=True)

    vout.loc[vout['From'].isin(address_list), 'Amount'] *= -1

    global btc_prices
    btc_prices = Prices()

    tokens = vout['Coin'].tolist()
    tokens.extend(vout['To Coin'].tolist())
    tokens = [x.upper() for x in list(set(tokens)) if
              x not in ["AUD", "BRL", "EUR", "GBP", "GHS", "HKD", "KES", "KZT", "NGN", "NOK", "PHP", "PEN", "RUB",
                        "TRY", "UGX",
                        "UAH", ""]]

    update_prices(btc_prices, tokens)

    btc_prices.convert_prices('EUR', tokens)

    vout.sort_index(inplace=True)
    price = btc_prices.to_pd_dataframe('EUR')
    price = price[~price.index.duplicated(keep='first')]
    for tok in tokens:
        temp_df = vout[vout['Coin'] == tok].copy()
        temp_df.index = [k.date() for k in temp_df.index]

        temp_df = temp_df.join(pd.DataFrame(price[tok]))
        temp_df['Fiat Price'] = temp_df['Amount'] * temp_df[tok]
        temp_df.loc[temp_df['To Coin'] == 'EUR', 'Fiat Price'] = temp_df.loc[
            temp_df['To Coin'] == 'EUR', 'To Amount']
        temp_df.index = vout[vout['Coin'] == tok].index
        vout.loc[vout['Coin'] == tok, 'Fiat Price'] = temp_df['Fiat Price']

    vout.loc[vout['To'].isin(address_list), 'Fee'] = 0
    vout['Tag Account'] = 'Bitcoin'
    return vout


def calculate_all(address_list, year_sel=None, name='BTC'):
    transactions = get_transactions_df(address_list=address_list)
    balances_in = ut.balances(transactions=transactions, year_sel=year_sel)
    balaces_fiat_in = ut.balances_fiat(balances=balances_in, prices=btc_prices, year_sel=year_sel)
    soglia_in = ut.soglia(balances=balances_in, prices=btc_prices, year_sel=year_sel)
    income_in = ut.income(transactions=transactions, type_out='crypto', name=name, year_sel=year_sel)
    income_in_fiat = ut.income(transactions=transactions, name=name, year_sel=year_sel)
    vout = {"transactions": transactions, "transactions_raw": transactions, "balances": balances_in,
            "balaces_fiat": balaces_fiat_in, "soglia": soglia_in,
            "income": income_in, "income_fiat": income_in_fiat}
    return vout
