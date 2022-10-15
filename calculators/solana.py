from utility.PricesClass import Prices, update_prices
import datetime as dt
import pandas as pd
import requests
from utility.tax_log import log
import utility.utils as ut

log.info('Solana calculator - updated 15/10/2022')

solana_prices = Prices()


def get_transactions_df(address):
    transfers = requests.get(
        f'https://public-api-test.solscan.io/account/solTransfers?account={address}&limit=10&offset=0')
    transfers = pd.DataFrame(transfers.json())
    From, To, Amount, timestamp, Fee = [], [], [], [], []
    vout = pd.DataFrame()
    for transaction in transfers['data']:
        From.append(transaction['src'])
        To.append(transaction['dst'])
        if transaction['src'] == address:
            Amount.append(-int(transaction['lamport']) / 10 ** 9)
            Fee.append(0.00005)
        else:
            Amount.append(int(transaction['lamport']) / 10 ** 9)
            Fee.append(0)
        timestamp.append(dt.datetime.fromtimestamp(transaction['blockTime']))

    vout['From'] = From
    vout['To'] = To
    vout['Amount'] = Amount
    vout['Coin'] = 'SOL'
    vout['To Amount'] = ''
    vout['To Coin'] = ''
    vout['Fee'] = Fee
    vout['Fee Currency'] = 'SOL'
    vout['Tag'] = 'Movement'
    vout['Tag Account'] = 'Solana'
    vout['Fiat Price'] = ''
    vout['Fiat Currency'] = 'EUR'
    vout.index = timestamp

    global solana_prices

    tokens = vout['Coin'].tolist()
    tokens.extend(vout['To Coin'].tolist())
    tokens = [x.upper() for x in list(set(tokens)) if x not in ut.fiat_list]

    was_updated = update_prices(solana_prices, tokens)

    if 'EUR' not in list(solana_prices.exchange_rates.keys()) or was_updated:
        solana_prices.convert_prices('EUR', tokens)

    vout.sort_index(inplace=True)
    price = solana_prices.to_pd_dataframe('EUR')
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

    vout.loc[vout['To'] == address, 'Fee'] = 0
    vout['Tag Account'] = 'Solana'

    return vout


def calculate_all(address: str, year_sel=None, name='Solana'):
    transactions = get_transactions_df(address=address)
    balances_in = ut.balances(transactions=transactions, year_sel=year_sel)
    balaces_fiat_in = ut.balances_fiat(balances=balances_in, prices=solana_prices, year_sel=year_sel)
    soglia_in = ut.soglia(balances_in=balances_in, prices=solana_prices, year_sel=year_sel)
    income_in = ut.income(transactions=transactions, type_out='crypto', name=name, year_sel=year_sel)
    income_in_fiat = ut.income(transactions=transactions, name=name, year_sel=year_sel)
    vout = {"transactions": transactions, "transactions_raw": transactions, "balances": balances_in,
            "balaces_fiat": balaces_fiat_in, "soglia": soglia_in,
            "income": income_in, "income_fiat": income_in_fiat}
    return vout
