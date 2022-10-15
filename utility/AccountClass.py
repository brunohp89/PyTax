import pandas as pd
import cryptodotcom
from old import terra, tron as trx, cryptodotcom_exchange as cdce
import ethereum as eth
import iotex as iotx
import tax_library as tx
import polkadot as dot
from calculators import bitcoin as btc, binancetax, coinbase, uphold, binance_beacon_bsc as bsc, vechain as vet, \
    cardano as cardano, cosmos as cm, cronos as cro
import near
import solana_tax as solx
import polygon as pg

class Account:
    def __init__(self, account, tax_year, address=None, beacon_address=None, address_list=None, stake_list=None, still_in_use=False):
        self.cashback_fiat = None
        self.cashback_cro = None
        if account.lower() in ["cryptodotcom", "cdc", "crypto.com", "crypto"]:
            self.transactions = cryptodotcom.get_transactions_df()
            self.transactions_raw = cryptodotcom.get_transactions_df(True)
            self.crypto_balance = cryptodotcom.get_balances(year_sel=tax_year)
            self.fiat_balance = cryptodotcom.get_balances_fiat(year_sel=tax_year)
            self.soglia = cryptodotcom.get_soglia(year_sel=tax_year)
            self.fiat_income = cryptodotcom.get_income(year_sel=tax_year)
            self.crypto_income = cryptodotcom.get_income(year_sel=tax_year, type_out='crypto')
            self.cashback_fiat = cryptodotcom.get_cashback(year_sel=tax_year)
            self.cashback_cro = cryptodotcom.get_cashback(year_sel=tax_year, type_out='cro')
            self.investment = tx.get_fiat_investment(self.transactions, year_sel=tax_year)
        elif account.lower() in ["binance", "bin"]:
            self.transactions = binancetax.get_transactions_df()
            self.transactions_raw = binancetax.get_transactions_df(True)
            self.crypto_balance = binancetax.get_balances(year_sel=tax_year)
            self.fiat_balance = binancetax.get_balances_fiat(year_sel=tax_year)
            self.soglia = binancetax.get_soglia(year_sel=tax_year)
            self.fiat_income = binancetax.get_income(year_sel=tax_year)
            self.crypto_income = binancetax.get_income(year_sel=tax_year, type_out='crypto')
            self.investment = tx.get_fiat_investment(self.transactions, year_sel=tax_year)
        elif account.lower() in ["coin", "coinbase"]:
            self.transactions = coinbase.get_transactions_df()
            self.transactions_raw = coinbase.get_transactions_df(True)
            self.crypto_balance = coinbase.get_balances(year_sel=tax_year)
            self.fiat_balance = coinbase.get_balances_fiat(year_sel=tax_year)
            self.soglia = coinbase.get_soglia(year_sel=tax_year)
            self.fiat_income = coinbase.get_income(year_sel=tax_year)
            self.crypto_income = coinbase.get_income(year_sel=tax_year, type_out='crypto')
            self.investment = tx.get_fiat_investment(self.transactions, year_sel=tax_year)
        elif account.lower() in ["uphold", "up"]:
            self.transactions = uphold.get_transactions_df()
            self.transactions_raw = uphold.get_transactions_df(True)
            self.crypto_balance = uphold.get_balances(year_sel=tax_year)
            self.fiat_balance = uphold.get_balances_fiat(year_sel=tax_year)
            self.soglia = uphold.get_soglia(year_sel=tax_year)
            self.fiat_income = uphold.get_income(year_sel=tax_year)
            self.crypto_income = uphold.get_income(year_sel=tax_year, type_out='crypto')
            self.investment = tx.get_fiat_investment(self.transactions, year_sel=tax_year)
        elif account.lower() in ["eth", "ethereum"]:
            self.transactions = eth.get_transactions_df(address)
            self.transactions_raw = self.transactions
            self.crypto_balance = eth.get_balances(address, year_sel=tax_year)
            self.fiat_balance = eth.get_balances_fiat(address, year_sel=tax_year)
            self.soglia = eth.get_soglia(address, year_sel=tax_year)
            self.fiat_income = eth.get_income(address, year_sel=tax_year)
            self.crypto_income = eth.get_income(address, year_sel=tax_year, type_out='crypto')
            self.investment = tx.get_fiat_investment(self.transactions, year_sel=tax_year)
        elif account.lower() in ["terra", "luna", "terrastation", "terra_station"]:
            self.transactions = terra.get_transactions_df(address)
            self.transactions_raw = self.transactions
            self.crypto_balance = terra.get_balances(address, year_sel=tax_year)
            self.fiat_balance = terra.get_balances_fiat(address, year_sel=tax_year)
            self.soglia = terra.get_soglia(address, year_sel=tax_year)
            self.fiat_income = terra.get_income(address, year_sel=tax_year)
            self.crypto_income = terra.get_income(address, year_sel=tax_year, type_out='crypto')
            self.investment = tx.get_fiat_investment(self.transactions, year_sel=tax_year)
        elif account.lower() in ["iotx", "iotex", "io", "tex"]:
            self.transactions = iotx.get_transactions_df()
            self.transactions_raw = iotx.get_transactions_df(True)
            self.crypto_balance = iotx.get_balances(year_sel=tax_year)
            self.fiat_balance = iotx.get_balances_fiat(year_sel=tax_year)
            self.soglia = iotx.get_soglia(year_sel=tax_year)
            self.fiat_income = iotx.get_income(year_sel=tax_year)
            self.crypto_income = iotx.get_income(year_sel=tax_year, type_out='crypto')
            self.investment = tx.get_fiat_investment(self.transactions, year_sel=tax_year)
        elif account.lower() in ["cardano", "ada", "card", "yoroi"]:
            self.transactions = cardano.get_transactions_and_staking(address_list, stake_list, still_in_use)
            self.transactions_raw = self.transactions
            self.crypto_balance = cardano.get_balances(self.transactions, year_sel=tax_year)
            self.fiat_balance = cardano.get_balances_fiat(self.transactions, year_sel=tax_year)
            self.soglia = cardano.get_soglia(self.fiat_balance, year_sel=tax_year)
            self.fiat_income = cardano.get_income(self.transactions, year_sel=tax_year)
            self.crypto_income = cardano.get_income(self.transactions, year_sel=tax_year, type_out='crypto')
            self.investment = tx.get_fiat_investment(self.transactions, year_sel=tax_year)
        elif account.lower() in ["tron", "trx", "tronlink", "tron_link"]:
            self.transactions = trx.get_transactions_df(address)
            self.transactions_raw = self.transactions
            self.crypto_balance = trx.get_balances(self.transactions, year_sel=tax_year)
            self.fiat_balance = trx.get_balances_fiat(self.transactions, year_sel=tax_year)
            self.soglia = trx.get_soglia(self.transactions, year_sel=tax_year)
            self.fiat_income = trx.get_income(address, self.transactions, year_sel=tax_year)
            self.crypto_income = trx.get_income(address, self.transactions, year_sel=tax_year, type_out='crypto')
            self.investment = tx.get_fiat_investment(self.transactions, year_sel=tax_year)
        elif account.lower() in ["binance_chain", "bsc", "smart_chain", "smartchain", "binancechain"]:
            self.transactions = bsc.get_transactions_df(address, beacon_address)
            self.transactions_raw = self.transactions
            self.crypto_balance = bsc.get_balances(address, beacon_address, year_sel=tax_year)
            self.fiat_balance = bsc.get_balances_fiat(address, beacon_address, year_sel=tax_year)
            self.soglia = bsc.get_soglia(address, beacon_address, year_sel=tax_year)
            self.fiat_income = bsc.get_income(address, beacon_address, year_sel=tax_year)
            self.crypto_income = bsc.get_income(address, beacon_address, year_sel=tax_year, type_out='crypto')
            self.investment = tx.get_fiat_investment(self.transactions, year_sel=tax_year)
        elif account.lower() in ["cronos", "cro", "crypto.org", "cryptodotorg"]:
            self.transactions = cro.get_transactions_df(address, beacon_address)
            self.transactions_raw = self.transactions
            self.crypto_balance = cro.get_balances(address, beacon_address, year_sel=tax_year)
            self.fiat_balance = cro.get_balances_fiat(address, beacon_address, year_sel=tax_year)
            self.soglia = cro.get_soglia(address, beacon_address, year_sel=tax_year)
            self.fiat_income = cro.get_income(address, beacon_address, year_sel=tax_year)
            self.crypto_income = cro.get_income(address, beacon_address, year_sel=tax_year, type_out='crypto')
            self.investment = tx.get_fiat_investment(self.transactions, year_sel=tax_year)
        elif account.lower() in ["polkadot", "dot", "polka"]:
            self.transactions = dot.get_transactions_df(address)
            self.transactions_raw = self.transactions
            self.crypto_balance = dot.get_balances(address, year_sel=tax_year)
            self.fiat_balance = dot.get_balances_fiat(address, year_sel=tax_year)
            self.soglia = dot.get_soglia(address, year_sel=tax_year)
            self.fiat_income = dot.get_income(address, year_sel=tax_year)
            self.crypto_income = dot.get_income(address, year_sel=tax_year, type_out='crypto')
            self.investment = tx.get_fiat_investment(self.transactions, year_sel=tax_year)
        elif account.lower() in ["bit", "btc", "bitcoin"]:
            self.transactions = btc.get_transactions_df(address_list)
            self.transactions_raw = self.transactions
            self.crypto_balance = btc.get_balances(address_list, year_sel=tax_year)
            self.fiat_balance = btc.get_balances_fiat(address_list, year_sel=tax_year)
            self.soglia = btc.get_soglia(address_list, year_sel=tax_year)
            self.fiat_income = btc.get_income(address_list, year_sel=tax_year)
            self.crypto_income = btc.get_income(address_list, year_sel=tax_year, type_out='crypto')
            self.investment = tx.get_fiat_investment(self.transactions, year_sel=tax_year)
        elif account.lower() in ["vechain", "ve", "vet"]:
            self.transactions = vet.get_transactions_df(address)
            self.transactions_raw = vet.get_transactions_df(address, raw=True)
            self.crypto_balance = vet.get_balances(address, year_sel=tax_year)
            self.fiat_balance = vet.get_balances_fiat(address, year_sel=tax_year)
            self.soglia = vet.get_soglia(address, year_sel=tax_year)
            self.fiat_income = vet.get_income(address, year_sel=tax_year)
            self.crypto_income = vet.get_income(address, year_sel=tax_year, type_out='crypto')
            self.investment = tx.get_fiat_investment(self.transactions, year_sel=tax_year)
        elif account.lower() in ["cdce", "cryprodoccomexchange", "cdc_exchange", "cdcexchange"]:
            self.transactions = cdce.get_transactions_df()
            self.transactions_raw = cdce.get_transactions_df(raw=True)
            self.crypto_balance = cdce.get_balances(year_sel=tax_year)
            self.fiat_balance = cdce.get_balances_fiat(year_sel=tax_year)
            self.soglia = cdce.get_soglia(year_sel=tax_year)
            self.fiat_income = cdce.get_income(year_sel=tax_year)
            self.crypto_income = cdce.get_income( year_sel=tax_year, type_out='crypto')
            self.investment = tx.get_fiat_investment(self.transactions, year_sel=tax_year)
        elif account.lower() in ["cosmos", "Cosmos", "atom", "Atom"]:
            self.transactions = cm.get_transactions_df(address)
            self.transactions_raw = cm.get_transactions_df(address)
            self.crypto_balance = cm.get_balances(address,year_sel=tax_year)
            self.fiat_balance = cm.get_balances_fiat(address,year_sel=tax_year)
            self.soglia = cm.get_soglia(address,year_sel=tax_year)
            self.fiat_income = cm.get_income(address,year_sel=tax_year)
            self.crypto_income = cm.get_income(address, year_sel=tax_year, type_out='crypto')
            self.investment = tx.get_fiat_investment(self.transactions, year_sel=tax_year)
        elif account.lower() in ["sol", "Solana", "SOL", "Sol", "solana"]:
            self.transactions = solx.get_transactions_df(address)
            self.transactions_raw = solx.get_transactions_df(address)
            self.crypto_balance = solx.get_balances(address, year_sel=tax_year)
            self.fiat_balance = solx.get_balances_fiat(address, year_sel=tax_year)
            self.soglia = solx.get_soglia(address, year_sel=tax_year)
            self.fiat_income = solx.get_income(address, year_sel=tax_year)
            self.crypto_income = solx.get_income(address, year_sel=tax_year, type_out='crypto')
            self.investment = tx.get_fiat_investment(self.transactions, year_sel=tax_year)
        elif account.lower() in ["near", "NEAR", "Near"]:
            self.transactions = near.get_transactions_df(address_list)
            self.transactions_raw = near.get_transactions_df(address_list)
            self.crypto_balance = near.get_balances(address_list, year_sel=tax_year)
            self.fiat_balance = near.get_balances_fiat(address_list, year_sel=tax_year)
            self.soglia = near.get_soglia(address_list, year_sel=tax_year)
            self.fiat_income = near.get_income(address_list, year_sel=tax_year)
            self.crypto_income = near.get_income(address_list, year_sel=tax_year, type_out='crypto')
            self.investment = tx.get_fiat_investment(self.transactions, year_sel=tax_year)
        elif account.lower() in ["polygon", "MATIC", "Polygon"]:
            self.transactions = pg.get_transactions_df(address)
            self.transactions_raw = pg.get_transactions_df(address)
            self.crypto_balance = pg.get_balances(address, year_sel=tax_year)
            self.fiat_balance = pg.get_balances_fiat(address, year_sel=tax_year)
            self.soglia = pg.get_soglia(address, year_sel=tax_year)
            self.fiat_income = pg.get_income(address, year_sel=tax_year)
            self.crypto_income = pg.get_income(address, year_sel=tax_year, type_out='crypto')
            self.investment = tx.get_fiat_investment(self.transactions, year_sel=tax_year)
        else:
            print("Wrong Account Name")
        self.soglia = pd.DataFrame(self.soglia.sum(axis=1))
        self.soglia.columns = [account]

    def write_file(self, name, path):
        if self.cashback_fiat is not None:
            tx.write_excel(f"{path}{name}.xlsx", Transazioni=self.transactions_raw, Saldo=self.crypto_balance,
                           Saldo_EUR=self.fiat_balance, Interessi=self.crypto_income, Interessi_EUR=self.fiat_income,
                           Cashback_EUR=self.cashback_fiat, Cashback_CRO=self.cashback_cro, Soglia=self.soglia)
        else:
            tx.write_excel(f"{path}{name}.xlsx", Transazioni=self.transactions_raw, Saldo=self.crypto_balance,
                           Saldo_EUR=self.fiat_balance, Interessi=self.crypto_income, Interessi_EUR=self.fiat_income,
                           Soglia=self.soglia)