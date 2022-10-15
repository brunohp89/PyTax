import numpy as np
import pandas as pd
import datetime as dt
import requests
from utility.utils import log

def str_to_datetime(date: str):
    try:
        if len(date) > 11:
            new_date = dt.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        else:
            new_date = dt.datetime.strptime(date, '%Y-%m-%d')

        return new_date

    except ValueError:
        log.info("Invalid format. Allowed formats are: YYYY-MM-DD and YYYY-MM-DD HH:MM:SS")


def datetime_to_str(date, hour_output=True):
    if hour_output:
        s = date.strftime('%Y-%m-%d %H:%M:%S')
    else:
        s = date.strftime('%Y-%m-%d')
    return str(s)


def to_binance_date_format(date_to_convert):
    month_out = None
    if date_to_convert.month == 1:
        month_out = 'Jan'
    elif date_to_convert.month == 2:
        month_out = 'Feb'
    elif date_to_convert.month == 3:
        month_out = 'Mar'
    elif date_to_convert.month == 4:
        month_out = 'Apr'
    elif date_to_convert.month == 5:
        month_out = 'May'
    elif date_to_convert.month == 6:
        month_out = 'Jun'
    elif date_to_convert.month == 7:
        month_out = 'Jul'
    elif date_to_convert.month == 8:
        month_out = 'Aug'
    elif date_to_convert.month == 9:
        month_out = 'Sep'
    elif date_to_convert.month == 10:
        month_out = 'Oct'
    elif date_to_convert.month == 11:
        month_out = 'Nov'
    elif date_to_convert.month == 12:
        month_out = 'Dec'
    return f'{date_to_convert.day} {month_out}, {date_to_convert.year}'


def uphold_date_to_datetime(date):
    month_out = None
    if date[4:7] == 'Jan':
        month_out = 1
    elif date[4:7] == 'Feb':
        month_out = 2
    elif date[4:7] == 'Mar':
        month_out = 3
    elif date[4:7] == 'Apr':
        month_out = 4
    elif date[4:7] == 'May':
        month_out = 5
    elif date[4:7] == 'Jun':
        month_out = 6
    elif date[4:7] == 'Jul':
        month_out = 7
    elif date[4:7] == 'Aug':
        month_out = 8
    elif date[4:7] == 'Sep':
        month_out = 9
    elif date[4:7] == 'Oct':
        month_out = 10
    elif date[4:7] == 'Nov':
        month_out = 11
    elif date[4:7] == 'Dec':
        month_out = 12
    date_out = dt.datetime(int(date[11:15]), month_out, int(date[8:10]), int(date[16:18]), int(date[19:21]),
                           int(date[22:24]))
    return date_out


def turn_age_into_dt(date_input):
    if 'days' in date_input:
        days = int(date_input.split(' days ')[0])
        date_input = date_input.split(' days ')[1]
    else:
        days = 0
    if 'hrs' in date_input:
        hours = int(date_input.split(' hrs ')[0])
        date_input = date_input.split(' hrs ')[1]
    else:
        hours = 0
    if 'minutes' in date_input:
        minutes = int(date_input.split(' minutes ')[0])
        date_input = date_input.split(' minutes ')[1]
    else:
        minutes = 0
    if 'secs' in date_input:
        seconds = int(date_input.split(' secs ')[0])
    else:
        seconds = 0

    return dt.datetime.now() - dt.timedelta(days=days, hours=hours+1, minutes=minutes, seconds=seconds)


def get_bnb(address):
    # FOR ADDRESSES STARTING WITH BNB (BINANCE CHAIN)
    output_bnb = requests.get(f"https://explorer.bnbchain.world/txs?address={address}")
    output_bnb = pd.read_html(output_bnb.text)
    output_bnb = output_bnb[0]

    output_bnb.index = [turn_age_into_dt(k) for k in output_bnb['Age']]

    output_bnb['Fee'] = -0.00075
    output_bnb['Value'] = [float(k.replace(' BNB', '')) for k in output_bnb['Value']]

    output_bnb['Fiat Price'] = 0
    output_bnb['Fiat'] = 'EUR'
    output_bnb['Fee Currency'] = 'BNB'
    output_bnb['To Coin'] = ''
    output_bnb['Coin'] = 'BNB'
    output_bnb['To Amount'] = ''
    output_bnb['Tag Account'] = 'BNB Beacon Chain'

    output_bnb.rename(columns={'Value': 'Amount', 'Type': 'Tag'}, inplace=True)

    output_bnb.drop(['TxHash', 'Height', 'Age', 'Unnamed: 5'], axis=1, inplace=True)

    output_bnb.loc[output_bnb['From'] == address, 'Amount'] *= -1

    output_bnb['Tag'] = 'Movement'

    output_bnb = output_bnb.reindex(
        columns=['From', 'To', 'Coin', 'Amount', 'To Coin', 'To Amount', 'Fiat Price',
                 'Fiat', 'Fee', 'Fee Currency', 'Tag'])
    output_bnb.sort_index(inplace=True)

    return output_bnb


def get_fiat_investment(transactions_df, currency='EUR', cummulative=True, year_sel='all', **credit_card_transactions):
    if credit_card_transactions is not None:
        ind_extra = []
        data_extra = []
        for cct in credit_card_transactions:
            cct1 = cct.replace("dt_", "")
            cct1 = cct1.replace("_", "-")
            ind_extra.append(str_to_datetime(cct1))
            data_extra.append(-credit_card_transactions[cct])
        FiatExtra = pd.DataFrame(pd.Series(data=data_extra, index=ind_extra, dtype=float))
        FiatExtra.index = [p.date() for p in FiatExtra.index]
        FiatExtra.columns = ['Amount']

    in_fiat = transactions_df[np.logical_and(np.logical_or(transactions_df['Coin'] == currency, transactions_df['To Coin'] == currency), transactions_df['To Coin'] != '')]
    NewAmount = in_fiat.loc[in_fiat['Coin'] != currency, 'To Amount'].tolist()
    in_fiat.loc[in_fiat['Coin'] != currency,'Amount'] = NewAmount

    if in_fiat[in_fiat['Amount'] > 0].shape[0] > 0 and in_fiat[in_fiat['Amount'] < 0].shape[0] > 0:
        in_fiat = in_fiat[in_fiat['Amount'] > 0]
        in_fiat['Amount'] *= -1
    if in_fiat.shape[0] > 0:
        in_fiat.index = [p.date() for p in in_fiat.index]
        if credit_card_transactions is not None:
            in_fiat = pd.concat([in_fiat, FiatExtra], axis=0)
        in_fiat = in_fiat.groupby(in_fiat.index).sum()
        in_fiat = pd.DataFrame(-in_fiat['Amount'])

        if year_sel != 'all':
            in_fiat = in_fiat[in_fiat.index >= dt.date(year_sel, 1, 1)]
            in_fiat = in_fiat[in_fiat.index <= dt.date(year_sel, 12, 31)]
        if cummulative:
            in_fiat = in_fiat.cumsum()
    else:
        if credit_card_transactions is not None:
            in_fiat = -FiatExtra.copy()
            if year_sel != 'all':
                in_fiat = in_fiat[in_fiat.index >= dt.date(year_sel, 1, 1)]
                in_fiat = in_fiat[in_fiat.index <= dt.date(year_sel, 12, 31)]
            if cummulative:
                in_fiat = in_fiat.cumsum()
    if in_fiat.shape[0] == 0:
        if year_sel == 'all':
            year_sel = 2021
        ind_out = pd.date_range(dt.date(year_sel, 1, 1),
                                dt.datetime.today().date() - dt.timedelta(days=1), freq='d')
        data = [0] * len(ind_out)
        in_fiat = pd.DataFrame(pd.Series(data=data, index=ind_out))
    in_fiat.columns = [currency]
    in_fiat.sort_index(inplace=True)

    return in_fiat


def write_excel(file_name, **sheets):
    excel_writer = pd.ExcelWriter(file_name, engine="xlsxwriter")
    for sheet in sheets:
        sheets[sheet].to_excel(excel_writer, sheet_name=sheet)

    excel_writer.close()
    log.info(f'Excel file output - {file_name}')


def calcolo_giacenza_media(df):
    return df.sum(axis=1)[df.sum(axis=1) != 0].sum(axis=0) / df.sum(axis=1)[df.sum(axis=1) != 0].shape[0]


def join_dfs(**df_to_join):
    vout = pd.DataFrame()
    for df in df_to_join:
        if df_to_join[df].shape[0] == 0:
            continue
        df_to_join[df].columns = [p.upper() for p in df_to_join[df].columns]
        if vout.shape[0] == 0:
            vout = df_to_join[df].copy()
        else:
            vout = vout.join(df_to_join[df], rsuffix='--R').copy()
        vout.iloc[0, :].fillna(0, inplace=True)
        vout.ffill(inplace=True)

    vout.columns = [l.replace("--R", "") for l in vout.columns]
    vout = vout.groupby(by=vout.columns, axis=1).sum()
    return vout

def concat_dfs(**df_to_concat):
    vout = pd.DataFrame()
    for df in df_to_concat:
        if df_to_concat[df].shape[0] == 0:
            continue
        if vout.shape[0] == 0:
            vout = df_to_concat[df].copy()
        else:
            if vout.shape[1] != df_to_concat[df].shape[1]:
                log.info(f'{df} --> PROBLEMI DI DIMENSIONE')
            vout = pd.concat([vout, df_to_concat[df]], axis=0)
    vout.sort_index(inplace=True)
    return vout

def soglia_superata(df):
    xts = df.copy()
    xts['Superata'] = [1 if x > 51645.69 else 0 for x in list(xts.iloc[:, 0])]
    num_days = 0
    days_tot = []
    for i in list(xts['Superata']):
        if i == 0:
            num_days = 0
        else:
            num_days += 1
        days_tot.append(num_days)
    if max(days_tot) >= 7:
        return 'SI'
    else:
        return 'NO'


def get_primo_ultimo_giorno(df, tax_year):
    if dt.datetime.today().date() <= dt.date(tax_year, 12, 31):
        ultimo_giorno = 'Non disponibile'
    else:
        ultimo_giorno = round(df.sum(axis=1)[-1], 2)
    primo_giorno = round(df.sum(axis=1)[0], 2)
    return [primo_giorno, ultimo_giorno]
