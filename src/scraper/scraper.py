from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import sys
import getopt

def parse_arguments(argv):
    arg_year = ''
    arg_help = '{0} -y <input>'.format(argv[0])

    try:
        opts, args = getopt.getopt(argv[1:], "hy:", ["help", "year="])
    except:
        print(arg_help)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # print the help message
            sys.exit(2)
        elif opt in ("-y", "--year"):
            arg_year = arg

    print('year:', arg_year)

    return int(arg_year)

def fetch_data(year, advanced = False):
    """Parses league information for a given year: fetch_data(year) -> DataFrame"""
    url = "https://www.basketball-reference.com/leagues/NBA_{}_totals.html".format(year)

    if advanced:
        url = "https://www.basketball-reference.com/leagues/NBA_{}_advanced.html".format(year)

    # this is the html from the given url
    html = urlopen(url)
    soup = BeautifulSoup(html)

    soup.findAll('tr', limit=2)
    column_headers = [th.getText() for th in soup.findAll('tr', limit=2)[0].findAll('th')]
    column_headers = column_headers[1:]
    data_rows = soup.findAll('tr')[2:]

    player_data = [[td.getText() for td in data_rows[i].findAll('td')]
                for i in range(len(data_rows))]

    data_frame = pd.DataFrame(player_data, columns=column_headers)

    data_frame = clean_data_frame(data_frame)
    return data_frame

def fetch_salaries(year):
    """Parses salary information for a given year: fetch_salaries(year) -> DataFrame"""

    url = "https://hoopshype.com/salaries/players/{}-{}/".format(year, year + 1)

    html = urlopen(url)
    soup = BeautifulSoup(html)

    salaries_table = soup.find_all("table", {"class": "hh-salaries-ranking-table hh-salaries-table-sortable responsive"})

    if len(salaries_table) == 0:
        print('No salaries table found')
        return
    else:
        salaries_table = salaries_table[0]

    salaries_table_body = salaries_table.find_all('tbody')

    if len(salaries_table_body) == 0:
        print('Salaries table body is empty')
        return
    else:
        salaries_table_body = salaries_table_body[0]

    data_rows = salaries_table_body.findAll('tr')

    player_data = [[td.getText().strip() for td in data_rows[i].findAll('td')]
                for i in range(len(data_rows))]

    data_frame = pd.DataFrame(player_data, columns=['Rank', 'Player', 'Salary', 'SalaryAdj'])
    data_frame = data_frame.drop(data_frame.columns[[0]],axis = 1)
    data_frame[data_frame.columns[1:]] = data_frame[data_frame.columns[1:]].replace('[\$,]', '', regex=True).astype(float)

    data_frame = clean_data_frame(data_frame)
    return data_frame

def clean_data_frame(data_frame):
    data_frame = data_frame[data_frame['Player'].notnull()]
    data_frame = data_frame._convert(numeric = True)
    data_frame = data_frame[:].fillna(0)
    data_frame = data_frame.drop_duplicates(['Player'], keep='first')

    return data_frame

if __name__ == "__main__":
    YEAR = parse_arguments(sys.argv)

    # download salaries
    salaries = fetch_salaries(YEAR)

    # download totals
    totals = fetch_data(YEAR)

    # download advanced
    advanced = fetch_data(YEAR, True)

    # drop empty columns that are used for visual separation on the web-site
    advanced = advanced.drop(advanced.columns[[23, 18]],axis = 1)

    cols_to_use = advanced.columns.difference(totals.columns)
    cols_to_use = cols_to_use.append(pd.Index(['Player']))

    stats = pd.merge(totals, advanced[cols_to_use], on='Player')
    stats = pd.merge(stats, salaries, on='Player')

    stats.to_csv('../../data/{}_advanced_plus_totals.csv'.format(YEAR), index=False)
