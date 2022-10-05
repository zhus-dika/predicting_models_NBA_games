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

def parse_totals(year):
    """Parses basic league information for a given year: parse_basic(year) -> DataFrame"""
    url = 'https://www.basketball-reference.com/leagues/NBA_{}_totals.html'.format(year)
    # this is the html from the given url
    html = urlopen(url)
    soup = BeautifulSoup(html)
    type(soup)
    soup.findAll('tr', limit=2)
    column_headers = [th.getText() for th in soup.findAll('tr', limit=2)[0].findAll('th')]
    column_headers = column_headers[1:]
    data_rows = soup.findAll('tr')[2:]
    type(data_rows)
    player_data = [[td.getText() for td in data_rows[i].findAll('td')]
                for i in range(len(data_rows))]

    totals = pd.DataFrame(player_data, columns=column_headers)
    return totals

def parse_advanced(year):
    """Parses advanced league information for a given year: parse_advanced(year) -> DataFrame"""
    url = "https://www.basketball-reference.com/leagues/NBA_{}_advanced.html".format(year)
    # this is the html from the given url
    html = urlopen(url)
    soup = BeautifulSoup(html)
    type(soup)
    soup.findAll('tr', limit=2)
    column_headers = [th.getText() for th in soup.findAll('tr', limit=2)[0].findAll('th')]
    column_headers = column_headers[1:]
    data_rows = soup.findAll('tr')[2:]
    type(data_rows)
    player_data = [[td.getText() for td in data_rows[i].findAll('td')]
                for i in range(len(data_rows))]

    advanced = pd.DataFrame(player_data, columns=column_headers)
    return advanced

if __name__ == "__main__":
    YEAR = parse_arguments(sys.argv)

    # download totals
    totals = parse_totals(YEAR)

    # clean-up data frame
    totals = totals[totals['Player'].notnull()]
    totals = totals._convert(numeric = True)
    totals = totals[:].fillna(0)
    totals = totals.drop_duplicates(['Player'], keep='first')

    # download advanced
    advanced = parse_advanced(YEAR)

    # clean-up data frame
    advanced = advanced[advanced['Player'].notnull()]
    advanced = advanced._convert(numeric=True)
    advanced = advanced[:].fillna(0)
    advanced = advanced.drop_duplicates(['Player'], keep='first')

    # drop empty columns that are used for visual separation on the web-site
    advanced = advanced.drop(advanced.columns[[23, 18]],axis = 1)

    cols_to_use = advanced.columns.difference(totals.columns)
    cols_to_use = cols_to_use.append(pd.Index(['Player']))

    stats = pd.merge(totals, advanced[cols_to_use], on='Player')
    stats.to_csv('../../data/{}_advanced_plus_totals.csv'.format(YEAR), index=False)
