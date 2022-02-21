from data_fetcher import DataFetcher
import pandas as pd
from textblob import TextBlob
from datetime import datetime
from datetime import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sb
import praw
from psaw import PushshiftAPI


###### get and stack stock data


def get_10_year_data(ticker):
    start_date = datetime(2010, 1, 1)
    end_date = datetime(2021, 1, 1)
    data = yf.download(ticker, start=start_date,
                       end=end_date)
    # plt.figure(figsize=(20, 8))
    # plt.title('Opening Prices OF {} from {} to {}'.format(ticker, start_date,
    #                                                       end_date))
    # plt.plot(data['Open'])
    # plt.show()
    return data


dis_data = get_10_year_data('DIS')
pep_data = get_10_year_data('PEP')
aal_data = get_10_year_data('AAL')
sap_data = get_10_year_data('^GSPC')

dis_data['ticker'] = 'DIS'
pep_data['ticker'] = 'PEP'
aal_data['ticker'] = 'AAL'
sap_data['ticker'] = 'S&P 500'

stock_data = sap_data.copy()
print(len(stock_data))

print(len(pep_data))
stock_data = stock_data.append(pep_data)
print(len(stock_data))

print(len(dis_data))
stock_data = stock_data.append(dis_data)
print(len(stock_data))

print(len(aal_data))
stock_data = stock_data.append(aal_data)
print(len(stock_data))

stock_data.to_csv('stock_data.csv')


############# combining twitter data

def add_csv(ticker):
    df = pd.read_csv(f'tweets/{ticker}.csv')
    print(len(df))
    df = df.append(pd.read_csv(f'tweets/{ticker}2.csv'))
    print(len(df))
    df = df.append(pd.read_csv(f'tweets/{ticker}3.csv'))
    print(len(df))
    df.to_csv(f'{ticker}_tweets.csv')
    return df


aal_tweets = add_csv('aal')
pep_tweets = add_csv('pepsi')
dis_tweets = add_csv('dis')

aal_tweets = aal_tweets.drop_duplicates(keep=False, inplace=True)
pep_tweets = pep_tweets.drop_duplicates(keep=False, inplace=True)
dis_tweets = dis_tweets.drop_duplicates(keep=False, inplace=True)


################ sentiment analysis twitter


def sentiment_analysis(ticker_df, ticker):
    assessment = ticker_df['sentence'].astype(str).apply(lambda x: TextBlob(x).sentiment_assessments)
    ticker_df['sentiment_polarity'] = assessment.apply(lambda x: x[0])
    ticker_df['sentiment_subjectivity'] = assessment.apply(lambda x: x[1])
    ticker_df['sentiment_assessment'] = assessment.apply(lambda x: x[2])
    ticker_df_sent = ticker_df[
        ['ticker', 'created_at', 'username', 'sentence', 'sentiment_polarity', 'sentiment_subjectivity',
         'sentiment_assessment']]
    ticker_df_sent = ticker_df_sent[
        (ticker_df_sent['sentiment_polarity'] != 0) & (ticker_df_sent['sentiment_subjectivity'] != 0)]
    return ticker_df_sent


aal_tweets = pd.read_csv('aal_tweets.csv')
aal_sent = sentiment_analysis(aal_tweets, 'aal')
pep_tweets = pd.read_csv('pepsi_tweets.csv')
pep_sent = sentiment_analysis(pep_tweets, 'pep')
dis_tweets = pd.read_csv('dis_tweets.csv')
dis_sent = sentiment_analysis(dis_tweets, 'dis')


def show_stats(df):
    df = df.groupby('created_at').describe()


################ sentiment analysis reddit

ticker = 'dis'

dis_reddit = pd.read_csv('Disney_reddit.csv')
dis_reddit = dis_reddit.append(pd.read_csv('Disney_reddit2.csv'))
dis_reddit['created_at'] = dis_reddit['created_utc'].apply(lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%d'))
dis_reddit.to_csv('dis_reddit.csv', index=None)

pep_reddit = pd.read_csv('pep_reddit.csv')
pep_reddit['created_at'] = pep_reddit['created_utc'].apply(lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%d'))
pep_reddit.to_csv('pep_reddit.csv', index=None)


def sentiment_analysis_reddit(ticker_df, ticker):
    assessment = ticker_df['title'].astype(str).apply(lambda x: TextBlob(x).sentiment_assessments)
    ticker_df['sentiment_polarity'] = assessment.apply(lambda x: x[0])
    ticker_df['sentiment_subjectivity'] = assessment.apply(lambda x: x[1])
    ticker_df['sentiment_assessment'] = assessment.apply(lambda x: x[2])

    ticker_df_sent = ticker_df[
        ['created_at', 'title', 'sentiment_polarity', 'sentiment_subjectivity', 'sentiment_assessment']]
    ticker_df_sent = ticker_df_sent[ticker_df_sent['language'] == 'en']
    ticker_df_sent = ticker_df_sent[
        (ticker_df_sent['sentiment_polarity'] != 0) & (ticker_df_sent['sentiment_subjectivity'] != 0)]
    ticker_df_sent.to_csv(f'{ticker}_sent_reddit_filtered.csv', index=None)

    return ticker_df_sent


pep_reddit_sent = sentiment_analysis_reddit(pep_reddit, 'pep')
dis_reddit_sent = sentiment_analysis_reddit(dis_reddit, 'dis')

################  stacking twitter and reddit data

ticker = 'aal'
df = pd.read_csv(f'{ticker}_sent.csv')
df[['created_at', 'Time', "Timezone"]] = df.created_at.str.split(expand=True)
aal_sent = df[
    ['created_at', 'tweet', 'sentiment_polarity', 'sentiment_subjectivity', 'sentiment_assessment']]

dis_sent = dis_sent.rename(columns={"tweet": "sentence"})
dis_sent['ticker'] = 'DIS'

pep_sent = pep_sent.rename(columns={"tweet": "sentence"})
pep_sent['ticker'] = 'PEP'

aal_sent1 = aal_sent.rename(columns={"tweet": "sentence"})
aal_sent1['ticker'] = 'AAL'

dis_reddit_sent = pd.read_csv('dis_sent_reddit.csv')
dis_reddit_sent = dis_reddit_sent.rename(columns={"title": "sentence"})
dis_reddit_sent['ticker'] = 'DIS'

pep_reddit_sent = pd.read_csv('pep_sent_reddit.csv')
pep_reddit_sent = pep_reddit_sent.rename(columns={"title": "sentence"})
pep_reddit_sent['ticker'] = 'PEP'

sentiment_data = aal_sent1.copy()
print(len(sentiment_data))

print(len(pep_sent))
sentiment_data = sentiment_data.append(pep_sent)
print(len(sentiment_data))

print(len(pep_reddit_sent))
sentiment_data_ = sentiment_data.append(pep_reddit_sent)
print(len(sentiment_data_))

print(len(dis_sent))
sentiment_data = sentiment_data_.append(dis_sent)
print(len(sentiment_data))

print(len(dis_reddit_sent))
sentiment_data = sentiment_data_.append(dis_reddit_sent)
print(len(sentiment_data))

sentiment_data_.to_csv('sentiment_data_all.csv')

############# analysing tweets

t = aal_tweets.sort_values(by=['replies_count', 'retweets_count', 'likes_count'], ascending=False)


############  filtered tweets sentiment


def username_filter_twitter(ticker_df, ticker):
    ticker_df = ticker_df[ticker_df['language'] == 'en']
    ticker_df = ticker_df[ticker_df['username'].isin(
        ["business", "wsj", "theeconomist", "forbes", 'reuters', 'businessinsider', 'ft', 'cnbc', 'markets',
         'yahoofinance', 'nytimes', 'harvardbiz', 'reutersbiz', 'nasdaq', 'dowjones', 'nyse', 'foxbusiness'])]
    ticker_df[['created_at', 'Time', "Timezone"]] = ticker_df.created_at.str.split(expand=True)
    ticker_df = ticker_df.rename(columns={"tweet": "sentence"})
    ticker_df['ticker'] = ticker.upper()
    ticker_df.to_csv(f'filtered_tweets/{ticker}_filtered_tweets.csv', index=None)
    ticker_df_sent = sentiment_analysis(ticker_df, ticker)
    ticker_df_sent.to_csv(f'{ticker}_sent_filtered.csv', index=None)
    return ticker_df_sent


aal_tweets = pd.read_csv('aal_tweets.csv')
aal_sent_filtered = username_filter_twitter(aal_tweets, 'aal')
pep_tweets = pd.read_csv('pepsi_tweets.csv')
pep_sent_filtered = username_filter_twitter(pep_tweets, 'pep')
dis_tweets = pd.read_csv('dis_tweets.csv')
dis_sent_filtered = username_filter_twitter(dis_tweets, 'dis')


############  filtered reddit sentiment


def username_filter_reddit(ticker_df, ticker):
    ticker_df = ticker_df.rename(columns={"title": "sentence", "author": "username"})
    # ticker_df = ticker_df[ticker_df['username'].isin(
    #     ["business", "wsj", "theeconomist", "forbes", 'reuters', 'businessinsider', 'ft', 'cnbc', 'markets',
    #      'yahoofinance', 'nytimes', 'harvardbiz', 'reutersbiz', 'nasdaq', 'dowjones', 'nyse', 'foxbusiness'])]
    ticker_df.to_csv(f'filtered_reddits/{ticker}_filtered_reddits.csv', index=None)
    ticker_df['ticker'] = ticker.upper()
    ticker_df_sent = sentiment_analysis(ticker_df, ticker)
    return ticker_df_sent


dis_reddit = pd.read_csv('dis_reddit.csv')
dis_reddit_sent_filtered = username_filter_reddit(dis_reddit, 'dis')
pep_reddit = pd.read_csv('pep_reddit.csv')
pep_reddit_sent_filtered = username_filter_reddit(pep_reddit, 'pep')

############# stack filtered data
###### twitter

sentiment_data = aal_sent_filtered.copy()
print(len(sentiment_data))

print(len(pep_sent_filtered))
sentiment_data = sentiment_data.append(pep_sent_filtered)
print(len(sentiment_data))

print(len(dis_sent_filtered))
sentiment_data = sentiment_data_.append(dis_sent_filtered)
print(len(sentiment_data))

sentiment_data.to_csv('sentiment_data_all_filtered_twitter.csv', index=None)

######  reddit

print(len(pep_reddit_sent_filtered))
sentiment_data = sentiment_data.append(pep_reddit_sent_filtered)
print(len(sentiment_data))

print(len(dis_reddit_sent_filtered))
sentiment_data = sentiment_data_.append(dis_reddit_sent_filtered)
print(len(sentiment_data))

sentiment_data.to_csv('sentiment_data_all_filtered_twitterreddit.csv', index=None)

