import GetOldTweets3 as got
def get_tweets (): 
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch('europe refugees')\
                                           .setSince("2015-05-01")\
                                           .setUntil("2015-09-30")\
                                           .setMaxTweets(1)
    tweet = got.manager.TweetManager.getTweets(tweetCriteria)[0]
    print(tweet.text)

get_tweets()