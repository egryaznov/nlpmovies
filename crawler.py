from lxml import html
from math import ceil
import re
import time
import json
import requests


def get_last_review_id():
    url = "https://kino.otzyv.ru/last.php"
    page = requests.get(url)
    tree = html.fromstring(page.content)
    last_ids = tree.xpath('//a[@itemprop="url"]/@href')
    return int(re.search('(\d+)', last_ids[0]).group(0))


def scrap(initial_review=1, reviews_dict=None, review_count=1, all=False,
            update=False):
    """
        Parses and returns the review dictionary of the following format:
            {string ID: (string TEXT, int RATING)}
        For example, if you run:
            dict = scrap()
            print(dict['1'])
        It will print you the very first review together with it's rating.
        Keyword arguments:
        initial_review -- first review to parse with.
        reviews_dict -- updates provided dictionary with more reviews.
        review_count -- how many reviews to scrap.
        all -- if True, then all available reviews will be scrapped starting
            from the `start_review_id`, disregarding `review_count`.
        update -- if True and `reviews_dict` != None, then sets initial_review
        to index of the last entry in `reviews_dict` + 1.
    """
    data = {} if reviews_dict is None else reviews_dict
    first_review = initial_review
    if (reviews_dict is not None) and (update):
        first_review = len(reviews_dict) + 1
        print('Updating provided dictionary...')
    last_review = get_last_review_id() if all else review_count + first_review
    count = last_review - first_review
    ten_percent_of_count = ceil(count / 10)
    print('Start scraping %d reviews with id from %d to %d.' % (count, first_review, last_review))
    elapsed = 0
# Gathering the data...
    for i in range(first_review, last_review + 1):
        start        = time.time()  # start of scrapping review in seconds
        try:
            url      = 'https://kino.otzyv.ru/read.php?id={}'.format(i)
        except Exception as e:
            print(e)
            time.sleep(5)
            start += 5
        page         = requests.get(url)
        tree         = html.fromstring(page.content)
        text         = tree.xpath('//p[@class="otzyv"]/text()')
        grade        = tree.xpath('//meta[@itemprop="ratingValue"]/@content')
        review_text  = text[0] if len(text) > 0 else 'NULL'
        review_grade = grade[0] if len(grade) > 0 else -1
        data[str(i)] = review_text, review_grade
    # Status stuff...
        t        = time.time() - start
        velocity = 1 / t  # reviews per second
        elapsed += t
        if (i % ten_percent_of_count == 0):
            prc          = round(100 * ((i - first_review) / count))
            remained     = (count - (i - first_review)) / velocity  # in seconds
            progress_bar = '[%s%s]' % ('=' * prc, ' ' * (100 - prc))
            print('+%d reviews, up next: %d' % (ten_percent_of_count, i))
            print('Speed: %0.2f reviews/second; Elapsed: %0.2f m; Remained: %0.2f m' % (velocity, elapsed / 60, remained / 60))
            print('Status: %s %d%%\n' % (progress_bar, prc))
# Saving cooked dictionary...
    print('Scraping of %d reviews completed in %d min (%d seconds).' % (i, elapsed / 60, elapsed))
    return data


def update(dict):
    return scrap(reviews_dict=dict, all=True, update=True)


def save(data, filename):
    with open(filename, 'w') as fp:
        json.dump(data, fp)
        fp.flush()


def load(filename):
    with open(filename, 'r') as fp:
        result = json.load(fp)
    return result


russian_filename = 'russian-reviews.json'
