from lxml import html
from math import ceil
from operator import itemgetter
import matplotlib.pyplot as plt
import nltk.stem.snowball as snowball
import re
import time
import json
import requests
import random


def get_last_review_id():
    """ Scans the web-page and returns the ID (number) of a last submitted review."""
    url = "https://kino.otzyv.ru/last.php"
    page = requests.get(url)
    tree = html.fromstring(page.content)
    last_ids = tree.xpath('//a[@itemprop="url"]/@href')
    return int(re.search('(\d+)', last_ids[0]).group(0))


def crawl(initial_review=1, reviews_dict=None, review_count=1, everything=False,
            update_dict=False):
    """
        Parses and returns the review dictionary of the following format:
            {string ID: (string TEXT, int RATING)}
        For example, if you run:
            dict = crawl()
            print(dict['1'])
        It will print you the very first review together with it's rating.
        Keyword arguments:
        initial_review -- first review to parse with.
        reviews_dict -- updates provided dictionary with more reviews.
        review_count -- how many reviews to scrap.
        everything -- if True, then all available reviews will be scrapped starting
            from the `start_review_id`, disregarding `review_count`.
        update -- if True and `reviews_dict` != None, then sets initial_review
        to index of the last entry in `reviews_dict` + 1.
    """
    data = {} if reviews_dict is None else reviews_dict
    first_review = initial_review
    if reviews_dict is not None and update_dict:
        first_review = len(reviews_dict) + 1
        print('Updating provided dictionary...')
    last_review = get_last_review_id() if everything else review_count + first_review
    count = last_review - first_review
    if count < 1:
        print('Dataset is up to date.')
        return reviews_dict
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
        if review_text is None:
            continue
    # Status building...
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


def update(d):
    """ Updates the dataset.
        Keyword arguments:
            dict -- the review dictionary to be updated.
        Returns:
            Updated dictionary.
        Usage case:
            When we need a way to quickly update the dataset, but we don't want
            to scrutinize arguments of the `crawl` method.
    """
    return crawl(reviews_dict=d, everything=True, update_dict=True)


def save(d, filename):
    """ Saves the dictionary to .json file `filename`.
        Keyword arguments:
            dict -- a dictionary to save.
            filename -- a string name of a file to which the dictionary will be
                saved.
    """
    with open(filename, 'w') as fp:
        json.dump(d, fp)
        fp.flush()


def load(filename):
    """ Loads the dictionary from the .json file
        Keyword arguments:
            filename -- a name of the file from which a dictionary will be
            loaded.
        Returns:
            The loaded dictionary.
    """
    with open(filename, 'r') as fp:
        result = json.load(fp)
    return result


def rating_distribution(data_dict):
    """ Collects and plots the distribution of the rating marks in the provided
        dataset.
        Keyword arguments:
            data_dict -- A dataset dictionary that is used to collect the
            distribution.
    """
    buckets = [[] for i in range(1, 12)]
    for id, text_rating_pair in data_dict.items():
        text, rating = text_rating_pair
        if rating == -1:
            b_index = 0
        else:
            b_index = int(rating)
        buckets[b_index].append(text)
    data = [len(b) for b in buckets]
    plt.bar(range(0, 11), data)
    plt.xlabel("Rating")
    plt.ylabel("# of Reviews")
    plt.title("Rating Distribution")
    plt.show()


def length_distribution(data_dict):
    """ Collects and plots the distribution of the revies length depentding on
    its rating in the provided dataset.
        Keyword arguments:
            data_dict -- A dataset dictionary that is used to collect the
            distribution.
    """
    buckets = [[] for i in range(1, 12)]
    for id, text_rating_pair in data_dict.items():
        text, rating = text_rating_pair
        if rating == -1:
            b_index = 0
        else:
            b_index = int(rating)
        buckets[b_index].append(text)
    data = [sum([len(t) for t in b]) / len(b) for b in buckets]
    plt.bar(range(0, 11), data)
    plt.xlabel("Rating")
    plt.ylabel('Average length of review')
    plt.title("Length Distribution")
    plt.show()


def clean(data_dict):
    """ Removes all unmarked reviews from the dataset.
        Keyword arguments:
            data_dict -- the dictionary dataset.
        Returns:
            Cleaned dataset.
    """
    unassessed_entries = [key for key, value in data_dict.items() if value[1] == -1]
    for key in unassessed_entries:
        del data_dict[key]
    return data_dict


def tokenize_review(text):
    """ Removes all junky symbols and returns lowercased stemmed tokens of `text`
        Keyword arguments:
            text -- a raw (unprocessed) string of a review.
        Returns:
            Lowercased stemmed tokens of `text`.
    """
    # Delete all useless punctuation signs
    text = re.sub('[^\w\d ]', ' ', text)
    # Only single space is allowed between words
    text = text.replace('  ', ' ').lower()
    stemmed_tokens = []
    ru_stemmer = snowball.RussianStemmer(ignore_stopwords=True)
    for token in text.split(' '):
        if len(token) == 0:
            continue
        stemmed_tokens.append(ru_stemmer.stem(token))
    return stemmed_tokens


def reviews_to_samples(data_dict, vocab, min_review_length=3, topn=20000):
    """ Splits a dictionary dataset into two tuples: tokenized reviews and ratings.
        Also skips all reviews with middle rating (5) and tokenizes each.
        Keyword arguments:
            data_dict -- a dictionary dataset.
            vocab -- TODO
        Returns:
            Two tuples: tokenized reviews and ratings.
    """
    x0 = []
    y0 = []
    review_index = 0
    digitized_dict = {}
    for key, value in data_dict.items():
        text, rating = value
        if rating == 5:  # Skip reviews with rating 5
            continue
        tokens = digitize(text, vocab, topn)
        if len(tokens) >= min_review_length:
            binary_rating = polarize(rating)
            digitized_dict[key] = [tokens, binary_rating]
            x0.append(tokens)
            y0.append(binary_rating)
        review_index += 1
    save(digitized_dict, 'cooked-dataset.json')
    return x0, y0


def polarize(rating, bad_end=4):
    """ Transforms a rating mark from [1, 2, ..., 10] into binary: positive (1)
    or negative (0).
        Keyword arguments:
            rating -- a mark from the set {1, 2, ..., 10}.
            bad_end=4 -- specifies the bad marks = {1, 2, ..., bad_end}.
        Returns:
            0 or 1 depending on the `rating`.
    """
    return 0 if int(rating) <= bad_end else 1


def build_vocabulary(dataset_filename='ru-reviews.json'):
    """ Constructs a vocabulary of the reviews corpora.
        Vocabulary has the following format: {STRING stemmed_word_token: INT
                frequency_index}.
        For example, the most frequent word in russian language is 'и':
            vocab = build_vocabulary()
            print(vocab['и'])
        Will print 0.
        Returns:
            Constructed vocabulary.
    """
    data_dict = load(dataset_filename)
    vocab = {}  # Format: {STRING word: INT frequency}
    for key, value in data_dict.items():
        review, rating = value
        for token in tokenize_review(review):
            if token in vocab:
                vocab[token] += 1
            else:
                vocab[token] = 1
# Sort vocabulary by frequency:
    sorted_key_value_pairs = sorted(vocab.items(), key=itemgetter(1), reverse=True)
# Set each value in vocabulary to its index in sorted_key_value_pairs:
    index = 0
    for pair in sorted_key_value_pairs:
        word, freq = pair
        vocab[word] = index
        index += 1
# Save vocabulary
    save(vocab, 'ru-vocab.json')
    return vocab


def digitize(review_text, vocab, topn=20000):
    """ Returns tokenized version of review where each token is replaced by
    its frequency rating. For example, if the 3 most frequent words
    are: 'Я' 'тебя' 'люблю', then if you run:
        t = 'Тебя люблю я'
        digitize(t, vocab)
    It would print:
        [1, 2, 0]
    This function is used as the first step in preparing raw text into
    appropriate form for LSTM model.
    Keyword arguments:
        review_text -- the string which contains the actual review in russian
            language.
        vocab -- a vocabulary used to identify the index of a token. Can be
            obtained by `build_vocabulary` function.
        topn -- if the index of a token is bigger than this value, then it will
        be discarded.
    Return:
        A tuple of natural numbers, representing the tokens.
    """
    digitized_review = []
    for token in tokenize_review(review_text):
        if (token in vocab) and (vocab[token] < topn):
            digitized_review.append(vocab[token])
    return digitized_review


def cook_data(dataset_filename='cooked-dataset.json',
                reviews_filename='ru-reviews.json',
                vocabulary_filename='ru-vocab.json', topn=20000,
                min_review_length=3,
                update_reviews=False,
                keep_value=1
              ):
    """ Prepare the dataset for supplying to LSTM.
        Keyword arguments:
            dataset_filename -- a name of the file which contains all movie
                reviews.
            vocabulary_filename -- a name of the file wich contains the
                vocabulary of tokens extracted from the dataset.
            topn -- keep only that much tokens from the top and discard all
                other.
            min_review_length -- if some reviews turns out to have length less
                than this, it will be discarted.
            update_reviews -- if true, then the review dataset and it's
                vocabulary will be updated firstly.
            keep_value -- the amount of data that will be used, other will be
                dropped.
    """
    cooked_reviews = []
    cooked_ratings = []
    print('Creating samples...')
    if update_reviews:
        reviews_dictionary = clean(update(load(reviews_filename)))
        save(reviews_dictionary, reviews_filename)
        vocab = build_vocabulary()
        save(vocab, vocabulary_filename)
        cooked_reviews, cooked_ratings = reviews_to_samples(reviews_dictionary, vocab, min_review_length=min_review_length, topn=topn)
    else:
        cooked_reviews, cooked_ratings = cooked_dict_to_samples(load(dataset_filename))
# Split cooked dataset in two parts for training and testing
    dataset_length = ceil(len(cooked_reviews) * keep_value)
    cooked_reviews = cooked_reviews[:dataset_length]
    cooked_ratings = cooked_ratings[:dataset_length]
    l = ceil(len(cooked_reviews) * 7 / 10)
    x_train, y_train, x_test, y_test = cooked_reviews[:l], cooked_ratings[:l], cooked_reviews[l:], cooked_ratings[l:]
# Normalize and shuffle these parts
    print('Normalizing...')
    x_train, y_train = normalize(x_train, y_train)
    x_test, y_test = normalize(x_test, y_test)
    return x_train, y_train, x_test, y_test


def normalize(reviews, ratings):
    """ If the number of reviews with and negative ratings is uneven, then
    equalize it by duplicating minor samples. Also shuffles the data for better
    sample distribution."""
    nb_positive = len([i for i in ratings if i == 1])
    nb_negative = len(ratings) - nb_positive
    x0 = reviews
    y0 = ratings
    if nb_positive != nb_negative:
        nb_duplications = abs(nb_positive - nb_negative)
        minor_rating = 0 if nb_positive > nb_negative else 1
        data = [[reviews[i], ratings[i]] for i in range(0, len(reviews))]
        for review, rating in data:
            if rating == minor_rating:
                data.append([review, rating])
                nb_duplications -= 1
                if nb_duplications == 0:
                    break
        random.shuffle(data)
        x0 = [sample[0] for sample in data]
        y0 = [sample[1] for sample in data]
    return x0, y0


def cooked_dict_to_samples(cooked_dict):
    cooked_reviews = []
    cooked_ratings = []
    for key, value in cooked_dict.items():
        cooked_reviews.append(value[0])
        cooked_ratings.append(int(value[1]))
    return cooked_reviews, cooked_ratings
