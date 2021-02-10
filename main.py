import glob
import logging
import pathlib
import string
import pandas as pd
import numpy as np
from bulstem.stem import BulStemmer
import bulstem.stemrules
import sklearn
import tqdm
#from stop_words import get_stop_words
from elasticsearch import Elasticsearch
import json
import os
import io
from pathlib import Path
import ntpath
from joblib import dump, load
#import elasticsnearch.helpers
from elasticsearch.helpers import streaming_bulk
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

genres = ['Фентъзи', 'Хорър (литература на ужаса)', 'Фантастика', 'Криминална литература',
          'Драма', 'Детска и юношеска литература', 'Любовен роман', 'Приключенска литература', 'Трилър', 'Философия',
          'Исторически роман', 'Хумор']

stop_words = [x for x in open('resources/bulgarianST.txt', 'r', encoding="utf-8-sig").read().split('\n')]
#unicode(x.strip(), 'utf-8')


logging.basicConfig(filename='books_errors.log', level=logging.INFO)
punctuation = string.punctuation
BULSTEM_DIR = pathlib.Path(bulstem.__file__).parent
RULES_2_PATH = BULSTEM_DIR / "stemrules" / "stem_rules_context_2_utf8.txt"


def dummy(text):
    return text


def create_json_file(path, literatureType, genre, year):
    try:
        file = io.open(path, mode="r", encoding="utf-8-sig")
    except FileNotFoundError:
        print(os.path.exists(path))
        file = io.open(path, mode="r", encoding="utf-8-sig")
    line = file.readline()
    author = line.strip()
    line = file.readline()
    title = ""

    while not (line == "\n"):
       title += line
       line = file.readline()
       if not line:
           logging.info(ntpath.split(path)[1])
           logging.info('Content:' + title)
           return

    title = title.strip()
    data = ""
    while not (line.startswith("	$id")):
        data = data + line
        line = file.readline()

    bookTry = {"title": title,
               "author": author,
               "literatureType": literatureType,
               "genre": genre,
               "year": year,
               "content": data,
               }
    try:
        with open("json_books/" + author+" - "+title+".txt", 'w', encoding='utf-8') as json_file:
            json.dump(bookTry, json_file, ensure_ascii=False)
    except OSError:
        keepcharacters = (' ', '.', '_', '-', ',')
        title = "".join(c for c in title if c.isalnum() or c in keepcharacters).rstrip()


def convert_to_json_files():
    X = ['A', 'Б','В','Г','Д','Е','Ж','З','И','Й','К','Л','М','О','П', 'Р','С','Т','У','Ф','Х','Ц','Ч','Ш','Щ','Ъ','Ю','Я']
    for letter in X:
        directory_in_str = '..\\..\\chitanka_books\\'+letter
        pathlist = Path(directory_in_str).rglob('*.txt')
        count = 0

        for path in pathlist:

            head, tail = ntpath.split(path)
            print(str(tail))
            tailString = os.path.splitext(str(tail))[0]

            try:
                characteristics = tailString.rsplit(')', 1)[0]
            except ValueError:
                characteristics = tailString

            literatureType, genre_year = characteristics.split('-', 1)
            try:
                genre, year = genre_year.rsplit('-', 1)
                year = year.strip()
            except ValueError:
                genre = genre_year
                year = ""

            genre = genre.strip()
            literatureType = literatureType.strip()

            create_json_file(path, literatureType[1:], genre, year)
            count = count + 1

    return count


def connect_elasticsearch():
    _es = None
    _es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
    if _es.ping():
        print('Yay Connect')
    else:
        print('Awww it could not connect!')
    return _es


if __name__ == '__main__':
          logging.basicConfig(level=logging.ERROR)


def create_index(es_object, index_name='books'):
    created = False
    # index settings
    settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
                "dynamic": "strict",
                "properties": {
                    "title": {
                        "type": "text"
                    },
                    "author": {
                        "type": "text"
                    },
                    "literatureType": {
                        "type": "text"
                    },
                    "genre": {
                        "type": "text"
                    },
                    "year": {
                        "type": "text"
                    },
                    "content": {
                        "type": "text"
                    }
                }
        }
    }
    try:
        if not es_object.indices.exists(index_name):
            # Ignore 400 means to ignore "Index Already Exist" error.
            es_object.indices.create(index=index_name, body=settings)
            print('Created Index')
            created = True
    except Exception as ex:
        print(str(ex))
    finally:
        return created


def store_record(elastic_object, index_name, record):
    try:
        outcome = elastic_object.index(index=index_name, body=record, op_type='create')
    except Exception as ex:
        print('Error in indexing data')
        print(str(ex))


def search(es_object, index_name, search):
    res = es_object.search(index=index_name, body=search)["hits"]["hits"]
    print(json.dumps(res, indent=4, ensure_ascii=False))


def generate_actions(letter):
    """Reads all the json text file in the current directory
    This function is passed into the bulk()
    helper to create many documents in sequence.
    """
    directory = os.fsencode('.')

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".txt") and filename.startswith(letter):
            file = io.open(filename, mode="r", encoding="utf-8-sig")
            doc = file.read()
            yield doc


def booksWithoutGenre():
    dir = "./"
    X = ['A', 'Б', 'В', 'Г', 'Д', 'E' 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ю', 'Я']

    for letter in X:
        for path in glob.iglob(dir+ letter + '*.txt'):
            with open(path, encoding='utf-8-sig') as json_file:
                data = json.load(json_file)
                stemmer = BulStemmer.from_file(RULES_2_PATH, min_freq=2, left_context=2)
                print(data['author'] + " - " + data['title'])
                if data["genre"] == "":
                    data['content'] = data['content'].lower()
                    data['content'] = word_tokenize(data['content'])
                    data['content'] = [stemmer.stem(w) for w in data['content']
                                       if w not in stop_words and w not in punctuation
                                       and w.isalpha()]

                    processedFile = {"title": data['title'],
                                     "author": data['author'],
                                     "genre": "",
                                     "tokenized_text": data['content']
                                     }

                    with open("./preprocessed_no_genre/" + data['author'] + " - " + data['title'] + "(preprocessed).txt", 'w',
                              encoding='utf-8') as json_file:
                        json.dump(processedFile, json_file, ensure_ascii=False)


def preprocessDocuments():

    dir = "."
    pathlist = Path(dir).rglob('*.txt')

    for path in pathlist:
        with open(path, encoding='utf-8-sig') as json_file:
            data = json.load(json_file)
            stemmer = BulStemmer.from_file(RULES_2_PATH, min_freq=2, left_context=2)
            if not (data["genre"] == ""):
                genre = data["genre"].split(',')[0]
                data['content'] = data['content'].lower()
                data['content'] = word_tokenize(data['content'])
                data['content'] = [stemmer.stem(w) for w in data['content']
                                   if w not in stop_words and w not in punctuation
                                   and w.isalpha()]
                print(data['content'])

                processedFile = {"title": data['title'],
                                 "author": data['author'],
                                 "genre": genre,
                                 "tokenized_text": data['content']
                           }

                with open("./preprocessed/"+data['author'] + " - " + data['title'] + "(preprocessed).txt", 'w',
                          encoding='utf-8') as json_file:
                    json.dump(processedFile, json_file, ensure_ascii=False)


def trainSVM():
    preprocessedDocumentsCount = 2000

    dir = "./preprocessed"

    categories_count = {genre: 0 for genre in genres}
    X_index = np.arange(0, preprocessedDocumentsCount)

    # Passing index array instead of the big feature matrix
    X_trainIndex, X_testIndex = train_test_split(X_index, test_size=0.25)
    X_trainIndex.sort()
    X_testIndex.sort()
    class_max_size = preprocessedDocumentsCount / 10
    print(class_max_size)

    preprocessedDocumentsCount = 0
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    testIndex = 0
    trainIndex = 0

    pathlist = Path(dir).rglob('*.txt')
    for path in pathlist:
        with open(path, encoding='utf-8-sig') as json_file:
            data = json.load(json_file)
            if data["genre"].startswith("Хумор"):
                curr_genre='Хумор'
            elif data["genre"].startswith("Философ"):
                print(data["genre"])
                curr_genre='Философия'
            else:
                curr_genre = data["genre"]
            if curr_genre in genres and categories_count[curr_genre] < class_max_size:
                categories_count[curr_genre] += 1
                if preprocessedDocumentsCount == X_trainIndex[trainIndex]:
                    trainIndex += 1
                    X_train.append(data["tokenized_text"])
                    Y_train.append([curr_genre])
                else:
                    testIndex += 1
                    X_test.append(data["tokenized_text"])
                    Y_test.append([curr_genre])
                    # print(data["author"])
                if preprocessedDocumentsCount == 1999:
                    break
                preprocessedDocumentsCount += 1

    print("data is read")
    print(Y_train)
    Encoder = LabelEncoder()  #Encode target labels with value between 0 and n_classes-1

    Train_Y = Encoder.fit_transform(Y_train)
    Test_Y = Encoder.fit_transform(Y_test)

    print("category is encoded")

    X = X_train + X_test

    print("data merged")
    Tfidf_vect = TfidfVectorizer(lowercase=False, tokenizer=dummy, min_df=0.02, max_features=9000)

    Tfidf_vect.fit(X)

    print("data is fitted and dumped")
    Train_X_Tfidf = Tfidf_vect.transform(X_train)
    Test_X_Tfidf = Tfidf_vect.transform(X_test)

    print("data is transformed")

    SVM = sklearn.svm.SVC(C=1,  kernel='linear', degree=3, gamma='auto')
    SVM.fit(Train_X_Tfidf, Train_Y)

    print("SVM fitted data")
    predictions_SVM = SVM.predict(Test_X_Tfidf)
    print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, Test_Y) * 100)

    user_input = input("Please select a folder:")
    folder = user_input
    if not os.path.exists(folder):
        os.makedirs(folder)
    dump(Encoder, folder + '/Encoder.joblib')
    dump(Tfidf_vect,  folder +'/vectorizer.joblib')
    print("tfifd_vect dumped")

    dump(SVM,  folder +'/svm.joblib')
    dump(Test_Y,  folder +'/test_data.joblib')
    dump(predictions_SVM,  folder +'/prediction.joblib')
    print("predictions_SVM dumped")


def demonstrate(folder, book):
   #book = "preprocessed_no_genre/" + book
    book = "test/" + book
    svm=load(folder + '/svm.joblib')
    with open(book, encoding='utf-8-sig') as json_file:
        data = json.load(json_file)

    encodedCategories=load(folder + '/Encoder.joblib')
    values = encodedCategories.classes_
    keys = encodedCategories.transform(encodedCategories.classes_)
    dictionary = dict(zip(keys, values))

    Tfidf_vect = load(folder+'/vectorizer.joblib')
    Test_X_Tfidf = Tfidf_vect.transform([data["tokenized_text"]])

    prediction = svm.predict(Test_X_Tfidf)[0]
    print(data["author"] + " - " + data["title"] + " : " + dictionary[prediction])


def index_documents():
    es = connect_elasticsearch()
    number_of_docs=37785
    print("Indexing documents...")
    progress = tqdm.tqdm(unit="docs", total=number_of_docs)
    successes = 0
    letters = ['Z', 'L', 'N', 'A', 'Б','В','Г','Д','Е','Ж','З','И','Й','К','Л','М','О','П', 'Р','С','Т','У','Ф','Х','Ц','Ч','Ш','Щ','Ъ','Ю','Я']
    for letter in letters:
        for ok, action in streaming_bulk(
            client=es, index="books", actions=generate_actions(letter), request_timeout=60
        ):
            progress.update(1)
            successes += ok
    print("Indexed %d/%d documents" % (successes, number_of_docs))


def make_query():
    es = connect_elasticsearch()
    user_input = input("Please choose query type: 1:")
    fuzziness=False

    if es is not None:
        if fuzziness == True:
            search_object = {'query': {
                                'match': {
                                    'content': {
                                        "query": user_input, "fuzziness":"AUTO"
                                               }
                                         }
                                      },
                                    "fields": ["author", "title"],
                                        "_source": False}
        else:
            search_object = {'query':
                {
                'match': {
                    'content': user_input
                        }
                },
                "fields": ["author", "title"],
                "_source": False
                                }

        # search_object = {'query':
        #                      {'query_string':
        #                           {'query': '(транквилизатори) OR (транквилизаториafaf)'}}}
        search(es, 'books', json.dumps(search_object))


def classifier_test(folder):
    #trainSVM()
    predictions_SVM=load(folder + '/prediction' +'.joblib')
    test_data=load(folder+'/test_data' +'.joblib')
    svm = load(folder + '/svm' + '.joblib')
    #print(svm.get_params(True))
    #print(len(predictions_SVM))
    print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, test_data) * 100)

    book1="Агата Кристи - Дамата с воала(preprocessed).txt"
    book2="Алеко Константинов - Разни хора, разни идеали I(preprocessed).txt"
    book3="Марк Твен - Писма от Земята(preprocessed).txt"
    book4="Айзък Азимов - Четириизмерната котка(preprocessed).txt"
    book5="Тери Пратчет - Автентичната котка(preprocessed).txt"
    book6="Стивън Кинг - Сънят на Харви(preprocessed).txt"
    book7="Ърнест Хемингуей - Зелените хълмове на Африка(preprocessed).txt"
    book8="Антоан дьо Сент-Екзюпери - Цитадела(preprocessed).txt"
    book9="Джеръм К. Джеръм - Какво струва да се покажеш любезен(preprocessed).txt"
    book10="Стивън Кинг - Дяволската котка(preprocessed).txt"
    book11="Емили Гифин - Нещо назаем(preprocessed).txt"
    book12="Христо Смирненски - Юноша(preprocessed).txt"
    book13="Рей Бредбъри - Градът, където никой не спира(preprocessed).txt"
    books = [book1, book2, book3, book4, book5, book6, book7, book8, book9, book10, book11, book12, book13]

    for book in books:
        demonstrate(folder, book)

def main():
    folder="experiments/"
    folder =folder + "svm(c1)"
    classifier_test(folder)
    #make_query()


if __name__ == '__main__':
    main()




