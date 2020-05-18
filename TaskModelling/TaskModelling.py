import gensim
import pandas as pd
import numpy as np
from tqdm import notebook
from nltk.corpus import stopwords
import scipy.spatial.distance as distance
import re
from TaskModelling.GetTasks import  GetTasks

class TaskModelling(object):
    def __init__(self,eddir, tddir):
        self.tasks=GetTasks.get_tasks(tddir)
        self.ddir=eddir
        print('Please wait until create the embedding file')
        self.embedding_index=self.get_glove_embed(self.ddir)
        self.cat_embed={}
        self.categories_task = {}
        self.stopWords = stopwords.words('english')

    def get_glove_embed(self,ddir):
        embedding_index={}
        with open(ddir, 'r', encoding='utf8',errors = 'ignore') as f:
            for line in notebook.tqdm(f):
                values = line.split()
                word = ''.join(values[:-300])
                coefs = np.asarray(values[-300:], dtype='float32')
                embedding_index[word]=coefs
        return embedding_index

    def add_newStopWords(self):

        cust_stops = ['(e.g.,', 'e.g.,', '(e.g.),']
        for cust_stop in cust_stops:
            self.stopWords.append(cust_stop)

    def get_counts_task(self,text):
        get_counts={}
        for sentence in text:
            sentence = re.sub(r'[^\w\s]','',sentence)
            for word in sentence.split():
                if word not in self.stopWords:
                    if word not in get_counts.keys():
                        get_counts[word]=1
                    else:
                        get_counts[word]=get_counts[word]+1
        get_counts=sorted(get_counts.items(), key=lambda x:x[1], reverse=True)
        top_words=[key for key,value in get_counts[:15]]
        return top_words

    def fit(self):
        self.add_newStopWords()
        tasks_c = {}
        for i, task in enumerate(self.tasks):
            self.tasks[task].append(task)
            tasks_c[f'task {i + 1}'] = self.get_counts_task(self.tasks[task])
        tasks_missings = {
            'task 1': ['incubation', 'stability', 'seasonality', 'season', 'winter', 'spring', 'summer', 'autumn','prediodically',
                       'carrying', 'transference', 'immunity', 'immune','quarantine','country', 'surface',],
            'task 2': ['danger', 'hazard', 'exposure','pregnancy','heart','prevent','weather','season','mask','disinfectant','hypertension','patient','prevention','precaution'],
            'task 3': ['genetic', 'origin','animal'],
            'task 4': ['interventions', 'non-pharmaceutical','drug','antiviral','ai'],
            'task 5': ['therapeutics'],
            'task 6': ['considerations', 'published'],
            'task 7':['phenotypic','tool','detect','adaptation'],
            'task 8': ['published', 'prevent', 'precaution', 'inhibit', 'block', 'avoid', 'forbid', 'avert','ethical'],
            'task 9': ['inter-sectoral', 'collaboration', 'published', 'sharing', 'information-sharing','rumors','misinformation']
        }

        for key, value in tasks_missings.items():
            for word in value:
                tasks_c[key].append(word)

        self.categories_task = {}
        for key, values in tasks_c.items():
            for word in values:
                if word not in self.categories_task.keys():
                    self.categories_task[word] = [key]
                else:
                    self.categories_task[word].append(key)
        self.cat_embed = {key: value for key, value in self.embedding_index.items() if key in self.categories_task.keys()}

        return self.cat_embed


    def get_text(self,text):
        text=[word for word in text.split(' ') if word not in self.stopWords]
        #text=' '.join([word for word in text])
        return text

    def check_distance(self,query):
        querys=self.get_text(query)
        #query_embed=embedding_index[query]
        distances={}
        for query in querys:
            if query in ['COVID-19', 'SARS-CoV-2', '2019-nCov', 'SARS Coronavirus 2' ,'2019 Novel Coronavirus']:
                query='Coronavirus'
            query_embed=self.embedding_index[query]
            for word, value in self.cat_embed.items():
                category=self.categories_task[word]
                sim=1-distance.cosine(query_embed, value)
                if sim>0.6:
                    #dist=query_embed.dot(value)
                    for cats in category:
                        #dist /= len(tasks_c[cats])
                        distances[cats] = distances.get(cats, 0) + sim
        #             dist /= len(tasks_c[category])
        #             distances[category] = distances.get(category, 0) + dist
        return distances