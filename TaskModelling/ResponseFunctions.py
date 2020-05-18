from TaskModelling.TaskModelling import TaskModelling
from TaskModelling.GetTasks import GetTasks
import os
import joblib
import datetime
import string
import scipy.spatial.distance as distance

class ResponseFunctions(object):
    def __init__(self, embedding_path, task_path, meta_dat):
        self.embeddings=TaskModelling(embedding_path,task_path)
        self.tasks = GetTasks.get_tasks(task_path)
        self.Cov_list = ['COVID-19', 'SARS-CoV-2', '2019-nCov', 'SARS Coronavirus 2', '2019 Novel Coronavirus', '2019nCoV',
                    'COVID19', 'SARSCoV2',
                    'SARSCoronavirus2', '2019NovelCoronavirus']

        self.Related_word = self.Cov_list + ['Virus', 'transmission', 'incubate', 'incubation', 'weather', 'summer', 'virus',
                                   'Corona', 'corona', 'prevent', 'mask', 'disinfectant']

        self.all_dicts, self.all_sums = self.read_dict_sum()
        self.task_questions = joblib.load('data/tasks/task_questions')
        self.new_embed=self.embeddings.fit()
        print('Loading Meta Data')
        self.meta=joblib.load(meta_dat)

    def read_dict_sum(self):
        def merge_files(path_d, path_s):
            all_dicts = {}
            all_sums = {}
            for path_d, path_s in zip(path_d, path_s):
                dicts = joblib.load(path_d)
                sums = joblib.load(path_s)
                all_dicts = {**all_dicts, **dicts}
                all_sums = {**all_sums, **sums}
            return all_dicts, all_sums

        datas = 'data/QAResults'
        files = os.listdir(os.path.join(datas))
        dicts = []
        summaries = []
        for file in files:
            if file.endswith('.dic'):
                dicts.append(os.path.join(datas, file))
            if file.endswith('.sum'):
                summaries.append(os.path.join(datas, file))
        return merge_files(dicts, summaries)

    def find_task_Summary(self,scores):
        text = 'The summary of 3500 paper that we know is' + os.linesep
        # task=int(scores[0][-1])
        text = text + self.all_sums[scores]
        return scores, text

    def find_task_ques(self,scores):
        text = 'My First answer is:'
        # task=int(scores[0][-1])
        rangeS = 2
        urls=[]
        for i in range(rangeS):
            row=self.all_dicts[scores].iloc[i, :]
            answ = row['answer_sent']
            text = text + answ
            url=list(self.meta[self.meta['Id']==row['Id']]['url'])[0]
            urls.append(url)
            if i < rangeS - 1:
                text = text + os.linesep + 'Another Answer is:'
        return scores, text, urls

    def End_conv(self):
        time = int(str(datetime.datetime.now().time())[:2])
        if time > 18 or time < 6:
            response = 'Have a Good evening'
        else:
            response = 'Have a Good Day'
        return response

    def check_distanceq(self,best, query):
        idx = best[0]
        query = query.translate(str.maketrans('', '', string.punctuation))
        querys = self.embeddings.get_text(query)
        distances = {}
        for query in querys:
            if query in self.Cov_list:
                query = 'Coronavirus'
            query_embed = self.embeddings.embedding_index[query]
            for i, sques in enumerate(self.task_questions[idx]):
                sques = sques.translate(str.maketrans('', '', string.punctuation))
                quests = self.embeddings.get_text(sques)
                for qw in quests:
                    if qw in self.Cov_list:
                        qw = 'Coronavirus'
                    value = self.embeddings.embedding_index[qw]
                    sim = 1 - distance.cosine(query_embed, value)
                    if sim > 0.6:
                        distances[i] = distances.get(i, 0) + sim
        return distances, idx
        # return task_questions[idx][sorted(distances.items(), key=lambda x:x[0],reverse=True)[0][0]]

    def sort_response(self,result, user_i):
        if len(result) < 1:
            response = "Sorry i couldn't understand your question. Can you give more detail?"
            quest = ''
            urls = ['']
        else:
            best = sorted(result.items(), key=lambda x: x[1], reverse=True)[0]
            # response=find_task_ques(best)
            #####
            distances, idx = self.check_distanceq(best, user_i)
            if len(distances) > 0:
                items = self.task_questions[idx][sorted(distances.items(), key=lambda x: x[0], reverse=True)[0][0]]
                #quest, response = self.find_task_Summary(items)
                quest, response, urls = self.find_task_ques(items)
            else:
                response = "Sorry i couldn't understand your question. Can you give more detail?"
                quest = ''
                urls=['']
        return quest, response,urls

