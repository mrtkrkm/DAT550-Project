from QAModels.Summary_Model import Summary_Model
import numpy as np

import torch
import pandas as pd
from transformers import BertTokenizer
from transformers import BertForQuestionAnswering
from tqdm import notebook


class QuestionAnswering(object):
    def __init__(self, bert_model_path, token_model_path, device):
        self.bert_model_path = bert_model_path
        self.token_model_path = token_model_path
        self.model = None
        self.tokenizer = None
        self.device = device
        self.create_models()
        self.sum_model = Summary_Model(summary_model_path=Bart, token_path=Bart, device=device)

    def create_models(self):
        self.model = BertForQuestionAnswering.from_pretrained(self.bert_model_path)
        self.tokenizer = BertTokenizer.from_pretrained(self.token_model_path)
        self.model = self.model.to(self.device)

    def get_answer_sentence(self, text, answer):
        text = text.lower()
        answer = answer.lower()
        split_byans = text.split(answer)
        if len(split_byans) == 1:
            return split_byans[0][split_byans[0].rfind(". ") + 1:] + " " + answer
        else:
            return split_byans[0][split_byans[0].rfind(". ") + 1:] + " " + answer + split_byans[1][
                                                                                    :split_byans[1].find(". ") + 1]

    def create_answer_text(self, tokens, start=0, stop=-1):
        tokens = tokens[start: stop]
        if '[SEP]' in tokens:
            sepind = tokens.index('[SEP]')
            tokens = tokens[sepind + 1:]
        txt = ' '.join(tokens)
        txt = txt.replace(' ##', '')
        txt = txt.replace('##', '')
        txt = txt.strip()
        txt = " ".join(txt.split())
        txt = txt.replace(' .', '.')
        txt = txt.replace('( ', '(')
        txt = txt.replace(' )', ')')
        txt = txt.replace(' - ', '-')
        txt_list = txt.split(' , ')
        txt = ''
        nTxtL = len(txt_list)
        if nTxtL == 1:
            return txt_list[0]
        newList = []
        for i, t in enumerate(txt_list):
            if i < nTxtL - 1:
                if t[-1].isdigit() and txt_list[i + 1][0].isdigit():
                    newList += [t, ',']
                else:
                    newList += [t, ', ']
            else:
                newList += [t]
        return ''.join(newList)

    def train_model_func(self, question, document):
        token_ids = self.tokenizer.encode(question, document)
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        answers = []
        nWords = len(document.split())
        cs = []
        if len(token_ids) > 512:
            # divIndex = len(token_ids) // 2
            divIndex = int(np.ceil(nWords / 2))
            all_sp = document.split()
            docs = [' '.join(all_sp[:divIndex]), ' '.join(all_sp[divIndex:])]
            token_ids = [self.tokenizer.encode(question, doc) for doc in docs]
        else:
            token_ids = [token_ids]

        for token_idp in token_ids:
            tokens = self.tokenizer.convert_ids_to_tokens(token_idp)
            sep_index = token_idp.index(self.tokenizer.sep_token_id)

            # Because of CLS
            questP = sep_index + 1
            secp = len(token_idp) - questP

            seq_types = [0] * questP + [1] * secp

            if len(seq_types) < 512:

                start_index, end_index = self.model(torch.tensor([token_idp]).to(self.device),
                                                    token_type_ids=torch.tensor([seq_types]) \
                                                    .to(self.device))
            else:
                start_index, end_index = self.model(torch.tensor([token_idp[:512]]).to(self.device),
                                                    token_type_ids=torch.tensor([seq_types[:512]]) \
                                                    .to(self.device))
            start_index = start_index[:, 1:-1]
            end_index = end_index[:, 1:-1]

            answer_start = torch.argmax(start_index)
            answer_end = torch.argmax(end_index)

            answer = self.create_answer_text(tokens, answer_start, answer_end + 2)

            answers.append(answer)

            c = start_index[0, answer_start].item() + end_index[0, answer_end].item()
            cs.append(c)

        maxC = max(cs)
        maxC_ind = cs.index(maxC)
        answer = answers[maxC_ind]
        return answer, maxC

    def train(self, documents, questions):
        df_dicts = {}
        summaries = {}
        for question in questions:
            ids = []
            answers = []
            confs = []
            abstracts = []
            sent_ans = []
            for document, abstract, id in notebook.tqdm(documents):
                answer, c = self.train_model_func(question, document)
                if answer != '':
                    ans_sent = self.get_answer_sentence(document, answer)
                    sent_ans.append(ans_sent)
                    ids.append(id)
                    answers.append(answer)
                    confs.append(c)
                    abstracts.append(abstract)

            question_df = pd.DataFrame(
                {'Id': ids, 'abstract': abstracts, 'Answer': answers, 'answer_sent': sent_ans, 'conf': confs})
            question_df = question_df.sort_values(by=['conf'], ascending=False).reset_index(drop=True)
            summary = self.sum_model.get_summary(question_df, 100)
            df_dicts[question] = question_df
            summaries[question] = summary
            # df_lists.append(question_df)
        return df_dicts, summaries