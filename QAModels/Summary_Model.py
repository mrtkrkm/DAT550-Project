import torch
import pandas as pd
import numpy as np
from transformers import BartForConditionalGeneration, BartConfig, BartTokenizer

import torch
import pandas as pd
import numpy as np
from transformers import BartForConditionalGeneration, BartConfig, BartTokenizer


class Summary_Model(object):
    def __init__(self, summary_model_path, token_path, device):
        self.summary_model_path = summary_model_path
        self.token_model_path = token_path
        self.model = None
        self.tokenizer = None
        self.device = device
        self.create_models()

    def create_models(self):
        self.model = BartForConditionalGeneration.from_pretrained(self.summary_model_path)
        self.tokenizer = BartTokenizer.from_pretrained(self.token_model_path)
        self.model = self.model.to(self.device)

    def get_summary(self, data, count):
        ### Top confident answer to article
        total_abstract = ''
        for i in range(len(data[:count])):
            abss, ans = data.loc[i, ['answer_sent', 'Answer']].values
            total_abstract += (abss + ".")
        ARTICLE_TO_SUMMARIZE = total_abstract

        ### Token the article, if larger than 1024, then split the article
        tokens = self.tokenizer.tokenize(ARTICLE_TO_SUMMARIZE)
        max_seq_length = 1024
        longer = 0
        all_tokens = []
        if len(tokens) > 1024:
            for i in range(0, len(tokens), max_seq_length):
                tokens_a = tokens[i:i + max_seq_length]
                one_token = self.tokenizer.batch_encode_plus([tokens_a], max_length=1024, return_tensors='pt')
                all_tokens.append(one_token)

        Summary_text = []

        ## decode the model output as summary text
        def decode_text(sum_ids):
            text = ''
            for g in sum_ids:
                text = text + self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            return text

        ## Summary model
        model = self.model.to(self.device)
        model.eval()
        Summary_text = ''
        for inputs in all_tokens:
            summary_ids = model.generate(inputs['input_ids'].to(self.device), num_beams=2, max_length=1000,
                                         early_stopping=True)
            Summary_text = Summary_text + " " + decode_text(summary_ids)

        if Summary_text == '':
            Summary_text = "Can't find summary of answer"

        return Summary_text



