import re
class GetInfo(object):
    def __init__(self, data, type):
        self.data=data
        self.type=type
        #self.get_author(data["authors"])

    def get_author(self, dat_authors):
        self.authors=[]
        for author in dat_authors:
            name=self.getName(author)
            self.authors.append(name)
        authors=','.join([x for x in self.authors])
        return authors

    def getName(self, dat_author):
        middle=' '.join(dat_author["middle"])
        if dat_author['middle']:
            return ' '.join([dat_author['first'], middle, dat_author['last']])
        else:
            return ' '.join([dat_author['first'], dat_author['last']])

    def get_text(self,data):
        f_text=""
        for text in data:
            f_text+=text["text"]+" "
        f_text=self.preprocess(f_text)
        return f_text

    def get_record(self,data):
        if self.type=='pdf':
            record=[data['paper_id'], data['metadata']['title'], self.get_author(data['metadata']['authors']), self.get_text(data["abstract"]),\
                   self.get_text(data["body_text"])]
        elif self.type=='pmc':
            record=[data['paper_id'], data['metadata']['title'], self.get_author(data['metadata']['authors']),self.get_text(data["body_text"])]
        return record

    def preprocess(self, text):

        stops=['doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', \
               'et', 'al', 'author', 'figure', 'rights', 'reserved', 'permission', 'used', 'using', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.', 'al.', 'Elsevier', 'PMC', 'CZI']


        xs = ['Table', 'Figure']

        # Remove Citations
        text = re.sub(r'\([^)[a-z]*]*\)', '', text)

        # Remove Citations
        text = re.sub(r'\[[^)[a-z]*]*\]', '', text)

        #Remove Table and Figure contents
        text = re.sub(r'(' + '|'.join([x for x in xs]) + ')\s(\w)*', '', text)

        #Remove additional stopwords
        text=re.sub(r'('+ '|'.join([x for x in stops]) +')\s*', '', text)

        return text
