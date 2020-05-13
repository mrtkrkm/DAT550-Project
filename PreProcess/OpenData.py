import os
import json
from tqdm import notebook
from GetInfo import GetInfo
import pandas as pd

class OpenData(object):
    def __init__(self, data_dir, types):
        if type(data_dir)==str:
            self.datadir=[data_dir]
        else:
            self.datadir = data_dir
        self.type=types
        #self.all_names=os.listdir(data_dir)
        self._read_data()

    def _read_data(self):
        self.all_files=[]
        for root in self.datadir:
            self.all_names = os.listdir(root)

            #startind=root.rindex('/')
            # type=root[-9:-6]
            # self.types.append(type)

            for files in notebook.tqdm(self.all_names):
                file_dir=root+files
                self.all_files.append(json.load(open(file_dir, 'rb')))
    
    def fit_transform(self):
        rec=[]
        for file in notebook.tqdm(self.all_files):
            gi=GetInfo(file, self.type)
            rec.append(gi.get_record(file))
        df=pd.DataFrame(rec)
        if self.type=='pdf':
            df.columns=["Id",'title','authors','abstract','body']
        else:
            df.columns = ["Id", 'title', 'authors', 'body']
        return df
    
    def a(self):
        a= GetInfo(self.all_files[0]["metadata"])
        return a.author()