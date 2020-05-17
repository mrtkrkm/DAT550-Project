from PreProcess.OpenData import OpenData
import pandas as pd

class Preprocess(object):
    def __init__(self, pdf_dirs, pmc_dirs):
        self.pdf_dir = pdf_dirs
        self.df_all=None
        self.pmc_dir =pmc_dirs

    def Create_merge_df(self, metadatadir):
        print('#'*20+'  Loading Pdf Data  '+'#' * 20)
        self.data_pdf = OpenData(data_dir=self.pdf_dir, types='pdf')

        print('#' * 20 + '  Creating Pdf DataFrame')
        self.df_pdf = self.data_pdf.fit_transform()

        print('#' * 20 + '  Loading Pmc Data  '+'#' * 20)
        self.data_pmc = OpenData(data_dir=self.pmc_dir, types='pmc')

        print('#' * 20 + '  Creating Pmc DataFrame  '+'#' * 20)
        self.df_pmc = self.data_pmc.fit_transform()

        print('#' * 20 + '  Creating Pmc DataFrame  '+'#' * 20)
        self.df_meta = pd.read_csv(metadatadir)

        print('#' * 20 + '  Merge Files  '+'#' * 20)
        self._merge_files()

        print('#' * 20 + '  Cleaning Data  '+'#' * 20)
        self._cleaning_part()

        return self.df_all

    def _merge_files(self):
        self.df_all = pd.merge(self.df_meta, self.df_pdf, left_on='sha', right_on='Id', how='right').drop(['sha', 'title_y', 'authors_y'],
                                                                                    axis=1)
        self.df_all.rename(columns={'authors_x': 'authors'}, inplace=True)

        self.df_all = pd.merge(self.df_all, self.df_pmc, left_on='pmcid', right_on='Id', how='left').drop(['Id_y', 'title_x', 'authors_y'],
                                                                                       axis=1)
        self.df_all.rename(columns={'authors_x': 'authors'}, inplace=True)

        self.df_all = self.df_all[self.df_all['body_y'] != '']

    def _cleaning_part(self):
        self.df_all.drop(columns=['has_pdf_parse', 'has_pmc_xml_parse', 'Microsoft Academic Paper ID', 'WHO #Covidence'],
                    inplace=True)
        self.df_all.loc[self.df_all.abstract_y.isnull() & (self.df_all.abstract_x != ''), 'abstract_y'] = self.df_all[
            (self.df_all.abstract_y.isnull()) & (self.df_all.abstract_x != '')].abstract_x

        self.df_all.rename(columns={'abstract_y': 'abstract'}, inplace=True)
        self.df_all.drop('abstract_x', axis=1, inplace=True)

        self.df_all.loc[self.df_all.body_y.notnull(), 'body_x'] = self.df_all.loc[self.df_all.body_y.notnull(), 'body_y']

        self.df_all.rename(columns={'body_x': 'body_text'}, inplace=True)
        self.df_all.drop('body_y', axis=1, inplace=True)

        self.df_all.rename(columns={'Id_x': 'Id', 'source_x': 'source'}, inplace=True)

        print('#' * 20 + '  Droping Dublicate Files  '+'#' * 20)
        self.df_all.drop_duplicates(['Id', 'body_text'], inplace=True)

        print('#' * 20 + '  Finding Relationship with Corona  '+'#' * 20)
        self.df_all['is_covid19'] = self.df_all.body_text.str.contains(
            'COVID-19|covid|sar cov 2|SARS-CoV-2|2019-nCov|2019 ncov|SARS Coronavirus 2|2019 Novel Coronavirus|coronavirus 2019| Wuhan coronavirus|wuhan pneumonia|wuhan virus',
            case=False)






