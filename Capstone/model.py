import gzip
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np


class Recommendation:
    
    def __init__(self):
        self.data = pd.read_pickle(open('pickle/processed_data.pkl','rb'))
        self.user_final_rating = pd.read_pickle(open('pickle/user_final_rating.pkl','rb'))
        with gzip.open('pickle/rfc.pkl', 'rb') as f:
            self.model = pd.read_pickle(f)
        self.raw_data = pd.read_csv("sample30.csv")

        self.raw_data["reviews_title"].fillna(" ", inplace=True)
        self.raw_data["reviews"] = self.raw_data[["reviews_title", "reviews_text"]].agg(" ".join, axis=1)
        
        self.data = pd.concat([self.raw_data[['id','reviews']],self.data], axis=1)
        _, self.i = np.unique(self.data.columns, return_index=True)
        self.data = self.data.iloc[:, self.i]
        
        
    def getTopProducts(self, user):
        items = self.user_final_rating.loc[user].sort_values(ascending=False)[0:20].index
        features = pickle.load(open('pickle/tfidf_vocab.pkl','rb'))
        vectorizer = TfidfVectorizer(vocabulary = features)
        temp=self.data[self.data.id.isin(items)]
        X = vectorizer.fit_transform(temp['reviews'])
        temp=temp[['id']]
        temp['prediction'] = self.model.predict(X)
        temp['prediction'] = temp['prediction'].map({1:1,-1:0})
        temp=temp.groupby('id').sum()
        temp['positive_percent']=temp.apply(lambda x: x['prediction']/sum(x), axis=1)
        final_list=temp.sort_values('positive_percent', ascending=False).iloc[:5,:].index
        return self.data[self.data.id.isin(final_list)][['id', 'brand',
                              'categories', 'manufacturer', 'name']].drop_duplicates().to_html(index=False)
    
