from flask import Flask,render_template,request
import numpy as np
import pickle


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)

def model(item):
                #read data 
               #writefile
        fn="datafile.txt"
        f=open(fn,'w')
        f.close()

        #read data 
        df=pd.read_csv("cleanedata - final.csv")
        features=['type','Price','Rating']
        for f in features:
            df[f]=df[f].fillna('')
        def combined_features(row):
            return str(row['type'])+"-"+str(row['Price'])+"-"+str(row['Rating'])

        df["combined_features"]=df.apply(combined_features,axis=1)

        cv=CountVectorizer()
        count_matrix=cv.fit_transform(df["combined_features"])
        #print("count matrix",count_matrix.toarray())

        cosine_sim=cosine_similarity(count_matrix)

        
        item=" "+item
        def get_index_from_title(title):
            return df[df.type==title]["Index"].values[0]
        mv=get_index_from_title(item)

        similar=list(enumerate(cosine_sim[mv]))
        sorted_similar=sorted(similar,key=lambda x:x[1],reverse=True)

        def get_name_from_index(index):
            f=open(fn,'a')
            k=[]    
            a=df[df.index==index]["Name"].values[0]
            #a=a+"\n"
            k.append(a)
            a=df[df.index==index]["Rating"].values[0]
            a=str(a)#+"\n"
            k.append(a)
            a=df[df.index==index]["Price"].values[0]
            a=str(a)#+"\n"
            k.append(a)
            for i in k:
                f.write(i+'\n')
            print(k)
            return k
        i=0
        l=[]
        for m in sorted_similar:
            l.append(get_name_from_index(m[0]))
            
            i=i+1
            if(i>15):
                break
        #print(l)




#model= pickle.load(open('cosinepickle.pkl','rb'))
#print(model)
@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    final_f=request.form.get("product name")
    #l=list(model.predict(final_f))
    '''
    pre=model.predict(final_f)
    #output=round(pre[0],2)'''
    model(final_f)
    f=open("datafile.txt","r")
    content=f.readlines()

    return render_template('index.html',pre_text= content)
    #print(model)


if __name__ == "__main__":
    app.run(debug=True)