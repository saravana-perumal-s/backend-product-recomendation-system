from flask import Flask,render_template,request
import numpy as np
import pickle
import json

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

f=open("result.json","w")

app = Flask(__name__)

main={"name":[],"price":[],"rate":[]}



def model(item):

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
        
        def get_index_from_title(title):
            return df[df.type==title]["Index"].values[0]
        mv=get_index_from_title(item)

        similar=list(enumerate(cosine_sim[mv]))
        sorted_similar=sorted(similar,key=lambda x:x[1],reverse=True)

        def get_name_from_index(index):
            global name
            global rate
            global price
          
            a=df[df.index==index]["Name"].values[0]
            #a=a+"\n"
            main["name"].append(a)
            a=df[df.index==index]["Rating"].values[0]
            a=str(a)#+"\n"
            main["rate"].append(a)
            a=df[df.index==index]["Price"].values[0]
            a=str(a)#+"\n"
            main["price"].append(a)
            return main
        i=0
        l=[]
        for m in sorted_similar:
            l.append(get_name_from_index(m[0]))
            
            i=i+1
            if(i>15):
                break
        



@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    final_f=request.form.get("product name")
    main["name"].clear()
    main["rate"].clear()
    main["price"].clear() 
    model(final_f)

    res=json.dumps(main,indent=4)
    with open("result.json","w") as n:
        json.dump(main,n)


    
    return render_template('index.html',pre_text=res)
    #print(model)
if __name__ == "__main__":
    app.run(debug=True)