import params as params
import csv
import sys

from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from flask import send_from_directory
from wtforms import Form, FieldList, FormField, IntegerField, SelectField, \
        StringField, TextAreaField, SubmitField
from wtforms import validators
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import scikitplot as skplt

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.metrics import mean_squared_error, roc_curve, auc, confusion_matrix, \
                            plot_confusion_matrix, classification_report, make_scorer, accuracy_score
import app as app
from flask import Flask, render_template , request, url_for ,session
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from random import randint
from flask.helpers import send_file
from jinja2 import Template
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
#from lwoku import RANDOM_STATE, N_JOBS, VERBOSE, get_prediction
import matplotlib.pyplot as plt
from os import path
import re
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
# Feature Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
import seaborn

RANDOM_STATE = 42
N_JOBS = -1
VERBOSE = 0

with open('templates/config.json','r') as c:
   params = json.load(c)["params"]

#plt.use('Agg')
app = Flask(__name__)
app.jinja_env.filters['zip'] = zip
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config['SQLALCHEMY_DATABASE_URI'] = params['server']
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db=SQLAlchemy(app)
app.secret_key = params['key']

app.config['SECRET_KEY'] = 'sosecret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
db.init_app(app)
db.create_all(app=app)


# this is the path to save dataset for Decision Tree
datapath = 'static/datasets/'
pathforplot = ''
app.config['data'] = datapath
app.config['plot'] = pathforplot
app.config['dataset_name'] = ''

pathfordataset = "static/data-preprocess/"
app.config['DFPr'] = pathfordataset
app.config['dataset_name_for_preprocessing'] = ''



app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config['SQLALCHEMY_DATABASE_URI'] = params['server']
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db=SQLAlchemy(app)
app.secret_key = params['key']



# Create models
db = SQLAlchemy()


class Login(db.Model):
    sno=db.Column(db.Integer,primary_key=True)
    name=db.Column(db.String(50),nullable=False)
    user=db.Column(db.String(50),nullable=False)
    password=db.Column(db.String(50),nullable=False)


@app.route('/')
def index():
    return render_template('front.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/createe')
def createe():
    return render_template('index.html')

@app.route('/create',methods=['GET','POST'])
def create():
      return render_template('data.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    l = os.listdir('templates/loans/')
    li = [x.split('.')[0] for x in l]
    return render_template("predict_loans_list_buttons.html", li=li)

@app.route('/viz',methods=['GET','POST'])
def viz():
    l = os.listdir('templates/loans/')
    li = [x.split('.')[0] for x in l]
    return render_template('visualize.html', li = li)

@app.route('/guide',methods=['GET','POST'])
def guide():
      return send_from_directory(directory='static/pdf/', path='user_guide.pdf')

@app.route("/logout")
def logout():
    session.pop('user',None)
    return render_template('front.html')

@app.route("/newuser",methods=['GET','POST'])
def new():
   if request.method=='POST':
      name=request.form['name']
      username=request.form['user']
      password=request.form['passw']
      x=Login(name=name,user=username,password=password)
      db.session.add(x)
      db.session.commit()
      return render_template('Login.html')
   return render_template('newuser.html')

@app.route('/')
def home():
   if('user' in session):
         return render_template('index.html')
   return render_template('login.html')


@app.route('/dashboard',methods=['GET','POST'])
def dashboard():
   if request.method=='POST':
      username=request.form['user']
      password=request.form['passw']
      allusers=Login.query.all()
      for i in allusers:
         if i.user==username and i.password == password:
            session['user']=username
            session['sno']=i.sno
            return render_template('index.html')
      return render_template('login.html')


@app.route('/supervised/classification/decisiontree/data', methods=['GET', 'POST'])
def decisiontreedataset():
    if request.method == 'POST':
        my_dataset = request.files['my_dataset']
        app.config['dataset_name'] = my_dataset.filename
        dataset_path = os.path.join(datapath, secure_filename(my_dataset.filename))
        my_dataset.save(dataset_path)
        get_dataset = os.path.join(app.config['data'], secure_filename(my_dataset.filename))
        df = pd.read_csv(get_dataset)
        missing = df.isnull().sum().to_frame('Individual count of missing value')

    return render_template('data.html',
                           my_dataset_head=df.head(5).to_html(classes='table table-striped table-hover x'),
                           missing=missing.to_html(classes='table table-striped table-hover x')
                           , dataset_describe=df.describe().to_html(classes='table table-striped table-hover x'),
                           col=df.columns.tolist()

                           )


@app.route('/supervised/classification/decisiontree/train', methods=['GET', 'POST'])
def dttrain():
    from sklearn import metrics
    if request.method == 'POST':
        get_dataset = os.path.join(app.config['data'],
                                   secure_filename(app.config['dataset_name']))
        feature = request.form.getlist('features')
        name_of_model = request.form.get('name_of_model')
        if os.path.exists(os.path.join(app.config['data'],secure_filename('%s.csv' %name_of_model))):
            pass

        else:
            os.rename(
                os.path.join(app.config['data'], secure_filename(app.config['dataset_name'])),
                os.path.join(app.config['data'], secure_filename('%s.csv' % name_of_model)))

        with open(r'rw_files/feature/%s_feature.txt' %name_of_model, 'w') as fp:
            for item in feature:
                fp.write("%s\n" % item)

        labelencode = request.form.getlist('label-encoding')

        predictlabel = request.form['predict-label']
        with open(r'rw_files/predict_label/%s_predict_label.txt' %name_of_model, 'w') as fp:
                fp.write("%s\n" % predictlabel)

        with open(r'rw_files/label_encode/%s_label_encode.txt' %name_of_model, 'w') as fp:
            for item in labelencode:
                fp.write("%s\n" % item)
        with open(r'rw_files/temp_gen_loan_name.txt', 'w') as fp:
                fp.write("%s\n" % name_of_model)
        df = pd.read_csv(os.path.join(app.config['data'],
                                   secure_filename('%s.csv' %name_of_model)))
        df = df.fillna(df.mean())
        le = LabelEncoder()
        df = df.dropna()
        diccsv = []
        import csv
        for item in labelencode:
            df[item] = le.fit_transform(df[item])
            dic = dict(zip(le.classes_[0:], np.arange(0, len(le.classes_))))
            diccsv.append(dic)


        x_feature = pd.DataFrame()
        i = 0
        for item in feature:
            x_feature.insert(i, item, df[item], allow_duplicates=False)
            i = i + 1

        for i in x_feature.columns:
            x_feature[i].fillna(int(x_feature[i].mean()), inplace=True)
        X = x_feature[feature]
        y = df["%s" % predictlabel]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        cv = KFold(n_splits=10)
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import BaggingClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.ensemble import VotingClassifier
        models = []
        #models.append(('xgb', XGBClassifier(n_estimators=150)))
        models.append(('abc', AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion="gini", max_depth=2, min_samples_leaf=5, splitter="random"),learning_rate=0.01, n_estimators=10)))
        models.append(('log', LogisticRegression(solver='lbfgs', max_iter=10000)))
        models.append(('gbc', GradientBoostingClassifier()))
        models.append(('bc', BaggingClassifier()))
        models.append(('svc', SVC(probability=True)))
        models.append(('rfc', RandomForestClassifier()))
        models.append(('extra', ExtraTreesClassifier(n_estimators=100, max_features=3)))

        model = VotingClassifier(models, voting='soft')
        model.fit(X_train, y_train)
        score = cross_val_score(model, X_test, y_test, cv=cv)
        preds = model.predict(X_test)
        #print(metrics.classification_report(y_test, preds))
        result = model.score(X_train, y_train)
        with open(r'rw_files/result/%s_result.txt' %name_of_model, 'w') as fp:
                fp.write("%s\n" % result)
        # accuracy on test data
        accuracytestdata = model.score(X_test, y_test)
        with open(r'rw_files/accuracy_test_data/%s_accuracytestdata.txt' %name_of_model, 'w') as fp:
                fp.write("%s\n" % accuracytestdata)
        import seaborn as sns
        sns.set_style('darkgrid')
        preds_train = model.predict(X_train)
        # calculate prediction probability
        prob_train = np.squeeze(model.predict_proba(X_train)[:, 1].reshape(1, -1))
        prob_test = np.squeeze(model.predict_proba(X_test)[:, 1].reshape(1, -1))
        # false positive rate, true positive rate, thresholds
        fpr1, tpr1, thresholds1 = metrics.roc_curve(y_test, prob_test, pos_label=1)
        fpr2, tpr2, thresholds2 = metrics.roc_curve(y_train, prob_train, pos_label=1)
        # auc score
        auc1 = metrics.auc(fpr1, tpr1)
        auc2 = metrics.auc(fpr2, tpr2)
        plt.figure(figsize=(8, 8))
        # plot auc
        plt.plot(fpr1, tpr1, color='blue', label='Test ROC curve area = %0.2f' % auc1)
        plt.plot(fpr2, tpr2, color='green', label='Train ROC curve area = %0.2f' % auc2)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.xlabel('False Positive Rate', size=14)
        plt.ylabel('True Positive Rate', size=14)
        plt.legend(loc='lower right')
        fig = plt.gcf()
        fig.savefig('static/plots/%splot.png' % name_of_model, dpi=1500)
        #img_name = '%splot' %name_of_model
        get_plot1 = '%splot' % name_of_model
        import pickle
        pickle.dump(model, open('saved_models/%s.pkl' %name_of_model, 'wb'))
        y_pred = model.predict(X_test)

        asbetweentestandpred = accuracy_score(y_test, y_pred) * 100
        with open(r'rw_files/asbetweentestandpred/%s_asbetweentestandpred.txt' %name_of_model, 'w') as fp:
                fp.write("%s\n" % asbetweentestandpred)
        cm = confusion_matrix(y_test, y_pred)
        with open(r'rw_files/cm/%s_cm.txt' %name_of_model, 'w') as fp:
                fp.write("%s\n" % cm)
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import f1_score

        precision = precision_score(y_test, y_pred, average='weighted')
        with open(r'rw_files/precision/%s_precision.txt' %name_of_model, 'w') as fp:
                fp.write("%s\n" % precision)
        recall = recall_score(y_test, y_pred, average='weighted')
        with open(r'rw_files/recall/%s_recall.txt' %name_of_model, 'w') as fp:
                fp.write("%s\n" % recall)
        score = f1_score(y_test, y_pred, average='weighted')
        with open(r'rw_files/score/%s_score.txt' %name_of_model, 'w') as fp:
                fp.write("%s\n" % score)

        return render_template('performancemetricsgenform.html', name_of_model=name_of_model,
                               result=result, accuracytestdata=accuracytestdata, getplot1=get_plot1,
                               asbetweentestandpred=asbetweentestandpred,
                               cm=cm, precision=precision, score=score, recall=recall
                               )





def getdic(loan_name, labelencode):
  get_dataset = os.path.join(app.config['data'], secure_filename('%s.csv' %loan_name))
  df = pd.read_csv(get_dataset)
  df = df.fillna(df.mean())
  le = LabelEncoder()
  df = df.dropna()



  diccsv = []
  import csv
  for item in labelencode:
    df[item] = le.fit_transform(df[item])
    dic = dict(zip(le.classes_[0:], np.arange(0, len(le.classes_))))
    diccsv.append(dic)

  return(diccsv)






@app.route('/preprocessing/dataadd', methods=['GET', 'POST'])
def dataadd():
    if request.method == 'POST':
        my_dataset = request.files['my_dataset']
        app.config['dataset_name_for_preprocessing'] = my_dataset.filename
        dataset_path = os.path.join(pathfordataset, secure_filename(my_dataset.filename))
        my_dataset.save(dataset_path)
        get_dataset = os.path.join(app.config['DFPr'], secure_filename(my_dataset.filename))
        df = pd.read_csv(get_dataset)

    return render_template('preprocessing.html'
                           , col=df.columns.tolist()
                           )


@app.route('/genform', methods=['GET', 'POST'])
def submitgenform():
    feature = []
    labelencode = []
    with open(r'rw_files/temp_gen_loan_name.txt', 'r') as file:
        loan_name = file.read()
    loan_name = loan_name[0:-1]
    with open(r'rw_files/predict_label/%s_predict_label.txt' %loan_name, 'r') as file:
        predict_label = file.read()
    predict_label = predict_label[0:-1]


    with open(r'rw_files/feature/%s_feature.txt' %loan_name, 'r') as fp:
        for line in fp:
            x = line[:-1]
            feature.append(x)
    with open(r'rw_files/label_encode/%s_label_encode.txt' %loan_name, 'r') as fp:
        for line in fp:
            x = line[:-1]
            labelencode.append(x)


    resfeature = [ele for ele in feature]
    for a in labelencode:
        if a in feature:
            resfeature.remove(a)

    reslabelencode = [ele for ele in labelencode]
    for a in [predict_label]:
        if a in labelencode:
            reslabelencode.remove(a)

    with open(r'rw_files/resfeature/%s_resfeature.txt.txt' % loan_name, 'w') as fp:
        for item in resfeature:
            fp.write("%s\n" % item)

    with open(r'rw_files/reslabelencode/%s_reslabelencode.txt' % loan_name, 'w') as fp:
        for item in reslabelencode:
            fp.write("%s\n" % item)


    diccsv = getdic(loan_name = loan_name, labelencode = reslabelencode)

    Func = open("templates/loans/%s.html" %loan_name, "w")
    with open('templates/ruf.html', 'r') as firstfile, open('templates/loans/%s.html' %loan_name, 'a') as secondfile:
        for line in firstfile:
            secondfile.write(line)
    return render_template("loans/%s.html"  %loan_name, feature=feature, resfeature=resfeature, reslabelencode = reslabelencode, diccsv = diccsv)

@app.route('/formlist', methods=['GET', 'POST'])
def genform():
    loan_name = request.form['genform']
    with open(r'rw_files/temp_gen_loan_name.txt', 'w') as fp:
        fp.write("%s\n" % loan_name)
    feature = []
    resfeature = []
    reslabelencode = []
    labelencode = []
    with open(r'rw_files/label_encode/%s_label_encode.txt' %loan_name, 'r') as fp:
        for line in fp:
            x = line[:-1]
            labelencode.append(x)
    with open(r'rw_files/feature/%s_feature.txt' %loan_name, 'r') as fp:
        for line in fp:
            x = line[:-1]
            feature.append(x)

    with open(r'rw_files/resfeature/%s_resfeature.txt.txt' %loan_name, 'r') as fp:
        for line in fp:
            x = line[:-1]
            resfeature.append(x)

    with open(r'rw_files/reslabelencode/%s_reslabelencode.txt' %loan_name, 'r') as fp:
        for line in fp:
            x = line[:-1]
            reslabelencode.append(x)

    diccsv = getdic(loan_name = loan_name, labelencode = reslabelencode)



    return render_template('loans/%s.html' %loan_name, feature=feature, resfeature=resfeature, reslabelencode = reslabelencode, diccsv = diccsv)

import pickle

@app.route('/predictloan', methods=['GET', 'POST'])
def predictloan():
    with open(r'rw_files/temp_gen_loan_name.txt', 'r') as file:
        loan_name = file.read()
    loan_name = loan_name[0:-1]
    model = pickle.load(open('saved_models/%s.pkl' %loan_name, 'rb'))
    final_features = [float(x) for x in request.form.values()]
    with open(r'rw_files/featureloop/%s_featureforloop.txt' % loan_name, 'w') as fp:
        for item in final_features:
            fp.write("%s\n" % item)
    #final_features = list(request.form.values())
    pred = model.predict([final_features])
    #pred = np.argmax(pred)
    if pred[0] == 1:
        return render_template('resultpredictapproved.html', pred = 'loan approved')
    else:
        return render_template('resultpredictnotapproved.html', pred='not loan approved')

@app.route('/notloop', methods=['GET', 'POST'])
def looploanamount():
    with open(r'rw_files/temp_gen_loan_name.txt', 'r') as file:
        loan_name = file.read()
    loan_name = loan_name[0:-1]
    get_dataset = os.path.join(app.config['data'], secure_filename('%s.csv' % loan_name))
    df = pd.read_csv(get_dataset)
    col = df.columns

    return render_template('getfieldnameloansug.html', col = col)

@app.route('/entersug', methods=['GET', 'POST'])
def selloanfield():
    feature = []
    featureforloop = []
    loan_amount = request.form['loan amount']
    with open(r'rw_files/temp_gen_loan_name.txt', 'r') as file:
        loan_name = file.read()
    loan_name = loan_name[0:-1]
    with open(r'rw_files/feature/%s_feature.txt' %loan_name, 'r') as fp:
        for line in fp:
            x = line[:-1]
            feature.append(x)
    with open(r'rw_files/featureloop/%s_featureforloop.txt' %loan_name, 'r') as fp:
        for line in fp:
            x = line[:-1]
            featureforloop.append(x)

    for i in range(len(feature)):
        if loan_amount == feature[i]:
            index = i

    model = pickle.load(open('saved_models/%s.pkl' %loan_name, 'rb'))

    featureforloop = [float(x) for x in featureforloop]

    oldloanamount = featureforloop[index]
    while(featureforloop[index]!=0):
        featureforloop[index] = featureforloop[index] - 1
        pred = model.predict([featureforloop])
        if pred[0] == 1:
            return render_template('suggestedloanamount.html', index = featureforloop[index])
            break
        #else:
            #return render_template('getfieldnameloansug.html', oldamount = oldloanamount)
            #continue

    return render_template('suggestedloanamount.html', index='no suggested amount')

@app.route('/rembut', methods=['GET', 'POST'])
def remloan():
    l = os.listdir('templates/loans/')
    li = [x.split('.')[0] for x in l]
    return render_template("fieldforremloan.html", li=li)

@app.route('/rembutsubmit', methods=['GET', 'POST'])
def remloanfin():
    loan_name_rem = request.form['remloan']
    os.remove('templates/loans/%s.html' %loan_name_rem)
    l = os.listdir('templates/loans/')
    li = [x.split('.')[0] for x in l]
    return render_template("predict_loans_list_buttons.html", li=li)


@app.route('/addbut', methods=['GET', 'POST'])
def addbut():
    return render_template('data.html')

@app.route('/permet', methods=['GET', 'POST'])
def permet():
    l = os.listdir('templates/loans/')
    li = [x.split('.')[0] for x in l]
    return render_template("permet_loans_list_buttons.html", li=li)

@app.route('/permetlist', methods=['GET', 'POST'])
def permetlist():
    loan_name = request.form['permet']
    with open(r'rw_files/result/%s_result.txt' %loan_name, 'r') as file:
        result = file.read()
    with open(r'rw_files/accuracy_test_data/%s_accuracytestdata.txt' %loan_name, 'r') as file:
        accuracytestdata = file.read()
    with open(r'rw_files/asbetweentestandpred/%s_asbetweentestandpred.txt' %loan_name, 'r') as file:
        asbetweentestandpred = file.read()
    with open(r'rw_files/cm/%s_cm.txt' %loan_name, 'r') as file:
        cm = file.read()
    with open(r'rw_files/precision/%s_precision.txt' %loan_name, 'r') as file:
        precision = file.read()
    with open(r'rw_files/recall/%s_recall.txt' %loan_name, 'r') as file:
        recall = file.read()
    with open(r'rw_files/score/%s_score.txt' %loan_name, 'r') as file:
        score = file.read()
    get_plot1 = '%splot' % loan_name

    return render_template('performancemetrics.html', name_of_model=loan_name,
                           result=result, accuracytestdata=accuracytestdata, getplot1=get_plot1,
                           asbetweentestandpred=asbetweentestandpred,
                           cm=cm, precision=precision, score=score, recall=recall
                           )














@app.route('/preprocessing/preprocessing', methods=['GET', 'POST'])
def uploadpreprocess():
    if request.method == 'POST':
        get_dataset = os.path.join(app.config['DFPr'],
                                   secure_filename(app.config['dataset_name_for_preprocessing']))
        feature = request.form.getlist('features')
        labelencode = request.form.getlist('labelencode')
        df = pd.read_csv(get_dataset)
        df = df.fillna(method='ffill')
        sc = StandardScaler()
        if len(feature):
            df[feature] = sc.fit_transform(df[feature])

        le = LabelEncoder()
        if len(labelencode):
            for item in labelencode:
                df[item] = le.fit_transform(df[item])

        trained_dataset = pd.DataFrame(df)

        trained_dataset.to_csv("static/data-preprocess/new/trained_dataset.csv")

        return render_template('preprocessing_output.html', data_shape=trained_dataset.shape,
                               table=trained_dataset.head(5).to_html(
                                   classes='table table-striped table-dark table-hover x'),
                               dataset_describe=trained_dataset.describe().to_html(
                                   classes='table table-striped table-dark table-hover x'))



if __name__ == '__main__':
   app.run(debug=True, host = '127.0.0.5')