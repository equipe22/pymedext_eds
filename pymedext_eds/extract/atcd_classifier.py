##
# 03/04/2020 - atcd_classifier.py - 1.2.0
# Ali Bellamine - contact@alibellamine.me
##

## Importation des librairies
import scipy
import pandas as pd
import numpy as np
from sklearn import preprocessing, metrics, pipeline, feature_extraction, tree
from sklearn.utils import validation
import os
import yaml
import glob
import pickle
import nltk
from nltk.corpus import stopwords

# Création des features - prend en entrée : la liste des features à créer, doc_references : le doc csv contenant la position de chaque document, data_dir : le dossier de travail
# Retourne data et son array features


class atcd_features_creation():
    def __init__ (self, doc_references, data_dir, features = []):
        self.features = features # liste des features à créer
        self.doc_references = pd.read_csv(doc_references)
        self.doc_references = self.doc_references.set_index("doc_id")

        self.data_dir = data_dir
        self.catalogKeys = ['%FUTURE%', '%DELTAYEAR%', '%DELTAMONTH%','%DELTAWEEK%','%DELTAMORETHANAYEAR%']
        
        customStopWords = ["d","d'","l","l'","L","L’", "’", ".", ",", "il","a"]
        #copied stopwords for access issue in shell
        with open("/export/home/cse180025/prod_information_extraction/data/terminologies/nltk_stopwords.txt","r") as h:
            stopwords = []
            for line in h.readlines():
                stopwords.append(line.strip())
                
            
        self.stopWords = set(stopwords+customStopWords)

        
    # Crée la feature : relative_loc : position relative d'un span dans le texte
    def getRelativeLocation(self, df):
        self.doc_references["fullpath"] = self.data_dir+self.doc_references["folder"].astype(str)+"/"+self.doc_references["file"]
        df = df.drop(["fullpath"], errors = "ignore", axis = 1).join(self.doc_references[["fullpath"]], on = "doc_id", rsuffix = "_sample", how = "left")
        df["span"] = df["span"].apply(lambda x: x[0])
        df["filesize"] = df["fullpath"].apply(lambda x: len(open(x).read()) if (pd.isnull(x) != True) else np.nan) # On déterminer la taille des documents
        filesize_mean = df['filesize'].mean(skipna = True)
        df["filesize"] = df["filesize"].apply(lambda x: filesize_mean if pd.isnull(x) else x) # On remplace les na par la moyenne
        df["relative_loc"] = df["span"].astype(int)/df["filesize"]
        df = df.drop(columns = ["fullpath","filesize"], axis = 1)

        return (df, [])
    
    def text_processing(self, df, text, new_text, stopWords):
        # Traitement du texte
        df[new_text] = df[text].str.replace(" {2,}"," ")
        df[new_text] = df[new_text].str.replace(" \d{1,} "," ")
        df[new_text] = df[new_text].str.replace("\'|’","")

        df[new_text] = df[new_text].apply(lambda x: ' '.join([word for word in x.split() if word not in stopWords])) # Suppression des stop words

        # Traitement du texte
        df[new_text] = df[new_text].str.replace(" {2,}"," ")

        return(df)

    def transformDate (self, df):
        # Calcul de l'écart entre les dates
        day = 24*3600*1000000000
        week = 7*day
        month = 30*day
        year = 364*day

        df["delta_date"] = pd.to_datetime(df["date"])-pd.to_datetime(df["note_datetime"]*1000000)
        df["delta_str"] = "%DELTAMORETHANAYEAR%"
        df.loc[df["delta_date"] >= pd.to_timedelta(-year),"delta_str"] = "%DELTAYEAR%"
        df.loc[df["delta_date"] >= pd.to_timedelta(-month),"delta_str"] = "%DELTAMONTH%"
        df.loc[df["delta_date"] >= pd.to_timedelta(-week),"delta_str"] = "%DELTAWEEK%"
        df.loc[df["delta_date"] > pd.to_timedelta(0),"delta_str"] = "%FUTURE%"

        return(df)
    
    def countTimeOccurence(self, x):
        mt = self.metadata.loc[x["doc_id"],"metadata"]

        countOccurence = (mt.loc[(mt["note_nlp_source_value"] == "DATE") &\
                                (mt["offset_begin"] >= x["span_sentence"][0]) &\
                                (mt["offset_end"] <= x["span_sentence"][1]), "delta_str"]
                          .value_counts()
                         )
        count = pd.DataFrame(np.zeros((1, len(self.catalogKeys))), dtype = int, columns = self.catalogKeys)

        if(countOccurence.shape[0] != 0):
            count[countOccurence.index] += countOccurence

        return(count.loc[0].tolist())
    
    """
        Utilisé pour l'entrainement - lorsque span_sentence est absent - Utile pour le vieu jeu d'entrainement
    """
    def countTimeOccurenceWithoutSpanSentence (self, x):
        catalog = {i:0 for i in self.catalogKeys}

        if (x["doc_id"] in self.metadata.index):
            mt = self.metadata["metadata"][x["doc_id"]]
            temp_str = x["sent_text"]
            temp_str = re.sub("\xa0", "", temp_str)
            temp_str = re.sub(" {2,}", " ", temp_str)

            mt["snippet_len"] = mt["snippet"].dropna().apply(len)
            for values in (mt[mt["note_nlp_source_value"] == "DATE"]
                           .dropna()[["snippet", "delta_str","snippet_len"]]
                           .sort_values(by = "snippet_len", ascending = False)
                           .values):
                values[0] = re.sub("/", " ", values[0])
                values[0] = re.sub("\xa0", "", values[0])
                values[0] = re.sub(" {2,}", " ", values[0])

                temp_str_2 = re.sub(values[0],values[1],temp_str)

                if (temp_str != temp_str_2):
                    catalog[values[1]] += 1
                    temp_str = temp_str_2

        return(list(catalog.values()))

    # Crée la feature sent_text_datechange : modifie les dates de sent_text par des dates relatives
    def writeHeidelTime(self, df):
        self.doc_references["fullpath"] = self.data_dir+self.doc_references["folder"].astype(str)+"/metadata/covid_"+self.doc_references.index.astype(str)+".json" # Localisation des json
        df = df.drop(["fullpath"], errors = "ignore", axis = 1).join(self.doc_references[["fullpath"]], on = "doc_id", rsuffix = "_sample", how = "left") # Ajout de la localisation des json au df

        # On charge le contenu des json dans metadata
        self.metadata = (df[["doc_id","fullpath"]]
                         .dropna(subset = ["fullpath"])
                         .drop_duplicates(subset = ["doc_id"])
                         .apply(lambda x: [x["doc_id"], pd.read_json(x["fullpath"])], axis = 1, result_type = "expand")
                         .rename(columns = {0: "doc_id", 1: "metadata"})
                         .set_index("doc_id")
                        ) # On charge le contenu des json dans metadata
        
        self.metadata["metadata"] = self.metadata["metadata"].apply(self.transformDate) # On transforme les date en str
        
        # On calcule pour chaque texte le nombre de chaque occurence
        if ("span_sentence" in df.columns):
            df = (df.join(df.apply(self.countTimeOccurence, axis = 1, result_type = "expand")
                          .rename(columns = {0:"%FUTURE%",1:"%DELTAYEAR%",2:"%DELTAMONTH%",3:"%DELTAWEEK%",4:"%DELTAMORETHANAYEAR%"})
                         )
                 )
        else:
            df = df.join(df.apply(self.countTimeOccurenceWithoutSpanSentence, axis = 1, result_type = "expand")
                         .rename(columns = {0:"%FUTURE%",1:"%DELTAYEAR%",2:"%DELTAMONTH%",3:"%DELTAWEEK%",4:"%DELTAMORETHANAYEAR%"}))
            
        df = df.drop(["fullpath"], errors = "ignore", axis = 1)
        
        return (df, self.catalogKeys)

    def fit (self, X, y = None):
        return self
    
    def transform (self, X):
        
        initial_features = [feature for feature in X.columns.values if feature not in self.features] # Permet de déterminer ce qui a été ajouté

        for feature in self.features:
            
            if (feature == "relative_loc"):
                (X, new_feature) = self.getRelativeLocation(X)
            elif (feature == 'sent_text'):
                (X, new_feature) = self.writeHeidelTime(X)
                
                # Traitement du texte
                X = self.text_processing(X, feature, feature, self.stopWords)
                
                self.features = self.features + new_feature # Ajoute les nouvelle features
                
        return (X, self.features)
    
    def fit_transform(self, X, y = None):
        return self.transform(X)
    
# Modèle - prend en entrée un DF contenant du texte, la variable textName précise la colonne du DF contenant le texte (pour réalisation d'un BOW) et diverses features au choix (contenu dans le tableau features)

class atcd_classifier():
    
    def __init__ (self, textName, features = []):
        
        self.text = textName
        
        self.features = features
        
        self.bow_model = pipeline.Pipeline([
            ("vect", feature_extraction.text.CountVectorizer(min_df = 10, ngram_range = (1,3))),
            ("tfidf", feature_extraction.text.TfidfTransformer())
        ]);
        
        self.lb = {}
        
        self.classifier = tree.DecisionTreeClassifier(max_depth = 10)
        self.fitted = False
        
    def preprocessing (self, X, train = False):
        
        temp_X = pd.DataFrame(X).reset_index()
        new_features = []
        
        if (train):
            tmpSpM = self.bow_model.fit_transform(temp_X[self.text]) # BOW
            
            for feature in self.features: # Intégration des autres features
                if (feature == self.text): # On intégre pas le texte une deuxième fois
                    continue

                if (temp_X.dtypes[feature] == object): # Pour les variables qualitative : on binarise la variable
                    temp_X[feature] = temp_X[feature].astype(str) # Prévient les mauvais types
                    self.lb[feature] = preprocessing.LabelBinarizer()
                    self.lb[feature].fit(temp_X[feature])
                    classes = np.core.defchararray.add(np.repeat(feature+'_',len(self.lb[feature].classes_)), self.lb[feature].classes_)
                    
                    new_features.append(pd.DataFrame(self.lb[feature].transform(temp_X[feature]), columns = classes))
                else: # Les autres features
                    new_features.append(temp_X[feature])                    
        else:
            tmpSpM = self.bow_model.transform(pd.DataFrame(X)[self.text])  # BOW
                
            for feature in self.features: # Intégration des autres features
                if (feature == self.text): # On intégre pas le texte une deuxième fois
                    continue
                
                if (temp_X.dtypes[feature] == object): # Pour les variables qualitative : on binarise la variable
                    temp_X[feature] = temp_X[feature].astype(str) # Prévient les mauvais types
                    classes = np.core.defchararray.add(np.repeat(feature+'_',len(self.lb[feature].classes_)), self.lb[feature].classes_)
                    
                    new_features.append(pd.DataFrame(self.lb[feature].transform(temp_X[feature]), columns = classes))
                else: # Les autres features
                    new_features.append(temp_X[feature])
                    
        # On ajoute les features au tmpSpM
        if(len(new_features) > 0):
            new_feature_SpM = scipy.sparse.csr_matrix(pd.concat(new_features, axis = 1))
            SpM = scipy.sparse.hstack((tmpSpM, new_feature_SpM))    
        else:
            SpM = tmpSpM

        return(SpM)
     
    def save_model(self, path):
        if(self.fitted):
            pickle.dump([self.classifier,self.bow_model, self.lb], open(path, "wb"))
        else:
            raise ValueError('Modèle non entrainé.')
        
        return self
        
    def load_model(self, path):
        if (os.path.isfile(path)):
            model = pickle.load(open(path, "rb"))
            self.classifier = model[0]
            self.bow_model = model[1]
            self.lb = model[2]
            self.fitted = True
        else:
            raise ValueError('Fichier de modèle inexistant.')
            
        return self
   
    def fit(self, X, y):
        X = self.preprocessing(X,True)
        self.classifier.fit(X, y)
        self.fitted = True

        return self
        
    def predict(self, X):
        
        X = self.preprocessing(X)
        y = self.classifier.predict(X)
        
        return(y)
    
    def predict_proba(self, X):
        
        X = self.preprocessing(X)
        y = self.classifier.predict_proba(X)
        
        return(y)

## Fonctions d'inférence et d'entrainement
# L'array data doit contenir sent_text, span, section_type et doc_id

# Data should be a DF containing sent_text, doc_references : localisation du document csv contenant la localisation des tous les fichiers txt, data_dir : dossier contenant les documents de travail, model_dump : pickle contenant le classifier entrainé
def infer_atcd(data, doc_references, data_dir, model_dump):
    """Data should be a DF containing sent_text, doc_references : localisation du document csv contenant la localisation des tous les fichiers txt, data_dir : dossier contenant les documents de travail, model_dump : pickle contenant le classifier entrainé"""
    text = "sent_text"
    features = ["section_type", "relative_loc"]
    features.append(text)
    
    data, features = atcd_features_creation(doc_references, data_dir, features).transform(data) # Création des features
    model = atcd_classifier(text, features).load_model(model_dump)
    
    y_hat = model.predict(data)
    
    return (y_hat)

# Data should be a DF containing sent_text, doc_references : localisation du document csv contenant la localisation des tous les fichiers txt, data_dir : dossier contenant les documents de travail, training csv, model_dump : pickle contenant le classifier entrainé
# Annotation is a panda df with doc_uid and atcd value 0 or 1
# Data should contains a doc_uid
def train_atcd(data, doc_references, data_dir, annotations, model_dump):
    
    text = "sent_text"
    features = ["section_type", "relative_loc"]
    features.append(text)
    
    data, features = atcd_features_creation(doc_references, data_dir, features).transform(data) # Création des features
    model = atcd_classifier(text, features)
    
    model.fit(data[features], data[annotations])
    model.save_model(model_dump)
        
    return("File save in {}".format(model_dump))


def infer_atcd_proba(data, doc_references, data_dir, model_dump):
    """Data should be a DF containing sent_text span sentence_span doc_id, doc_references : localisation du document csv contenant la localisation des tous les fichiers txt, data_dir : dossier contenant les documents de travail, model_dump : pickle contenant le classifier entrainé"""
    text = "sent_text"
    features = ["section_type", "relative_loc"]
    features.append(text)
    
    data, features = atcd_features_creation(doc_references, data_dir, features).transform(data) # Création des features
    model = atcd_classifier(text, features).load_model(model_dump)
    y_hat = model.predict_proba(data)
    
    return (y_hat)

def infer_atcd(data, doc_references, data_dir, model_dump):
    """Data should be a DF containing sent_text span sentence_span doc_id, doc_references : localisation du document csv contenant la localisation des tous les fichiers txt, data_dir : dossier contenant les documents de travail, model_dump : pickle contenant le classifier entrainé"""
    text = "sent_text"
    features = ["section_type", "relative_loc"]
    features.append(text)

    data, features = atcd_features_creation(doc_references, data_dir, features).transform(data) # Création des features
    model = atcd_classifier(text, features).load_model(model_dump)
    y_hat = model.predict(data)
    
    return (y_hat)