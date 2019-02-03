# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 11:23:40 2018

@author: Home
"""

""" Programme de création de l'API : on définit une page web sur laquelle
l'utilisateur pourra interagir en inscrivant un titre de film : par la 
méthode des K plus proches voisins, on lui retourne 5 films similaires. """




from flask import Flask, request, render_template, url_for
import pandas as pd
import numpy as np


df=pd.read_csv("film_kmean.csv",sep=',',engine='python')

''' récupération de la matrice des distances calculée dans le programme 
script_reco_kppv'''

matrice_distancekppv=np.load("matrice_cosinus_lt.npy")



app = Flask(__name__)

''' options '''

app.config.from_object('config')

''' page d'accueil '''

@app.route('/')
def home():
    return render_template("page_recom.html")

''' fonction qui va récupérer le titre du film inscrit par l'utilisateur,
et retourné la même page que précédemment, avec les titres similaires, renvoyés
par la méthode des plus proches voisins'''


@app.route('/', methods=['POST'])
def text_box():
    text = request.form['text']
    modif_text = text.lower()
    
    indices_film=pd.Series(df.index,index=df['movie_title']).drop_duplicates()
    ind=indices_film[modif_text]
 
    liste_cosinsim=list(enumerate(matrice_distancekppv[ind]))    
    
    # tri des la liste ci-dessus en fonction de la note de similarité
    liste_cosinsim=sorted(liste_cosinsim, key=lambda x: x[1], reverse=True)
    
    # récupération des premiers films similaires ( le film à l'indice 0 est exclu puisqu'il s'agit du film indiqué)
    liste_cosinsim=liste_cosinsim[1:6]
    indices_film=[ i[0] for i in liste_cosinsim]
    
    #retour des films similaires
    films_similaire=df.iloc[indices_film]    
    
    liste=[]
    for x in films_similaire['movie_title']:
        liste.append(x)
        
  
    return render_template("page_recom.html" , premier_film=liste[0],deuxieme_film=liste[1],troisieme_film=liste[2],quatrieme_film=liste[3],cinquieme_film=liste[4])


if __name__ == '__main__':
    app.run()
    
    
   
    
