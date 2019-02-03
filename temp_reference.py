# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 17:07:16 2018

@author: Home
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances

from scipy.spatial.distance import pdist
from sklearn.neighbors import DistanceMetric


df=pd.read_csv("film_data.csv",sep=',',low_memory=False)


df=df.drop(['gross','content_rating','budget','aspect_ratio','movie_imdb_link'],axis=1)

#&print(df.info())

realisateur=LabelBinarizer()
acteur=LabelBinarizer()
categorie=LabelBinarizer()



df_categorie=categorie.fit_transform(df['genres'])



''' nettoyage des colonnes utilisées pour affiner la recommandation : suppression
des majuscules et des espaces pour les noms des acteurs et du réalisateur'''

def nettoyage(colonne):
    if isinstance(colonne,list):
        return [str.lower(i.replace(" ","")) for i in colonne]
    else:
        if isinstance(colonne,str):
            return str.lower(colonne.replace(" ",""))
        else:
            return ''


df['director_name']=df['director_name'].apply(nettoyage)
df['actor_1_name']=df['actor_1_name'].apply(nettoyage)





''' tri des réalisateurs pour ne conserver que ceux qui ont fait plus de 1 film '''
freq=pd.value_counts(df.director_name.values,sort=True)
liste_conserver=freq[freq>1]
liste_nom_realisateur=pd.Series(liste_conserver.index)
masque=df.director_name.isin(liste_nom_realisateur)

df['realisateur']=df.director_name[masque]
df['realisateur']=df['realisateur'].fillna('')


''' encodage du nom des réalisateurs '''

df_realisateur=realisateur.fit_transform(df['realisateur'])


''' tri des acteurs pour ne conserver que ceux qui ont fait plus de 1 film '''

freq_acteur1=pd.value_counts(df.actor_1_name.values,sort=True)
liste_acteur=freq_acteur1[freq_acteur1>1]
liste_nom_acteur=pd.Series(liste_acteur.index)

masq_acteur=df.actor_1_name.isin(liste_nom_acteur)

df['acteur_1']=df.actor_1_name[masq_acteur]
df['acteur_1']=df['acteur_1'].fillna('')

''' encodage du nom des acteurs '''

df_acteur=acteur.fit_transform(df['acteur_1'])


''' insertion des colonnes encodées des acteurs et réalisateurs dans le dataframe de base '''

df_onehot_categorie=pd.DataFrame(df_categorie,columns=["categorie_"+str(int(i)) for i in range(df_categorie.shape[1])])

df_onehot_acteur=pd.DataFrame(df_acteur,columns=["acteur_"+str(int(i)) for i in range(df_acteur.shape[1])])

df_onehot_realisateur=pd.DataFrame(df_realisateur,columns=["realisateur_"+str(int(i)) for i in range(df_realisateur.shape[1])])


df=pd.concat([df,df_onehot_acteur],axis=1)
df=pd.concat([df,df_onehot_realisateur],axis=1)
df=pd.concat([df,df_onehot_categorie],axis=1)



''' fonction de nettoyage des titres de films : accents et caractères particuliers posent problème'''

def nettoyage_titrefilm(colonne):
    if isinstance(colonne,list):
        return [str.lower(i.replace("\xa0","")) for i in colonne]
    else:
        if isinstance(colonne, str):
            return str.lower(colonne.replace("\xa0",""))
        else:
            return ''



df['movie_title']=df['movie_title'].apply(nettoyage_titrefilm)



''' établissement d'une liste de films déterminés dont on connait parfaitement les plus proches voisins :
    on va chercher à tester les différentes distances pour établir la plus adéquats dans le problème'''

df_harrypotter=df.iloc[[9,114,115,195,199,202,206,285],:]

df_starwars=df.iloc[[4,236,237,240,1536,2051,3024,3329],:]

df_jamesbond=df.iloc[[2,12,30,149,172,252,286,717,1166,1577,1634,2774,2944,3281,3375,3465,4537],:]

df_panda=df.iloc[[139,154,184],:]

df_batman=df.iloc[[3,66,120,217,309,441,1400,4017,4457]]


df_assemble=pd.concat([df_harrypotter,df_starwars,df_jamesbond,df_panda,df_batman])

df_partie=df_assemble.iloc[:,25:]


df_assemble=df_assemble.reset_index()




''' on établit trois types de distances différentes : euclidienne, manhattan et celle appelé similarité-cosinus'''


distance_euclid=euclidean_distances(df_partie,df_partie)

distance_cosinsim=cosine_similarity(df_partie,df_partie)

distance_manhattan=manhattan_distances(df_partie,df_partie)

dist=DistanceMetric.get_metric("canberra")

distance_canberra=dist.pairwise(df_partie,df_partie)

#distance_canberra=pdist(df_partie,metric='canberra')



''' on créé trois fonctions pour tester les trois différentes "recommandations" avec les trois différentes distances'''


def proche_euclid(titre):

    indices_film=pd.Series(df_assemble.index,index=df_assemble['movie_title']).drop_duplicates()
    ind=indices_film[titre]
    
    liste_euclid=list(enumerate(distance_euclid[ind]))
    
    # tri des la liste ci-dessus en fonction de la note de similarité
    liste_euclid=sorted(liste_euclid, key=lambda x: x[1], reverse=True)
    
    # récupération des premiers films similaires ( le film à l'indice 0 est exclu puisqu'il s'agit du film indiqué)
    liste_euclid=liste_euclid[1:5]
    
    indices_film=[ i[0] for i in liste_euclid]
    
    #retour des 10 films similaires
    print(df_assemble['movie_title'].iloc[indices_film])




def proche_cosinsim(titre):
    
    indices_film=pd.Series(df_assemble.index,index=df_assemble['movie_title']).drop_duplicates()
    ind=indices_film[titre]
    
    
    liste_cosinsim=list(enumerate(distance_cosinsim[ind]))
    
    # tri des la liste ci-dessus en fonction de la note de similarité
    liste_cosinsim=sorted(liste_cosinsim, key=lambda x: x[1], reverse=True)
    
    # récupération des premiers films similaires ( le film à l'indice 0 est exclu puisqu'il s'agit du film indiqué)
    liste_cosinsim=liste_cosinsim[1:5]
    
    indices_film=[ i[0] for i in liste_cosinsim]
    
    #retour des 10 films similaires
    print(df_assemble['movie_title'].iloc[indices_film])    
    
    
    
    
def proche_manhattan(titre):
    
    indices_film=pd.Series(df_assemble.index,index=df_assemble['movie_title']).drop_duplicates()
    ind=indices_film[titre]
    
    liste_manhattan=list(enumerate(distance_manhattan[ind]))
    
    # tri des la liste ci-dessus en fonction de la note de similarité
    liste_manhattan=sorted(liste_manhattan, key=lambda x: x[1], reverse=True)
    
    # récupération des premiers films similaires ( le film à l'indice 0 est exclu puisqu'il s'agit du film indiqué)
    liste_manhattan=liste_manhattan[1:5]
    
    indices_film=[ i[0] for i in liste_manhattan]
    
    #retour des 10 films similaires
    print(df_assemble['movie_title'].iloc[indices_film])    




def proche_canberra(titre):
    
    indices_film=pd.Series(df_assemble.index,index=df_assemble['movie_title']).drop_duplicates()
    ind=indices_film[titre]
    
    liste_canberra=list(enumerate(distance_canberra[ind]))
    
    # tri des la liste ci-dessus en fonction de la note de similarité
    liste_canberra=sorted(liste_canberra, key=lambda x: x[1], reverse=True)
    
    # récupération des premiers films similaires ( le film à l'indice 0 est exclu puisqu'il s'agit du film indiqué)
    liste_canberra=liste_canberra[1:5]
    
    indices_film=[ i[0] for i in liste_canberra]
    
    #retour des 10 films similaires
    print(df_assemble['movie_title'].iloc[indices_film])



proche_euclid('kung fu panda')
proche_cosinsim('kung fu panda')
proche_manhattan('kung fu panda')   
proche_canberra('kung fu panda')