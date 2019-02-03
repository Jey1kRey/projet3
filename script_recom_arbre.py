# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 17:31:26 2018

@author: Jérôme
"""

""" Programme utilisant l'algorithme de classification hiérarchique ascendant
sur la base de données des films """




import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.cluster import AgglomerativeClustering
#import matplotlib.pyplot as plt
#from scipy.cluster.hierarchy import dendrogram, linkage, fcluster



df=pd.read_csv("film_data.csv",sep=',',low_memory=False)

df=df.drop(['gross','budget','aspect_ratio','movie_imdb_link'],axis=1)


''' définition des encodeurs '''

realisateur=LabelBinarizer()
acteur=LabelBinarizer()
categorie=LabelBinarizer()
keywords=LabelBinarizer()
couleur=LabelBinarizer()
contenu_note=LabelBinarizer()
pays=LabelBinarizer()

df_categorie=categorie.fit_transform(df['genres'])


"""--------------------préparation de la base de données---------------"""



''' nettoyage des colonnes utilisées pour affiner la recommandation : suppression
des majuscules et des espaces pour les noms des acteurs et du réalisateur, nettoyage
des cellules vides'''


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
df['plot_keywords']=df['plot_keywords'].apply(nettoyage)
df['color']=df['color'].apply(nettoyage)
df['content_rating']=df['content_rating'].apply(nettoyage)
df['country']=df['country'].apply(nettoyage)



df_couleur=couleur.fit_transform(df['color'])
df_cont_rat=contenu_note.fit_transform(df['content_rating'])
df_pays=pays.fit_transform(df['country'])

def nettoyage_titrefilm(colonne):
    if isinstance(colonne,list):
        return [str.lower(i.replace("\xa0","")) for i in colonne]
    else:
        if isinstance(colonne, str):
            return str.lower(colonne.replace("\xa0",""))
        else:
            return ''



df['movie_title']=df['movie_title'].apply(nettoyage_titrefilm)



''' fonction qui va permettre de définir un masque de tri pour les noms des acteurs et réalisateurs '''

def tri_colonne(colonne):
    
    freq=pd.value_counts(colonne.values,sort=True)
    liste_conserver=freq[freq>=3]
    liste_nom_conserver=pd.Series(liste_conserver.index)
    masque=colonne.isin(liste_nom_conserver)
    
    return masque


''' tri des réalisateurs et encodage'''

masque_realisateur=tri_colonne(df.director_name)
df['realisateur']=df.director_name[masque_realisateur]
df['realisateur']=df['realisateur'].fillna('')

df_realisateur=realisateur.fit_transform(df['realisateur'])
    
    
    
''' tri des acteurs_1 pour ne conserver que ceux qui ont fait plus de 3 films '''

masq_acteur=tri_colonne(df.actor_1_name)
df['acteur_1']=df.actor_1_name[masq_acteur]
df['acteur_1']=df['acteur_1'].fillna('')

df_acteur=acteur.fit_transform(df['acteur_1'])




''' tri des acteurs_2 pour ne conserver que ceux qui ont fait plus de 3 films '''

masq_acteur2=tri_colonne(df.actor_2_name)
df['acteur_2']=df.actor_2_name[masq_acteur2]
df['acteur_2']=df['acteur_2'].fillna('')

df_acteur2=acteur.fit_transform(df['acteur_2'])


''' tri des acteurs_3 pour ne conserver que ceux présents dans  3 films et plus '''

masq_acteur3=tri_colonne(df.actor_3_name)
df['acteur_3']=df.actor_3_name[masq_acteur3]
df['acteur_3']=df['acteur_3'].fillna('')

df_acteur3=acteur.fit_transform(df['acteur_3'])



''' ajout de la colonne plot keywords '''

freq_keywords=pd.value_counts(df.plot_keywords.values,sort=True)
liste_keywords=freq_keywords[freq_keywords>1]
liste_plotkeywords=pd.Series(liste_keywords.index)

masq_keywords=df.plot_keywords.isin(liste_plotkeywords)

df['motscles']=df.plot_keywords[masq_keywords]
df['motscles']=df['motscles'].fillna('')


df_keywords=keywords.fit_transform(df['motscles'])


''' vérification des tailles des matrices obtenues afin de limiter le nombre de variables à considérer'''

#print('taille de la matrice realisateur : ', df_realisateur.shape)
#print('taille de la matrice acteur 1 : ', df_acteur.shape)
#print('taille de la matrice acteur 2 : ', df_acteur2.shape)
#print('taille de la matrice acteur 3 : ', df_acteur3.shape)
#print('la taille de la matrice genre est :',df_categorie.shape)
#print('la taille de la matrice content rating est :',df_cont_rat.shape)
#print('la taille de la matrice couleur est :',df_couleur.shape)
#print('la taille de la matrice motsclés est : ', df_keywords.shape)
#print('la taille de la matrice pays est : ', df_pays.shape)



''' insertion des colonnes encodées des acteurs et réalisateurs dans le dataframe de base '''

df_onehot_categorie=pd.DataFrame(df_categorie,columns=["categorie_"+str(int(i)) for i in range(df_categorie.shape[1])])

df_onehot_acteur=pd.DataFrame(df_acteur,columns=["acteur_"+str(int(i)) for i in range(df_acteur.shape[1])])

df_onehot_realisateur=pd.DataFrame(df_realisateur,columns=["realisateur_"+str(int(i)) for i in range(df_realisateur.shape[1])])

df_onehot_keywords=pd.DataFrame(df_keywords,columns=["mots_cles_"+str(int(i)) for i in range(df_keywords.shape[1])])

df_onehot_acteur2=pd.DataFrame(df_acteur2,columns=["acteur2_"+str(int(i)) for i in range(df_acteur2.shape[1])])

df_onehot_acteur3=pd.DataFrame(df_acteur3,columns=["acteur3_"+str(int(i)) for i in range(df_acteur3.shape[1])])

df_onehot_couleur=pd.DataFrame(df_couleur,columns=["couleur_"+str(int(i)) for i in range(df_couleur.shape[1])])

#df_onehot_cont_rat=pd.DataFrame(df_cont_rat,columns=["cont_rat_"+str(int(i)) for i in range(df_cont_rat.shape[1])])
#df_onehot_pays=pd.DataFrame(df_pays,columns=["cont_rat_"+str(int(i)) for i in range(df_pays.shape[1])])


df=pd.concat([df,df_onehot_acteur],axis=1)
df=pd.concat([df,df_onehot_realisateur],axis=1)
df=pd.concat([df,df_onehot_categorie],axis=1)
df=pd.concat([df,df_onehot_keywords],axis=1)
df=pd.concat([df,df_onehot_acteur2],axis=1)
df=pd.concat([df,df_onehot_acteur3],axis=1)
df=pd.concat([df,df_onehot_couleur],axis=1)
#df=pd.concat([df,df_onehot_cont_rat],axis=1)
#df=pd.concat([df,df_onehot_pays],axis=1)


df_calcul=df.iloc[:,29:]

matrice=np.array(df_calcul)



"""------------------------- clustering --------------------------"""


''' mise en place de l'arbre hiérarchique ascendant : plusieurs tests ont été effectués sur un nombre
de clusters différents, des paramètres différents : ceux ci-dessous ont donné les meilleurs résultats'''

#partition=AgglomerativeClustering(n_clusters=650,affinity='cosine',linkage='average')
partition=AgglomerativeClustering(n_clusters=650,affinity='cosine',linkage='complete')
#partition=AgglomerativeClustering(n_clusters=600,affinity='cosine',linkage='average')
#partition=AgglomerativeClustering(n_clusters=600,affinity='cosine',linkage='complete')


partition.fit_predict(matrice)


colonne_cluster=partition.labels_

''' récupération des numéros de cluster dans une colonne et insertion dans le dataframe '''

df_cluster=pd.Series(colonne_cluster,name='label_cluster')

df=pd.concat([df,df_cluster],axis=1)


''' export de la colonne titre de film et numéro de clusters correspondant dans un fichier '''
df_film=df.iloc[:,[10,2864]]

df_export=df_film.set_index(['movie_title'])


df_export.to_csv("film_completecah.csv",sep=',')



''' test du tracé du dendrogramme 
z=sch.linkage(matrice, method='complete',metric='cosine')
plt.figure(figsize=(10, 10))  
plt.title("Dendogramme")  
dend = sch.dendrogram(z) 
'''

"""---------------------------- test ------------------------------"""


""" fonction pour les nombreux tests des différents paramètres de l'arbre hiérarchique ascendant

def recommandation(titre):

    
    indices_film=pd.Series(df.index,index=df['movie_title']).drop_duplicates()
    ind=indices_film[titre]
    
    cluster_film=pd.Series(df['label_cluster'],index=df.index)
    titre_cluster=cluster_film[ind]
        
    df_id_cluster=df[df['label_cluster']==titre_cluster]

        
    #liste_cosinsim=list(enumerate(matrice_cosin[ind]))
    
        
    # tri des la liste ci-dessus en fonction de la note de similarité
    #liste_cosinsim=sorted(liste_cosinsim, key=lambda x: x[1], reverse=True)
    
    
    # récupération des premiers films similaires ( le film à l'indice 0 est exclu puisqu'il s'agit du film indiqué)
    #liste_cosinsim=liste_cosinsim[1:20]
    
    #indices_film=[ i[0] for i in liste_cosinsim]
    
    #retour des 10 films similaires
    #films_similaire=df.iloc[indices_film]
     
    print(df_id_cluster['movie_title'].head(5))
    print(df_id_cluster['label_cluster'].head(5))

recommandation('goldeneye')
"""