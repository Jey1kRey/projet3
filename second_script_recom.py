# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 11:02:49 2018

@author: Home
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity


df=pd.read_csv("film_data.csv",sep=',',low_memory=False)


''' ce programme va effectuer une recommandation en comparant le film présenté
à des films similaires, tant en terme de genre que de catégories ( plot_keywords)'''


decompte=CountVectorizer(stop_words='english')

# on remplace les NaN par des espaces vides

df['plot_keywords']=df['plot_keywords'].fillna('')

matrice=decompte.fit_transform(df['plot_keywords'])



# on va récupérer une distance similarité cosinus
# calcul de la matrice similarité cosinus

cosinsim=cosine_similarity(matrice,matrice)


# on récupère les indices pour chaque nom de films pour pouvoir les identifier et les rechercher facilement
'''
liste=[]
for element in df['movie_title']:
    liste.append(element)
'''


indices_film=pd.Series(df.index,index=df['movie_title']).drop_duplicates()


#print(indices_film)

#print(indices_film.index)

#print(indices_film.loc['Spectre\xa0'])



''' fonction recommandation : '''




def recommandation(titre, sim_cosin=cosinsim):
    
    # récupération de l'index du titre en paramètre
    indices_film=pd.Series(df.index,index=df['movie_title']).drop_duplicates()
    
    ind=indices_film[titre]
    
    # récupération de la comparaison entre ce film et les films dans la base de données
    liste_similarite=list(enumerate(sim_cosin[ind]))
    
    # tri des la liste ci-dessus en fonction de la note de similarité
    liste_similarite=sorted(liste_similarite, key=lambda x: x[1], reverse=True)
    
    # récupération des premiers films similaires
    liste_similarite=liste_similarite[1:11]
    
    # récupération des indices des films
    indices_film=[ i[0] for i in liste_similarite]
    
    return df['movie_title'].iloc[indices_film]


print(recommandation('Avatar\xa0'))
