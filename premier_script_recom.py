# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 09:25:28 2018

@author: Home
"""


import pandas as pd
import numpy as np




df=pd.read_csv("film_data.csv",sep=',',low_memory=False)


''' ce programme va chercher à fournir des recommandations basées sur les films
les plus populaires : pour cela on va établir la liste des 250 premiers 
films populaires ( 5% de la base de données : peut enuite être réduit encore)'''

# définition d'une fonction de pondération : elle permet de calculer
# la véritable 'valeur' du film selon la note fonction du nbr de votes

def note_ponderee(matrice,nbrvotes,moy_notes):
    nb_vote=matrice['num_voted_users']
    moy=matrice['imdb_score']
    
    formule=(nb_vote/(nb_vote+nbrvotes) * moy) + ( nbrvotes/(nbrvotes+nb_vote) * moy_notes)
    
    return formule




# calcul de la moyenne des notes attribuées

moyenne_note=df['imdb_score'].mean()


# calcul du nombre de votes nécessaires pour faire partie des 25% meilleurs films

nbr_votes=df['num_voted_users'].quantile(0.75)


# on range les potentiels meilleurs films dans un nouveau dataframe


film_prem=df.copy().loc[df['num_voted_users']>=nbr_votes]


# on retourne une liste de films à partir de la liste ci-dessus sur laquelle on a appliquée la note pondérée

film_prem['note']=note_ponderee(film_prem,nbr_votes,moyenne_note)

# on trie cette liste suivant la colonne 'note' créée, puis on affiche les 20 premiers films du classement

film_prem=film_prem.sort_values('note',ascending=False)

print(film_prem[['movie_title','num_voted_users','imdb_score','note']].head(20))



