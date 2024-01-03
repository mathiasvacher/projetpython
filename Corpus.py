import pickle
import pandas as pd
import re
import numpy as np
from numpy.linalg import norm

''' IL EST DECONSEILLE D'UTILISER LE SINGLETON
def singleton(cls):
    instance = [None]

    def wrapper(*args, **kwargs):
        if instance[0] is None:
            instance[0] = cls(*args, **kwargs)
        return instance[0]

    return wrapper


@singleton
'''


class Corpus:
    # Initialisation du Corpus (tous les attributs sont prives)
    def __init__(self, nom, authors, id2doc, ndoc, naut):
        self.__nom = nom
        self.__authors = authors
        self.__id2doc = id2doc
        self.__ndoc = ndoc
        self.__naut = naut
        self.__chaine = ""
        self.__vocab = dict()

    def getNom(self):
        return self.__nom

    def getAuthors(self):
        return self.__authors

    def getId2Doc(self):
        return self.__id2doc

    def getNdoc(self):
        return self.__ndoc

    def getNaut(self):
        return self.__naut

    def getVocab(self):
        return self.__vocab

    # Trie le corpus selon la date de ses documents et affiche les ndoc premiers
    def trie_date(self, ndoc):
        dates_dico = list()
        for doc in self.__id2doc.values():
            dates_dico.append(doc.getDate().date())
        dates_dico.sort(reverse=True)
        for i in range(ndoc):
            print(dates_dico[i])

    # Trie le corpus selon le titre de ses documents et affiche les ndoc premiers
    def trie_titre(self, ndoc):
        titres_dico = list()
        for doc in self.__id2doc.values():
            titres_dico.append(doc.getTitre())
        titres_dico.sort()
        for i in range(ndoc):
            print(titres_dico[i])

    # Sauvegarde le corpus dans un fichier pickle
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    # Charge le corpus depuis un fichier pickle
    def load(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    # Retourne les passages du mot cle dans les documents
    def search(self, motcle, window=10):
        # On regarde si la chaine existe deja pour ce corpus
        if self.__chaine == "":
            # Si non on créé la chaine
            list_chaine = list()
            for doc in self.__id2doc.values():
                list_chaine.append(doc.getTexte())
            self.__chaine = " ".join(list_chaine)
        # On recupere les valeurs des positions
        find = re.finditer(motcle, self.__chaine)
        for f in find:
            # On affiche le passage à +- 10 caracteres
            print(self.__chaine[f.start() - window:f.end() + window])
        return self.__chaine[f.start() - window:f.end() + window]

    # Retourne un tableau avec le motcle et les contextes gauches et droites (mots autour)
    def concorde(self, motcle, window):
        contexte_gauche = list()
        motif = list()
        contexte_droit = list()
        # On regarde si la chaine existe deja sur ce Corpus
        if self.__chaine == "":
            # Si non on créé la chaine
            list_chaine = list()
            for doc in self.__id2doc.values():
                list_chaine.append(doc.getTexte())
            self.__chaine = " ".join(list_chaine)
        # On recupere les valeurs des positions
        find = re.finditer(motcle, self.__chaine)
        for f in find:
            # On ajoute le contexte gauche
            contexte_gauche.append(self.__chaine[f.start() - window: f.start()])
            # On ajoute le mot cle
            motif.append(f.group(0))
            # On ajoute le contexte droit
            contexte_droit.append(self.__chaine[f.end(): f.end() + window])
        # On en fait une liste
        data = {"context gauche": contexte_gauche, "motif trouvé": motif, "contexte droit": contexte_droit}
        # Puis un DataFrame
        df = pd.DataFrame(data)
        print(df)
        return df
    
    # Méthode pour récupérer les statistiques spécifiques au mot
    def get_word_statistics(self, word):
        occurrences = 0
        doc_frequency = 0

        for doc in self.__id2doc.values():
            # On récupère le texte du document
            texte = doc.getTexte()
            # On le nettoie
            cleaned_texte = self.nettoyer_texte(texte)
            # On le découpe en mots
            mots = re.split("[ \t,;\.]", cleaned_texte)

            if word in mots:
                occurrences += mots.count(word)
                doc_frequency += 1
        return {
            'Mot': word,
            'Occurrences totales': occurrences,
            'Nombre de documents contenant ce mot (Document Frequency)': doc_frequency
        }

    # Nettoie la chaine en parametre
    def nettoyer_texte(self, chaine):
        # Mise en minuscule
        chaine = chaine.lower()
        # Suppression des retours a la ligne
        chaine = chaine.replace("\n", "")
        # Suppression des chiffres
        chaine = re.sub("[0-9]", "", chaine)
        # Suppression de la ponctuation
        chaine = re.sub(r'[^\w\s]', '', chaine)
        return chaine

    # Affichage de statistiques (nombre de mots dans le corps, mots les plus fréquents)
    def stats(self, n):
        set_voc = set()  # Un set pour le vocabulaire sans doublon
        list_voc = list()  # Une liste pour les doublons
        vocabulary = dict()
        for doc in self.__id2doc.values():
            # On recupere le texte de chaque document
            texte = doc.getTexte()
            # On le nettoie
            cleaned_texte = self.nettoyer_texte(texte)
            # On le decoupe en mots
            mots = re.split("[ \t,;\.]", cleaned_texte)
            for mot in mots:
                # On ajoute les mots dans le vocabulaire
                set_voc.add(mot)
                # On ajoute les mots dans la liste
                list_voc.append(mot)
        # On retire les mots vides
        set_voc.remove("")
        i = 0  # id des mots
        for mot_unique in set_voc:
            vocabulary[mot_unique] = i
            i += 1
        print("Il y a " + str(len(vocabulary)) + " mots dans le dictionnaire et le corpus.")
        term_df = pd.DataFrame(list_voc, columns=["mots"])
        print("Les " + str(n) + " mots les plus fréquents du corpus sont : ")
        freq = term_df["mots"].value_counts()
        print(freq.head(n))

    # Remplir le vocabulaire du corpus puis calcule les matrices TF et TF-IDF Documents x Mots
    def remplir_voc(self):
        set_voc = set()  # Un set pour le vocabulaire sans doublon
        list_voc = list()  # Une liste pour les doublons
        dic_voc = dict()  # Un dictionnaire pour document frequency

        for doc in self.__id2doc.values():
            doc_done = list()  # On teste si on deja vu le mot dans ce document
            doc_voc = dict()  # Un dictionnaire pour la matrice
            # On recupere le texte de chaque document
            texte = doc.getTexte()
            # On le nettoie
            cleaned_texte = self.nettoyer_texte(texte)
            # On le decoupe en mots
            mots = re.split("[ \t,;\.]", cleaned_texte)
            for mot in mots:
                # On ajoute les mots dans le vocabulaire
                set_voc.add(mot)
                # On ajoute les mots dans la liste
                list_voc.append(mot)

                # On ajoute les mots dans le dictionnaire avec leur frequence
                if mot in doc_done:
                    pass
                else:
                    doc_done.append(mot)
                    if dic_voc.get(mot):
                        dic_voc[mot] = dic_voc[mot] + 1
                    else:
                        dic_voc[mot] = 1

        # On retire les mots vides
        set_voc.remove("")
        i = 0  # id des mots
        term_df = pd.DataFrame(list_voc, columns=["mots"])
        for mot_unique in set_voc:
            dictionnaire = {"id": i, "nombre d'occurences totales": term_df['mots'].value_counts()[mot_unique],
                            "nombre de documents contenant ce mot (document frequency)": dic_voc[mot_unique]}
            self.__vocab[mot_unique] = dictionnaire
            i += 1

        # vocabulaire avec le nombre d'occurence totales
        print(self.__vocab)

    # Creer la matrice tf du corpus
    def matrix_tf(self):
        # DataFrame de taille document * nombre mots (equivalent de mat_TF)
        df_tf = pd.DataFrame(np.zeros((self.__ndoc, len(self.__vocab))), columns=list(self.__vocab))

        # Indice pour la matrice
        index_tf = 0
        for doc_actual in self.__id2doc.values():
            # On récupere le texte
            texte = doc_actual.getTexte()
            # On le nettoie
            cleaned_texte = self.nettoyer_texte(texte)
            # On le decoupe en mots
            mots = re.split("[ \t,;\.]", cleaned_texte)
            for mot in mots:
                if mot == '':
                    continue
                # On calcule le term frequency du mot
                df_tf[mot][index_tf] = df_tf[mot][index_tf] + (1 / len(mots))
            index_tf += 1

        print(df_tf)
        return df_tf

    # Creer la matrice tfidf du corpus
    def matrix_tfidf(self, matrix_tf):
        idf = dict()

        for mot in self.__vocab:
            doc_contains = 0
            # test = dic_voc[mot]
            for doc_actual in self.__id2doc.values():
                if mot in doc_actual.getTexte().split():
                    doc_contains += 1

            idf[mot] = np.log10(self.__ndoc / (doc_contains + 1))

        # print(idf)

        df_tf_idf = matrix_tf.copy()

        for mot in self.__vocab:
            for i in range(self.__ndoc):
                df_tf_idf[mot][i] = matrix_tf[mot][i] * idf[mot]

        print(df_tf_idf)
        return df_tf_idf

    # Calcul le vecteur TF-IDF du mot clé par rapport aux Documents du Corpus
    def vector_transformation(self, mot_cle):
        tf_list = list()  # vecteur tf pour le motcle
        doc_contains = 0  # documents qui contiennent le mot
        idf = list()  # vecteur idf pour le motcle
        tf_idf = [None] * self.__ndoc  # liste pour le vecteur final

        for doc in self.__id2doc.values():
            tf = 0  # term frequency du document
            # On récupere le texte
            texte = doc.getTexte()
            # On le nettoie
            cleaned_texte = self.nettoyer_texte(texte)
            # On le decoupe en mots
            mots = re.split("[ \t,;\.]", cleaned_texte)

            for mot in mots:
                if mot == mot_cle:
                    # On calcule le nombre de fois que le mot apparait
                    tf += 1

            # On calcule le term frequency du document
            tf_list.append(tf / len(mots))

            # Si le mot est dans le doc on augment son df
            if mot_cle in doc.getTexte().split():
                doc_contains += 1

            # On peut alors calculer le idf
            idf.append(np.log10(self.__ndoc / (doc_contains + 1)))

        tf_list.copy()

        # On peut alors calculer le tf_idf de chaque document
        for i in range(self.__ndoc):
            tf_idf[i] = tf_list[i] * idf[i]

        print(tf_idf)
        return tf_idf

    # Calcule la similarité de Cosinus d'un vecteur et de la matrice TF-IDF du Corpus
    def cosine_similarity(self, vector, df_tf_id):
        # On transpose la matrice tf_idf pour pouvoir faire le produit
        df_tfidT = df_tf_id.transpose()
        # On le transforme en numpy array
        tf_idf = df_tfidT.to_numpy()
        # Pareil pour le vector
        vector_cle = np.asarray(vector)
        # On le transpose pour fitter avec la matrice
        vector_cleT = vector_cle.transpose()

        # compute cosine similarity
        cosine = np.dot(tf_idf, vector_cleT) / (norm(tf_idf, axis=1) * norm(vector_cleT))
        print("Cosine Similarity:\n", cosine)

        # trier les scores de similarité
        similarity_scores = sorted(cosine, reverse=True)

        # afficher les meilleurs résultats (les 10 premiers)
        print(similarity_scores[:10])

    # Retourne les documents ou sont present le mot
    def presence_doc(self, motcle):
        doc_present = list()  # listes des documents
        for doc in self.__id2doc.values():
            # texte propre
            texte = self.nettoyer_texte(doc.getTexte())
            # Si le mot cle est dans le texte
            if motcle in texte:
                # alors on l'ajoute
                doc_present.append(doc.getTitre())
        return doc_present

    # Methode appeler quand on print le Corpus
    def __repr__(self):
        return "Le nombre de document du corpus " + self.__nom + " est de " + str(self.__ndoc)
