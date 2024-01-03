class Author:
    # Initialisation de l'Auteur (tous les attributs sont prives)
    def __init__(self, name,  ndoc, production):
        self.__name = name
        self.__ndoc = ndoc
        self.__production = production

    # Alimente le dictionnaire production de l'Auteur
    def add(self, document):
        self.__production[self.__ndoc] = document
        self.__ndoc += 1

    # Affiche des stats sur l'auteur (son nombre de documents et la taille moyenne de ceux-ci)
    def get_author_stats(self):
        word_counter = 0
        for doc in self.__production.values():
            word_counter += len(doc.getTexte())
        taille_moyenne = word_counter / self.__ndoc
        print("Le nombre de document de", self.__name, "est de", self.__ndoc, "et la taille moyenne de ceux-ci de",
              taille_moyenne)

    # Methode appeler quand on print l'Auteur
    def __str__(self):
        return "Le nom de l'auteur est : " + self.__name + " il a écrit " + str(self.__ndoc) + " articles."
