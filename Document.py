class Document:
    # Initialisation du Document (tous les attributs sont prives)
    def __init__(self, titre, auteur, date, url, texte="vide"):
        self.__titre = titre
        self.__auteur = auteur
        self.__date = date
        self.__url = url
        self.__texte = texte
        self.__list_mots = list()

    def getDate(self):
        return self.__date

    def getTitre(self):
        return self.__titre

    def getTexte(self):
        return self.__texte

    def getType(self):
        pass

    # Affichage de toutes les informations du Document
    def affichage(self):
        print("Titre: " + self.__titre)
        print("Auteur: " + self.__auteur)
        print("Date: " + self.__date)
        print("URL: " + self.__url)
        print("Texte: " + self.__texte)

    # Methode appeler quand on print le Document
    def __str__(self):
        return "Le titre du doc est : " + self.__titre


class RedditDocument(Document):
    # Initialisation du RedditDocument avec le mot cle super
    def __init__(self, titre, auteur, date, url, texte):
        super().__init__(titre=titre, auteur=auteur, date=date, url=url, texte=texte)
        self.__nombre_commentaires = 0

    def getNombreCommentaires(self):
        return self.__nombre_commentaires

    def setNombreCommentaires(self, nombre_commentaires):
        self.__nombre_commentaires = nombre_commentaires

    def getType(self):
        return "Document Reddit"

    def __str__(self):
        return "Le titre du doc Reddit est : " + self.__titre + " et son nombre de commentaire est de : " + str(
            self.__nombreCommentaires)


class ArxivDocument(Document):
    # Initialisation du ArxivDocument avec le mot cle super
    def __init__(self, titre, auteur, date, url, texte):
        super().__init__(titre=titre, auteur=auteur, date=date, url=url, texte=texte)
        self.__coauteurs = list()

    def getCoauteurs(self):
        return self.__coauteurs

    def setCoauteurs(self, coauteurs):
        self.__coauteurs = coauteurs

    def getType(self):
        return "Document Arxiv"

    def __str__(self):
        return "Le titre du doc Arxiv est : " + self.__titre + " et sa liste de coauteurs : " + self.__coauteurs
    

class DocumentGenerator:
    @staticmethod
    def factory(type, titre, auteur, date, url, texte):
        if type == "Document Reddit":
            return RedditDocument(titre, auteur, date, url, texte)
        if type == "Document Arxiv":
            return ArxivDocument(titre, auteur, date, url, texte)

        assert 0, "Erreur : " + type
