# Test de la creation d'un Document
import Document
from datetime import datetime

docReddit = Document.RedditDocument("DocReddit", "Arthur", datetime.now(), "URL",
                                  "Blablabla")

docArxiv = Document.ArxivDocument("DocArxiv", "Cyrielle", datetime.now(), "URL",
                                  "Blobloblo")

# Test de la fonction setNombreCommentaires
docReddit.setNombreCommentaires(9)
print(docReddit.getNombreCommentaires())

# Test de la fonction setCoauteurs
coauteurs = ["Parmentier", "Barailler"]
docArxiv.setCoauteurs(coauteurs)
print(docArxiv.getCoauteurs())

# Test de la creation d'un Auteur
import Author

authReddit = Author.Author("Arthur", 0, {})
authArxiv = Author.Author("Cyrielle", 0, {})

# Test de la fonction add
authReddit.add(docReddit)
authReddit.add(docArxiv)
authArxiv.add(docArxiv)

# Test de la fonction get_author_stats
authReddit.get_author_stats()
authArxiv.get_author_stats()

# Test de creation d'un corpus
import Corpus

id2auth = dict()
id2doc = dict()

corpus = Corpus.Corpus("CorpusTest", id2auth, id2doc, len(id2doc), len(id2auth))
print(corpus)
# Test de la méthode load
corpus = corpus.load("corpus.pkl")
print(corpus)
# Test de la méthode trie_date
corpus.trie_date(3)
# Test de la méthode trie_titre
corpus.trie_titre(3)
# Test de la fonction search
corpus.search("machine")
# Test de la fonction concorde
corpus.concorde("machine", 10)

# Test de la fonction nettoyer_texte
chaine = corpus.nettoyer_texte("J'aime mangé des Br@$ !")
print(chaine)

# Test de la fonction stats
corpus.stats(10)

# Test de la fonction remplir voc
corpus.remplir_voc()
print(corpus.getVocab())

# Test de la fonction matrix_tf
matrix_tf = corpus.matrix_tf()
# Test de la fonction matrix_tfidf
matrix_tfidf = corpus.matrix_tfidf(matrix_tf)

# Test de la fonction vector_transformation
vector = corpus.vector_transformation("machine")
# Test de la fonction cosine_similarity
corpus.cosine_similarity(vector, matrix_tfidf)

# Test de la fonction presence_doc
print(corpus.presence_doc("machine"))
print(len(corpus.presence_doc("machine")))
