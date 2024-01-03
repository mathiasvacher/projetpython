""" IMPORT INITULE MAINTENANT QU'ON UTILISE LE CORPUS.PKL
import praw
import urllib
import xmltodict
import Document
import Author
from datetime import datetime
"""
import pandas as pd
import Corpus
from dash import Dash, html, dcc, dash_table
from dash.dependencies import Input, Output, State

""" CE CODE A SERVI POUR L'AVANCEMENT DES TPS 1 à 4
docs = list()
dico = dict()
list_reddit = list()
list_arxiv = list()

# Dictionnaire contenant des documents
id2doc = dict()
# Indice permettant d'indexe le dictionnaire
indice_id2doc = 0

# Dictionnaire contenant des auteurs
id2auth = dict()

# REDDIT
reddit = praw.Reddit(client_id='7sSBv66krDGjQFKjBp33kg', client_secret='HV57BmGVkSgO1XUX9sQx34qGWZIJYA', user_agent='romane_touzain')


# On recupere les 10 hot posts du subreddit MachineLearning
sub_ml = reddit.subreddit('MachineLearning').hot(limit=10)

for post in sub_ml:
    # On recupere le titre
    title_reddit = post.title
    # On enleve les retours a la ligne
    title_reddit = title_reddit.replace("\n", " ")
    docs.append(title_reddit)
    list_reddit.append(title_reddit)
    # Ajout du document au RedditDocument
    docReddit = Document.RedditDocument(post.title, post.author, datetime.fromtimestamp(post.created), post.url,
                                  post.selftext)
    post.comments.replace_more(limit=None)
    # On recupere le nombre de commentaires
    nombre_commentaires = len(post.comments)
    docReddit.setNombreCommentaires(nombre_commentaires)

    # On ajoute le RedditDocument dans le dictionnaire id2doc
    id2doc[indice_id2doc] = docReddit
    indice_id2doc += 1

    # Partie avec les auteurs
    # Vérifier si l'auteur est connue
    if post.author.name not in id2auth:
        # Si non on créé un auteur et on l'ajoute au dictionnaire
        authReddit = Author.Author(post.author.name, 0, {})
        authReddit.add(docReddit)
        id2auth[post.author.name] = authReddit
    else:
        # Si oui on ajoute le document à l'auteur
        authReddit = id2auth.get(post.author.name)
        authReddit.add(docReddit)
        id2auth[post.author.name] = authReddit
print(docs)
print(id2doc)
print(id2auth)

# ARXIV
# URL des 10 derniers posts sur le sujet machine learning
url = "http://export.arxiv.org/api/query?search_query=all:machine+AND+all:learning&start=0&max_results=10"
# On recupere les donneses de de l'url
data = urllib.request.urlopen(url)
# On les transforme au format XML
xml = data.read().decode("utf-8")
# Puis au format JSON
json = xmltodict.parse(xml, "utf-8")

# On recupere les documents Arxiv
docs_arxiv = json["feed"]["entry"]

for doc in docs_arxiv:
    # On recupere le titre
    title_arxiv = doc["title"]
    # On enleve les retours a la ligne
    title_arxiv.replace("\n", " ")
    docs.append(title_arxiv)
    list_arxiv.append(title_arxiv)
    # Ajout du document au ArxivDocument
    docArxiv = Document.ArxivDocument(doc["title"], doc["author"], datetime.strptime(doc["published"], "%Y-%m-%dT%H:%M:%SZ"),
                                 doc["link"][0]["@href"], doc["summary"])

    # On ajoute le ArxivDocument dans le dictionnaire id2doc
    id2doc[indice_id2doc] = docArxiv
    indice_id2doc += 1

    # Partie avec les auteurs
    # On vérifie s'il y a plus d'un auteur
    if len(doc["author"]) > 1:
        # Si oui on rajoute les coauteurs au document
        docArxiv.setCoauteurs(doc["author"])
        for author in doc["author"]:
            # On vérifie alors si un auteur est connu
            if author.get("name") not in id2auth:
                # Si non on créé un auteur et on l'ajoute au dictionnaire
                authArxiv = Author.Author(author.get("name"), 0, {})
                authArxiv.add(docArxiv)
                id2auth[author.get("name")] = authArxiv
            else:
                # Si oui on ajoute le document à l'auteur
                authArxiv = id2auth.get(author.get("name"))
                authArxiv.add(docArxiv)
                id2auth[author.get("name")] = authArxiv
    else:
        # Si non on recupere le nom de l'auteur
        author = doc["author"]
        # On verifie si l'auteur est connu
        if author.get("name") not in id2auth:
            # Si non on créé un auteur et on l'ajoute au dictionnaire
            authArxiv = Author.Author(author.get("name"), 0, {})
            authArxiv.add(docArxiv)
            id2auth[author.get("name")] = authArxiv
        else:
            # Si oui on ajoute le document à l'auteur
            authArxiv = id2auth.get(author.get("name"))
            authArxiv.add(docArxiv)
            id2auth[author.get("name")] = authArxiv

print(id2doc)
print(id2auth)

dico["Reddit"] = list_reddit
dico["Arxiv"] = list_arxiv

print(docs)
print(dico)

df = pd.DataFrame.from_dict(dico)
print(df)

# On enregistre le DataFrame sur notre disque
df.to_csv("textes.csv", sep="\t")

# Maintenant on est plus obliges d'utiliser l'API, on a directement nos donnees ici
df_disk = pd.read_csv("textes.csv", sep="\t")

# On fait -1 car on a la ligne index en plus
taille_corpus = df_disk.shape[0] * (df_disk.shape[1] - 1)
print("La taille du corpus est de :", taille_corpus)
"""
''' Méthode pour récupérer les listes via des colonnes
list_reddit = df_disk["Reddit"].tolist()
list_arxiv = df_disk["Arxiv"].tolist()
'''
"""
# Autre méthode
corpus = list_reddit + list_arxiv

for doc in corpus:
    count = 0  # compteur de caractere
    mots = doc.split(" ")  # mots du documents
    for mot in mots:
        count += len(mot)
    print("Le nombre de mot est de :", len(mots))

    if count < 20:
        corpus.remove(doc)

chaine_finale = ", ".join(corpus)
print(chaine_finale)

cor = Corpus.Corpus("C1", id2auth, id2doc, len(id2doc), len(id2auth))
cor.save("corpus.pkl")
"""
# Dictionnaire vide pour creer un corpus
id2auth = dict()
id2doc = dict()

# Corpus vide
cor2 = Corpus.Corpus("C2", id2auth, id2doc, len(id2doc), len(id2auth))
# On charge le corpus déjà disponible
cor2 = cor2.load("corpus.pkl")

# On recupere le vocabulaire du corpus
vocab = cor2.getVocab()
# On le transforme en DataFrame
df = pd.DataFrame.from_dict(vocab)
column_names = df.columns.values.tolist()
# On le transpose pour un meilleur affichage
df = df.transpose()
# On rajoute une colonne pour les Mots
df.insert(0, "Mot", column_names)

# Application Dash
app = Dash(__name__)

app.layout = html.Div([
    # 1ere partie
    html.H1("Moteur de recherche de documents"),
    html.H4("Demandez un mot clé pour afficher les documents dans lesquels il apparait :"),
    dcc.Input(id="query", placeholder="Entrez un mot-clef", type="text"),
    html.Button("Rechercher", id="search-button"),
    html.Div(id="output"),

    # 2eme partie
    html.H4("Demandez un mot clé pour afficher son vecteur TF-IDF par rapport au corpus :"),
    dcc.Input(id="query1", placeholder="Entrez un mot-clef", type="text"),
    html.Button("Rechercher", id="search-button1"),
    html.Div(id="output1"),

    # 3eme partie
    html.H4(children='Statistiques sur les mots du Corpus'),

    # records --> mot clé qui dit qu'on prend tout le tableau
    dash_table.DataTable(
        df.to_dict("records"),
        columns=[{"id": str(i), "name": str(i)} for i in df.columns if i != "Mot"],  # Excluez la colonne "Mot"
        id="tbl",
        style_cell={"textAlign": "left"},
        page_size=15,
        css=[{
            "selector": "td.cell--selected, td.focused",
            "rule": "background-color: #a8dadc !important;"
        }, {
            "selector": "td.cell--selected, td.focused",
            "rule": "border-color: #2b5a5c !important;"
    }]
    ),  
    html.Div(children=[
        html.P()
    ], id="tbl-out"),
])


@app.callback(
    Output("output", "children"),
    [Input("search-button", "n_clicks")],
    [State("query", "value")]
)
def update_output(n_clicks, query):
    if not query:
        return "Aucun mot-clef saisie"
    else:
        documents = cor2.presence_doc(query)
        if len(documents) == 0:
            return "Aucun document trouvé"
        else:
            return html.Ul([html.Li(doc) for doc in documents])


@app.callback(
    Output("output1", "children"),
    [Input("search-button1", "n_clicks")],
    [State("query1", "value")]
)
def update_output(n_clicks, query1):
    if not query1:
        return "Aucun mot-clef saisie"
    else:
        vector_tfidf = cor2.vector_transformation(query1)
        if len(vector_tfidf) == 0:
            return "Aucun vecteur trouvé"
        else:
            return html.Ul([html.Li(vec) for vec in vector_tfidf])
        
# Fonction de rappel pour récupérer et afficher les statistiques du mot
@app.callback(
    Output("tbl-out", "children"),
    [Input("query", "value")]
)
def display_word_statistics(query):
    if not query:
        return "Aucun mot-clef saisi"

    # Utilisez la méthode ajoutée dans la classe Corpus pour récupérer les statistiques du mot
    word_statistics = cor2.get_word_statistics(query)

    # Créez une liste de paragraphes pour afficher les statistiques
    stats_paragraphs = [html.P(f"{key}: {value}") for key, value in word_statistics.items()]

    return stats_paragraphs


if __name__ == '__main__':
    app.run_server(debug=True)
