import os
import math
import re
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import networkx as nx
from pyvis.network import Network
from collections import Counter
import community.community_louvain as community_louvain
import spacy

# ---------------------------------------------------------------------
# Configuration spaCy et modèle CamemBERT
# ---------------------------------------------------------------------
spacy_nlp = spacy.load("fr_core_news_sm")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MAX_LENGTH = 512
NB_NEIGHBORS_DEFAULT = 20
TOP_N_DEFAULT = 100

MODEL_NAME = "camembert-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# ---------------------------------------------------------------------
# Variables globales
# ---------------------------------------------------------------------
selected_filepath = ""
text_widget = None

# Variables Tkinter pour les options
pooling_method = None      # "mean", "weighted", "max", "attention", "sif"
var_intra = None           # bool: ajouter des arêtes intracommunautaires (optionnel)
var_stopwords = None       # bool: utiliser stopwords
var_lemmatisation = None   # bool: utiliser la lemmatisation

# ---------------------------------------------------------------------
# Fonctions d'affichage et de sélection de fichier
# ---------------------------------------------------------------------
def afficher_message(msg):
    """Affiche un message dans la zone de texte ou dans la console."""
    global text_widget
    if text_widget is not None:
        text_widget.insert(tk.END, msg + "\n")
        text_widget.see(tk.END)
    else:
        print(msg)

def selectionner_fichier():
    """Ouvre une boîte de dialogue pour sélectionner un fichier texte."""
    global selected_filepath
    path = filedialog.askopenfilename(filetypes=[("Fichiers texte", "*.txt"), ("Tous", "*.*")])
    if path:
        selected_filepath = path
        afficher_message("Fichier sélectionné : " + selected_filepath)

# ---------------------------------------------------------------------
# Prétraitement du texte
# ---------------------------------------------------------------------
def pretraiter_phrase(phrase):
    """Traite la phrase avec spaCy (lemmatisation et retrait des stopwords)."""
    doc = spacy_nlp(phrase)
    if var_lemmatisation.get():
        if var_stopwords.get():
            tokens = [token.lemma_.lower() for token in doc
                      if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop]
        else:
            tokens = [token.lemma_.lower() for token in doc
                      if token.pos_ in ["NOUN", "PROPN"]]
    else:
        if var_stopwords.get():
            tokens = [token.text.lower() for token in doc
                      if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop]
        else:
            tokens = [token.text.lower() for token in doc
                      if token.pos_ in ["NOUN", "PROPN"]]
    return " ".join(tokens)

def extraire_termes_frequents(texte, top_n):
    """Extrait les termes les plus fréquents (NOUN, PROPN) du texte."""
    doc = spacy_nlp(texte)
    if var_lemmatisation.get():
        if var_stopwords.get():
            tokens = [token.lemma_.lower() for token in doc
                      if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop and len(token.text) >= 4]
        else:
            tokens = [token.lemma_.lower() for token in doc
                      if token.pos_ in ["NOUN", "PROPN"] and len(token.text) >= 4]
    else:
        if var_stopwords.get():
            tokens = [token.text.lower() for token in doc
                      if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop and len(token.text) >= 4]
        else:
            tokens = [token.text.lower() for token in doc
                      if token.pos_ in ["NOUN", "PROPN"] and len(token.text) >= 4]
    freq = Counter(tokens)
    return dict(freq.most_common(top_n))

def normaliser_texte(text):
    """Normalise le texte en supprimant les lignes indésirables et les espaces superflus."""
    lignes = text.splitlines()
    lignes_filtrees = [l for l in lignes if not l.strip().startswith("****")]
    texte_filtre = " ".join(lignes_filtrees)
    return re.sub(r'\s+', ' ', texte_filtre).strip().lower()

def split_text_into_sentences(text):
    """Découpe le texte en phrases basées sur la ponctuation (. ! ?)."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

# ---------------------------------------------------------------------
# Fonctions d'embedding (avec CamemBERT et diverses méthodes de pooling)
# ---------------------------------------------------------------------
def encoder_phrase(phrase):
    """Encode une phrase via CamemBERT selon la méthode de pooling choisie."""
    from collections import Counter
    phrase_pretraitee = pretraiter_phrase(phrase)
    inputs = tokenizer(phrase_pretraitee, return_tensors="pt", truncation=True,
                       max_length=MAX_LENGTH, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
    method = pooling_method.get()
    if method == "mean":
        emb = outputs.last_hidden_state.mean(dim=1)
        return emb.cpu().numpy().squeeze(0)
    elif method == "weighted":
        tokens_pretraite = phrase_pretraitee.split()
        freq_dict = Counter(tokens_pretraite)
        tokens_ids = inputs['input_ids'][0]
        tokens_from_ids = tokenizer.convert_ids_to_tokens(tokens_ids)
        if tokens_from_ids[0] in tokenizer.all_special_tokens:
            tokens_from_ids = tokens_from_ids[1:]
            outputs.last_hidden_state = outputs.last_hidden_state[:, 1:, :]
        if tokens_from_ids[-1] in tokenizer.all_special_tokens:
            tokens_from_ids = tokens_from_ids[:-1]
            outputs.last_hidden_state = outputs.last_hidden_state[:, :-1, :]
        weights = []
        for token in tokens_from_ids:
            if token.startswith("▁"):
                word = token[1:]
                weights.append(freq_dict.get(word, 1))
            else:
                weights.append(weights[-1] if weights else 1)
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=outputs.last_hidden_state.device)
        weights_tensor = weights_tensor.unsqueeze(0).unsqueeze(-1)
        weighted_embeds = (outputs.last_hidden_state * weights_tensor).sum(dim=1)
        normalization = weights_tensor.sum()
        emb = weighted_embeds / (normalization if normalization != 0 else 1)
        return emb.cpu().numpy().squeeze(0)
    elif method == "max":
        emb = outputs.last_hidden_state.max(dim=1)[0]
        return emb.cpu().numpy().squeeze(0)
    elif method == "attention":
        token_embeds = outputs.last_hidden_state
        norms = torch.norm(token_embeds, dim=2)
        att_weights = torch.softmax(norms, dim=1)
        weighted_embeds = (token_embeds * att_weights.unsqueeze(2)).sum(dim=1)
        return weighted_embeds.cpu().numpy().squeeze(0)
    elif method == "sif":
        a = 0.001
        tokens_pretraitee = phrase_pretraitee.split()
        if len(tokens_pretraitee) == 0:
            return outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze(0)
        from collections import Counter
        counts = Counter(tokens_pretraitee)
        total_tokens = len(tokens_pretraitee)
        sif_weights_list = [a / (a + counts[token] / total_tokens) for token in tokens_pretraitee]
        tokens_ids = inputs['input_ids'][0]
        tokens_from_ids = tokenizer.convert_ids_to_tokens(tokens_ids)
        if tokens_from_ids[0] in tokenizer.all_special_tokens:
            tokens_from_ids = tokens_from_ids[1:]
            outputs.last_hidden_state = outputs.last_hidden_state[:, 1:, :]
        if tokens_from_ids[-1] in tokenizer.all_special_tokens:
            tokens_from_ids = tokens_from_ids[:-1]
            outputs.last_hidden_state = outputs.last_hidden_state[:, :-1, :]
        sif_weights = []
        for token in tokens_from_ids:
            if token.startswith("▁"):
                sif_weights.append(sif_weights_list.pop(0) if sif_weights_list else 1)
            else:
                sif_weights.append(sif_weights[-1] if sif_weights else 1)
        sif_weights_tensor = torch.tensor(sif_weights, dtype=torch.float32, device=outputs.last_hidden_state.device)
        sif_weights_tensor = sif_weights_tensor.unsqueeze(0).unsqueeze(-1)
        weighted_embeds = (outputs.last_hidden_state * sif_weights_tensor).sum(dim=1)
        normalization = sif_weights_tensor.sum()
        emb = weighted_embeds / (normalization if normalization != 0 else 1)
        return emb.cpu().numpy().squeeze(0)
    else:
        emb = outputs.last_hidden_state.mean(dim=1)
        return emb.cpu().numpy().squeeze(0)

def encoder_contextuel_simplifie(texte, mot_cle):
    """
    Calcule l'embedding du mot-clé en contexte en moyennant les embeddings
    de toutes les phrases contenant le mot.
    Si aucune phrase n'est trouvée, encode simplement le mot-clé.
    """
    sentences = split_text_into_sentences(texte)
    pertinentes = [s for s in sentences if mot_cle.lower() in s.lower()]
    afficher_message(f"Nombre de phrases contextuelles pour '{mot_cle}' : {len(pertinentes)}")
    if not pertinentes:
        return encoder_phrase(mot_cle)
    embeddings = [encoder_phrase(s) for s in pertinentes]
    return np.mean(embeddings, axis=0)

def encoder_terme_par_contexte(terme, texte):
    """
    Pour un terme donné, encode les phrases le contenant et retourne
    (embedding moyen, liste des phrases contextuelles).
    """
    sentences = split_text_into_sentences(texte)
    context_sentences = [s for s in sentences if terme.lower() in s.lower()]
    if not context_sentences:
        return encoder_phrase(terme), []
    embeddings = [encoder_phrase(s) for s in context_sentences]
    return np.mean(embeddings, axis=0), context_sentences

def cosine_similarity(v1, v2):
    """
    Calcule la similarité cosinus entre deux vecteurs.
    Retourne 0 si la valeur calculée est négative.
    """
    sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return max(sim, 0)

# ---------------------------------------------------------------------
# Fonction manquante : Construction des voisins contextuels
# ---------------------------------------------------------------------
def construire_voisins_contextuel(texte, mot_cle, embedding_keyword):
    """
    Extrait les termes fréquents du texte et, pour chacun, calcule
    la similarité cosinus avec l'embedding du mot-clé.
    Retourne une liste de tuples (terme, similarité, passages),
    ainsi que le dictionnaire de fréquences et un cache des embeddings.
    """
    top_n = int(entry_nb_termes.get())
    freq_dict = extraire_termes_frequents(texte, top_n)
    candidats = list(freq_dict.keys())
    afficher_message(f"DEBUG: {len(candidats)} candidats fréquents extraits du corpus.")
    voisins = []
    cache = {}
    for idx, t in enumerate(candidats):
        if t.lower() == mot_cle.lower():
            continue
        if t in cache:
            emb_t, passages = cache[t]
        else:
            emb_t, passages = encoder_terme_par_contexte(t, texte)
            cache[t] = (emb_t, passages)
        sim = cosine_similarity(embedding_keyword, emb_t)
        afficher_message(f"Candidat : {t} - Similarité : {sim:.4f}")
        voisins.append((t, sim, passages))
        progress_bar['maximum'] = len(candidats)
        progress_bar['value'] = idx + 1
    voisins = sorted(voisins, key=lambda x: x[1], reverse=True)[:int(entry_voisins.get())]
    return voisins, freq_dict, cache

# ---------------------------------------------------------------------
# Hiérarchisation et Construction des graphes
# ---------------------------------------------------------------------
def hierarchiser_voisins_cosinus(voisins):
    """
    Classe les voisins en fonction du 75ème quantile.
    Si la similarité >= Q75, le voisin est "central", sinon "périphérique".
    Retourne une liste de tuples (mot, sim, passages, catégorie).
    """
    if not voisins:
        return []
    scores = np.array([sim for mot, sim, passages in voisins])
    q75 = np.quantile(scores, 0.75)
    resultats = []
    for mot, sim, passages in voisins:
        if sim >= q75:
            cat = "central"
        else:
            cat = "périphérique"
        resultats.append((mot, sim, passages, cat))
    return resultats

def creer_graphe_general_keyword(voisins_hier, mot_cle, freq_dict):
    """
    Construit un graphe général (layout spring) en mode mot-clé.
    Le mot-clé est le nœud central et chaque voisin y est relié.
    """
    G = nx.Graph()
    G.add_node(mot_cle, frequency=freq_dict.get(mot_cle, 0), categorie="mot_clé")
    for mot, sim, passages, cat in voisins_hier:
        G.add_node(mot, frequency=freq_dict.get(mot, 0), categorie=cat)
        G.add_edge(mot_cle, mot, weight=sim)
    return G

def creer_graphe_circulaire_keyword(voisins_hier, mot_cle, freq_dict):
    """
    Construit un graphe circulaire (étoilé) avec le mot-clé au centre.
    """
    G = nx.Graph()
    G.add_node(mot_cle, frequency=freq_dict.get(mot_cle, 0), categorie="mot_clé")
    for mot, sim, passages, cat in voisins_hier:
        G.add_node(mot, frequency=freq_dict.get(mot, 0), categorie=cat)
        G.add_edge(mot_cle, mot, weight=sim)
    return G

def colorer_communautes(G, freq_dict):
    """
    Applique l'algorithme de Louvain pour détecter les communautés
    et assigne une couleur à chaque nœud.
    """
    partition = community_louvain.best_partition(G)
    palette = ["#8A2BE2", "#9370DB", "#BA55D3", "#DA70D6", "#D8BFD8"]
    for node in G.nodes():
        comm = partition.get(node, 0)
        color = palette[comm % len(palette)]
        G.nodes[node]["frequency"] = freq_dict.get(node, 0)
        G.nodes[node]["color"] = color
    return G, partition

def assigner_layout_classique(G):
    """Positionne les nœuds avec spring_layout et adapte les coordonnées pour Pyvis."""
    positions = nx.spring_layout(G, seed=42)
    positions_dict = {}
    for node, coord in positions.items():
        positions_dict[node] = {"x": float(coord[0]*1000), "y": float(coord[1]*1000)}
    return positions_dict

def assigner_layout_etoile(G, central, centre=(300,300), rayon=300):
    """
    Place le mot-clé au centre et répartit ses voisins en cercle autour.
    """
    positions = {central: {"x": centre[0], "y": centre[1]}}
    voisins = list(G.neighbors(central))
    n = len(voisins)
    if n > 0:
        angle_gap = 2 * math.pi / n
        for i, node in enumerate(voisins):
            angle = i * angle_gap
            x = centre[0] + rayon * math.cos(angle)
            y = centre[1] + rayon * math.sin(angle)
            positions[node] = {"x": x, "y": y}
    return positions

def nx_vers_pyvis_general(G, positions):
    """
    Convertit le graphe en un graphique Pyvis pour le layout général.
    """
    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")
    net.set_options('{ "physics": { "enabled": false } }')
    for node, data in G.nodes(data=True):
        freq = data.get("frequency", 0)
        color = data.get("color", "#8A2BE2")
        cat = data.get("categorie", "inconnu")
        label = f"{node}\nFreq: {freq}\nCat: {cat}"
        size = 20 + (freq*2 if isinstance(freq, (int, float)) else 0)
        pos = positions.get(node, {"x":300, "y":300})
        net.add_node(node, label=label, title=label,
                     x=pos["x"], y=pos["y"], color=color, size=size)
    for u, v, data in G.edges(data=True):
        weight = float(data.get("weight", 0))
        net.add_edge(u, v, value=weight, title=f"Sim: {weight:.4f}")
    return net

def nx_vers_pyvis_circulaire(G, positions, mot_cle):
    """
    Convertit le graphe en un graphique Pyvis de type "étoile" avec le mot-clé au centre.
    """
    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")
    net.set_options('{ "physics": { "enabled": false } }')
    factor = 30
    for node, data in G.nodes(data=True):
        freq = data.get("frequency", 0)
        color = data.get("color", "#8A2BE2")
        cat = data.get("categorie", "inconnu")
        if node.lower() == mot_cle.lower():
            color = "#FFFFFF"  # le mot-clé s'affiche en blanc
        label = f"{node}\nFreq: {freq}\nCat: {cat}"
        pos = positions.get(node, {"x":300, "y":300})
        net.add_node(node, label=label, title=label,
                     x=pos["x"], y=pos["y"], color=color, size=30)
    for u, v, data in G.edges(data=True):
        sim = float(data.get("weight", 0))
        width = sim * factor
        net.add_edge(u, v, title=f"Sim: {sim:.4f}", value=width)
    return net

def sauvegarder_resultats(voisins, filename):
    """Sauvegarde la liste des voisins avec leur similarité dans un fichier texte."""
    with open(filename, "w", encoding="utf-8") as f:
        for mot, sim, passages in voisins:
            f.write(f"{mot}\t{sim:.4f}\n")
            if passages:
                for passage in passages:
                    f.write(f"    {passage}\n")
            f.write("\n")

# ---------------------------------------------------------------------
# Analyse par mot-clé (seule analyse conservée)
# ---------------------------------------------------------------------
def analyser_fichier_mot_cle(texte):
    """
    Analyse par mot-clé :
      1. Récupère le mot-clé depuis l'interface.
      2. Calcule l'embedding contextuel du mot-clé.
      3. Extrait les termes fréquents et calcule leur similarité cosinus.
      4. Trie et limite le nombre de voisins.
      5. Applique la hiérarchisation par quantile (Q75) pour distinguer le noyau central des éléments périphériques.
      6. Construit deux graphes (général et circulaire) et génère les fichiers HTML correspondants.
    """
    mot_cle = entry_noeud_central.get().strip().lower()
    afficher_message(f"Analyse du mot-clé : {mot_cle}")

    # Calcul de l'embedding contextuel du mot-clé
    emb_keyword = encoder_contextuel_simplifie(texte, mot_cle)
    afficher_message(f"Embedding du mot-clé (norme) : {np.linalg.norm(emb_keyword):.4f}")

    # Extraction des voisins à partir des termes fréquents
    voisins, freq_dict, cache = construire_voisins_contextuel(texte, mot_cle, emb_keyword)
    nb_voisins = int(entry_voisins.get()) if entry_voisins.get().isdigit() else NB_NEIGHBORS_DEFAULT
    voisins = voisins[:nb_voisins]
    afficher_message(f"{len(voisins)} voisins positifs sélectionnés.")

    # Sauvegarde des voisins dans un fichier texte
    sauvegarder_resultats(voisins, "context_neighbors.txt")
    afficher_message("Fichier texte généré : context_neighbors.txt")

    # Hiérarchisation des voisins via quantile (Q75)
    voisins_hier = hierarchiser_voisins_cosinus(voisins)
    for mot, sim, _, cat in voisins_hier:
        afficher_message(f"{mot}: {sim:.4f} ({cat})")

    # Construction du graphe général (layout spring)
    G_general = creer_graphe_general_keyword(voisins_hier, mot_cle, freq_dict)
    G_general, partition = colorer_communautes(G_general, freq_dict)
    positions_general = assigner_layout_classique(G_general)
    net_general = nx_vers_pyvis_general(G_general, positions_general)
    html_file_general = "graph_general.html"
    net_general.write_html(html_file_general)
    afficher_message("Graphe général généré : " + os.path.abspath(html_file_general))

    # Construction du graphe circulaire (étoilé, mot-clé au centre)
    G_circulaire = creer_graphe_circulaire_keyword(voisins_hier, mot_cle, freq_dict)
    G_circulaire, partition = colorer_communautes(G_circulaire, freq_dict)
    if var_intra.get():
        afficher_message("Option arêtes intracommunautaires cochée (non implémentée).")
    positions_circulaire = assigner_layout_etoile(G_circulaire, mot_cle)
    net_circulaire = nx_vers_pyvis_circulaire(G_circulaire, positions_circulaire, mot_cle)
    html_file_circulaire = "graph_circulaire.html"
    net_circulaire.write_html(html_file_circulaire)
    afficher_message("Graphe circulaire généré : " + os.path.abspath(html_file_circulaire))

# ---------------------------------------------------------------------
# Lanceur principal
# ---------------------------------------------------------------------
def analyser_fichier():
    """Lit le fichier sélectionné, normalise le texte et lance l'analyse par mot-clé."""
    if not selected_filepath:
        afficher_message("Erreur : veuillez sélectionner un fichier.")
        return
    with open(selected_filepath, "r", encoding="utf-8") as f:
        texte_brut = f.read().strip()
    texte_normalise = normaliser_texte(texte_brut)
    analyser_fichier_mot_cle(texte_normalise)

# ---------------------------------------------------------------------
# Interface Tkinter (Analyse par mot-clé uniquement)
# ---------------------------------------------------------------------
root = tk.Tk()
root.title("Analyse textuelle – Analyse par mot-clé")
root.geometry("700x1100")

frame_params = ttk.LabelFrame(root, text="Paramètres", padding="10")
frame_params.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
ttk.Button(frame_params, text="Sélectionner un fichier", command=selectionner_fichier).grid(row=0, column=0, columnspan=2, pady=5)
ttk.Label(frame_params, text="Nombre de voisins/termes (max) :").grid(row=1, column=0, sticky=tk.W)
entry_voisins = ttk.Entry(frame_params, width=10)
entry_voisins.insert(0, "20")
entry_voisins.grid(row=1, column=1, sticky=tk.W)
ttk.Label(frame_params, text="Nombre de termes à analyser :").grid(row=2, column=0, sticky=tk.W)
entry_nb_termes = ttk.Entry(frame_params, width=10)
entry_nb_termes.insert(0, "100")
entry_nb_termes.grid(row=2, column=1, sticky=tk.W)
ttk.Label(frame_params, text="(Le seuil de similarité n'est pas utilisé ici)").grid(row=3, column=0, sticky=tk.W)

frame_pretraitement = ttk.LabelFrame(root, text="Options de prétraitement", padding="10")
frame_pretraitement.grid(row=1, column=0, padx=10, pady=10, sticky=tk.W+tk.E)
var_stopwords = tk.BooleanVar(value=True)
var_lemmatisation = tk.BooleanVar(value=True)
ttk.Checkbutton(frame_pretraitement, text="Utiliser stopwords", variable=var_stopwords).grid(row=0, column=0, sticky=tk.W)
ttk.Checkbutton(frame_pretraitement, text="Utiliser lemmatisation", variable=var_lemmatisation).grid(row=0, column=1, sticky=tk.W)

frame_embedding = ttk.LabelFrame(root, text="Méthode d'embedding", padding="10")
frame_embedding.grid(row=2, column=0, padx=10, pady=10, sticky=tk.W+tk.E)
pooling_method = tk.StringVar(value="mean")
ttk.Radiobutton(frame_embedding, text="Mean pooling", variable=pooling_method, value="mean").grid(row=0, column=0, sticky=tk.W)
ttk.Radiobutton(frame_embedding, text="Weighted pooling (fréquence)", variable=pooling_method, value="weighted").grid(row=0, column=1, sticky=tk.W)
ttk.Radiobutton(frame_embedding, text="Max pooling", variable=pooling_method, value="max").grid(row=0, column=2, sticky=tk.W)
ttk.Radiobutton(frame_embedding, text="Attention pooling", variable=pooling_method, value="attention").grid(row=0, column=3, sticky=tk.W)
ttk.Radiobutton(frame_embedding, text="SIF pooling", variable=pooling_method, value="sif").grid(row=0, column=4, sticky=tk.W)

frame_analysis = ttk.LabelFrame(root, text="Mot-clé", padding="10")
frame_analysis.grid(row=3, column=0, padx=10, pady=10, sticky=tk.W+tk.E)
ttk.Label(frame_analysis, text="Mot-clé :").grid(row=0, column=0, sticky=tk.W)
entry_noeud_central = ttk.Entry(frame_analysis, width=20)
entry_noeud_central.insert(0, "soins")
entry_noeud_central.grid(row=0, column=1, sticky=tk.W)
ttk.Label(frame_analysis, text="(Le mot-clé défini ici sera analysé)").grid(row=0, column=2, sticky=tk.W)
var_intra = tk.BooleanVar(value=False)
ttk.Checkbutton(frame_analysis, text="Ajouter arêtes intracommunautaires", variable=var_intra).grid(row=1, column=0, columnspan=2, sticky=tk.W)

ttk.Button(root, text="Lancer l'analyse", command=analyser_fichier).grid(row=4, column=0, pady=10)
progress_bar = ttk.Progressbar(root, orient="horizontal", mode="determinate")
progress_bar.grid(row=5, column=0, padx=10, pady=10, sticky="ew")
text_widget = scrolledtext.ScrolledText(root, width=80, height=20)
text_widget.grid(row=6, column=0, padx=10, pady=10)

root.mainloop()
