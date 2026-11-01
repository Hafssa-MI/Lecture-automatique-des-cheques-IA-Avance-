"""Lecture automatique des chèques : démonstration des quatre jalons.

Lancement : streamlit run app.py
"""
import html
import json
import os
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy.ndimage import gaussian_filter, rotate

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from tensorflow import keras  # noqa: E402

# ======================================================================================
# Constantes (identiques à celles des notebooks)
# ======================================================================================
RACINE = Path(__file__).parent
DOSSIER_MODELES = RACINE / "models"
FICHIER_EXEMPLES = RACINE / "data" / "exemples.npz"
FICHIER_JOURNAL = RACINE / "data" / "journal.jsonl"

IMG_H, IMG_W, L_MAX = 32, 128, 5
EMPTY = 10                                  # position vide (jalons 1 à 3)
T_MAX, PAD, DEBUT, FIN = 6, 10, 11, 12      # jalon 4 : jetons de la séquence
NOMS_POS = ["unités", "dizaines", "centaines", "milliers", "dizaines de milliers"]

ENSEMBLE_J3 = ["aug_corr50", "multi_lam0.5", "multi_chaud_lam2_120ep", "aug_corr100"]
PAIRE_J4 = ["seq_gru128", "seq_gru128_s1"]
PAIRE_J4_AUG = ["seq_gru128_aug", "seq_gru128_aug_s1"]      # extension facultative

MULTI_J3 = [
    ("λ = 0,1", "multi_lam0.1"),
    ("λ = 0,5", "multi_lam0.5"),
    ("λ = 1", "multi_lam1"),
    ("λ = 2", "multi_lam2"),
    ("λ = 4", "multi_lam4"),
    ("λ = 2, départ à chaud", "multi_chaud_lam2_120ep"),
]
# Validation (journal du jalon 3) : exactitude A, exactitude B, AUC A, AUC B
REF_MULTI = {
    "multi_lam0.1": (0.749, 0.703, 0.608, 0.648),
    "multi_lam0.5": (0.756, 0.743, 0.761, 0.731),
    "multi_lam1": (0.678, 0.685, 0.918, 0.749),
    "multi_lam2": (0.653, 0.625, 0.955, 0.810),
    "multi_lam4": (0.582, 0.563, 0.993, 0.846),
    "multi_chaud_lam2_120ep": (0.762, 0.748, 0.883, 0.725),
}

# Seuil du détecteur du jalon 2 selon le rapport de coûts (validation) :
# seuil, fausses alertes, rappel, fraudes manquées sur 285
COUTS_J2 = {
    "5 à 10": (0.503, "0,3 %", "0,989", 3),
    "50 (retenu)": (0.0163, "1,9 %", "0,993", 2),
    "200 et plus": (0.0007, "8,4 %", "1,000", 0),
}
COUT_DEFAUT = "50 (retenu)"

SYSTEMES = {
    "j1": dict(nom="Jalon 1 : MLP", fichiers=["mlp_base_80ep"]),
    "j2": dict(nom="Jalon 2 : CNN", fichiers=["cnn_final_80ep"]),
    "j2d": dict(nom="Jalon 2 : CNN après débruitage", fichiers=["cnn_final_80ep", "debruiteur_v4"]),
    "j3e": dict(nom="Jalon 3 : ensemble de 4 CNN", fichiers=ENSEMBLE_J3),
    "j3m": dict(nom="Jalon 3 : multitâche (λ = 2)", fichiers=["multi_lam2"]),
    "j4g": dict(nom="Jalon 4 : GRU seul", fichiers=["seq_gru128"]),
    "j4p": dict(nom="Jalon 4 : paire de GRU", fichiers=PAIRE_J4),
    "j4a": dict(nom="Jalon 4, extension : paire de GRU avec dégradations", fichiers=PAIRE_J4_AUG),
}

DETECTEURS = [
    dict(cle="d2", nom="Jalon 2 : détecteur dédié", fichier="detecteur_blocs4", multi=False,
         ref="AUC 0,9996 (A, test) ; 0,58 (B, validation)"),
    dict(cle="d3", nom="Jalon 3 : détecteur dédié avec augmentation", fichier="detecteur_aug", multi=False,
         ref="AUC 0,9974 (A, test) ; 0,892 (B, test)"),
] + [
    dict(cle=f"m_{f}", nom=f"Jalon 3 : multitâche, {lab}", fichier=f, multi=True,
         ref=(f"AUC {REF_MULTI[f][2]:.3f} (A) ; {REF_MULTI[f][3]:.3f} (B), validation").replace(".", ","))
    for lab, f in MULTI_J3
]

PAGES = [
    "Vue d'ensemble",
    "Atelier de lecture",
    "Jalon 1 : MLP",
    "Jalon 2 : CNN et détecteur",
    "Jalon 3 : scanner B et multitâche",
    "Jalon 4 : lecture en séquence",
    "Détection d'altération",
    "Évaluation par lot",
    "Journal d'expériences",
]
PAGES_SANS_CHEQUE = {"Vue d'ensemble", "Évaluation par lot", "Journal d'expériences"}

ENCRE, ACCENT, GRIS, TRAIT, FOND = "#1B2A3A", "#1F4E79", "#5B6B7B", "#D9DEE5", "#F6F8FA"
COULEURS_AB = ["#1F4E79", "#B9822E"]
PALETTE = ["#1F4E79", "#6F9BBF", "#2E7D6B", "#9AA7B4", "#B9822E", "#7A5C99", "#A33B3B", "#4D6A80"]

st.set_page_config(page_title="Lecture automatique des chèques", layout="wide")

CSS = """
<style>
.stApp { font-family: "Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif; color: #1B2A3A; }
.block-container { padding-top: 4.5rem; padding-bottom: 3rem; max-width: 1180px; }
footer { visibility: hidden; }
.stAppDeployButton { display: none; }
[data-testid="stSidebar"] { border-right: 1px solid #D9DEE5; }
[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }

.marque { padding: 0 0 1rem 0; margin-bottom: 1rem; border-bottom: 1px solid #D9DEE5; }
.marque-t { font-size: 1.05rem; font-weight: 650; color: #1B2A3A; line-height: 1.3; }
.marque-s { font-size: .76rem; color: #5B6B7B; margin-top: .35rem; letter-spacing: .01em; }
.sb-titre { font-size: .72rem; font-weight: 650; letter-spacing: .08em; text-transform: uppercase;
            color: #5B6B7B; margin: 1.1rem 0 .4rem 0; }
.equipe { margin-top: .9rem; padding-top: .7rem; border-top: 1px solid #D9DEE5; font-size: .74rem;
          color: #33414F; line-height: 1.7; letter-spacing: .02em; }
.equipe b { display: block; font-size: .68rem; font-weight: 650; letter-spacing: .08em;
            text-transform: uppercase; color: #5B6B7B; margin-bottom: .2rem; }

.entete { margin-bottom: 1.6rem; padding-bottom: 1rem; border-bottom: 1px solid #D9DEE5; }
.entete .surtitre { font-size: .74rem; font-weight: 650; letter-spacing: .09em; text-transform: uppercase;
                    color: #1F4E79; margin-bottom: .35rem; line-height: 1.5; }
.entete h1 { font-size: 1.85rem; font-weight: 650; color: #1B2A3A; margin: 0; padding: 0; letter-spacing: -.015em; }
.entete p { color: #5B6B7B; margin: .5rem 0 0 0; font-size: .98rem; max-width: 52rem; }

.section { font-size: 1.02rem; font-weight: 650; color: #1B2A3A; margin: 2rem 0 .8rem 0; padding-bottom: .4rem;
           border-bottom: 1px solid #D9DEE5; }
.section small { font-weight: 400; color: #5B6B7B; font-size: .85rem; margin-left: .5rem; }

.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: .8rem; margin: .4rem 0 1rem 0; }
.kpi { background: #FFFFFF; border: 1px solid #D9DEE5; border-radius: 6px; padding: .8rem 1rem .85rem 1rem;
       border-top: 3px solid #1F4E79; }
.kpi-l { font-size: .72rem; font-weight: 600; letter-spacing: .05em; text-transform: uppercase; color: #5B6B7B; }
.kpi-v { font-size: 1.55rem; font-weight: 650; color: #1B2A3A; margin-top: .25rem; font-variant-numeric: tabular-nums; }
.kpi-s { font-size: .8rem; color: #5B6B7B; margin-top: .15rem; }

.badge { display: inline-block; padding: .12rem .55rem; border-radius: 3px; font-size: .78rem; font-weight: 600;
         letter-spacing: .01em; border: 1px solid transparent; white-space: nowrap; }
.badge.ok { background: #E8F3EE; color: #1F6A4B; border-color: #C5E0D2; }
.badge.warn { background: #FBF1DE; color: #8A5A0B; border-color: #EBD3A2; }
.badge.bad { background: #F8E7E7; color: #8F2F2F; border-color: #E6BFBF; }
.badge.neutre { background: #EEF2F6; color: #33414F; border-color: #D9DEE5; }
.puces { display: flex; flex-wrap: wrap; gap: .4rem; margin: .2rem 0 .9rem 0; }

table.tbl { width: 100%; border-collapse: collapse; font-size: .9rem; margin: .3rem 0 .6rem 0; }
table.tbl th { text-align: left; font-size: .72rem; font-weight: 650; letter-spacing: .05em; text-transform: uppercase;
               color: #5B6B7B; background: #F6F8FA; padding: .55rem .7rem; border-bottom: 1px solid #D9DEE5;
               border-top: 1px solid #D9DEE5; }
table.tbl td { padding: .55rem .7rem; border-bottom: 1px solid #E8ECF1; color: #1B2A3A; vertical-align: top;
               font-variant-numeric: tabular-nums; }
table.tbl tr:hover td { background: #FAFBFC; }

.note { border-left: 3px solid #1F4E79; background: #F6F8FA; padding: .7rem 1rem; color: #33414F;
        font-size: .92rem; margin: .8rem 0; border-radius: 0 4px 4px 0; }
.note.alerte { border-left-color: #B9822E; background: #FDF8EE; }
.legende { color: #5B6B7B; font-size: .84rem; }
[data-testid="stImage"] img { border: 1px solid #D9DEE5; border-radius: 4px; }

.hero { position: relative; overflow: hidden; border: 1px solid #D9DEE5; border-radius: 12px;
        padding: 2.4rem 2.6rem 2rem 2.6rem; margin-bottom: 2rem;
        background: linear-gradient(135deg, #F1F6FB 0%, #FFFFFF 60%); box-shadow: 0 1px 2px rgba(27,42,58,.05); }
.hero::after { content: ""; position: absolute; right: 0; top: 0; width: 45%; height: 100%;
               background-image: radial-gradient(#BFCEDD 1.1px, transparent 1.1px); background-size: 16px 16px;
               -webkit-mask-image: linear-gradient(to left, #000, transparent);
               mask-image: linear-gradient(to left, #000, transparent); opacity: .7; }
.hero > * { position: relative; z-index: 1; }
.hero-puces { display: flex; flex-wrap: wrap; gap: .45rem; margin-bottom: 1.1rem; }
.hero-puces span { font-size: .72rem; font-weight: 650; letter-spacing: .08em; text-transform: uppercase;
                   color: #1F4E79; border: 1px solid #C5D3E2; background: #FFFFFF; padding: .2rem .65rem; border-radius: 999px; }
.hero h1 { font-size: 2.5rem; font-weight: 700; letter-spacing: -.025em; line-height: 1.1; color: #1B2A3A;
           margin: 0 0 1rem 0; padding: 0; }
.hero-texte { font-size: 1.05rem; line-height: 1.65; color: #4A5A6A; max-width: 46rem; margin: 0; }
.hero-equipe { margin-top: 1.3rem; font-size: .8rem; letter-spacing: .06em; color: #33414F; font-weight: 600; }
.hero-equipe i { font-style: normal; color: #A9B6C3; margin: 0 .6rem; }
.hero-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1.5rem;
              margin-top: 1.8rem; padding-top: 1.3rem; border-top: 1px solid #D9DEE5; }
.hero-stat b { display: block; font-size: 1.7rem; font-weight: 700; color: #1F4E79; letter-spacing: -.02em;
               font-variant-numeric: tabular-nums; }
.hero-stat span { font-size: .8rem; color: #5B6B7B; }

.grille { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: .9rem; margin: .4rem 0 1rem 0; }
.carte-pb { border: 1px solid #D9DEE5; border-radius: 10px; padding: 1.1rem 1.2rem; background: #FFFFFF; }
.carte-pb .nb { font-size: .72rem; font-weight: 700; letter-spacing: .08em; color: #1F4E79; }
.carte-pb h4 { margin: .35rem 0 .4rem 0; font-size: 1.05rem; font-weight: 650; color: #1B2A3A; padding: 0; }
.carte-pb p { margin: 0; font-size: .9rem; line-height: 1.55; color: #4A5A6A; }

.etape { display: grid; grid-template-columns: 56px 1fr; gap: 1rem; margin-bottom: 1.1rem; }
.etape-num { position: relative; display: flex; justify-content: center; }
.etape-num span { width: 44px; height: 44px; border-radius: 50%; background: #1F4E79; color: #FFFFFF; font-weight: 700;
                  display: flex; align-items: center; justify-content: center; font-size: 1.05rem;
                  box-shadow: 0 0 0 6px #E6EEF6; z-index: 1; }
.etape:not(:last-child) .etape-num::after { content: ""; position: absolute; left: calc(50% - 1px); top: 52px;
                  bottom: -1.2rem; width: 2px; background: linear-gradient(#1F4E79, #C9D6E3); }
.etape-corps { border: 1px solid #D9DEE5; border-radius: 10px; padding: 1.2rem 1.4rem; background: #FFFFFF; }
.etape-tete { display: flex; justify-content: space-between; align-items: flex-start; gap: 1.2rem; flex-wrap: wrap;
              padding-bottom: .9rem; margin-bottom: .9rem; border-bottom: 1px solid #E8ECF1; }
.etape-sur { font-size: .72rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; color: #1F4E79; }
.etape-titre { font-size: 1.2rem; font-weight: 650; color: #1B2A3A; margin-top: .15rem; }
.etape-tech { font-size: .88rem; color: #5B6B7B; margin-top: .1rem; }
.etape-res { text-align: right; }
.etape-val { font-size: 1.9rem; font-weight: 700; color: #1B2A3A; letter-spacing: -.02em; line-height: 1;
             font-variant-numeric: tabular-nums; }
.etape-mes { font-size: .76rem; color: #5B6B7B; margin-top: .3rem; }
.etape-grille { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 1.3rem; }
.etape-grille b { display: block; font-size: .7rem; font-weight: 700; letter-spacing: .07em; text-transform: uppercase;
                  color: #5B6B7B; margin-bottom: .35rem; }
.etape-grille div { font-size: .9rem; line-height: 1.55; color: #33414F; }
.etape-grille div.suite { border-left: 3px solid #B9822E; padding-left: .9rem; background: #FDF8EE; border-radius: 0 6px 6px 0; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ======================================================================================
# Mise en forme
# ======================================================================================
def esc(x):
    return html.escape(str(x))


def fr(x, d=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{d}f}".replace(".", ",")


def badge(texte, genre="neutre"):
    return f'<span class="badge {genre}">{esc(texte)}</span>'


def entete(titre, sous_titre="", surtitre=""):
    st.markdown(
        f'<div class="entete"><div class="surtitre">{esc(surtitre)}</div><h1>{esc(titre)}</h1>'
        f"<p>{esc(sous_titre)}</p></div>",
        unsafe_allow_html=True,
    )


def section(titre, precision=""):
    petit = f"<small>{esc(precision)}</small>" if precision else ""
    st.markdown(f'<div class="section">{esc(titre)}{petit}</div>', unsafe_allow_html=True)


def note(texte, genre=""):
    st.markdown(f'<div class="note {genre}">{texte}</div>', unsafe_allow_html=True)


def legende(texte):
    st.markdown(f'<span class="legende">{texte}</span>', unsafe_allow_html=True)


def kpis(items):
    """items : liste de (libellé, valeur, précision)."""
    blocs = "".join(
        f'<div class="kpi"><div class="kpi-l">{esc(l)}</div><div class="kpi-v">{esc(v)}</div>'
        f'<div class="kpi-s">{esc(s)}</div></div>'
        for l, v, s in items
    )
    st.markdown(f'<div class="kpis">{blocs}</div>', unsafe_allow_html=True)


def tableau(df, brut=()):
    """Tableau HTML ; les colonnes de `brut` contiennent du HTML de confiance (badges)."""
    ent = "".join(f"<th>{esc(c)}</th>" for c in df.columns)
    lignes = []
    for _, r in df.iterrows():
        cellules = "".join(f"<td>{r[c] if c in brut else esc(r[c])}</td>" for c in df.columns)
        lignes.append(f"<tr>{cellules}</tr>")
    st.markdown(
        f'<table class="tbl"><thead><tr>{ent}</tr></thead><tbody>{"".join(lignes)}</tbody></table>',
        unsafe_allow_html=True,
    )


def style_graphique(chart, hauteur=280):
    return (
        chart.properties(height=hauteur)
        .configure(background="#FFFFFF")
        .configure_view(strokeWidth=0)
        .configure_axis(labelColor=GRIS, titleColor=GRIS, gridColor="#E8ECF1", domainColor=TRAIT,
                        tickColor=TRAIT, labelFontSize=11, titleFontSize=11, titleFontWeight="normal")
        .configure_legend(labelColor=GRIS, titleColor=GRIS, labelFontSize=11, orient="bottom", title=None)
    )


def afficher_graphique(chart, hauteur=280):
    chart = style_graphique(chart, hauteur)
    try:
        st.altair_chart(chart, theme=None, width="stretch")
    except TypeError:                       # versions plus anciennes de Streamlit
        st.altair_chart(chart, theme=None, use_container_width=True)


def graphique_lignes(df, x_titre, y_titre, cible=None, domaine=(0, 1), ordre_series=None, hauteur=280,
                     couleurs=None):
    """df : colonnes x (texte), y (nombre), serie (texte)."""
    series = ordre_series or list(dict.fromkeys(df["serie"]))
    base = alt.Chart(df).encode(
        x=alt.X("x:O", title=x_titre, axis=alt.Axis(labelAngle=0), sort=list(dict.fromkeys(df["x"]))),
        y=alt.Y("y:Q", title=y_titre, scale=alt.Scale(domain=list(domaine))),
        color=alt.Color("serie:N", scale=alt.Scale(domain=series, range=(couleurs or PALETTE)[:len(series)]),
                        legend=alt.Legend(title=None)),
        tooltip=[alt.Tooltip("serie:N", title="Série"), alt.Tooltip("x:O", title=x_titre),
                 alt.Tooltip("y:Q", title=y_titre, format=".3f")],
    )
    couches = [base.mark_line(strokeWidth=2.2), base.mark_point(size=55, filled=True)]
    if cible is not None:
        couches.append(alt.Chart(pd.DataFrame({"c": [cible]})).mark_rule(strokeDash=[5, 4], color=GRIS).encode(y="c:Q"))
    afficher_graphique(alt.layer(*couches), hauteur)


# ======================================================================================
# Modèles, exemples, images
# ======================================================================================
@st.cache_resource(show_spinner="Chargement du modèle")
def charger(nom):
    chemin = DOSSIER_MODELES / f"{nom}.keras"
    if not chemin.exists():
        return None
    return keras.models.load_model(chemin, compile=False)


def existe(nom):
    return (DOSSIER_MODELES / f"{nom}.keras").exists()


def existe_tous(noms):
    return all(existe(n) for n in noms)


def systemes_disponibles():
    return [k for k, s in SYSTEMES.items() if existe_tous(s["fichiers"])]


@st.cache_data
def charger_exemples():
    if not FICHIER_EXEMPLES.exists():
        return None
    d = np.load(FICHIER_EXEMPLES)
    sortie = {}
    for s in ("A", "B"):
        sortie[s] = dict(
            X=d[f"X_{s}"].astype("float32") / 255.0,
            montant=d[f"montant_{s}"].astype(str),
            altere=d[f"altere_{s}"],
            type=d[f"type_{s}"] if f"type_{s}" in d.files else None,
        )
    return sortie


def preparer_image(fichier):
    """Image téléversée -> tableau 32 x 128 dans [0, 1], encre claire sur fond sombre."""
    img = Image.open(fichier).convert("L").resize((IMG_W, IMG_H), Image.LANCZOS)
    x = np.asarray(img, dtype="float32") / 255.0
    if x.mean() > 0.5:          # encre sombre sur fond clair : on inverse
        x = 1.0 - x
    return x


def vignette(x, facteur=4):
    img = Image.fromarray((np.clip(x, 0, 1) * 255).astype("uint8"))
    return img.resize((IMG_W * facteur, IMG_H * facteur), Image.NEAREST)


def corrompre(X, rng, rot_max=2.0, flou=(0.6, 1.4), fond=(0.09, 0.20), contraste=(0.55, 0.85),
              raies=(0.02, 0.07), lambda_=(10, 30), taches=0.04, bruit=(0.03, 0.07), bruit_flou=0.7):
    """Simulateur du scanner B du jalon 3 (même fonction que dans le notebook)."""
    N, H, W, _ = X.shape
    out = np.empty_like(X)
    yy, xx = np.mgrid[0:H, 0:W].astype("float32")
    for i in range(N):
        img = rotate(X[i, :, :, 0], rng.uniform(-rot_max, rot_max), reshape=False, order=1, mode="constant", cval=0.0)
        img = gaussian_filter(img, rng.uniform(*flou))
        img = rng.uniform(*fond) + rng.uniform(*contraste) * img
        th, lam, ph = rng.uniform(0.3, 1.2), rng.uniform(*lambda_), rng.uniform(0, 2 * np.pi)
        img = img + rng.uniform(*raies) * np.sin(2 * np.pi * (xx * np.cos(th) + yy * np.sin(th)) / lam + ph)
        t = gaussian_filter(rng.normal(0, 1, (H, W)), 5)
        img = img + taches * t / (t.std() + 1e-8)
        n = gaussian_filter(rng.normal(0, 1, (H, W)), bruit_flou)
        n = n / (n.std() + 1e-8)
        img = img + rng.uniform(*bruit) * n
        out[i, :, :, 0] = np.clip(img, 0, 1)
    return out


def imiter_scanner_b(x, graine):
    X = x[None, :, :, None].astype("float32")
    return corrompre(X, np.random.default_rng(graine))[0, :, :, 0]


# ======================================================================================
# Lecture par positions (jalons 1 à 3)
# ======================================================================================
def sorties_dict(modele, P):
    return P if isinstance(P, dict) else dict(zip(modele.output_names, P))


def probs_positions(modele, X):
    P = sorties_dict(modele, modele.predict(X, verbose=0, batch_size=256))
    return np.stack([P[f"pos{k}"] for k in range(L_MAX)], axis=1)          # (N, 5, 11)


def lire_probs(probs):
    cls = probs.argmax(2)
    conf = probs.max(2).prod(1)                                             # produit des 5 maxima
    lus = np.array(["".join(str(c) for c in ligne[::-1] if c != EMPTY) for ligne in cls])
    return lus, conf


def debruiter(X):
    return np.clip(charger("debruiteur_v4").predict(X, verbose=0, batch_size=256), 0.0, 1.0)


def resultat(lus, conf, rang=None, accord=None, **autres):
    n = len(lus)
    return dict(lus=np.asarray(lus), conf=np.asarray(conf, dtype=float),
                rang=np.asarray(conf if rang is None else rang, dtype=float),
                accord=np.ones(n, bool) if accord is None else np.asarray(accord, bool), **autres)


def lecture_positions(nom, X):
    probs = probs_positions(charger(nom), X)
    lus, conf = lire_probs(probs)
    return resultat(lus, conf, probs=probs)


def lecture_ensemble(X):
    probs = [probs_positions(charger(n), X) for n in ENSEMBLE_J3]
    lus, conf = lire_probs(np.mean(probs, axis=0))
    membres = []
    for n, p in zip(ENSEMBLE_J3, probs):
        l, c = lire_probs(p)
        membres.append(dict(nom=n, lus=l, conf=c))
    return resultat(lus, conf, probs=np.mean(probs, axis=0), membres=membres)


# ======================================================================================
# Lecture en séquence (jalon 4) : décodage glouton pas à pas
# ======================================================================================
def construire_inference(m):
    cellule = "gru" if isinstance(m.get_layer("rnn_dec"), keras.layers.GRU) else "lstm"
    unites = m.get_layer("rnn_dec").units
    sorties_enc = [m.get_layer("h0").output] + ([m.get_layer("c0").output] if cellule == "lstm" else [])
    enc = keras.Model(m.input[0], sorties_enc)
    tok = keras.Input((1,), dtype="int32")
    etats_in = [keras.Input((unites,)) for _ in sorties_enc]
    res = m.get_layer("rnn_dec")(m.get_layer("emb")(tok), initial_state=etats_in)
    pas = keras.Model([tok] + etats_in, [m.get_layer("jeton")(res[0])] + list(res[1:]))
    return enc, pas


@st.cache_resource(show_spinner="Chargement du modèle")
def inference_sequence(nom):
    m = charger(nom)
    return None if m is None else construire_inference(m)


def decoder_glouton(enc, pas, X, lot=256):
    N = len(X)
    jetons = np.zeros((N, T_MAX), dtype=int)
    probs = np.zeros((N, T_MAX), dtype="float32")
    for d0 in range(0, N, lot):
        xb = X[d0:d0 + lot]
        etats = enc.predict_on_batch(xb)
        etats = [etats] if not isinstance(etats, (list, tuple)) else list(etats)
        tok = np.full((len(xb), 1), DEBUT, dtype="int32")
        for t in range(T_MAX):
            sortie = pas.predict_on_batch([tok] + etats)
            p = sortie[0][:, 0, :]
            etats = list(sortie[1:])
            suivant = p.argmax(1)
            jetons[d0:d0 + len(xb), t] = suivant
            probs[d0:d0 + len(xb), t] = p.max(1)
            tok = suivant[:, None].astype("int32")
    return jetons, probs


def lire_sequence(jetons, probs):
    """Montant lu, confiance (produit des probabilités jusqu'au jeton de fin inclus)."""
    lus, conf = [], np.zeros(len(jetons))
    for i in range(len(jetons)):
        s, c = "", 1.0
        for t in range(T_MAX):
            c *= probs[i, t]
            if jetons[i, t] >= PAD:
                break
            s += str(jetons[i, t])
        lus.append(s)
        conf[i] = c
    return np.array(lus), conf


def lecture_sequence(noms, X):
    membres = []
    for nom in noms:
        enc, pas = inference_sequence(nom)
        jetons, probs = decoder_glouton(enc, pas, X)
        lus, conf = lire_sequence(jetons, probs)
        membres.append(dict(nom=nom, lus=lus, conf=conf, jetons=jetons, probs=probs))
    if len(membres) == 1:
        m = membres[0]
        return resultat(m["lus"], m["conf"], membres=membres)
    a, b = membres
    accord = a["lus"] == b["lus"]
    produit = a["conf"] * b["conf"]
    lus = np.where(a["conf"] >= b["conf"], a["lus"], b["lus"])
    rang = np.where(accord, 1.0 + produit, 0.5 * np.maximum(a["conf"], b["conf"]))
    return resultat(lus, produit, rang=rang, accord=accord, membres=membres)


# ======================================================================================
# Point d'entrée de lecture, avec mémoire pour l'interface
# ======================================================================================
def lire(cle, X):
    if cle == "j1":
        return lecture_positions("mlp_base_80ep", X)
    if cle == "j2":
        return lecture_positions("cnn_final_80ep", X)
    if cle == "j2d":
        return lecture_positions("cnn_final_80ep", debruiter(X))
    if cle == "j3e":
        return lecture_ensemble(X)
    if cle == "j3m":
        return lecture_positions("multi_lam2", X)
    if cle == "j4g":
        return lecture_sequence(["seq_gru128"], X)
    if cle == "j4p":
        return lecture_sequence(PAIRE_J4, X)
    if cle == "j4a":
        return lecture_sequence(PAIRE_J4_AUG, X)
    raise KeyError(cle)


def octets(X):
    return np.ascontiguousarray(X, dtype="float32").tobytes()


@st.cache_data(show_spinner=False, max_entries=256)
def lire_une(cle, x_octets):
    X = np.frombuffer(x_octets, dtype="float32").reshape(1, IMG_H, IMG_W, 1)
    return lire(cle, X)


@st.cache_data(show_spinner=False, max_entries=512)
def score_alteration(fichier, multi, x_octets, n):
    X = np.frombuffer(x_octets, dtype="float32").reshape(n, IMG_H, IMG_W, 1)
    m = charger(fichier)
    P = m.predict(X, verbose=0, batch_size=256)
    if multi:
        P = sorties_dict(m, P)["alt"]
    return np.asarray(P).ravel()


@st.cache_data(show_spinner=False, max_entries=64)
def debruiter_une(x_octets):
    X = np.frombuffer(x_octets, dtype="float32").reshape(1, IMG_H, IMG_W, 1)
    return debruiter(X)[0, :, :, 0]


def seuil_detecteur(d, cout=COUT_DEFAUT):
    return COUTS_J2[cout][0] if d["cle"] == "d2" else 0.5


# ======================================================================================
# Mesures
# ======================================================================================
def taux_automatisation(lus, montants, rang, seuil=0.99):
    ordre = np.argsort(-rang, kind="stable")
    ok = (lus[ordre] == montants[ordre]).astype(float)
    cum = np.cumsum(ok) / np.arange(1, len(ok) + 1)
    bons = np.where(cum >= seuil)[0]
    return float((bons.max() + 1) / len(ok)) if len(bons) else 0.0


def exactitude_par_longueur(lus, montants):
    lg = np.char.str_len(montants)
    return {n: float(np.mean(lus[lg == n] == montants[lg == n])) for n in range(1, L_MAX + 1) if (lg == n).any()}


def courbe_couverture(lus, montants, rang):
    ordre = np.argsort(-rang, kind="stable")
    ok = (lus[ordre] == montants[ordre]).astype(float)
    return np.arange(1, len(ok) + 1) / len(ok), np.cumsum(ok) / np.arange(1, len(ok) + 1)


def auc(y, s):
    y = np.asarray(y).astype(bool)
    if y.all() or (~y).all():
        return float("nan")
    r = pd.Series(np.asarray(s)).rank().to_numpy()
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def decision(conf, seuil, accord=True):
    ok = accord and conf >= seuil
    return badge("Traitement automatique", "ok") if ok else badge("Revue par un opérateur", "warn")


def verdict(lecture, montant):
    if montant is None:
        return badge("Non évalué", "neutre")
    return badge("Correct", "ok") if lecture == montant else badge("Incorrect", "bad")


# ======================================================================================
# Choix du chèque (menu de gauche)
# ======================================================================================
def _hasard(cle, n):
    st.session_state[cle] = int(np.random.randint(0, n))


def choisir_cheque():
    sb = st.sidebar
    sb.markdown('<div class="sb-titre">Chèque à lire</div>', unsafe_allow_html=True)
    exemples = charger_exemples()
    options = ["Image téléversée"]
    if exemples is not None:
        options = ["Exemple, scanner A", "Exemple, scanner B"] + options
    source = sb.radio("Source", options, label_visibility="collapsed", key="source")

    c = None
    if source.startswith("Exemple"):
        s = "A" if source.endswith("A") else "B"
        d = exemples[s]
        n = len(d["X"])
        cle = f"i_{s}"
        if cle not in st.session_state:
            st.session_state[cle] = 0
        sb.number_input(f"Numéro de l'exemple (0 à {n - 1})", min_value=0, max_value=n - 1, step=1, key=cle)
        sb.button("Exemple au hasard", key=f"hasard_{s}", on_click=_hasard, args=(cle, n))
        i = int(st.session_state[cle])
        type_alt = int(d["type"][i]) if d["type"] is not None else None
        c = dict(x=d["X"][i], scanner=s, montant=str(d["montant"][i]), altere=int(d["altere"][i]),
                 type=type_alt, origine=f"Exemple {i}")
    else:
        fichier = sb.file_uploader("Image du chèque", type=["png", "jpg", "jpeg"])
        scanner = sb.radio("Scanner d'origine", ["A", "B"], horizontal=True, key="scanner_up")
        if fichier is not None:
            c = dict(x=preparer_image(fichier), scanner=scanner, montant=None, altere=None, type=None,
                     origine="Image téléversée")

    if c is not None:
        c["simule"] = False
        if c["scanner"] == "A":
            sb.markdown('<div class="sb-titre">Simulation</div>', unsafe_allow_html=True)
            if sb.checkbox("Dégrader l'image comme le scanner B", key="simu",
                           help="Applique le simulateur du jalon 3 (rotation, flou, fond, contraste, raies, "
                                "taches, grain) à une image du scanner A."):
                graine = sb.number_input("Graine de la dégradation", 0, 9999, 0, key="graine_simu")
                c["x"] = imiter_scanner_b(c["x"], int(graine))
                c["simule"] = True
    return c


def reglage_seuil():
    sb = st.sidebar
    sb.markdown('<div class="sb-titre">Décision</div>', unsafe_allow_html=True)
    return sb.slider("Seuil de confiance", 0.0, 1.0, 0.80, 0.01,
                     help="Au-dessus du seuil, et si les modèles s'accordent, le chèque est traité automatiquement.")


def etat_modeles():
    tous = set()
    for s in SYSTEMES.values():
        tous.update(s["fichiers"])
    tous.update(d["fichier"] for d in DETECTEURS)
    obligatoires = tous - set(PAIRE_J4_AUG)
    absents = sorted(n for n in obligatoires if not existe(n))
    st.sidebar.markdown('<div class="sb-titre">Modèles</div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        f'<span class="legende">{len(obligatoires) - len(absents)} sur {len(obligatoires)} fichiers présents '
        f"dans le dossier models.</span>", unsafe_allow_html=True)
    if absents:
        with st.sidebar.expander("Fichiers absents"):
            st.markdown("\n".join(f"- {n}.keras" for n in absents))


def bloc_cheque(c):
    if c is None:
        st.info("Choisissez un exemple ou téléversez une image dans le menu de gauche.")
        return False
    gauche, droite = st.columns([3, 2])
    with gauche:
        st.image(vignette(c["x"], 4), width=IMG_W * 4)
    with droite:
        puces = [badge(f"Scanner {c['scanner']}", "neutre"), badge(c["origine"], "neutre")]
        if c["simule"]:
            puces.append(badge("Dégradation simulée", "warn"))
        if c["altere"] is not None:
            if c["altere"]:
                t = f", type {c['type']}" if c.get("type") not in (None, 0) else ""
                puces.append(badge(f"Chèque altéré{t}", "warn"))
            else:
                puces.append(badge("Chèque normal", "ok"))
        st.markdown(f'<div class="puces">{"".join(puces)}</div>', unsafe_allow_html=True)
        kpis([("Montant réel", c["montant"] if c["montant"] is not None else "inconnu",
               "étiquette du jeu de données" if c["montant"] is not None else "image téléversée")])
    return True


# ======================================================================================
# Éléments communs aux pages
# ======================================================================================
def table_positions(probs):
    lignes = []
    for k in range(L_MAX):
        cl = int(probs[k].argmax())
        lignes.append({"Position": NOMS_POS[k], "Lecture": "vide" if cl == EMPTY else str(cl),
                       "Probabilité": fr(float(probs[k].max()))})
    return pd.DataFrame(lignes)


def distribution_position(probs, cle):
    choix = st.selectbox("Distribution de probabilité d'une position", NOMS_POS, key=cle)
    k = NOMS_POS.index(choix)
    df = pd.DataFrame({"classe": [str(i) for i in range(10)] + ["vide"], "p": probs[k].astype(float)})
    ch = alt.Chart(df).mark_bar(color=ACCENT, size=22).encode(
        x=alt.X("classe:O", title="Classe", sort=list(df["classe"]), axis=alt.Axis(labelAngle=0)),
        y=alt.Y("p:Q", title="Probabilité", scale=alt.Scale(domain=[0, 1])),
        tooltip=[alt.Tooltip("classe:O", title="Classe"), alt.Tooltip("p:Q", title="Probabilité", format=".3f")])
    afficher_graphique(ch, 200)


def ligne_lecture(nom, r, c, seuil):
    lus, conf, acc = r["lus"][0], float(r["conf"][0]), bool(r["accord"][0])
    return [nom, lus or "vide", fr(conf), decision(conf, seuil, acc), verdict(lus, c["montant"])]


def est_scanner_b(c):
    return c["scanner"] == "B" or c["simule"]


def table_detecteurs(c, cout=COUT_DEFAUT, limiter=None):
    x_oct = octets(c["x"][None, :, :, None])
    lignes = []
    for d in DETECTEURS:
        if limiter and d["cle"] not in limiter:
            continue
        if not existe(d["fichier"]):
            continue
        s = float(score_alteration(d["fichier"], d["multi"], x_oct, 1)[0])
        seuil = seuil_detecteur(d, cout)
        alerte = s >= seuil
        lignes.append([d["nom"], fr(s, 4), fr(seuil, 4),
                       badge("Altéré", "warn") if alerte else badge("Normal", "ok"), d["ref"]])
    return pd.DataFrame(lignes, columns=["Détecteur", "Score", "Seuil", "Verdict", "Performance de référence"])


def vue_verite_alteration(c):
    if c["altere"] is not None:
        legende(f"Vérité terrain : chèque {'altéré' if c['altere'] else 'normal'}. Les scores ne sont pas des "
                "probabilités calibrées : les détecteurs sont entraînés avec des poids de classe.")


def avertissement_scanner(c):
    if est_scanner_b(c):
        note("Les jalons 1 et 4 ont été entraînés uniquement sur des images du scanner A (voir leurs pages) : leur "
             "lecture d'une image de type scanner B n'est pas fiable. Le jalon 2 y ajoute un débruitage, le "
             "jalon 3 entraîne les lecteurs sur des images de A dégradées pour imiter B.", "alerte")


# ======================================================================================
# Pages
# ======================================================================================
JALONS = [
    dict(num="1", titre="Lecture par positions", technique="MLP, cinq têtes softmax",
         fait="Un perceptron multicouche reçoit l'image aplatie (4 096 valeurs) et lit le montant par cinq têtes, "
              "des unités aux dizaines de milliers. La confiance est le produit des cinq probabilités maximales.",
         constat="Les montants d'un chiffre sont lus à 78 %, ceux de quatre et cinq chiffres presque jamais. Chaque "
                 "pixel a son propre poids : le réseau réapprend chaque chiffre à chaque endroit. La confiance est "
                 "calibrée mais peu discriminante.",
         suite="Il faut partager les poids sur toute l'image : c'est le principe de la convolution.",
         valeur="0,116", mesure="montant exact, scanner A (cible 0,85)"),
    dict(num="2", titre="Convolutions, détection et débruitage", technique="CNN à quatre blocs, détecteur, auto-encodeur",
         fait="Un CNN à quatre blocs remplace le MLP avec dix fois moins de paramètres (231 k). Un détecteur "
              "d'altération dédié est réglé par un compromis de coûts : une fraude manquée vaut 50 revues. Un "
              "auto-encodeur débruite les images du scanner B.",
         constat="Le montant exact passe à 0,699 et la détection atteint 0,9996 d'AUC. Sur le scanner B, le même "
                 "lecteur tombe à 0,001 ; le débruitage le remonte à 0,633.",
         suite="Débruiter ne suffit pas : il faut apprendre au lecteur lui-même à supporter le nouveau scanner.",
         valeur="0,699", mesure="montant exact, scanner A (cible 0,85)"),
    dict(num="3", titre="Un nouveau scanner, deux tâches", technique="Augmentation, ensemble de 4 CNN, multitâche",
         fait="Deux voies sont comparées : réentraîner avec les 600 images étiquetées de B, ou imiter B en dégradant "
              "les images de A. Le système retenu combine un ensemble de quatre CNN et un détecteur entraîné avec la "
              "même augmentation. Un modèle multitâche unique est aussi étudié.",
         constat="L'imitation de B bat l'ajustement fin, sans aucune étiquette de B et sans oublier A. L'ensemble "
                 "atteint 0,715 sur B. Le multitâche ne résout pas le compromis entre lecture et détection.",
         suite="Reste la limite des montants longs : cinq têtes doivent toutes avoir raison. Lire en séquence évite "
               "cette contrainte.",
         valeur="0,715", mesure="montant exact, scanner B (cible 0,70)"),
    dict(num="4", titre="Lire le montant comme une séquence", technique="Encodeur-décodeur GRU, paire de modèles",
         fait="Un encodeur-décodeur à cellules GRU lit le montant chiffre par chiffre et décide lui-même de "
              "s'arrêter. Deux modèles de graines différentes forment une paire : leur accord classe les chèques les "
              "plus sûrs en tête.",
         constat="À budget égal, GRU et LSTM sont équivalents, et la séquence dépasse nettement la lecture par "
                 "positions sur les longs montants. La paire atteint 0,829 avec 3,74 Mo et 27 à 28 ms par chèque.",
         suite="La cible de 0,85 est manquée de 0,021. Piste suivante : une attention entre le décodeur et les "
               "colonnes de l'image.",
         valeur="0,829", mesure="montant exact, scanner A (cible 0,85)"),
]


def bandeau_accueil():
    stats = [("4", "jalons successifs"), ("0,116 → 0,829", "montant exact, scanner A"),
             ("2", "scanners à lire"), ("47 %", "des montants ont 4 ou 5 chiffres")]
    blocs = "".join(f'<div class="hero-stat"><b>{esc(v)}</b><span>{esc(l)}</span></div>' for v, l in stats)
    st.markdown(
        '<div class="hero">'
        '<div class="hero-puces"><span>ENSET Mohammedia</span><span>M2 SDIA</span><span>Advanced AI</span></div>'
        "<h1>Lecture automatique des chèques</h1>"
        '<p class="hero-texte">Une banque fictive veut lire automatiquement le montant de ses chèques, repérer ceux '
        "qui ont été altérés et décider quand la lecture est assez sûre pour se passer d'un opérateur. Le même "
        "problème est repris quatre fois, chaque jalon partant de la limite mesurée au précédent.</p>"
        '<div class="hero-equipe">HAFSSA MIFTAH IDRISSI<i>/</i>HANANE AIT LHAJ<i>/</i>FATIMA ZAHRA LASFAR</div>'
        f'<div class="hero-stats">{blocs}</div></div>', unsafe_allow_html=True)


def cartes_probleme():
    cartes = [("01", "Lire", "Le montant, de 1 à 5 chiffres, sur des images de 32 × 128 pixels en niveaux de gris."),
              ("02", "Détecter", "Les chèques altérés, environ 10 % du total, sans laisser passer de fraude."),
              ("03", "Décider", "Quand la confiance suffit pour traiter sans opérateur, avec une précision de 99 %."),
              ("04", "S'adapter", "Un second scanner, plus pâle, flou et granuleux, avec très peu d'étiquettes.")]
    blocs = "".join(f'<div class="carte-pb"><div class="nb">{n}</div><h4>{esc(t)}</h4><p>{esc(p)}</p></div>'
                    for n, t, p in cartes)
    st.markdown(f'<div class="grille">{blocs}</div>', unsafe_allow_html=True)


def parcours_jalons():
    etapes = []
    for j in JALONS:
        etapes.append(
            f'<div class="etape"><div class="etape-num"><span>{j["num"]}</span></div><div class="etape-corps">'
            f'<div class="etape-tete"><div><div class="etape-sur">Jalon {j["num"]}</div>'
            f'<div class="etape-titre">{esc(j["titre"])}</div><div class="etape-tech">{esc(j["technique"])}</div></div>'
            f'<div class="etape-res"><div class="etape-val">{esc(j["valeur"])}</div>'
            f'<div class="etape-mes">{esc(j["mesure"])}</div></div></div>'
            f'<div class="etape-grille"><div><b>Ce qui a été construit</b>{esc(j["fait"])}</div>'
            f'<div><b>Ce que les mesures ont montré</b>{esc(j["constat"])}</div>'
            f'<div class="suite"><b>Ce qui mène à la suite</b>{esc(j["suite"])}</div></div></div></div>')
    st.markdown("".join(etapes), unsafe_allow_html=True)
    legende("Résultats au test, évalué une seule fois par jalon.")


def page_vue_ensemble():
    bandeau_accueil()
    section("Le problème", "quatre difficultés à résoudre")
    cartes_probleme()
    section("Le parcours en quatre jalons", "chaque étape part d'une limite mesurée à la précédente")
    parcours_jalons()

    section("Résultats au test et cibles")
    ok, ko = badge("Atteinte", "ok"), badge("Non atteinte", "warn")
    tableau(pd.DataFrame([
        ["1", "MLP, cinq têtes de position", "Montant exact (A)", "0,116", "0,85", ko],
        ["1", "", "Automatisation à 99 % (A)", "0,005", "0,40", ko],
        ["2", "CNN à 4 blocs, détecteur, auto-encodeur", "Montant exact (A)", "0,699", "0,85", ko],
        ["2", "", "AUC des altérations (A)", "0,9996", "0,95", ok],
        ["2", "", "Montant exact (B, débruité)", "0,633", "0,70", ko],
        ["3", "Ensemble de 4 CNN, détecteur dédié, augmentation", "Montant exact (B)", "0,715", "0,70", ok],
        ["3", "", "Automatisation (B)", "0,260", "0,15", ok],
        ["3", "", "AUC des altérations (A)", "0,9974", "0,95", ok],
        ["4", "Paire de GRU, encodeur-décodeur", "Montant exact (A)", "0,829", "0,85", ko],
        ["4", "", "Automatisation à 99 % (A)", "0,469", "0,40", ok],
        ["4", "", "Taille des modèles", "3,74 Mo", "5 Mo", ok],
        ["4", "", "Latence par chèque", "27 à 28 ms", "50 ms", ok],
    ], columns=["Jalon", "Technique retenue", "Indicateur", "Résultat", "Cible", "Statut"]), brut=["Statut"])
    legende("Tous les réglages ont été faits sur la validation ; le jeu de test n'a été utilisé qu'une seule fois "
            "par jalon, après déclaration du système retenu dans le journal d'expériences.")

    section("Exactitude selon le nombre de chiffres", "scanner A, test")
    donnees = {"Jalon 1 : MLP": [0.777, 0.290, 0.113, 0.007, 0.0],
               "Jalon 2 : CNN": [0.969, 0.926, 0.840, 0.615, 0.310],
               "Jalon 3 : ensemble": [0.982, 0.945, 0.892, 0.713, 0.443],
               "Jalon 4 : paire de GRU": [0.987, 0.939, 0.911, 0.806, 0.578]}
    df = pd.DataFrame([dict(serie=s, x=str(i + 1), y=v) for s, vals in donnees.items() for i, v in enumerate(vals)])
    gauche, droite = st.columns([3, 2])
    with gauche:
        graphique_lignes(df, "Nombre de chiffres du montant", "Montant exact", cible=0.85)
    with droite:
        st.markdown(
            "L'écart entre jalons se creuse avec la longueur du montant. Le MLP lit les montants d'un chiffre "
            "et presque aucun montant de quatre ou cinq chiffres ; le CNN partage ses poids sur toute l'image ; "
            "la lecture en séquence (jalon 4) laisse le décodeur choisir lui-même la longueur. Les montants de "
            "quatre et cinq chiffres représentent 47 % du test : c'est là que se joue la cible de 0,85.")
        legende("Ligne en tirets : cible de 0,85.")

    section("Données")
    st.markdown(
        "Les images (32 × 128 pixels, niveaux de gris) viennent de chiffres MNIST assemblés en montants de 1 à "
        "5 chiffres. Le scanner A est le scanner de référence (15 000 images d'entraînement). Le scanner B produit "
        "des images plus pâles, floues et granuleuses (600 images étiquetées, utilisées uniquement pour la "
        "validation et le test au jalon 3). Environ 10 % des chèques sont altérés.")

    section("Contenu de l'application")
    tableau(pd.DataFrame([
        ["Atelier de lecture", "Un même chèque lu par tous les systèmes, avec décision et détection d'altération."],
        ["Jalons 1 à 4", "Essai, méthode et résultats de chaque jalon, avec ses propres modèles."],
        ["Détection d'altération", "Détecteurs dédiés et multitâche, seuil selon le coût d'une fraude manquée."],
        ["Évaluation par lot", "Mesure de l'exactitude et de l'automatisation sur un échantillon, par système."],
        ["Journal d'expériences", "Les expériences enregistrées au cours du projet, consultables et filtrables."],
    ], columns=["Page", "Contenu"]))


def page_atelier(c, seuil):
    entete("Atelier de lecture",
           "Le chèque choisi dans le menu de gauche est lu par tous les systèmes construits pendant les quatre "
           "jalons, puis soumis aux détecteurs d'altération.", "Démonstration comparative")
    if not bloc_cheque(c):
        return
    x_oct = octets(c["x"][None, :, :, None])

    section("Lecture du montant")
    cles = ["j1", "j2"] + (["j2d"] if est_scanner_b(c) else []) + ["j3e", "j3m", "j4g", "j4p", "j4a"]
    lignes, absents = [], []
    for cle in cles:
        s = SYSTEMES[cle]
        if not existe_tous(s["fichiers"]):
            if cle != "j4a":
                absents.append(s["nom"])
            continue
        lignes.append(ligne_lecture(s["nom"], lire_une(cle, x_oct), c, seuil))
    if lignes:
        if c["montant"] is not None:
            n_ok = sum("Correct" in str(l[4]) for l in lignes)
            kpis([("Systèmes corrects", f"{n_ok} sur {len(lignes)}", "lecture identique au montant réel"),
                  ("Traités automatiquement", f"{sum('automatique' in str(l[3]) for l in lignes)} sur {len(lignes)}",
                   f"au seuil de confiance de {fr(seuil, 2)}")])
        tableau(pd.DataFrame(lignes, columns=["Système", "Lecture", "Confiance", "Décision", "Vérification"]),
                brut=["Décision", "Vérification"])
    if absents:
        legende("Systèmes non évalués faute de fichiers de modèle : " + ", ".join(absents) + ".")
    avertissement_scanner(c)

    section("Détection d'altération")
    df = table_detecteurs(c, limiter={"d2", "d3", "m_multi_lam2"})
    if len(df):
        tableau(df, brut=["Verdict"])
    vue_verite_alteration(c)
    legende("Seuil du détecteur du jalon 2 : 0,0163 (une fraude manquée coûte 50 revues). Seuil 0,5 pour les autres. "
            "Tous les détecteurs, dont les six modèles multitâches, sont sur la page Détection d'altération.")


def page_j1(c, seuil):
    entete("Jalon 1 : lecture par positions avec un MLP",
           "Un perceptron multicouche lit le montant à travers cinq têtes de position.", "Jalon 1")
    essai, methode, resultats = st.tabs(["Essai", "Méthode", "Résultats"])
    with essai:
        if bloc_cheque(c) and existe("mlp_base_80ep"):
            r = lire_une("j1", octets(c["x"][None, :, :, None]))
            lus, conf = r["lus"][0], float(r["conf"][0])
            section("Lecture")
            kpis([("Lecture", lus or "vide", "montant lu"), ("Confiance", fr(conf), "produit des cinq maxima"),
                  ("Vérification", "inconnue" if c["montant"] is None else ("correct" if lus == c["montant"] else "incorrect"),
                   f"montant réel : {c['montant']}" if c["montant"] is not None else "")])
            st.markdown(decision(conf, seuil), unsafe_allow_html=True)
            section("Détail par position")
            gauche, droite = st.columns([2, 3])
            with gauche:
                tableau(table_positions(r["probs"][0]))
            with droite:
                distribution_position(r["probs"][0], "dist_j1")
        elif c is not None:
            st.warning("Fichier de modèle absent du dossier models : mlp_base_80ep.keras")
    with methode:
        st.markdown(
            "L'image est aplatie en 4 096 valeurs, puis traverse trois couches denses (512, 256 et 128 neurones, "
            "ReLU, dropout 0,3). Cinq têtes softmax de 11 classes terminent le réseau : une par position, des "
            "unités aux dizaines de milliers, la classe 10 signifiant « position vide ». Le montant lu est obtenu "
            "en prenant la classe la plus probable de chaque tête. La confiance est le produit des cinq "
            "probabilités maximales.")
        st.markdown("Entraînement : 80 époques, Adam au taux 1e-3, lots de 128, décalages horizontaux aléatoires de "
                    "±4 pixels, perte égale à la somme des cinq entropies croisées.")
        note("Le modèle chargé dans cette application est le MLP de référence du jalon 2, qui reprend l'architecture "
             "et le budget d'entraînement du modèle final du jalon 1 (2 268 983 paramètres).")
    with resultats:
        section("Test, scanner A", "évalué une seule fois")
        tableau(pd.DataFrame([["Montant exact", "0,116", "0,85"], ["Automatisation à 99 %", "0,005", "0,40"],
                              ["AUC de la confiance", "0,881", ""]], columns=["Indicateur", "Résultat", "Cible"]))
        section("Montant exact selon le nombre de chiffres")
        df = pd.DataFrame([dict(serie="Jalon 1 : MLP", x=str(i + 1), y=v)
                           for i, v in enumerate([0.777, 0.290, 0.113, 0.007, 0.0])])
        graphique_lignes(df, "Nombre de chiffres du montant", "Montant exact", cible=0.85, hauteur=230)
        st.markdown(
            "Le MLP donne un poids distinct à chaque pixel : il doit réapprendre chaque chiffre à chaque endroit "
            "de l'image, et un montant n'est correct que si les cinq têtes ont toutes raison. Il lit bien les "
            "montants d'un chiffre et presque aucun montant de quatre ou cinq chiffres.")


def page_j2(c, seuil):
    entete("Jalon 2 : CNN, détecteur d'altération et débruitage",
           "Un réseau convolutif remplace le MLP ; un détecteur signale les chèques altérés ; un auto-encodeur "
           "nettoie les images du scanner B.", "Jalon 2")
    essai, methode, resultats = st.tabs(["Essai", "Méthode", "Résultats"])
    with essai:
        if bloc_cheque(c):
            x_oct = octets(c["x"][None, :, :, None])
            if existe("cnn_final_80ep"):
                section("Lecture du montant")
                r = lire_une("j2", x_oct)
                lus, conf = r["lus"][0], float(r["conf"][0])
                kpis([("Lecture", lus or "vide", "montant lu"), ("Confiance", fr(conf), "produit des cinq maxima")])
                st.markdown(decision(conf, seuil), unsafe_allow_html=True)
                gauche, droite = st.columns([2, 3])
                with gauche:
                    tableau(table_positions(r["probs"][0]))
                with droite:
                    distribution_position(r["probs"][0], "dist_j2")
            if existe("detecteur_blocs4"):
                section("Détection d'altération")
                cout = st.select_slider("Rapport de coûts (fraude manquée / fausse alerte)",
                                        options=list(COUTS_J2), value=COUT_DEFAUT, key="cout_j2")
                s = float(score_alteration("detecteur_blocs4", False, x_oct, 1)[0])
                seuil_d, fa, rappel, manquees = COUTS_J2[cout]
                kpis([("Score", fr(s, 4), "sortie du détecteur"), ("Seuil", fr(seuil_d, 4), "fixé sur la validation"),
                      ("Verdict", "Altéré" if s >= seuil_d else "Normal", "")])
                legende(f"À ce rapport de coûts : {fa} de fausses alertes, rappel {rappel}, {manquees} fraudes "
                        "manquées sur 285 (validation).")
                vue_verite_alteration(c)
            if existe("debruiteur_v4") and existe("cnn_final_80ep"):
                section("Débruitage")
                propre = debruiter_une(x_oct)
                avant, apres = st.columns(2)
                r0 = lire_une("j2", x_oct)
                r1 = lire_une("j2d", x_oct)
                with avant:
                    st.markdown("**Image d'origine**")
                    st.image(vignette(c["x"], 3), width=IMG_W * 3)
                    st.markdown(f"Lecture : **{r0['lus'][0] or 'vide'}** (confiance {fr(float(r0['conf'][0]))})")
                with apres:
                    st.markdown("**Après débruitage**")
                    st.image(vignette(propre, 3), width=IMG_W * 3)
                    st.markdown(f"Lecture : **{r1['lus'][0] or 'vide'}** (confiance {fr(float(r1['conf'][0]))})")
                legende("Le débruiteur a été entraîné sur des images de A dégradées ; il est utile surtout pour "
                        "les images de type scanner B.")
    with methode:
        section("Lecteur convolutif")
        st.markdown(
            "Quatre blocs (convolution 3 × 3, max-pooling 2 × 2) à 32, 64, 64 et 64 filtres, un aplatissement, une "
            "couche dense de 128 neurones et les cinq têtes de position. Le modèle compte 230 967 paramètres, soit "
            "dix fois moins que le MLP. Les convolutions partagent leurs poids sur toute l'image ; le champ réceptif "
            "passe de 22 à 46 pixels avec le quatrième bloc, ce qui a donné environ 0,14 de gain en validation. "
            "L'aplatissement est indispensable : avec une moyenne globale, la position est perdue et le modèle "
            "n'apprend rien.")
        section("Détecteur d'altération")
        st.markdown(
            "Même corps convolutif, une sortie sigmoïde et des poids de classe (environ 10 % de chèques altérés). "
            "Le seuil minimise le coût 1 × fausses alertes + 50 × fraudes manquées sur la validation.")
        section("Débruitage du scanner B")
        st.markdown(
            "Les images de A sont dégradées (rotation, flou, fond relevé, contraste réduit, raies, taches, grain) "
            "avec des statistiques calées sur celles de B, sans utiliser ses étiquettes. Un auto-encodeur convolutif "
            "à connexions directes apprend à retrouver l'image propre, avec une sortie linéaire et une perte "
            "pondérée sur l'encre. Une première version à sortie sigmoïde s'était effondrée en images noires.")
    with resultats:
        section("Test", "évalué une seule fois")
        tableau(pd.DataFrame([
            ["Montant exact, scanner A", "0,699 ± 0,007", "0,85"],
            ["AUC des altérations, scanner A", "0,9996", "0,95"],
            ["Rappel au seuil de coût", "0,996 (2,2 % de fausses alertes)", ""],
            ["Montant exact, scanner B sans débruitage", "0,001", ""],
            ["Montant exact, scanner B avec débruitage", "0,633 ± 0,009", "0,70"],
        ], columns=["Indicateur", "Résultat", "Cible"]))
        section("Choix du seuil d'alerte", "validation")
        tableau(pd.DataFrame([[k, fr(v[0], 4), v[1], v[2], v[3]] for k, v in COUTS_J2.items()],
                             columns=["Rapport de coûts", "Seuil", "Fausses alertes", "Rappel", "Fraudes manquées"]))
        section("Montant exact selon le nombre de chiffres", "scanner A")
        df = pd.DataFrame([dict(serie="Jalon 2 : CNN", x=str(i + 1), y=v)
                           for i, v in enumerate([0.969, 0.926, 0.840, 0.615, 0.310])])
        graphique_lignes(df, "Nombre de chiffres du montant", "Montant exact", cible=0.85, hauteur=230)
        st.markdown(
            "Le test (0,699) est inférieur à la validation (0,757) qui a servi à choisir l'architecture. L'écart à "
            "la cible vient des montants de quatre et cinq chiffres, qui forment 47 % du test.")


def page_j3(c, seuil):
    entete("Jalon 3 : un nouveau scanner, deux tâches",
           "Adapter la lecture au scanner B sans étiquettes, puis lire et détecter les altérations avec un seul "
           "réseau (multitâche) ou deux réseaux dédiés.", "Jalon 3")
    essai, multi, methode, resultats = st.tabs(["Essai", "Multitâche", "Méthode", "Résultats"])
    with essai:
        if bloc_cheque(c):
            x_oct = octets(c["x"][None, :, :, None])
            if existe_tous(ENSEMBLE_J3):
                section("Lecture par l'ensemble de 4 CNN")
                r = lire_une("j3e", x_oct)
                lus, conf = r["lus"][0], float(r["conf"][0])
                kpis([("Lecture de l'ensemble", lus or "vide", "moyenne des probabilités"),
                      ("Confiance", fr(conf), "produit des cinq maxima")])
                st.markdown(decision(conf, seuil), unsafe_allow_html=True)
                tableau(pd.DataFrame([[m["nom"], m["lus"][0] or "vide", fr(float(m["conf"][0]))]
                                      for m in r["membres"]], columns=["Modèle", "Lecture", "Confiance"]))
                legende("L'ensemble moyenne les probabilités des quatre modèles avant de lire chaque position.")
            df = table_detecteurs(c, limiter={"d2", "d3"})
            if len(df):
                section("Détection d'altération")
                tableau(df, brut=["Verdict"])
                if est_scanner_b(c):
                    legende("Sur le scanner B, le détecteur du jalon 2 est proche du hasard (AUC 0,58 en validation) ; "
                            "celui du jalon 3 atteint 0,89 au test.")
    with multi:
        page_multitache(c, seuil)
    with methode:
        st.markdown("Le lecteur du jalon 2 tombe à 0,003 sur le scanner B. Deux familles de solutions ont été comparées.")
        section("Utiliser les 600 images étiquetées de B")
        st.markdown(
            "Entraînement depuis zéro, têtes seules, extraction de caractéristiques (convolutions gelées) et "
            "ajustement fin à plusieurs taux d'apprentissage, à partir du meilleur lecteur du jalon 2. Le meilleur "
            "ajustement (taux 3e-4) atteint environ 0,60 sur B mais fait retomber A de 0,757 à 0,58 : le modèle "
            "oublie le scanner de départ.")
        section("Imiter B à partir de A, sans étiquette de B")
        st.markdown(
            "Les images de A sont dégradées pour ressembler à B (rotation, flou, fond, contraste, raies, taches, "
            "grain) et un lecteur est entraîné sur un mélange moitié images propres, moitié images dégradées. Il lit "
            "B à 0,735 en validation et conserve 0,762 sur A. Le même principe appliqué au détecteur d'altération le "
            "fait passer de 0,58 à 0,94 d'AUC sur B.")
        section("Systèmes retenus")
        st.markdown(
            "Lecture : moyenne des probabilités de quatre CNN entraînés avec l'augmentation (un lecteur à mélange "
            "50/50, un lecteur 100 % dégradé, deux modèles multitâches). Détection : un détecteur dédié avec "
            "augmentation. Le modèle multitâche unique a été testé mais le compromis entre lecture et détection n'a "
            "pas pu être résolu par la pondération de la perte.")
    with resultats:
        section("Comparaison des stratégies", "validation")
        tableau(pd.DataFrame([
            ["Lecteur de départ (3 blocs, 15 époques)", "0", "0,003", "0,388"],
            ["Lecteur du jalon 2 (4 blocs)", "0", "0,003", "0,757"],
            ["Entraînement de zéro sur B", "600", "0,048", "n/a"],
            ["Têtes seules", "600", "0,055", "n/a"],
            ["Extraction (convolutions gelées)", "600", "0,195", "n/a"],
            ["Ajustement fin (lr 1e-3, 3e-4, 1e-4)", "600", "0,606 / 0,601 / 0,575", "0,58 (lr 3e-4)"],
            ["Augmentation 100 % dégradé", "0", "0,703", "0,654"],
            ["Augmentation 50/50", "0", "0,735", "0,762"],
            ["Augmentation 50/50 puis ajustement fin", "600", "0,750", "0,759"],
            ["Ensemble de 4 lecteurs", "0", "0,788", "0,815"],
        ], columns=["Stratégie", "Étiquettes de B", "Montant exact B", "Montant exact A"]))
        section("Test", "évalué une seule fois")
        tableau(pd.DataFrame([
            ["Montant exact, scanner B (ensemble)", "0,715", "0,70"],
            ["Automatisation, scanner B (ensemble)", "0,260", "0,15"],
            ["Montant exact, scanner A (ensemble)", "0,772", ""],
            ["Montant exact, scanner B (un seul lecteur)", "0,675", ""],
            ["AUC des altérations, scanner A (détecteur dédié)", "0,9974", "0,95"],
            ["AUC des altérations, scanner B (détecteur dédié)", "0,892 [0,872 ; 0,913]", ""],
            ["Multitâche λ = 2 : montant exact B / AUC A", "0,56 / 0,949", ""],
        ], columns=["Indicateur", "Résultat", "Cible"]))
        st.markdown(
            "L'ensemble gagne environ 0,04 sur un lecteur seul. Les deux cibles de lecture sont atteintes ; l'AUC de "
            "détection est atteinte sur A mais descend à 0,89 sur B.")


def graphiques_lambda():
    lams = ["0,1", "0,5", "1", "2", "4"]
    cles = ["multi_lam0.1", "multi_lam0.5", "multi_lam1", "multi_lam2", "multi_lam4"]
    lecture = pd.DataFrame([dict(serie=f"Scanner {s}", x=lam, y=REF_MULTI[k][i])
                            for lam, k in zip(lams, cles) for s, i in (("A", 0), ("B", 1))])
    detection = pd.DataFrame([dict(serie=f"Scanner {s}", x=lam, y=REF_MULTI[k][i])
                              for lam, k in zip(lams, cles) for s, i in (("A", 2), ("B", 3))])
    g, d = st.columns(2)
    with g:
        st.markdown("**Lecture : montant exact (validation)**")
        graphique_lignes(lecture, "Poids λ de la perte d'altération", "Montant exact", cible=0.70, hauteur=240,
                         couleurs=COULEURS_AB)
    with d:
        st.markdown("**Détection : AUC (validation)**")
        graphique_lignes(detection, "Poids λ de la perte d'altération", "AUC", cible=0.95, domaine=(0.5, 1), hauteur=240,
                         couleurs=COULEURS_AB)
    legende("Lignes en tirets : cibles de 0,70 (lecture sur B) et de 0,95 (AUC sur A). Aucun λ n'atteint les deux.")


def page_multitache(c, seuil):
    st.markdown(
        "Un seul réseau convolutif lit le montant par cinq têtes de position et prédit en même temps si le chèque "
        "est altéré. La perte totale est la somme des cinq entropies croisées et de λ fois la perte d'altération : "
        "λ règle le compromis entre les deux tâches.")
    section("Compromis lecture et détection", "validation")
    graphiques_lambda()
    tableau(pd.DataFrame(
        [[lab, fr(REF_MULTI[f][0]), fr(REF_MULTI[f][1]), fr(REF_MULTI[f][2]), fr(REF_MULTI[f][3])]
         for lab, f in MULTI_J3],
        columns=["Modèle", "Exact A", "Exact B", "AUC A", "AUC B"]))
    note("La détection s'améliore avec λ (AUC sur A de 0,61 à 0,99), la lecture se dégrade au-delà de 0,5. λ = 2 "
         "est le plus petit poids qui atteint la cible de détection sur A en validation (0,955). Au test, ce modèle "
         "manque les deux cibles (AUC sur A 0,949, lecture sur B 0,56) : le système retenu sépare donc la lecture "
         "(ensemble de 4 CNN) et la détection (détecteur dédié).")

    section("Essai sur le chèque choisi")
    if not bloc_cheque(c):
        return
    dispo = [(lab, f) for lab, f in MULTI_J3 if existe(f)]
    if not dispo:
        st.warning("Aucun fichier de modèle multitâche dans le dossier models.")
        return
    etiquettes = [l for l, _ in dispo]
    if len(etiquettes) == 1:
        lab = etiquettes[0]
    else:
        lab = st.select_slider("Modèle multitâche", options=etiquettes,
                               value="λ = 2" if "λ = 2" in etiquettes else etiquettes[0], key="lam_multi")
    f = dict(dispo)[lab]
    x_oct = octets(c["x"][None, :, :, None])
    X = c["x"][None, :, :, None]
    probs = probs_positions(charger(f), X)
    lus, conf = lire_probs(probs)
    s = float(score_alteration(f, True, x_oct, 1)[0])
    kpis([("Lecture", lus[0] or "vide", "branche de lecture"), ("Confiance", fr(float(conf[0])), ""),
          ("Score d'altération", fr(s, 4), "branche de détection"),
          ("Verdict", "Altéré" if s >= 0.5 else "Normal", "seuil 0,5")])
    st.markdown(decision(float(conf[0]), seuil), unsafe_allow_html=True)
    vue_verite_alteration(c)

    section("Même chèque : multitâche contre systèmes dédiés")
    lignes = [[f"Multitâche, {lab}", lus[0] or "vide", fr(float(conf[0])), fr(s, 4)]]
    if existe_tous(ENSEMBLE_J3):
        r = lire_une("j3e", x_oct)
        lignes.append(["Ensemble de 4 CNN + détecteur dédié", r["lus"][0] or "vide", fr(float(r["conf"][0])),
                       fr(float(score_alteration("detecteur_aug", False, x_oct, 1)[0]), 4)
                       if existe("detecteur_aug") else "n/a"])
    tableau(pd.DataFrame(lignes, columns=["Système", "Lecture", "Confiance", "Score d'altération"]))


def page_j4(c, seuil):
    entete("Jalon 4 : lire le montant comme une séquence",
           "Un encodeur-décodeur à cellules GRU lit les chiffres un par un et décide lui-même de s'arrêter.", "Jalon 4")
    essai, methode, resultats = st.tabs(["Essai", "Méthode", "Résultats"])
    with essai:
        if bloc_cheque(c):
            x_oct = octets(c["x"][None, :, :, None])
            if existe_tous(PAIRE_J4):
                r = lire_une("j4p", x_oct)
                lus, produit, accord = r["lus"][0], float(r["conf"][0]), bool(r["accord"][0])
                section("Lecture par la paire de GRU")
                kpis([("Lecture", lus or "vide", "modèle le plus confiant"),
                      ("Accord des deux modèles", "oui" if accord else "non", ""),
                      ("Produit des confiances", fr(produit), "")])
                st.markdown(decision(produit, seuil, accord), unsafe_allow_html=True)
                section("Décodage pas à pas")
                noms_jeton = {PAD: "remplissage", DEBUT: "début", FIN: "fin"}
                colonnes = st.columns(2)
                for col, m in zip(colonnes, r["membres"]):
                    with col:
                        lignes = []
                        for t in range(T_MAX):
                            j = int(m["jetons"][0][t])
                            lignes.append([t + 1, noms_jeton.get(j, str(j)), fr(float(m["probs"][0][t]))])
                            if j >= PAD:
                                break
                        st.markdown(f"**{m['nom']}** : lecture {m['lus'][0] or 'vide'}, confiance {fr(float(m['conf'][0]))}")
                        tableau(pd.DataFrame(lignes, columns=["Pas", "Jeton choisi", "Probabilité"]))
                legende("Le décodeur choisit à chaque pas le jeton le plus probable (décodage glouton) et s'arrête "
                        "au jeton de fin.")
                if est_scanner_b(c):
                    note("Ces modèles n'ont vu que des images du scanner A, sans dégradation de type scanner B : la "
                         "lecture d'une image de B ou simulée n'est pas fiable. La page Évaluation par lot mesure "
                         "l'écart.", "alerte")
            else:
                st.warning("Fichiers de modèle absents du dossier models : " + ", ".join(
                    f"{n}.keras" for n in PAIRE_J4 if not existe(n)))
            if existe_tous(PAIRE_J4_AUG):
                section("Extension : mêmes modèles avec les dégradations du scanner B", "hors protocole des jalons")
                r = lire_une("j4a", x_oct)
                kpis([("Lecture", r["lus"][0] or "vide", "paire entraînée avec dégradations"),
                      ("Produit des confiances", fr(float(r["conf"][0])), ""),
                      ("Accord", "oui" if bool(r["accord"][0]) else "non", "")])
    with methode:
        note("<b>Point de départ des modèles.</b> Aucun modèle des jalons précédents n'est réutilisé : les deux GRU "
             "sont entraînés de zéro, uniquement sur le scanner A, avec pour seule augmentation des décalages "
             "horizontaux de ±4 pixels (journal d'expériences, jalon 4). Le lecteur augmenté avec des images de A "
             "dégradées pour imiter B est un résultat du jalon 3 ; il n'entre pas dans le jalon 4.")
        st.markdown(
            "Le montant est lu comme une suite de jetons : les chiffres 0 à 9, un jeton de remplissage (10), un "
            "jeton de début (11) et un jeton de fin (12). La longueur du montant n'est plus fixée : le modèle décide "
            "lui-même de s'arrêter.")
        section("Architecture")
        st.markdown(
            "Un tronc convolutif de trois blocs transforme l'image en 32 colonnes de 256 valeurs. Un encodeur GRU "
            "bidirectionnel lit ces colonnes ; ses états finaux, avant et arrière, forment un résumé de l'image qui "
            "initialise l'état du décodeur. Le décodeur est un GRU de 128 unités avec un plongement de dimension 32 "
            "et une sortie softmax sur les 13 jetons. Le modèle compte 449 389 paramètres.")
        section("Entraînement et décodage")
        st.markdown(
            "L'entraînement utilise le forçage par l'enseignant : à chaque pas, le décodeur reçoit le vrai chiffre "
            "précédent. Au test, il reçoit son propre choix précédent (décodage glouton). La confiance est le produit "
            "des probabilités des jetons choisis jusqu'au jeton de fin.")
        section("Paire de modèles")
        st.markdown(
            "Deux GRU entraînés avec des graines différentes lisent le chèque. Si leurs lectures sont identiques, le "
            "chèque est classé en tête avec le produit de leurs confiances ; sinon la lecture du modèle le plus "
            "confiant est conservée mais le chèque passe en fin de file, vers un opérateur.")
    with resultats:
        section("Test, scanner A", "évalué une seule fois")
        tableau(pd.DataFrame([
            ["Montant exact", "0,829 [0,819 ; 0,840]", "0,798", "0,85"],
            ["Automatisation à 99 %", "0,469 [0,400 ; 0,512]", "0,326", "0,40"],
            ["Taille des modèles", "3,74 Mo", "1,87 Mo", "5 Mo"],
            ["Latence par chèque", "27 à 28 ms", "environ 14 ms", "50 ms"],
        ], columns=["Indicateur", "Paire de GRU", "GRU seul", "Cible"]))
        section("Montant exact selon le nombre de chiffres", "scanner A")
        df = pd.DataFrame([dict(serie="Paire de GRU", x=str(i + 1), y=v)
                           for i, v in enumerate([0.987, 0.939, 0.911, 0.806, 0.578])])
        graphique_lignes(df, "Nombre de chiffres du montant", "Montant exact", cible=0.85, hauteur=230)
        section("Comparaisons en validation")
        tableau(pd.DataFrame([
            ["Séquence GRU 128 (3 graines)", "449 k", "0,828 ± 0,018", "0,402 ± 0,022"],
            ["Séquence LSTM 102 (3 graines)", "447 k", "0,815 ± 0,010", "0,401 ± 0,008"],
            ["Positions, dense 128, même tronc", "1,11 M", "0,514", "0,106"],
            ["Positions, dense 48, budget égal", "452 k", "0,388", "0,057"],
        ], columns=["Modèle", "Paramètres", "Montant exact", "Automatisation"]))
        st.markdown(
            "GRU et LSTM sont indiscernables à budget égal. Sur le même tronc, la lecture en séquence dépasse "
            "nettement la lecture par positions, surtout sur les montants longs. La cible d'exactitude (0,85) est "
            "manquée de 0,021 au test ; la validation surestimait le test d'environ 0,05.")
        section("Analyse d'erreurs", "validation")
        st.markdown(
            "Le décodeur s'arrête à la bonne longueur pour 97 % des chèques. Les erreurs restantes sont surtout des "
            "chiffres faux à la bonne longueur, le plus souvent un seul. Les chèques altérés sont moins bien lus "
            "(0,764 contre 0,836 au test). Quand les deux modèles s'accordent (72,5 % des chèques), la lecture est "
            "exacte dans 95,1 % des cas ; quand ils divergent, dans 50,7 %.")


def page_detection(c):
    entete("Détection d'altération",
           "Trois familles de détecteurs ont été construites : un détecteur dédié entraîné sur A propre (jalon 2), un "
           "détecteur dédié avec augmentation (jalon 3) et les têtes d'altération des modèles multitâches (jalon 3).",
           "Transversal")
    if not bloc_cheque(c):
        return
    section("Seuil d'alerte du détecteur du jalon 2", "choix du coût")
    cout = st.select_slider("Rapport de coûts (fraude manquée / fausse alerte)", options=list(COUTS_J2),
                            value=COUT_DEFAUT, key="cout_det")
    seuil_d, fa, rappel, manquees = COUTS_J2[cout]
    kpis([("Seuil", fr(seuil_d, 4), "validation"), ("Fausses alertes", fa, ""), ("Rappel", rappel, ""),
          ("Fraudes manquées", f"{manquees} sur 285", "validation")])

    section("Scores sur le chèque choisi")
    df = table_detecteurs(c, cout)
    if len(df):
        tableau(df, brut=["Verdict"])
        n_alertes = int(df["Verdict"].str.contains("Altéré").sum())
        legende(f"{n_alertes} détecteur(s) sur {len(df)} signalent ce chèque comme altéré.")
    else:
        st.warning("Aucun fichier de détecteur dans le dossier models.")
    vue_verite_alteration(c)
    if est_scanner_b(c):
        note("Sur une image de type scanner B, le détecteur du jalon 2 (entraîné sur A propre) est proche du hasard : "
             "AUC 0,58 en validation. Les détecteurs entraînés avec augmentation gardent un AUC de 0,89 à 0,94.",
             "alerte")

    section("Types d'altération")
    if c.get("type") is not None:
        legende(f"Type d'altération de ce chèque : {c['type']} (0 = chèque normal).")
    st.markdown(
        "Le jeu de données contient deux types d'altération. Au jalon 2, le détecteur manque 2 fraudes sur 495 au "
        "test (type 1 : 0 sur 214 ; type 2 : 2 sur 281). Au jalon 1, le type 1 était le plus mal lu par le MLP "
        "(0,028 de montant exact).")

    section("Compromis du multitâche", "validation")
    graphiques_lambda()


def page_lot():
    entete("Évaluation par lot",
           "Mesure de l'exactitude et de l'automatisation de chaque système sur un échantillon d'exemples, pour "
           "comparer les jalons sur les mêmes chèques.", "Mesure")
    exemples = charger_exemples()
    if exemples is None:
        st.info("Le fichier data/exemples.npz est absent : l'évaluation par lot a besoin d'un échantillon d'images.")
        return
    dispo = systemes_disponibles()
    if not dispo:
        st.warning("Aucun modèle dans le dossier models.")
        return
    c1, c2 = st.columns([1, 3])
    scanner = c1.radio("Scanner", ["A", "B"], horizontal=True, key="lot_scanner")
    d = exemples[scanner]
    n_max = len(d["X"])
    n = c1.slider("Nombre de chèques", min(10, n_max), n_max, min(n_max, 200), key="lot_n") if n_max > 10 else n_max
    defaut = [k for k in ("j1", "j2", "j2d", "j3e", "j4p") if k in dispo and (scanner == "B" or k != "j2d")]
    choix = c2.multiselect("Systèmes à comparer", dispo, default=defaut, format_func=lambda k: SYSTEMES[k]["nom"])
    if st.button("Lancer l'évaluation", key="lot_go") and choix:
        X = d["X"][:n, :, :, None].astype("float32")
        montants = d["montant"][:n]
        resultats = {}
        barre = st.progress(0.0)
        for j, cle in enumerate(choix):
            lire(cle, X[:1])                                    # échauffement, hors mesure
            t0 = time.perf_counter()
            r = lire(cle, X)
            ms = (time.perf_counter() - t0) / n * 1000
            exact = float(np.mean(r["lus"] == montants))
            resultats[cle] = dict(
                exact=exact, marge=1.96 * np.sqrt(exact * (1 - exact) / n),
                auto=taux_automatisation(r["lus"], montants, r["rang"]), ms=ms,
                longueurs=exactitude_par_longueur(r["lus"], montants),
                courbe=courbe_couverture(r["lus"], montants, r["rang"]))
            barre.progress((j + 1) / len(choix))
        barre.empty()
        alterations = {}
        x_oct = octets(X)
        for det in DETECTEURS:
            if existe(det["fichier"]):
                alterations[det["nom"]] = auc(d["altere"][:n], score_alteration(det["fichier"], det["multi"], x_oct, n))
        st.session_state["lot"] = dict(scanner=scanner, n=n, resultats=resultats, alterations=alterations)

    lot = st.session_state.get("lot")
    if not lot:
        st.info("Choisissez le scanner, la taille de l'échantillon et les systèmes, puis lancez l'évaluation.")
        return
    res = lot["resultats"]
    noms = [SYSTEMES[k]["nom"] for k in res]
    section(f"Résultats sur {lot['n']} chèques du scanner {lot['scanner']}", "échantillon de validation")
    tableau(pd.DataFrame([
        [SYSTEMES[k]["nom"], fr(r["exact"]), f"± {fr(r['marge'])}", fr(r["auto"]), fr(r["ms"], 1)]
        for k, r in res.items()],
        columns=["Système", "Montant exact", "Marge à 95 %", "Automatisation à 99 %", "Temps par chèque (ms)"]))
    legende("L'automatisation à 99 % est la part des chèques, classés du plus sûr au moins sûr, que l'on peut traiter "
            "tant que la précision cumulée reste d'au moins 99 %. Sur un petit échantillon elle est très instable. "
            "Les temps sont mesurés en lot, sur cette machine, et ne sont pas ceux du jalon 4.")
    if lot["scanner"] == "B":
        note("Les jalons 1 et 4 n'ont été entraînés que sur le scanner A : comparez leurs lignes à celles des systèmes "
             "du jalon 3, entraînés avec des images de A dégradées pour imiter B.", "alerte")

    section("Exactitude selon le nombre de chiffres")
    df = pd.DataFrame([dict(serie=SYSTEMES[k]["nom"], x=str(L), y=v)
                       for k, r in res.items() for L, v in r["longueurs"].items()])
    graphique_lignes(df, "Nombre de chiffres du montant", "Montant exact", ordre_series=noms)

    section("Précision cumulée selon la part de chèques traités")
    df = pd.DataFrame([dict(serie=SYSTEMES[k]["nom"], traite=float(t), precision=float(p))
                       for k, r in res.items() for t, p in zip(*r["courbe"])])
    base = alt.Chart(df).encode(
        x=alt.X("traite:Q", title="Part des chèques traités automatiquement (du plus sûr au moins sûr)",
                scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("precision:Q", title="Précision cumulée", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("serie:N", scale=alt.Scale(domain=noms, range=PALETTE[:len(noms)]), legend=alt.Legend(title=None)))
    cible = alt.Chart(pd.DataFrame({"y": [0.99]})).mark_rule(strokeDash=[5, 4], color=GRIS).encode(y="y:Q")
    afficher_graphique(alt.layer(base.mark_line(strokeWidth=2), cible), 300)
    legende("Ligne en tirets : précision de 99 %. Plus une courbe reste haut longtemps, plus le système peut automatiser.")

    if lot["alterations"]:
        section("Détection d'altération", "AUC sur le même échantillon")
        tableau(pd.DataFrame([[k, fr(v, 4)] for k, v in lot["alterations"].items()], columns=["Détecteur", "AUC"]))
        legende("Un échantillon de 200 chèques contient environ 20 chèques altérés : l'AUC est indicative.")


@st.cache_data
def charger_journal():
    if not FICHIER_JOURNAL.exists():
        return None
    lignes = []
    for ligne in FICHIER_JOURNAL.read_text(encoding="utf-8").splitlines():
        ligne = ligne.strip()
        if ligne:
            try:
                lignes.append(json.loads(ligne))
            except json.JSONDecodeError:
                pass
    if not lignes:
        return None
    df = pd.json_normalize(lignes)
    for col in df.columns:
        df[col] = df[col].map(lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
    return df


def page_journal():
    entete("Journal d'expériences",
           "Les expériences enregistrées au cours du projet, avec leurs paramètres et leurs résultats, échecs compris.",
           "Traçabilité")
    df = charger_journal()
    if df is None:
        st.info("Le fichier data/journal.jsonl est absent. Copiez-y le journal d'expériences du projet.")
        return
    comptes = df["jalon"].value_counts().sort_index()
    kpis([("Expériences", str(len(df)), "entrées du journal")])
    c1, c2 = st.columns([1, 2])
    jalon = c1.selectbox("Jalon", ["Tous"] + [int(j) for j in comptes.index], key="j_jalon")
    texte = c2.text_input("Filtrer par nom d'expérience", key="j_texte")
    vue = df if jalon == "Tous" else df[df["jalon"] == jalon]
    if texte:
        vue = vue[vue["nom"].astype(str).str.contains(texte, case=False, regex=False)]
    vue = vue.dropna(axis=1, how="all")
    legende(f"{len(vue)} entrées affichées.")
    st.dataframe(vue, hide_index=True)
    st.download_button("Télécharger la sélection (CSV)", vue.to_csv(index=False).encode("utf-8"),
                       "journal_selection.csv", "text/csv", key="j_dl")


# ======================================================================================
# Navigation
# ======================================================================================
st.sidebar.markdown(
    '<div class="marque"><div class="marque-t">Lecture automatique<br>des chèques</div>'
    '<div class="marque-s">M2 SDIA, Advanced AI, ENSET Mohammedia</div>'
    '<div class="equipe"><b>Réalisé par</b>HAFSSA MIFTAH IDRISSI<br>HANANE AIT LHAJ<br>FATIMA ZAHRA LASFAR</div>'
    '</div>', unsafe_allow_html=True)
page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed", key="page")

if page in PAGES_SANS_CHEQUE:
    cheque, seuil_conf = None, 0.80
else:
    cheque = choisir_cheque()
    seuil_conf = reglage_seuil()
etat_modeles()

if page == "Vue d'ensemble":
    page_vue_ensemble()
elif page == "Atelier de lecture":
    page_atelier(cheque, seuil_conf)
elif page == "Jalon 1 : MLP":
    page_j1(cheque, seuil_conf)
elif page == "Jalon 2 : CNN et détecteur":
    page_j2(cheque, seuil_conf)
elif page == "Jalon 3 : scanner B et multitâche":
    page_j3(cheque, seuil_conf)
elif page == "Jalon 4 : lecture en séquence":
    page_j4(cheque, seuil_conf)
elif page == "Détection d'altération":
    page_detection(cheque)
elif page == "Évaluation par lot":
    page_lot()
else:
    page_journal()