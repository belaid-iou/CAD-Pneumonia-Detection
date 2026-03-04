# 🩻 Système de Diagnostic Assisté par Ordinateur (DAO) — Détection de Pneumonie

## 📋 Description

Ce projet implémente un **Système de Diagnostic Assisté par Ordinateur (DAO)** pour la détection automatique de la **pneumonie** à partir de radiographies thoraciques. Il utilise des techniques d'**apprentissage automatique classique** (Machine Learning) avec une **architecture en cascade** pour minimiser les diagnostics manqués.

Le projet contient **deux notebooks complets**, chacun adapté à un jeu de données différent :

| Notebook | Jeu de Données | Images | Spécificités |
|----------|---------------|--------|-------------|
| `CAD_Pneumonia_Detection.ipynb` | RSNA Pneumonia (Kaggle) | 26 684 | Masques de segmentation, boîtes englobantes, CSV |
| `CAD_ChestXray_Pneumonia.ipynb` | Chest X-Ray (Kermany et al.) | 5 856 | Sous-types bactérienne/virale, classification ternaire |

---

## 🏗️ Architecture du Système

```
┌───────────────────────────────────────────────────────────────┐
│                    SYSTÈME DAO EN CASCADE                      │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│  Étape 1 : Classifieur Primaire (Random Forest Optimisé)       │
│     ├── Négatif (confiant) → Normal                            │
│     └── Positif → Étape 2                                      │
│                                                                │
│  Étape 2 : Classifieur de Sous-type / Détection                │
│     └── Analyse approfondie du cas positif                     │
│                                                                │
│  Étape 3 : Vérification Secondaire (Haute Sensibilité)         │
│     ├── Confirme les négatifs primaires                         │
│     └── Récupère les positifs manqués                          │
│                                                                │
└───────────────────────────────────────────────────────────────┘
```

---

## 📂 Structure du Projet

```
projet-ia/
├── CAD_Pneumonia_Detection.ipynb     # Notebook 1 — Dataset RSNA
├── CAD_ChestXray_Pneumonia.ipynb     # Notebook 2 — Dataset Chest X-Ray
├── requirements.txt                  # Dépendances Python
├── README.md                         # Ce fichier
├── .gitignore                        # Fichiers ignorés par Git
├── archive (3)/                      # Dataset RSNA (non inclus dans Git)
│   ├── stage2_train_metadata.csv
│   ├── stage2_test_metadata.csv
│   ├── Training/
│   │   ├── Images/
│   │   └── Masks/
│   └── Test/
├── chest_Xray/                       # Dataset Kermany (non inclus dans Git)
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── test/
│   └── val/
├── saved_models/                     # Modèles RSNA (générés à l'exécution)
└── saved_models_chestxray/           # Modèles Chest X-Ray (générés à l'exécution)
```

---

## 📊 Jeux de Données

### 1. RSNA Pneumonia Detection (Notebook 1)
- **Source** : [Kaggle — RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge)
- **Taille** : 26 684 images d'entraînement (1024×1024 PNG)
- **Labels** : CSV avec métadonnées, boîtes englobantes, masques de segmentation
- **Classes** : Normal / Pneumonie (binaire)

### 2. Chest X-Ray (Notebook 2)
- **Source** : [Kaggle — Chest X-Ray Images (Kermany et al.)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Taille** : 5 856 images (JPEG, tailles variables)
- **Labels** : Organisation par dossiers (NORMAL / PNEUMONIA)
- **Classes** : Normal / Pneumonie Bactérienne / Pneumonie Virale (ternaire)

> ⚠️ **Note** : Les jeux de données ne sont **pas inclus** dans ce dépôt (trop volumineux). Téléchargez-les depuis Kaggle et placez-les dans les dossiers correspondants.

---

## 🔬 Méthodologie (Commune aux deux notebooks)

Chaque notebook suit un pipeline complet de 17 sections :

1. **Introduction & Contexte Médical**
2. **Configuration de l'Environnement & Imports**
3. **Exploration des Données (EDA)**
4. **Prétraitement & Ingénierie des Caractéristiques**
5. **Séparation des Données**
6. **ACP & Construction des Pipelines**
7. **Exploration des Modèles & Entraînement**
8. **Évaluation sur l'Ensemble de Validation**
9. **Validation Croisée (5-Fold Stratifiée)**
10. **Réglage des Hyperparamètres (RandomizedSearchCV)**
11. **Optimisation du Seuil & Caractéristiques HOG**
12. **Persistance des Modèles (joblib)**
13. **Classification Ternaire / Avancée**
14. **Architecture en Cascade du DAO**
15. **Évaluation Finale sur l'Ensemble de Test**
16. **Discussion & Conclusions**
17. **Export, Rapports & Contrôle de Version**

### Modèles Utilisés
- Régression Logistique
- SVM (Noyau RBF)
- Random Forest
- Gradient Boosting

### Techniques Clés
- **ACP** (Analyse en Composantes Principales) pour la réduction de dimensionnalité
- **HOG** (Histogrammes de Gradients Orientés) pour l'extraction de caractéristiques
- **Augmentation de données** pour gérer le déséquilibre des classes
- **Optimisation du seuil** pour maximiser le rappel (sensibilité clinique)
- **Système en cascade** pour minimiser les faux négatifs

---

## 🚀 Installation & Exécution

### Prérequis
- Python 3.10+
- Conda (recommandé) ou pip

### Installation avec Conda

```bash
# Créer l'environnement
conda create -n pneumonia_env python=3.10 -y
conda activate pneumonia_env

# Installer les dépendances
pip install -r requirements.txt

# Enregistrer le kernel Jupyter
python -m ipykernel install --user --name pneumonia_env --display-name "Python (pneumonia_env)"
```

### Installation avec pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Exécution

```bash
jupyter notebook
# Ou dans VS Code : ouvrir le .ipynb et sélectionner le kernel pneumonia_env
```

---

## 📈 Résultats Attendus

| Métrique | RSNA (Notebook 1) | Chest X-Ray (Notebook 2) |
|----------|-------------------|--------------------------|
| Exactitude | ~85-90% | ~88-93% |
| Rappel (Sensibilité) | ~90-95% (cascade) | ~92-97% (cascade) |
| F1-Score | ~85-90% | ~90-94% |

> Les résultats varient selon l'exécution. Le système en cascade améliore significativement le rappel.

---

## 👨‍💻 Auteur

**[Votre Nom]**  
Projet de Diagnostic Assisté par Ordinateur — 2025

---

## 📄 Licence

Ce projet est à but éducatif et de recherche. Les jeux de données ont leurs propres licences respectives.

---

## 📚 Références

1. Kermany, D.S., et al. (2018). *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*. Cell, 172(5), 1122-1131.
2. RSNA Pneumonia Detection Challenge. *Radiological Society of North America*. Kaggle.
3. Scikit-learn: Machine Learning in Python. Pedregosa et al., JMLR 12, 2011.
