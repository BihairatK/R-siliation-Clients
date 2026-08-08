# Prédiction de la résiliation d'un portefeuille assurantiel

Projet de Data Science visant à modéliser et comprendre le churn (résiliation) sur un portefeuille clients d'assurance, à partir de données comportementales, contractuelles et relationnelles.



## 1. Problématique métier

Dans le secteur assurantiel, la fidélisation client est un enjeu majeur : acquérir un nouveau client coûte significativement plus cher que d'en retenir un existant. L'objectif de ce projet est de répondre à une question opérationnelle concrète :

> **Quels clients d'un portefeuille d'assurance présentent un risque élevé de résiliation, et quels sont les facteurs qui expliquent ce risque ?**

Concrètement, il s'agit de construire un modèle capable de :
- **Détecter en amont** les contrats à risque de résiliation, pour permettre à l'assureur d'agir de façon proactive (relance commerciale, offre de fidélisation, ajustement tarifaire, amélioration du parcours client) ;
- **Comprendre les leviers** de la résiliation, afin d'orienter des actions correctives sur les processus internes (gestion des sinistres, tarification, canaux de distribution) plutôt que de subir l'attrition.

Le sujet a été volontairement traité comme un **problème de classification binaire déséquilibré** (~10,6 % de résiliés sur 25 000 contrats), représentatif de la réalité du churn en assurance.

## 2. Pourquoi ce jeu de données

Le jeu de données (`Portefeuille_assures.csv`, 25 000 contrats, 17 variables) a été choisi car il réunit les caractéristiques typiques d'un portefeuille assurantiel réel tout en restant exploitable pédagogiquement :
- **Variables contractuelles** : type de garantie, ancienneté, prime annuelle, mode de paiement, nombre de contrats au foyer ;
- **Variables relationnelles / comportementales** : canal de souscription, engagement digital, score NPS, nombre de contacts avec le conseil ou le service réclamation ;
- **Variables liées à la gestion des sinistres** : nombre de sinistres sur 3 ans, délai de règlement, satisfaction post-règlement ;
- **Une cible claire et binaire** (`resiliation`), directement alignée avec la problématique métier.

Ce jeu combine ainsi des signaux "prix" (hausse de prime), des signaux "expérience client" (satisfaction, NPS, réclamations) et des signaux "fidélité" (ancienneté), ce qui permet une analyse riche des causes de résiliation plutôt qu'une simple prédiction boîte noire.

## 3. Démarche et méthodologie

Le projet suit un pipeline classique de Data Science, structuré en 6 étapes :

1. **Chargement et audit qualité** : contrôle des doublons et des valeurs manquantes. Deux variables (`delai_reglement_moyen_jours`, `satisfaction_reglement`) présentent ~45 % de valeurs manquantes, cohérent avec le fait qu'elles ne concernent que les clients ayant déclaré un sinistre.
2. **Nettoyage** : imputation par la médiane pour les variables numériques, par une catégorie `"InfoManquante"` pour les variables catégorielles (le caractère manquant étant lui-même une information potentiellement porteuse de sens).
3. **Analyse exploratoire (EDA)** : distribution de la cible, taux de résiliation par segment (garantie, canal de souscription, mode de paiement), effet de la hausse de prime, effet de l'ancienneté, matrice de corrélation.
4. **Préprocessing** : séparation train/test (80/20, stratifiée), encodage One-Hot des variables catégorielles, sans standardisation des variables numériques afin de préserver l'interprétabilité directe des coefficients et des valeurs SHAP.
5. **Modélisation** : entraînement et optimisation (RandomizedSearchCV, validation croisée stratifiée à 5 plis, métrique ROC-AUC) de trois modèles complémentaires :
   - **Régression logistique** (`class_weight="balanced"`) — modèle de référence, interprétable ;
   - **Random Forest** (`class_weight="balanced"`) ;
   - **XGBoost** (`scale_pos_weight`) — pour capter les interactions non linéaires.
6. **Interprétation** : comparaison des performances (ROC-AUC, F1, précision, rappel), importance des variables (coefficients / feature importances), puis analyse **SHAP** (valeurs globales et locales) pour comprendre le sens et l'intensité de l'effet de chaque variable, y compris sur le modèle non linéaire (XGBoost).

## 4. Résultats et insights

### Performances des modèles

| Modèle | ROC-AUC | F1-score | Précision | Rappel |
|---|---|---|---|---|
| Régression logistique | 0.723 | 0.309 | 0.201 | **0.676** |
| XGBoost | 0.720 | 0.313 | 0.215 | 0.572 |
| Random Forest | 0.713 | **0.317** | **0.229** | 0.516 |

Les trois modèles obtiennent des performances proches (ROC-AUC ~0.71–0.72), signe que le signal capté est majoritairement **linéaire et additif** plutôt que porté par des interactions complexes. La régression logistique, avec son fort rappel, est particulièrement adaptée à un usage métier où l'on préfère **détecter un maximum de clients à risque**, quitte à générer davantage de faux positifs.

### Principaux enseignements

- **L'ancienneté est le frein n°1 à la résiliation** : le taux de résiliation chute de ~15,4 % (0–6 mois) à ~2,4 % (10 ans et plus). Les nouveaux clients sont donc la population la plus fragile.
- **Le canal de souscription est déterminant** : les contrats souscrits via un **comparateur en ligne** résilient 2,5 fois plus (17,5 %) que ceux souscrits en **agence** (7,0 %). Ce canal ressort comme la variable la plus influente dans les trois modèles.
- **La hausse de prime a un effet clair et croissant** : le taux de résiliation passe de ~9,2 % (baisse de prime) à ~14,2 % (hausse de 20-30 %) — une relation quasi monotone confirmée par SHAP.
- **Le mode de paiement mensuel** est associé à un taux de résiliation plus élevé (12,6 %) que le paiement trimestriel (8,6 %) ou annuel (7,9 %), probablement lié à une plus grande sensibilité au prix perçu et à une plus faible barrière à la sortie.
- **L'expérience client compte autant que le prix** : un nombre élevé de contacts au service réclamation, une faible satisfaction post-règlement et un score NPS bas sont tous associés à un risque de résiliation accru (confirmé par les coefficients logistiques et les valeurs SHAP).
- **Le type de garantie et l'âge de l'assuré** ont un effet marginal, la garantie Santé étant légèrement plus exposée (11,6 %) que les contrats multi-garanties (10,3 %), probablement du fait d'un effet de fidélisation lié au cumul de contrats.

## 5. Recommandations

1. **Prioriser la rétention sur les 12 premiers mois** : mettre en place un parcours d'onboarding renforcé et des points de contact proactifs pour les nouveaux souscripteurs, période où le risque de résiliation est maximal.
2. **Traiter différemment les clients issus des comparateurs en ligne** : ce canal capte une clientèle plus volatile et sensible au prix ; envisager des actions de fidélisation dédiées (multi-équipement, avantages différenciants) dès la souscription.
3. **Encadrer les hausses de prime** : au-delà d'un certain seuil (~10-15 %), accompagner la hausse d'une communication pédagogique ou d'une offre de lissage, plutôt que de l'appliquer telle quelle.
4. **Encourager les modes de paiement annualisés** via des incitations tarifaires, qui semblent corrélés à un meilleur engagement contractuel.
5. **Surveiller les signaux d'insatisfaction en temps réel** (réclamations, NPS, satisfaction post-sinistre) pour déclencher des actions correctives ciblées avant la résiliation plutôt qu'après.
6. **Déployer le modèle en scoring mensuel** sur le portefeuille actif, avec un focus sur le rappel (régression logistique) pour maximiser la détection des clients à risque, en acceptant un taux de faux positifs plus élevé compte tenu du coût asymétrique entre perte d'un client et coût d'une action de rétention.

## 6. Pistes d'amélioration du projet

- **Rééquilibrage de la classe minoritaire** (SMOTE, undersampling) pour comparer avec l'approche `class_weight`/`scale_pos_weight` actuelle.
- **Feature engineering** : croisement ancienneté × canal, création d'un score composite d'insatisfaction, historique de hausses de prime cumulées.
- **Calibration des probabilités** (Platt scaling / isotonic regression) pour un scoring exploitable directement en seuils métier.
- **Analyse coût-bénéfice** : définir un seuil de décision optimal en fonction du coût réel d'une action de rétention vs. le coût de la perte d'un client.
- **Test d'un modèle de survie** (Cox, Kaplan-Meier) pour modéliser non seulement *si* mais *quand* un client résilie.
- **Mise en production** : packaging du pipeline (préprocessing + modèle) via `joblib`/`MLflow`, et suivi de la dérive du modèle dans le temps (data drift, performance drift).

---

*Projet réalisé dans le cadre d'une simulation de problématique de résiliation sur un portefeuille assurantiel, à des fins d'apprentissage en Data Science / Machine Learning.*
