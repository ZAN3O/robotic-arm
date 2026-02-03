# 🧠 Architecture Robotique : Analyse Technique

En tant que **Lead Tech Robotique**, voici l'analyse détaillée de notre script de contrôle (`arm.py` / `master_controller_smart.py`). Ce document est destiné à servir de référence pour toute l'équipe.

---

## 1. La Chaîne Cinématique Manuelle (`create_manual_chain`)

### ❓ Pourquoi ne pas juste lire l'URDF ?
L'URDF (*Unified Robot Description Format*) est un fichier XML qui décrit physiquement le robot. Cependant, les solveurs de cinématique inverse (IK) génériques comme `ikpy` ont parfois du mal à parser correctement les limites complexes ou les repères imbriqués de l'URDF brut.

**La Solution :** On redéfinit "manuellement" la chaîne dans le code Python.
Cela nous donne un contrôle absolu sur :
*   **L'origine exacte** (`BASE_HEIGHT = 0.11`) : On sait que notre robot n'est pas posé à z=0 mais que son premier axe de rotation est surélevé.
*   **Les axes de rotation** : On force la définition (ex: `rotation=[0, 1, 0]` pour l'épaule) pour éviter que le solveur ne "devine" mal.
*   **Les limites (Bounds)** : On peut restreindre les mouvements logiciellement (ex: empêcher le poignet de faire un 360°) même si le moteur physique le permettrait.

C'est une "Carte Simplifiée" du robot que l'on donne au cerveau mathématique pour qu'il calcule plus vite et plus juste.

---

## 2. Le Calcul IK (Cinématique Inverse)

### 📐 Le Concept
*   **Cinématique Directe (FK)** : *"Je mets le moteur A à 30° et B à 45°. Où est ma main ?"* -> Facile, c'est de la géométrie pure.
*   **Cinématique Inverse (IK)** : *"Je veux mettre ma main en (x=0.2, y=0.1, z=0.05). Quels angles donner aux moteurs A et B ?"* -> Difficile, il y a souvent plusieurs solutions (coude en haut vs coude en bas).

### ⚙️ Dans le code
La fonction `solve_ik(target_xyz)` utilise un algorithme d'optimisation (souvent une descente de gradient ou Levenberg-Marquardt via `ikpy`).
1.  Elle prend la position cible (x, y, z).
2.  Elle cherche la combinaison d'angles $(\theta_1, \theta_2, \theta_3, \theta_4)$ qui minimise la distance entre le bout du bras et la cible.
3.  Elle retourne ces angles cibles.

> **Note Junior :** C'est pour ça que parfois le robot fait des mouvements bizarres. Si la cible est hors de portée, l'IK fait de son mieux pour s'en approcher, ce qui peut tordre le bras.

---

## 3. L'Interpolation (`Lerp`) dans `move_smooth`

Si on envoyait directement les angles calculés par l'IK aux moteurs, le robot se téléporterait ou ferait un mouvement violent (à-coup) car les moteurs essaieraient d'y aller à vitesse maximale.

### 🌊 Le Lissage (Lerp)
On utilise l'interpolation linéaire (**L**inear **Int**erpolation).
Au lieu de passer de A à B instantanément, on découpe le trajet en `N` petites étapes.

```python
cmd = start_angles + (target_angles - start_angles) * t
```

*   `t` varie de 0.0 (début) à 1.0 (fin).
*   À chaque boucle (240 fois par seconde), on calcule une position intermédiaire "un peu plus proche" de la fin.

Cela crée un mouvement fluide, naturel et préserve les moteurs virtuels (et réels !).

---

## 4. La "Triche" sur le Z (Hauteur)

### 🕵️ Le Problème
En simulation (et en vrai), atteindre `z=0` (le sol) est risqué.
*   Si on demande `z=0` et que l'IK calcule `z=-0.001` (erreur d'arrondi), le robot rentre dans la table -> **EXPLOSION PHYSIQUE**.
*   Notre robot a une base physique (`BASE_HEIGHT`). L'IK doit savoir que l'origine du bras n'est pas au sol.

### 🎯 La Stratégie
1.  On définit l'origine de la chaîne à `z=BASE_HEIGHT` (0.11m). Ainsi, quand on demande `z=0`, le bras descend vraiment vers le sol relatif à sa base.
2.  On demande une cible à `z=0.005` (5 millimètres). C'est assez bas pour attraper le cube (dont le centre est à ~1.5cm) mais assez haut pour ne jamais toucher la table. C'est notre marge de sécurité ("Safety Deck").

---

## 5. La Logique de la Pince (Gripper)

### 🖐️ Pourquoi des signes inversés ?
Dans notre URDF, la pince est constituée de deux doigts coulissants sur un axe.
*   Le doigt **Gauche** se déplace vers la droite (axe positif `+`) pour fermer.
*   Le doigt **Droit** se déplace vers la gauche (axe négatif `-`) pour fermer.

Pour fermer la pince, on doit donc envoyer :
*   `Doigt G : +valeur`
*   `Doigt D : -valeur`

Si on envoyait `+` aux deux, ils bougeraient tous les deux vers la droite, et la pince ne se fermerait pas (elle se décalerait).

---

## 🚀 Résumé pour l'IA Future

Ce script est robuste car il sépare clairement :
1.  **Le Cerveau (IK)** : Qui calcule où aller.
2.  **Le Système Nerveux (Lerp)** : Qui lisse le mouvement.
3.  **Le Corps (PyBullet)** : Qui exécute avec la physique (gravité, friction).

Pour la suite ("Smart Pick & Place"), nous gardons cette base motrice intacte et nous remplaçons simplement le générateur de coordonnées aléatoires par un "Cerveau Cognitif" (Vision + NLP).
