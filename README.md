# Robotic Arm

Projet de bras robotique 4-DOF avec :
- cinematique inverse
- simulation PyBullet
- detection d'objets par vision classique OpenCV
- detection open-vocabulary avec modeles Hugging Face
- commande vocale en francais
- communication reseau vers Raspberry Pi / Arduino
- entrainement RL pour le grasping

Le depot contient a la fois des briques modulaires et des scripts de test rapides. Le point important : tout n'est pas encore branche dans une seule pipeline complete, mais chaque sous-systeme peut etre lance et valide separement.

## Vue d'ensemble

Les modules principaux sont :
- `kinematics.py` : chaine cinematique, IK, limites articulaires
- `simulation.py` : jumeau numerique PyBullet du bras et de son environnement
- `perception.py` : detection couleur/forme avec OpenCV
- `vision.py` : capture camera et homographie pixel -> monde
- `live_detection.py` : detection temps reel OpenCV, selective, avec trackbars
- `live_detection_hf.py` : detection open-vocabulary avec Hugging Face
- `brain_controller.py` : orchestration simulation + reseau + voix + vision
- `voice_control.py` : reconnaissance vocale Google Speech, optimisee FR
- `network.py` : envoi des commandes en mode simule ou ZeroMQ
- `IA.py` : apprentissage par renforcement pour saisir / lever un objet
- `calibrate.py` : outil PyBullet pour regler la pose et la pince
- `requete.py` : envoi simple d'une commande ZeroMQ vers le robot

Documentation locale utile :
- `VOICE_CONTROL_GUIDE.md`
- `ROBOT_ARCHITECTURE_EXPLAINED.md`

## Prerequis

- Python 3.10+ recommande
- macOS / Linux / Windows
- Webcam si tu veux tester la vision
- Micro si tu veux tester la voix
- Apple Silicon supporte `mps` pour l'inference PyTorch

## Installation

Creation d'un environnement virtuel :

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

Installation de base pour simulation + vision classique :

```bash
pip install numpy opencv-python pybullet ikpy pyzmq pyserial SpeechRecognition gymnasium pillow
```

Si tu veux utiliser la detection IA Hugging Face :

```bash
pip install torch torchvision torchaudio transformers pillow
```

Si tu veux utiliser l'entrainement RL :

```bash
pip install "stable-baselines3[extra]"
```

Notes :
- `pyaudio` peut demander une installation systeme selon la plateforme.
- Au premier lancement de `live_detection_hf.py`, le modele est telecharge depuis Hugging Face.

## Demarrage rapide

### 1. Demo principale

Lance le controleur principal en mode demo :

```bash
python brain_controller.py
```

Mode interactif PyBullet :

```bash
python brain_controller.py --gui
```

Mode avec camera initialisee :

```bash
python brain_controller.py --camera
```

Important :
- `brain_controller.py --camera` initialise la camera, mais ne fait pas encore une boucle complete de detection live + action robot.
- Pour tester la vision en direct, utilise plutot `live_detection.py` ou `live_detection_hf.py`.

### 2. Vision classique OpenCV

Detection par couleur / forme avec filtres et trackbars :

```bash
python live_detection.py
```

Ce script est utile pour :
- valider la webcam
- regler la selectivite
- limiter les faux positifs simples

### 3. Vision IA Hugging Face

Detection open-vocabulary pour des objets plus complexes, par exemple :
- stylo
- cahier
- colle
- regle
- gomme

Lancement par defaut :

```bash
python live_detection_hf.py
```

Exemple avec labels explicites :

```bash
python live_detection_hf.py --labels "stylo, cahier, colle, regle, gomme, pen, notebook, glue stick, ruler, eraser"
```

Exemple plus leger pour limiter la charge :

```bash
python live_detection_hf.py --infer-width 640 --infer-height 360 --infer-fps 2 --target-fps 15
```

Le script :
- utilise `IDEA-Research/grounding-dino-tiny`
- essaye `mps` automatiquement sur Mac Apple Silicon
- affiche le FPS d'affichage et le FPS d'inference
- permet de regler les seuils via trackbars

Limite pratique :
- la detection open-vocabulary est beaucoup plus lourde que l'OpenCV classique
- le flux video peut rester fluide, mais le nombre de detections par seconde peut rester bas selon le modele et la resolution

### 4. Calibration et simulation

Outil de calibration manuel dans PyBullet :

```bash
python calibrate.py
```

Test du module perception sans camera :

```bash
python perception.py
```

Test du module vision / homographie :

```bash
python vision.py
```

### 5. Commande vocale

Mode vocal dans le controleur principal :

```bash
python brain_controller.py --voice
```

Le module vocal :
- ecoute en francais
- utilise un wake word type `bonjour bras`
- detecte des intentions du type `prends`, `pose`, `ouvre`, `ferme`, `home`, `stop`

Voir aussi `VOICE_CONTROL_GUIDE.md`.

### 6. Communication reseau

Le controleur principal supporte plusieurs modes :
- `simulated` : affiche seulement les commandes
- `zmq` : envoi reel via ZeroMQ
- `serial` : prevu dans l'architecture, a verifier selon ton setup reel

Exemple :

```bash
python brain_controller.py --network simulated
python brain_controller.py --network zmq
```

Envoi rapide d'une commande predefinie vers un endpoint ZeroMQ :

```bash
python requete.py
python requete.py idle
python requete.py custom
```

Pense a verifier l'IP / port dans `requete.py` et `network.py` avant un usage reel.

### 7. Entrainement IA / RL

Lancement de l'entrainement :

```bash
python IA.py --train
```

Entrainer plus longtemps :

```bash
python IA.py --train --steps 500000
```

Tester un modele entraine :

```bash
python IA.py --test
```

Visualiser l'environnement :

```bash
python IA.py --demo
```

L'entrainement sauvegarde automatiquement dans `models/` et reprend si un checkpoint compatible existe.

## Structure du projet

```text
.
├── README.md
├── IA.py
├── brain_controller.py
├── calibrate.py
├── grasp_planner.py
├── kinematics.py
├── live_detection.py
├── live_detection_hf.py
├── network.py
├── perception.py
├── requete.py
├── simulation.py
├── vision.py
├── voice_control.py
├── arduino_arm.urdf
├── models/
├── logs/
└── docs annexes (*.md)
```

## Etat actuel du projet

Ce depot est deja utile pour :
- tester la cinematique et la simulation
- prototyper la vision classique et la vision IA
- valider des commandes vocales
- preparer un pipeline pick-and-place

En revanche, il faut garder en tete que :
- la detection camera live est aujourd'hui principalement testee via des scripts dedies
- la detection Hugging Face n'est pas encore connectee automatiquement a une boucle complete de grasping
- certaines parties reseau / hardware dependent encore de ton setup reel

## Fichiers a lire ensuite

Si tu veux comprendre le projet sans tout ouvrir dans l'IDE :
- `ROBOT_ARCHITECTURE_EXPLAINED.md` pour la logique globale
- `VOICE_CONTROL_GUIDE.md` pour le vocal
- `simulation.py` pour le jumeau numerique
- `live_detection_hf.py` pour la vision moderne

## Pistes d'amelioration

Les prochaines etapes logiques seraient :
- ajouter un vrai `requirements.txt` ou `pyproject.toml`
- brancher la detection live au `brain_controller`
- sauvegarder la calibration camera dans un fichier
- ajouter un mode pick-and-place base sur les detections Hugging Face
