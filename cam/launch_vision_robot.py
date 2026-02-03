"""
launch_vision_robot.py - Interactive Launcher

Menu interactif pour lancer facilement le système de vision robot.
"""

import os
import sys
import subprocess


BANNER = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║     🤖 ROBOT ARM avec VISION RÉELLE                      ║
║                                                           ║
║     Détection: YOLOv8 (1000+ objets)                     ║
║     Caméra: Webcam ou iPhone                             ║
║     Contrôle: Texte ou Voix                              ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
"""

MENU = """
🎯 MENU PRINCIPAL

1. 🎥 Test Caméra Simple (webcam)
2. 📐 Calibration Caméra (OBLIGATOIRE 1ère fois)
3. 🎮 Mode Interactif (Texte)
4. 🎤 Mode Vocal (Voice Control)
5. 📱 iPhone Camera Mode
6. ⚙️  Configuration Avancée
7. 📚 Aide & Documentation
8. ❌ Quitter

Choix:"""


def check_installation():
    """Vérifie l'installation des dépendances."""
    print("\n🔍 Vérification des dépendances...\n")
    
    errors = []
    
    # Check Python packages
    try:
        import cv2
        print("✅ OpenCV installé")
    except ImportError:
        errors.append("opencv-python")
        print("❌ OpenCV manquant")
    
    try:
        from ultralytics import YOLO
        print("✅ YOLOv8 (ultralytics) installé")
    except ImportError:
        errors.append("ultralytics")
        print("❌ YOLOv8 manquant")
    
    try:
        import torch
        print("✅ PyTorch installé")
    except ImportError:
        errors.append("torch")
        print("❌ PyTorch manquant")
    
    try:
        import pybullet
        print("✅ PyBullet installé")
    except ImportError:
        errors.append("pybullet")
        print("❌ PyBullet manquant")
    
    try:
        from ikpy.chain import Chain
        print("✅ ikpy installé")
    except ImportError:
        errors.append("ikpy")
        print("❌ ikpy manquant")
    
    # Check optional
    try:
        import clip
        print("✅ CLIP installé (optionnel)")
    except ImportError:
        print("⚠️  CLIP non installé (optionnel)")
    
    if errors:
        print(f"\n⚠️  Dépendances manquantes: {', '.join(errors)}")
        print("\nInstaller avec:")
        print(f"pip install {' '.join(errors)}")
        return False
    
    print("\n✅ Toutes les dépendances sont installées!\n")
    return True


def test_camera():
    """Test caméra simple."""
    print("\n🎥 Test de la caméra...")
    print("Appuyez sur 'q' pour quitter\n")
    
    try:
        subprocess.run([
            sys.executable, "vision_advanced.py", "--demo", "webcam"
        ])
    except FileNotFoundError:
        print("❌ vision_advanced.py non trouvé!")
    except KeyboardInterrupt:
        print("\n⚠️ Test interrompu")


def calibrate_camera():
    """Lance la calibration."""
    print("\n📐 CALIBRATION CAMÉRA")
    print("=" * 60)
    print("INSTRUCTIONS:")
    print("1. Place 4 marqueurs visibles à des positions CONNUES")
    print("2. Mesure leurs coordonnées X,Y en mètres")
    print("3. Clique sur chaque marqueur dans la caméra")
    print("4. Entre les coordonnées quand demandé")
    print("5. Appuie sur 'c' pour calibrer")
    print("=" * 60)
    
    input("\nAppuie sur Entrée quand prêt...")
    
    try:
        subprocess.run([
            sys.executable, "master_controller_vision.py",
            "--mode", "calibrate"
        ])
    except FileNotFoundError:
        print("❌ master_controller_vision.py non trouvé!")
    except KeyboardInterrupt:
        print("\n⚠️ Calibration interrompue")


def interactive_mode():
    """Mode interactif texte."""
    print("\n🎮 MODE INTERACTIF")
    print("\nCommandes: pick <objet> | place | home | scan | quit\n")
    
    try:
        subprocess.run([
            sys.executable, "master_controller_vision.py",
            "--mode", "interactive"
        ])
    except FileNotFoundError:
        print("❌ master_controller_vision.py non trouvé!")
    except KeyboardInterrupt:
        print("\n⚠️ Session interrompue")


def voice_mode():
    """Mode vocal."""
    print("\n🎤 MODE VOCAL")
    print("\nActivation: 'Bonjour bras'")
    print("Commandes: 'Prends le <objet>' | 'Pose' | 'Maison'\n")
    
    try:
        subprocess.run([
            sys.executable, "master_controller_vision.py",
            "--mode", "voice"
        ])
    except FileNotFoundError:
        print("❌ master_controller_vision.py non trouvé!")
    except KeyboardInterrupt:
        print("\n⚠️ Mode vocal interrompu")


def iphone_mode():
    """Mode iPhone camera."""
    print("\n📱 MODE IPHONE CAMERA")
    print("\nAssurez-vous que:")
    print("- iPhone et Mac sur même compte iCloud")
    print("- Bluetooth activé sur les deux")
    print("- iPhone à proximité\n")
    
    input("Appuie sur Entrée quand prêt...")
    
    try:
        subprocess.run([
            sys.executable, "master_controller_vision.py",
            "--mode", "interactive",
            "--camera", "iphone"
        ])
    except FileNotFoundError:
        print("❌ master_controller_vision.py non trouvé!")
    except KeyboardInterrupt:
        print("\n⚠️ Session interrompue")


def advanced_config():
    """Configuration avancée."""
    print("\n⚙️  CONFIGURATION AVANCÉE\n")
    print("1. Changer modèle YOLO (nano/small/medium)")
    print("2. Ajuster seuil de confiance")
    print("3. Tester différentes caméras")
    print("4. Retour")
    
    choice = input("\nChoix: ").strip()
    
    if choice == "1":
        print("\nModèles disponibles:")
        print("  n = nano (rapide, moins précis)")
        print("  s = small (équilibré)")
        print("  m = medium (lent, très précis)")
        
        model_choice = input("\nChoix (n/s/m): ").strip().lower()
        models = {'n': 'yolov8n.pt', 's': 'yolov8s.pt', 'm': 'yolov8m.pt'}
        
        if model_choice in models:
            model = models[model_choice]
            print(f"\nLancement avec {model}...")
            
            try:
                subprocess.run([
                    sys.executable, "master_controller_vision.py",
                    "--mode", "interactive",
                    "--model", model
                ])
            except:
                print("❌ Erreur de lancement")
        else:
            print("❌ Choix invalide")
    
    elif choice == "3":
        print("\nTest caméras disponibles...\n")
        
        import cv2
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                print(f"✅ Caméra {i}: {int(width)}x{int(height)}")
                cap.release()
            else:
                print(f"❌ Caméra {i}: Non disponible")


def show_help():
    """Affiche l'aide."""
    print("\n📚 AIDE & DOCUMENTATION\n")
    print("🔗 Fichiers importants:")
    print("  - VISION_SETUP_GUIDE.md : Guide complet")
    print("  - requirements_vision.txt : Dépendances")
    print("  - vision_advanced.py : Module de détection")
    print("  - master_controller_vision.py : Contrôleur principal")
    
    print("\n💡 Quick Start:")
    print("  1. Installer: pip install -r requirements_vision.txt")
    print("  2. Tester caméra: Option 1 du menu")
    print("  3. Calibrer: Option 2 (OBLIGATOIRE)")
    print("  4. Utiliser: Option 3 ou 4")
    
    print("\n🎯 Objets reconnaissables:")
    print("  COCO dataset (80 objets):")
    print("  - Kitchen: bottle, cup, fork, knife, bowl, banana, apple")
    print("  - Office: laptop, mouse, keyboard, phone, book, scissors")
    print("  - Toys: teddy bear, ball, frisbee")
    print("  - Et bien plus...")
    
    print("\n🔧 Troubleshooting:")
    print("  - 'Camera not found' → Ferme Zoom/Teams/etc")
    print("  - 'Object not found' → Vérifie calibration")
    print("  - 'YOLO missing' → pip install ultralytics")
    
    input("\nAppuie sur Entrée pour continuer...")


def main():
    """Main menu loop."""
    # Check if in correct directory
    if not os.path.exists("arduino_arm.urdf"):
        print("⚠️  ATTENTION: Fichier arduino_arm.urdf non trouvé!")
        print("   Lance ce script depuis le dossier robotic-arm/\n")
    
    # Banner
    print(BANNER)
    
    # Check installation
    if not check_installation():
        print("\n⚠️  Installation incomplète!")
        print("   Installe les dépendances avec:")
        print("   pip install -r requirements_vision.txt\n")
        
        if input("Continuer quand même? (o/n): ").lower() != 'o':
            return
    
    # Main loop
    while True:
        print(MENU, end=" ")
        
        try:
            choice = input().strip()
        except KeyboardInterrupt:
            print("\n\n👋 Au revoir!")
            break
        
        if choice == "1":
            test_camera()
        elif choice == "2":
            calibrate_camera()
        elif choice == "3":
            interactive_mode()
        elif choice == "4":
            voice_mode()
        elif choice == "5":
            iphone_mode()
        elif choice == "6":
            advanced_config()
        elif choice == "7":
            show_help()
        elif choice == "8":
            print("\n👋 Au revoir!")
            break
        else:
            print("\n❓ Choix invalide")
        
        print()  # Blank line


if __name__ == "__main__":
    main()
