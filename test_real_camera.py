"""
Script de test CAMERA REELLE & DETECTION.
Utilise le module 'cam' pour ouvrir la webcam et détecter les objets.
"""
import sys
import os

# Ajout du dossier 'cam' au path pour les imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'cam'))

try:
    from vision_advanced import demo_webcam_detection, demo_find_object
    print("✅ Module vision chargé depuis 'cam/'.")
except ImportError as e:
    print(f"❌ Erreur d'import : {e}")
    sys.exit(1)

if __name__ == "__main__":
    print("="*60)
    print("🎥 TEST CAMERA & DETECTION")
    print("="*60)
    print("1. Detection en temps réel (Webcam)")
    print("2. Chercher un objet spécifique")
    print("3. Quitter")
    
    choice = input("\nVotre choix [1/2/3] : ").strip()
    
    if choice == "1":
        demo_webcam_detection()
    elif choice == "2":
        demo_find_object()
    else:
        print("Fin.")
