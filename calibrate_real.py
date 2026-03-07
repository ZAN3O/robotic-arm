"""
calibrate_real.py - Interactive Camera-to-Robot Calibration

Place markers at known positions on the desk (measured from robot base).
Click on them in the camera view, enter world coordinates,
and save the homography to calibration.json.

Usage:
    python calibrate_real.py
    python calibrate_real.py --camera-id 1
"""

import cv2
import numpy as np
import json
import argparse
import os
from vision import CameraCapture, HomographyTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="Calibration caméra → robot")
    parser.add_argument("--camera-id", type=int, default=1)
    parser.add_argument("--output", type=str, default="calibration.json")
    return parser.parse_args()


def prompt_action(num_points):
    """Prompt user for next action in terminal (since OpenCV keys don't work with input())."""
    print()
    print(f"   📊 {num_points} points enregistrés")
    print("   ─────────────────────────────────")
    print("   [Clic]  → Ajouter un point")
    if num_points >= 4:
        print("   [c]     → Calibrer et sauvegarder")
    if num_points > 0:
        print("   [u]     → Annuler le dernier point")
    print("   [q]     → Quitter")
    print()
    print("   👆 Clique sur un marqueur dans la fenêtre caméra...")
    print("      (ou tape c/u/q ici puis Entrée)")
    
    return None  # non-blocking, handled in loop


def main():
    args = parse_args()
    
    print("=" * 60)
    print("📷 CALIBRATION CAMÉRA → ROBOT (Monde Réel)")
    print("=" * 60)
    print()
    print("Mode d'emploi:")
    print("  1. Place des marqueurs à des positions connues sur le bureau")
    print("     (mesure avec une règle depuis la base du robot)")
    print("  2. Clique sur chaque marqueur dans la vue caméra")
    print("  3. Entre les coordonnées monde (X, Y en mètres) dans le terminal")
    print("     X = devant/derrière, Y = gauche/droite")
    print("  4. Minimum 4 points, plus = meilleure précision")
    print("  5. Actions par le terminal : c=calibrer, u=annuler, q=quitter")
    print()
    
    # Camera
    cap = CameraCapture(args.camera_id, 1920, 1080)
    if not cap.open():
        print("❌ Impossible d'ouvrir la caméra")
        return
    
    pixel_points = []
    world_points = []
    click_pos = [None]
    waiting_for_click = True
    
    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_pos[0] = (x, y)
    
    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Calibration", on_click)
    
    transformer = HomographyTransformer()
    calibrated = False
    
    print("👆 Clique sur un marqueur dans la fenêtre caméra...")
    print("   (ou tape c/u/q ici puis Entrée)\n")
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        display = frame.copy()
        
        # Draw existing points
        for i, ((px, py), (wx, wy)) in enumerate(zip(pixel_points, world_points)):
            color = (0, 255, 0)
            cv2.drawMarker(display, (px, py), color, cv2.MARKER_CROSS, 30, 2)
            cv2.circle(display, (px, py), 8, color, 2)
            label = f"P{i+1}: ({wx:.2f}, {wy:.2f})m"
            cv2.putText(display, label, (px + 15, py - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # If calibrated, draw grid
        if calibrated:
            for gx in np.arange(0.05, 0.35, 0.05):
                for gy in np.arange(-0.15, 0.20, 0.05):
                    px, py = transformer.world_to_pixels(gx, gy)
                    if 0 <= px < frame.shape[1] and 0 <= py < frame.shape[0]:
                        cv2.circle(display, (px, py), 3, (255, 0, 255), -1)
                        cv2.putText(display, f"({gx:.2f},{gy:.2f})", (px+5, py-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1)
        
        # HUD
        h, w = display.shape[:2]
        cv2.rectangle(display, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.putText(display, f"Points: {len(pixel_points)}/4+  |  Clique sur un marqueur, puis entre les coords dans le terminal",
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        
        status = "CALIBRE - Sauvegarde OK" if calibrated else "En attente de calibration"
        status_color = (0, 255, 0) if calibrated else (0, 200, 255)
        cv2.putText(display, status, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        cv2.imshow("Calibration", display)
        cv2.waitKey(1)
        
        # Handle click → go to terminal for coordinates
        if click_pos[0] is not None:
            px, py = click_pos[0]
            click_pos[0] = None
            
            print(f"\n📍 Point {len(pixel_points)+1} au pixel ({px}, {py})")
            print("   Coordonnées robot (mètres depuis la base):")
            try:
                wx = float(input("   X (devant = positif) : "))
                wy = float(input("   Y (gauche = positif) : "))
                
                pixel_points.append((px, py))
                world_points.append((wx, wy))
                print(f"   ✅ Ajouté: pixel ({px}, {py}) → monde ({wx:.3f}, {wy:.3f})m")
                calibrated = False
            except ValueError:
                print("   ❌ Entrée invalide, point ignoré")
            
            # After adding a point, ask what to do next
            action = _ask_next_action(len(pixel_points))
            if action == "c":
                calibrated = _do_calibrate(transformer, pixel_points, world_points, w, h, args)
            elif action == "u":
                _do_undo(pixel_points, world_points)
                calibrated = False
            elif action == "q":
                break
            # otherwise: continue (user will click another point)
    
    cap.close()
    cv2.destroyAllWindows()


def _ask_next_action(num_points):
    """Ask user what to do next via terminal."""
    print()
    print(f"   📊 {num_points} point(s) enregistré(s)")
    print("   ─────────────────────────────────────")
    options = "   Actions:  "
    if num_points >= 4:
        options += "[c] Calibrer  "
    if num_points > 0:
        options += "[u] Annuler dernier  "
    options += "[q] Quitter  [Entrée] Continuer"
    print(options)
    
    choice = input("\n   → Ton choix : ").strip().lower()
    return choice


def _do_calibrate(transformer, pixel_points, world_points, w, h, args):
    """Run calibration and save."""
    if len(pixel_points) < 4:
        print("   ❌ Il faut au moins 4 points!")
        return False
    
    if transformer.calibrate(pixel_points, world_points):
        # Test: center of image
        center_px = w // 2
        center_py = h // 2
        wx, wy, wz = transformer.pixels_to_world(center_px, center_py)
        print(f"\n📐 Test: centre image ({center_px}, {center_py}) → monde ({wx:.3f}, {wy:.3f})m")
        
        # Save
        calib_data = {
            "homography_matrix": transformer.H.tolist(),
            "pixel_points": pixel_points,
            "world_points": world_points,
            "camera_id": args.camera_id,
            "resolution": [1920, 1080],
            "table_z": 0.0
        }
        
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
        with open(output_path, 'w') as f:
            json.dump(calib_data, f, indent=2)
        
        print(f"💾 Calibration sauvegardée dans {output_path}")
        print("   Tu peux maintenant lancer real_arm_controller.py !")
        return True
    
    return False


def _do_undo(pixel_points, world_points):
    """Undo last point."""
    if pixel_points:
        rp = pixel_points.pop()
        rw = world_points.pop()
        print(f"   ↩️ Annulé: pixel {rp} → monde {rw}")
    else:
        print("   ❌ Aucun point à annuler")


if __name__ == "__main__":
    main()
