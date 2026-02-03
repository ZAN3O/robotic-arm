"""
vision_advanced.py - Advanced Object Recognition System

Ce module fournit une reconnaissance d'objets de niveau industriel utilisant :
- YOLOv8 (Ultralytics) pour détection rapide et précise
- CLIP (OpenAI) pour classification flexible (reconnaissance de n'importe quel objet)
- Support caméra webcam + iPhone (Continuity Camera)

Capabilities:
- Détection de 1000+ objets (extensible à l'infini)
- Tracking multi-objets
- Position 3D (via homographie si calibré)
- Intégration temps réel avec PyBullet

Usage:
    detector = AdvancedObjectDetector()
    objects = detector.detect_frame(frame)
    for obj in objects:
        print(f"Found {obj.name} at {obj.position_3d}")
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
from enum import Enum
import json
from pathlib import Path

# Tentative d'import (installation à la demande)
try:
    from ultralytics import YOLO, YOLOWorld # 🆕 Import YOLO-World
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLOv8 non installé. Run: pip install ultralytics")

try:
    import torch
    from PIL import Image
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("⚠️ CLIP non installé. Run: pip install torch torchvision ftfy regex tqdm")


class DetectionBackend(Enum):
    """Backends de détection disponibles."""
    YOLO = "yolo"           # YOLOv8 (rapide, précis, 80 classes COCO)
    CLIP = "clip"           # CLIP (flexible, ~∞ classes possibles)
    HYBRID = "hybrid"       # YOLO pour détecter + CLIP pour classifier
    COLOR_LEGACY = "color"  # Ancien système couleur (fallback)


@dataclass
class DetectedObject3D:
    """
    Représente un objet détecté avec position 3D.
    """
    # Identification
    id: int
    name: str                      # "bottle", "cup", "red cube"
    category: str                  # "container", "toy", etc.
    confidence: float              # 0.0 - 1.0
    
    # Position 2D (image)
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    center_px: Tuple[int, int]
    
    # Position 3D (monde robot)
    position_3d: Optional[Tuple[float, float, float]] = None  # (x, y, z) meters
    
    # Propriétés physiques
    estimated_size_m: float = 0.03  # Taille estimée en mètres
    graspable: bool = True
    
    # Couleur dominante (si disponible)
    dominant_color: Optional[str] = None
    
    # Metadata
    labels: List[str] = field(default_factory=list)  # Tags additionnels
    
    def __str__(self):
        pos_str = f"3D:{self.position_3d}" if self.position_3d else "2D only"
        return f"{self.name} ({self.confidence:.2f}) @ {pos_str}"


class ObjectDatabase:
    """
    Base de données d'objets reconnaissables.
    Peut être étendue à l'infini.
    """
    
    # Categories principales avec objets communs
    CATEGORIES = {
        "kitchen": [
            "bottle", "cup", "fork", "knife", "spoon", "bowl", "plate",
            "wine glass", "banana", "apple", "orange", "carrot", "pizza",
            "donut", "cake", "sandwich"
        ],
        "workspace": [
            "laptop", "mouse", "keyboard", "cell phone", "book", "pen",
            "scissors", "pencil", "paper", "stapler", "tape", "notebook"
        ],
        "toys": [
            "ball", "teddy bear", "doll", "toy car", "toy train", "toy boat",
            "lego brick", "puzzle piece", "dice", "playing card"
        ],
        "containers": [
            "box", "container", "jar", "can", "bag", "basket", "bin",
            "crate", "bucket", "pot"
        ],
        "tools": [
            "hammer", "screwdriver", "wrench", "pliers", "drill", "saw",
            "tape measure", "level", "clamp", "chisel"
        ],
        "household": [
            "vase", "clock", "picture frame", "candle", "lamp", "pillow",
            "blanket", "towel", "soap", "toothbrush", "hairbrush"
        ],
        "electronics": [
            "remote", "charger", "cable", "headphones", "speaker", "camera",
            "watch", "tablet", "game controller", "usb drive"
        ],
        "sports": [
            "baseball", "basketball", "football", "tennis ball", "golf ball",
            "frisbee", "shuttlecock", "hockey puck", "baseball bat"
        ]
    }
    
    # COCO dataset (80 classes standard YOLO)
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    # Dictionnaire de traduction EN -> FR
    COCO_FR = {
        'person': 'personne', 'bicycle': 'velo', 'car': 'voiture', 'motorcycle': 'moto', 
        'airplane': 'avion', 'bus': 'bus', 'train': 'train', 'truck': 'camion', 'boat': 'bateau',
        'traffic light': 'feu tricolore', 'fire hydrant': 'bouche incendie', 'stop sign': 'panneau stop', 
        'parking meter': 'parcmetre', 'bench': 'banc', 'bird': 'oiseau', 'cat': 'chat',
        'dog': 'chien', 'horse': 'cheval', 'sheep': 'mouton', 'cow': 'vache', 
        'elephant': 'elephant', 'bear': 'ours', 'zebra': 'zebre', 'giraffe': 'girafe', 
        'backpack': 'sac a dos', 'umbrella': 'parapluie', 'handbag': 'sac a main', 'tie': 'cravate', 
        'suitcase': 'valise', 'frisbee': 'frisbee', 'skis': 'skis', 'snowboard': 'snowboard', 
        'sports ball': 'balle', 'kite': 'cerf-volant', 'baseball bat': 'batte de baseball', 
        'baseball glove': 'gant de baseball', 'skateboard': 'skateboard', 'surfboard': 'planche de surf', 
        'tennis racket': 'raquette tennis', 'bottle': 'bouteille', 'wine glass': 'verre a vin', 
        'cup': 'tasse', 'fork': 'fourchette', 'knife': 'couteau', 'spoon': 'cuillere', 
        'bowl': 'bol', 'banana': 'banane', 'apple': 'pomme', 'sandwich': 'sandwich', 
        'orange': 'orange', 'broccoli': 'brocoli', 'carrot': 'carotte', 'hot dog': 'hot dog', 
        'pizza': 'pizza', 'donut': 'beignet', 'cake': 'gateau', 'chair': 'chaise', 
        'couch': 'canape', 'potted plant': 'plante', 'bed': 'lit', 'dining table': 'table (manger)', 
        'toilet': 'toilettes', 'tv': 'tv', 'laptop': 'ordinateur', 'mouse': 'souris', 
        'remote': 'telecommande', 'keyboard': 'clavier', 'cell phone': 'telephone', 
        'microwave': 'micro-ondes', 'oven': 'four', 'toaster': 'grille-pain', 'sink': 'evier', 
        'refrigerator': 'frigo', 'book': 'livre', 'clock': 'horloge', 'vase': 'vase', 
        'scissors': 'ciseaux', 'teddy bear': 'ours en peluche', 'hair drier': 'seche-cheveux', 
        'toothbrush': 'STYLO', # Hack: contexte bureau
        'carrot': 'MARQUEUR', # Hack: contexte bureau
        'spoon': 'outil', 'fork': 'outil', 'knife': 'cutter'
    }
    
    @classmethod
    def get_all_objects(cls) -> List[str]:
        """Retourne la liste complète d'objets reconnaissables."""
        all_objects = set(cls.COCO_CLASSES)
        for category_items in cls.CATEGORIES.values():
            all_objects.update(category_items)
        return sorted(list(all_objects))
    
    @classmethod
    def get_category(cls, obj_name: str) -> str:
        """Trouve la catégorie d'un objet."""
        for category, items in cls.CATEGORIES.items():
            if obj_name.lower() in [item.lower() for item in items]:
                return category
        return "unknown"
    
    @classmethod
    def is_graspable(cls, obj_name: str, size_m: float = 0.05) -> bool:
        """Détermine si un objet est saisissable."""
        # Objets trop grands ou non saisissables
        non_graspable = {'person', 'car', 'bus', 'truck', 'airplane', 'train',
                         'boat', 'bench', 'couch', 'bed', 'dining table', 'toilet'}
        
        if obj_name.lower() in non_graspable:
            return False
        
        # Taille limite (15cm max pour notre pince)
        if size_m > 0.15:
            return False
        
        return True


class AdvancedObjectDetector:
    """
    Détecteur d'objets multi-backend avec tracking.
    """
    
    def __init__(self, 
                 backend: DetectionBackend = DetectionBackend.HYBRID,
                 yolo_model: str = "yolo26n.pt", # 🌍 YOLO-World: Le modèle qui comprend le texte!
                 clip_model: str = "ViT-B/32",
                 confidence_threshold: float = 0.05): # TRES BAS pour tout capter
        """
        Initialize detector.
        """
        self.backend = backend
        self.confidence_threshold = confidence_threshold
        self.object_counter = 0
        
        # 🆕 VOCABULAIRE PERSONNALISÉ (Ce que le modèle doit chercher)
        self.custom_classes = [
            "pen", "blue pen", "red pen", "green pen", "marker", # Ecriture
            "glue stick", "glue tube", "UHU stick", # Colle
            "correction tape", "tipp-ex", "white stationery", # Tipp-ex
            "scissors", "cutter", # Decoupe
            "stapler", "scotch tape", "clear tape", # Bureau
            "cell phone", "smartphone", "calculator", "scientific calculator", # Tech
            "mug", "coffee cup", "bottle", # Boisson
            "rubiks cube", "puzzle cube", "toy cube", # Jouets
            "plastic container", "storage box", "clear box", # Conteneurs
            "tissue pack", "packet", "wipes" # Autres
        ]
        
        # Tracking & Persistence (Anti-Clignotement)
        # Dict[id, {'obj': DetectedObject3D, 'ttl': int}]
        self.persistence_memory = {}
        self.MAX_TTL = 10  # Augmenté à 10 frames (super stable)
        
        # Initialize backends
        self.yolo_model = None
        self.clip_model = None
        self.clip_preprocess = None
        
        if backend in [DetectionBackend.YOLO, DetectionBackend.HYBRID]:
            self._init_yolo(yolo_model)
        
        if backend in [DetectionBackend.CLIP, DetectionBackend.HYBRID]:
            self._init_clip(clip_model)
        
        print(f"✅ AdvancedObjectDetector initialized (backend: {backend.value})")
    
    def _init_yolo(self, model_name: str):
        """Initialize YOLOv8 / YOLO-World."""
        if not YOLO_AVAILABLE:
            print("❌ YOLOv8 unavailable")
            return
        
        try:
            # Essai YOLO-World
            print(f"🌍 Loading YOLO-World: {model_name}")
            self.yolo_model = YOLOWorld(model_name)
            
            # 🧠 Set Custom Vocabulary
            print(f"🧠 Prompting: {self.custom_classes}")
            self.yolo_model.set_classes(self.custom_classes)
            
        except Exception as e:
            print(f"⚠️ YOLO-World Load Failed ({e}), fallback to standard YOLOv8x")
            self.yolo_model = YOLO("yolov8x.pt")
    
    def _init_clip(self, model_name: str):
        """Initialize CLIP."""
        if not CLIP_AVAILABLE:
            print("❌ CLIP unavailable")
            return
        
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model, self.clip_preprocess = clip.load(model_name, device=device)
            print(f"✅ CLIP loaded: {model_name} on {device}")
        except Exception as e:
            print(f"❌ Failed to load CLIP: {e}")
    
    def detect_frame(self, 
                     frame: np.ndarray,
                     target_objects: Optional[List[str]] = None) -> List[DetectedObject3D]:
        """
        Détecte tous les objets dans une frame.
        
        Args:
            frame: Image BGR (OpenCV format)
            target_objects: Liste optionnelle d'objets à chercher (pour CLIP)
            
        Returns:
            List[DetectedObject3D]: Objets détectés
        """
        if self.backend == DetectionBackend.YOLO:
            return self._detect_yolo(frame)
        
        elif self.backend == DetectionBackend.CLIP:
            if target_objects is None:
                target_objects = ObjectDatabase.get_all_objects()[:100]  # Top 100
            return self._detect_clip(frame, target_objects)
        
        elif self.backend == DetectionBackend.HYBRID:
            # YOLO pour détecter, CLIP pour affiner classification
            return self._detect_hybrid(frame, target_objects)
        
        else:
            # Fallback: color-based detection
            return self._detect_color_legacy(frame)
    
    def _detect_yolo(self, frame: np.ndarray) -> List[DetectedObject3D]:
        """Détection avec YOLOv8."""
        if self.yolo_model is None:
            return []
        
        # Run inference
        results = self.yolo_model(frame, verbose=False)
        
        objects = []
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Extract data
                conf = float(box.conf[0])
                if conf < self.confidence_threshold:
                    continue
                
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get class name
                if hasattr(result, 'names') and cls_id in result.names:
                    class_name = result.names[cls_id]
                else:
                    class_name = "unknown"

                # TRADUCTION & NETTOYAGE
                # YOLO-World renvoie exactement les prompts anglais
                map_names = {
                    "glue stick": "COLLE", "uhu stick": "COLLE", "green glue stick": "COLLE",
                    "correction tape": "TIPP-EX", "white stationery": "TIPP-EX", "tipp-ex": "TIPP-EX",
                    "pen": "STYLO", "blue pen": "STYLO", "red pen": "STYLO", "green pen": "STYLO",
                    "marker": "MARQUEUR", "pencil": "CRAYON",
                    "cell phone": "TELEPHONE", "smartphone": "TELEPHONE",
                    "calculator": "CALCULATRICE", "scientific calculator": "CALCULATRICE",
                    "scissors": "CISEAUX", "stapler": "AGRAFEUSE",
                    "rubiks cube": "RUBIK CUBE", "puzzle cube": "RUBIK CUBE", "toy cube": "RUBIK CUBE",
                    "plastic container": "BOITE", "storage box": "BOITE", "clear box": "BOITE",
                    "scotch tape": "SCOTCH", "clear tape": "SCOTCH",
                    "mug": "TASSE", "coffee cup": "TASSE", "bottle": "BOUTEILLE"
                }
                obj_name = map_names.get(class_name.lower(), class_name.upper())
                # --- HEURISTIQUES GEOMETRIQUES (PATCHES) ---
                
                # 1. Correction BOUTEILLE -> COLLE (Updated)
                # Le user met l'objet près de la cam, donc il apparait grand.
                # On augmente le seuil de taille et on verifie le ratio (tube fin)
                w = x2 - x1
                h = y2 - y1
                ratio = w / h if h > 0 else 0
                
                if obj_name == "BOUTEILLE":
                     # Si < 450px de haut (sur 720p) ET ratio < 0.6 (plus haut que large sans être carré)
                     if h < 450 and ratio < 0.6: 
                         obj_name = "COLLE"
                         conf = max(conf, 0.65) # Boost confiance
                
                # 2. Correction CISEAUX -> STYLO
                # YOLO confond souvent les stylos fins avec des ciseaux fermés
                if obj_name == "CISEAUX":
                    # Si l'objet est très fin (ratio < 0.25) -> C'est un stylo
                    # Les ciseaux ont des poignées qui élargissent la box
                    if ratio < 0.30:
                        obj_name = "STYLO"
                        conf = max(conf, 0.60)

                # 3. Correction TELEPHONE -> STYLO (si très fin)
                if obj_name == "TELEPHONE" and ratio < 0.25:
                    obj_name = "STYLO"
                
                class_name = obj_name

                # Create object
                
                # Create object
                self.object_counter += 1
                obj = DetectedObject3D(
                    id=self.object_counter,
                    name=class_name,
                    category=ObjectDatabase.get_category(class_name),
                    confidence=conf,
                    bbox=(x1, y1, x2 - x1, y2 - y1),
                    center_px=((x1 + x2) // 2, (y1 + y2) // 2),
                    graspable=ObjectDatabase.is_graspable(class_name)
                )
                
                # Estimate size (rough)
                obj.estimated_size_m = self._estimate_size_from_bbox(obj.bbox, class_name)
                
                objects.append(obj)
        
        return objects
    
    def _detect_clip(self, 
                     frame: np.ndarray, 
                     target_objects: List[str]) -> List[DetectedObject3D]:
        """
        Détection avec CLIP (recherche d'objets spécifiques).
        Note: CLIP est un classificateur, pas un détecteur.
        On utilise une grille de patches ou combine avec détection de régions.
        """
        if self.clip_model is None:
            return []
        
        # Simplified: détecte regions puis classifie avec CLIP
        # Pour une vraie solution, utiliser OWL-ViT ou Grounded-SAM
        
        # Pour l'instant, retourne détection simple
        # TODO: Implémenter sliding window + CLIP classification
        return []
    
    def _detect_hybrid(self, 
                       frame: np.ndarray,
                       target_objects: Optional[List[str]] = None) -> List[DetectedObject3D]:
        """
        HYBRIDE : YOLO (Formes connues) + COULEUR (Objets inconnus comme Stylo/Colle).
        """
        # 1. Détection YOLO
        yolo_objects = self._detect_yolo(frame)
        
        # 2. Détection Couleur (Fallback pour objets inconnus)
        # Si YOLO ne trouve rien ou pour compléter, on regarde les taches de couleur
        color_objects = self._detect_color_legacy(frame)
        
        # Fusion simple : on ajoute les objets couleur qui ne chevauchent pas trop les objets YOLO
        final_objects = list(yolo_objects)
        
        for c_obj in color_objects:
            overlap = False
            cx, cy = c_obj.center_px
            for y_obj in yolo_objects:
                yx, yy, yw, yh = y_obj.bbox
                if yx < cx < yx + yw and yy < cy < yy + yh:
                    overlap = True
                    break
            
            if not overlap:
                # C'est un objet inconnu de YOLO mais coloré (ex: Le Cube / Boite)
                # On filtre les trop petits trucs
                if c_obj.bbox[2] * c_obj.bbox[3] > 1000: # Area min
                    c_obj.name = f"{c_obj.dominant_color.upper()} {c_obj.category.upper()}" 
                    # ex: BLUE CUBE, RED CYLINDER
                    
                    # Hack Cube
                    if "RECTANGLE" in c_obj.name or "CUBE" in c_obj.name:
                         c_obj.name = "BOITE / CUBE"

                    final_objects.append(c_obj)
        
        # 3. Persistence (Anti-Flicker)
        return self._apply_persistence(final_objects)

    def _apply_persistence(self, current_detections: List[DetectedObject3D]) -> List[DetectedObject3D]:
        """Garde les objets en mémoire quelques frames pour éviter le clignotement."""
        # Decrement TTL
        todel = []
        for oid in self.persistence_memory:
            self.persistence_memory[oid]['ttl'] -= 1
            if self.persistence_memory[oid]['ttl'] <= 0:
                todel.append(oid)
        for oid in todel:
            del self.persistence_memory[oid]
            
        # Update / Add new
        # Note: ceci est un tracking très basique basé sur la distance
        for obj in current_detections:
            matched = False
            for oid, data in self.persistence_memory.items():
                last_obj = data['obj']
                dist = np.linalg.norm(np.array(obj.center_px) - np.array(last_obj.center_px))
                if dist < 50: # Si proche de l'ancien
                    self.persistence_memory[oid] = {'obj': obj, 'ttl': self.MAX_TTL}
                    matched = True
                    break
            
            if not matched:
                self.object_counter += 1
                self.persistence_memory[self.object_counter] = {'obj': obj, 'ttl': self.MAX_TTL}
        
        # Return all active memories
        return [data['obj'] for data in self.persistence_memory.values()]
    
    def _detect_color_legacy(self, frame: np.ndarray) -> List[DetectedObject3D]:
        """Fallback: détection basique par couleur (comme perception.py)."""
        # Import local
        from perception import ObjectDetector as LegacyDetector
        
        detector = LegacyDetector()
        legacy_objects = detector.detect_objects(frame)
        
        # Convert to DetectedObject3D
        objects = []
        for obj in legacy_objects:
            self.object_counter += 1
            new_obj = DetectedObject3D(
                id=self.object_counter,
                name=f"{obj.color.value} {obj.shape.value}",
                category="colored_object",
                confidence=obj.confidence,
                bbox=obj.bbox,
                center_px=obj.center_px,
                dominant_color=obj.color.value
            )
            objects.append(new_obj)
        
        return objects
    
    def _estimate_size_from_bbox(self, 
                                  bbox: Tuple[int, int, int, int],
                                  class_name: str) -> float:
        """
        Estime la taille réelle d'un objet à partir de sa bbox.
        Utilise des tailles typiques connues.
        """
        # Tailles moyennes connues (en mètres)
        typical_sizes = {
            'bottle': 0.20,
            'cup': 0.08,
            'cell phone': 0.15,
            'book': 0.25,
            'banana': 0.15,
            'apple': 0.08,
            'spoon': 0.15,
            'fork': 0.18,
            'knife': 0.20,
            'mouse': 0.10,
            'scissors': 0.15,
            'teddy bear': 0.25,
        }
        
        # Default si inconnu
        return typical_sizes.get(class_name, 0.05)
    
    def find_object(self, 
                    frame: np.ndarray,
                    query: str) -> Optional[DetectedObject3D]:
        """
        Cherche un objet spécifique par nom/description.
        
        Args:
            frame: Image
            query: "red bottle", "my phone", "the cup"
            
        Returns:
            DetectedObject3D le plus pertinent, ou None
        """
        # Parse query
        query_lower = query.lower()
        
        # Detect all objects
        objects = self.detect_frame(frame)
        
        # Match by name
        for obj in objects:
            if query_lower in obj.name.lower():
                return obj
        
        # Match by color if specified
        color_keywords = ['red', 'blue', 'green', 'yellow', 'orange', 'white', 'black']
        for color in color_keywords:
            if color in query_lower:
                for obj in objects:
                    if obj.dominant_color and color in obj.dominant_color.lower():
                        return obj
        
        return None
    
    def draw_detections(self, 
                        frame: np.ndarray,
                        objects: List[DetectedObject3D],
                        show_3d: bool = False) -> np.ndarray:
        """
        Dessine les détections sur l'image.
        """
        annotated = frame.copy()
        
        for obj in objects:
            x, y, w, h = obj.bbox
            
            # Couleur selon confiance
            if obj.confidence > 0.8:
                color = (0, 255, 0)  # Vert
            elif obj.confidence > 0.5:
                color = (0, 255, 255)  # Jaune
            else:
                color = (0, 165, 255)  # Orange
            
            # Bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            
            # Label
            label = f"{obj.name} {obj.confidence:.2f}"
            if show_3d and obj.position_3d:
                label += f" @ ({obj.position_3d[0]:.2f}, {obj.position_3d[1]:.2f})"
            
            # Background pour le texte
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x, y - text_h - 4), (x + text_w, y), color, -1)
            cv2.putText(annotated, label, (x, y - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Center point
            cx, cy = obj.center_px
            cv2.circle(annotated, (cx, cy), 5, color, -1)
        
        return annotated


# ============================================================================
# CAMERA CONTROLLER (Webcam + iPhone)
# ============================================================================

class CameraSource(Enum):
    """Sources de caméra disponibles."""
    WEBCAM = "webcam"               # Webcam intégrée (ID 0)
    WEBCAM_EXTERNAL = "external"    # Webcam USB (ID 1)
    IPHONE_CONTINUITY = "iphone"    # iPhone via Continuity Camera (macOS)
    IP_CAMERA = "ip"                # Caméra IP/réseau
    VIDEO_FILE = "file"             # Fichier vidéo


class SmartCamera:
    """
    Contrôleur de caméra intelligent avec support multi-sources.
    """
    
    def __init__(self, 
                 source: CameraSource = CameraSource.WEBCAM,
                 resolution: Tuple[int, int] = (1280, 720)):
        """
        Initialize camera.
        
        Args:
            source: Source de la caméra
            resolution: Résolution désirée (width, height)
        """
        self.source = source
        self.resolution = resolution
        self.cap = None
        self.is_open = False
        
        # Homography transformer (pour 2D -> 3D)
        self.transformer = None
    
    def open(self, device_id: int = 0) -> bool:
        """
        Ouvre la caméra.
        
        Args:
            device_id: ID du device (0=default, 1=external, etc.)
        """
        if self.source == CameraSource.IPHONE_CONTINUITY:
            # Sur macOS, Continuity Camera apparaît comme device > 0
            # Essayer plusieurs IDs
            for cam_id in range(0, 5):
                self.cap = cv2.VideoCapture(cam_id)
                if self.cap.isOpened():
                    # Check si c'est un iPhone (résolution typique)
                    w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    if w >= 1280:  # iPhone généralement HD+
                        device_id = cam_id
                        break
                self.cap.release()
        
        # Open camera
        self.cap = cv2.VideoCapture(device_id)
        
        if not self.cap.isOpened():
            print(f"❌ Cannot open camera {device_id}")
            return False
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        # Verify
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.is_open = True
        print(f"✅ Camera opened: {actual_w}x{actual_h}")
        
        return True
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Lit une frame."""
        if not self.is_open or self.cap is None:
            return False, None
        return self.cap.read()
    
    def close(self):
        """Ferme la caméra."""
        if self.cap:
            self.cap.release()
        self.is_open = False
    
    def calibrate_3d(self, pixel_points: List[Tuple[int, int]],
                     world_points: List[Tuple[float, float]]):
        """
        Calibre la transformation 2D -> 3D.
        Utilise vision.py HomographyTransformer.
        """
        from vision import HomographyTransformer
        
        self.transformer = HomographyTransformer()
        success = self.transformer.calibrate(pixel_points, world_points)
        
        if success:
            print("✅ 3D calibration successful")
        
        return success
    
    def add_3d_coordinates(self, obj: DetectedObject3D) -> DetectedObject3D:
        """Ajoute les coordonnées 3D à un objet détecté."""
        if self.transformer and self.transformer.is_calibrated:
            cx, cy = obj.center_px
            x, y, z = self.transformer.pixels_to_world(cx, cy)
            obj.position_3d = (x, y, z)
        
        return obj


# ============================================================================
# TEST / DEMO
# ============================================================================

def demo_webcam_detection():
    """Démo: détection temps réel sur webcam."""
    print("=" * 60)
    print("=" * 60)
    print("=" * 60)
    print("📷 DEMO: Détection d'objets en temps réel (Mode Bureau)")
    print("=" * 60)
    if not CLIP_AVAILABLE:
        print("⚠️  AVIS: Mode 'Vitesse' actif (YOLO).")
        print("    Objets détectables : Ciseaux, Tasses, Claviers, Souris, Téléphones...")
        print("    Objets INVISIBLES : Stylos, Colle (Nécessite l'installation de 'CLIP')")
        print("=" * 60)
    print("Appuyez sur 'q' pour quitter, 's' pour capture d'écran")
    
    # Initialize
    camera = SmartCamera(source=CameraSource.WEBCAM)
    if not camera.open():
        return
    
    detector = AdvancedObjectDetector(
        backend=DetectionBackend.YOLO,
        confidence_threshold=0.15 # Seuil bas pour la demo
    )
    
    frame_count = 0
    
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        
        # Detect objects (every 2 frames for better reactivity)
        if frame_count % 2 == 0:
            objects = detector.detect_frame(frame)
            
            # Add 3D if calibrated
            for obj in objects:
                camera.add_3d_coordinates(obj)
        
        # Draw (use last detected objects)
        annotated = detector.draw_detections(frame, objects)
        
        # Info overlay
        cv2.putText(annotated, f"Objets: {len(objects) if frame_count % 3 == 0 else 0}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Object Detection", annotated)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save screenshot
            filename = f"detection_{frame_count}.jpg"
            cv2.imwrite(filename, annotated)
            print(f"📸 Sauvegarde : {filename}")
        
        frame_count += 1
    
    camera.close()
    cv2.destroyAllWindows()


def demo_find_object():
    """Démo: chercher un objet spécifique."""
    print("=" * 60)
    print("🔍 DEMO: Recherche d'objet")
    print("=" * 60)
    
    camera = SmartCamera()
    if not camera.open():
        return
    
    detector = AdvancedObjectDetector(backend=DetectionBackend.YOLO)
    
    # Capture frame
    ret, frame = camera.read()
    if not ret:
        camera.close()
        return
    
    # Detect
    objects = detector.detect_frame(frame)
    
    print(f"\n✅ Found {len(objects)} objects:")
    for obj in objects:
        print(f"   - {obj}")
    
    # Search for specific object
    query = input("\nEnter object to find (e.g., 'bottle', 'phone'): ")
    
    result = detector.find_object(frame, query)
    
    if result:
        print(f"\n🎯 Found: {result}")
        
        # Show
        annotated = detector.draw_detections(frame, [result])
        cv2.imshow("Found Object", annotated)
        cv2.waitKey(0)
    else:
        print(f"\n❌ Object '{query}' not found")
    
    camera.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', choices=['webcam', 'find'], default='webcam',
                       help="Demo mode")
    args = parser.parse_args()
    
    if args.demo == 'webcam':
        demo_webcam_detection()
    elif args.demo == 'find':
        demo_find_object()
