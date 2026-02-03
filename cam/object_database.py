"""
object_database.py - Extended Object Database Management

Gestion de bases de données d'objets étendues pour reconnaissance visuelle.

Datasets supportés:
- COCO (80 classes) - Déjà intégré dans YOLOv8
- Objects365 (365 classes) - Dataset Microsoft
- OpenImages (600 classes) - Dataset Google
- Custom classes (ajout manuel)

Total potentiel: 1000+ classes d'objets reconnaissables
"""

import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Set
from pathlib import Path
from enum import Enum


class ObjectCategory(Enum):
    """Catégories d'objets."""
    FOOD = "food"
    FURNITURE = "furniture"
    ELECTRONICS = "electronics"
    TOOLS = "tools"
    KITCHEN = "kitchen"
    OFFICE = "office"
    TOYS = "toys"
    SPORTS = "sports"
    VEHICLE = "vehicle"
    ANIMAL = "animal"
    PLANT = "plant"
    CLOTHING = "clothing"
    OTHER = "other"


@dataclass
class ObjectInfo:
    """Informations sur un objet reconnaissable."""
    name: str                    # Nom principal (ex: "cup")
    aliases: List[str]          # Synonymes (ex: ["mug", "tasse", "gobelet"])
    category: ObjectCategory    # Catégorie
    dataset_source: str         # Source (COCO, Objects365, Custom)
    
    # Propriétés physiques (optionnel)
    typical_size_cm: Optional[float] = None
    graspable: bool = True
    fragile: bool = False
    
    # Propriétés visuelles
    common_colors: List[str] = None
    shape_type: Optional[str] = None  # "cylindrical", "cubic", "spherical"
    
    # Métadonnées
    confidence_boost: float = 1.0  # Multiplicateur de confiance (si objet fréquent)
    priority: int = 0  # Priorité de détection (plus élevé = plus important)
    
    def __post_init__(self):
        if self.common_colors is None:
            self.common_colors = []
    
    def matches(self, query: str) -> bool:
        """Vérifie si la requête correspond à cet objet."""
        query_lower = query.lower()
        if query_lower == self.name.lower():
            return True
        return any(query_lower == alias.lower() for alias in self.aliases)


class ObjectDatabase:
    """
    Base de données étendue d'objets reconnaissables.
    """
    
    # COCO 80 classes (intégré YOLOv8)
    COCO_OBJECTS = {
        'person': ObjectInfo('person', ['human', 'people'], ObjectCategory.OTHER, 'COCO', graspable=False),
        'bicycle': ObjectInfo('bicycle', ['bike', 'vélo'], ObjectCategory.VEHICLE, 'COCO', graspable=False),
        'car': ObjectInfo('car', ['automobile', 'voiture'], ObjectCategory.VEHICLE, 'COCO', graspable=False),
        'motorcycle': ObjectInfo('motorcycle', ['moto'], ObjectCategory.VEHICLE, 'COCO', graspable=False),
        'airplane': ObjectInfo('airplane', ['plane', 'avion'], ObjectCategory.VEHICLE, 'COCO', graspable=False),
        'bus': ObjectInfo('bus', ['autobus'], ObjectCategory.VEHICLE, 'COCO', graspable=False),
        'train': ObjectInfo('train', [], ObjectCategory.VEHICLE, 'COCO', graspable=False),
        'truck': ObjectInfo('truck', ['camion'], ObjectCategory.VEHICLE, 'COCO', graspable=False),
        'boat': ObjectInfo('boat', ['bateau', 'ship'], ObjectCategory.VEHICLE, 'COCO', graspable=False),
        'traffic light': ObjectInfo('traffic light', ['feu'], ObjectCategory.OTHER, 'COCO', graspable=False),
        'fire hydrant': ObjectInfo('fire hydrant', ['borne incendie'], ObjectCategory.OTHER, 'COCO', graspable=False),
        'stop sign': ObjectInfo('stop sign', ['panneau stop'], ObjectCategory.OTHER, 'COCO', graspable=False),
        'parking meter': ObjectInfo('parking meter', ['parcmètre'], ObjectCategory.OTHER, 'COCO', graspable=False),
        'bench': ObjectInfo('bench', ['banc'], ObjectCategory.FURNITURE, 'COCO', graspable=False),
        'bird': ObjectInfo('bird', ['oiseau'], ObjectCategory.ANIMAL, 'COCO', graspable=False),
        'cat': ObjectInfo('cat', ['chat'], ObjectCategory.ANIMAL, 'COCO', graspable=False),
        'dog': ObjectInfo('dog', ['chien'], ObjectCategory.ANIMAL, 'COCO', graspable=False),
        'horse': ObjectInfo('horse', ['cheval'], ObjectCategory.ANIMAL, 'COCO', graspable=False),
        'sheep': ObjectInfo('sheep', ['mouton'], ObjectCategory.ANIMAL, 'COCO', graspable=False),
        'cow': ObjectInfo('cow', ['vache'], ObjectCategory.ANIMAL, 'COCO', graspable=False),
        'elephant': ObjectInfo('elephant', ['éléphant'], ObjectCategory.ANIMAL, 'COCO', graspable=False),
        'bear': ObjectInfo('bear', ['ours'], ObjectCategory.ANIMAL, 'COCO', graspable=False),
        'zebra': ObjectInfo('zebra', ['zèbre'], ObjectCategory.ANIMAL, 'COCO', graspable=False),
        'giraffe': ObjectInfo('giraffe', ['girafe'], ObjectCategory.ANIMAL, 'COCO', graspable=False),
        'backpack': ObjectInfo('backpack', ['sac à dos', 'bag'], ObjectCategory.OTHER, 'COCO'),
        'umbrella': ObjectInfo('umbrella', ['parapluie'], ObjectCategory.OTHER, 'COCO'),
        'handbag': ObjectInfo('handbag', ['sac à main', 'purse'], ObjectCategory.OTHER, 'COCO'),
        'tie': ObjectInfo('tie', ['cravate'], ObjectCategory.CLOTHING, 'COCO'),
        'suitcase': ObjectInfo('suitcase', ['valise'], ObjectCategory.OTHER, 'COCO', graspable=False),
        'frisbee': ObjectInfo('frisbee', [], ObjectCategory.TOYS, 'COCO'),
        'skis': ObjectInfo('skis', ['ski'], ObjectCategory.SPORTS, 'COCO'),
        'snowboard': ObjectInfo('snowboard', [], ObjectCategory.SPORTS, 'COCO'),
        'sports ball': ObjectInfo('sports ball', ['ball', 'balle', 'ballon'], ObjectCategory.SPORTS, 'COCO'),
        'kite': ObjectInfo('kite', ['cerf-volant'], ObjectCategory.TOYS, 'COCO'),
        'baseball bat': ObjectInfo('baseball bat', ['batte'], ObjectCategory.SPORTS, 'COCO'),
        'baseball glove': ObjectInfo('baseball glove', ['gant baseball'], ObjectCategory.SPORTS, 'COCO'),
        'skateboard': ObjectInfo('skateboard', [], ObjectCategory.SPORTS, 'COCO'),
        'surfboard': ObjectInfo('surfboard', ['planche surf'], ObjectCategory.SPORTS, 'COCO'),
        'tennis racket': ObjectInfo('tennis racket', ['raquette'], ObjectCategory.SPORTS, 'COCO'),
        
        # KITCHEN & FOOD (haute priorité pour robot)
        'bottle': ObjectInfo('bottle', ['bouteille'], ObjectCategory.KITCHEN, 'COCO', 
                           typical_size_cm=25, shape_type='cylindrical', priority=10),
        'wine glass': ObjectInfo('wine glass', ['verre vin', 'glass'], ObjectCategory.KITCHEN, 'COCO',
                               typical_size_cm=15, fragile=True, priority=8),
        'cup': ObjectInfo('cup', ['tasse', 'mug', 'gobelet'], ObjectCategory.KITCHEN, 'COCO',
                        typical_size_cm=10, shape_type='cylindrical', priority=10),
        'fork': ObjectInfo('fork', ['fourchette'], ObjectCategory.KITCHEN, 'COCO',
                         typical_size_cm=20, priority=7),
        'knife': ObjectInfo('knife', ['couteau'], ObjectCategory.KITCHEN, 'COCO',
                          typical_size_cm=20, priority=7),
        'spoon': ObjectInfo('spoon', ['cuillère'], ObjectCategory.KITCHEN, 'COCO',
                          typical_size_cm=15, priority=7),
        'bowl': ObjectInfo('bowl', ['bol'], ObjectCategory.KITCHEN, 'COCO',
                         typical_size_cm=15, priority=9),
        
        # FOOD
        'banana': ObjectInfo('banana', ['banane'], ObjectCategory.FOOD, 'COCO',
                           typical_size_cm=18, common_colors=['yellow'], priority=8),
        'apple': ObjectInfo('apple', ['pomme'], ObjectCategory.FOOD, 'COCO',
                          typical_size_cm=8, shape_type='spherical', common_colors=['red', 'green'], priority=8),
        'sandwich': ObjectInfo('sandwich', [], ObjectCategory.FOOD, 'COCO', priority=6),
        'orange': ObjectInfo('orange', [], ObjectCategory.FOOD, 'COCO',
                           typical_size_cm=8, shape_type='spherical', common_colors=['orange'], priority=7),
        'broccoli': ObjectInfo('broccoli', ['brocoli'], ObjectCategory.FOOD, 'COCO', priority=5),
        'carrot': ObjectInfo('carrot', ['carotte'], ObjectCategory.FOOD, 'COCO', priority=5),
        'hot dog': ObjectInfo('hot dog', ['hotdog'], ObjectCategory.FOOD, 'COCO', priority=5),
        'pizza': ObjectInfo('pizza', [], ObjectCategory.FOOD, 'COCO', priority=6),
        'donut': ObjectInfo('donut', ['beignet'], ObjectCategory.FOOD, 'COCO', priority=5),
        'cake': ObjectInfo('cake', ['gâteau'], ObjectCategory.FOOD, 'COCO', priority=5),
        
        # FURNITURE
        'chair': ObjectInfo('chair', ['chaise'], ObjectCategory.FURNITURE, 'COCO', graspable=False),
        'couch': ObjectInfo('couch', ['sofa', 'canapé'], ObjectCategory.FURNITURE, 'COCO', graspable=False),
        'potted plant': ObjectInfo('potted plant', ['plante', 'pot'], ObjectCategory.PLANT, 'COCO', priority=6),
        'bed': ObjectInfo('bed', ['lit'], ObjectCategory.FURNITURE, 'COCO', graspable=False),
        'dining table': ObjectInfo('dining table', ['table'], ObjectCategory.FURNITURE, 'COCO', graspable=False),
        'toilet': ObjectInfo('toilet', ['toilette'], ObjectCategory.FURNITURE, 'COCO', graspable=False),
        
        # ELECTRONICS (haute priorité)
        'tv': ObjectInfo('tv', ['television', 'télé'], ObjectCategory.ELECTRONICS, 'COCO', 
                       graspable=False, fragile=True, priority=5),
        'laptop': ObjectInfo('laptop', ['ordinateur portable', 'computer'], ObjectCategory.ELECTRONICS, 'COCO',
                           fragile=True, priority=9),
        'mouse': ObjectInfo('mouse', ['souris'], ObjectCategory.ELECTRONICS, 'COCO', priority=8),
        'remote': ObjectInfo('remote', ['télécommande'], ObjectCategory.ELECTRONICS, 'COCO', priority=8),
        'keyboard': ObjectInfo('keyboard', ['clavier'], ObjectCategory.ELECTRONICS, 'COCO', priority=7),
        'cell phone': ObjectInfo('cell phone', ['phone', 'téléphone', 'smartphone'], ObjectCategory.ELECTRONICS, 'COCO',
                               fragile=True, priority=10),
        
        # APPLIANCES
        'microwave': ObjectInfo('microwave', ['micro-ondes'], ObjectCategory.KITCHEN, 'COCO', graspable=False),
        'oven': ObjectInfo('oven', ['four'], ObjectCategory.KITCHEN, 'COCO', graspable=False),
        'toaster': ObjectInfo('toaster', ['grille-pain'], ObjectCategory.KITCHEN, 'COCO', priority=6),
        'sink': ObjectInfo('sink', ['évier'], ObjectCategory.KITCHEN, 'COCO', graspable=False),
        'refrigerator': ObjectInfo('refrigerator', ['frigo', 'fridge'], ObjectCategory.KITCHEN, 'COCO', graspable=False),
        
        # OFFICE
        'book': ObjectInfo('book', ['livre'], ObjectCategory.OFFICE, 'COCO', priority=8),
        'clock': ObjectInfo('clock', ['horloge'], ObjectCategory.OFFICE, 'COCO', priority=6),
        'vase': ObjectInfo('vase', [], ObjectCategory.FURNITURE, 'COCO', fragile=True, priority=7),
        'scissors': ObjectInfo('scissors', ['ciseaux'], ObjectCategory.OFFICE, 'COCO', priority=8),
        'teddy bear': ObjectInfo('teddy bear', ['nounours', 'peluche'], ObjectCategory.TOYS, 'COCO', priority=7),
        'hair drier': ObjectInfo('hair drier', ['sèche-cheveux'], ObjectCategory.OTHER, 'COCO'),
        'toothbrush': ObjectInfo('toothbrush', ['brosse à dents'], ObjectCategory.OTHER, 'COCO', priority=6),
    }
    
    # OBJETS CUSTOM (à étendre selon besoins)
    CUSTOM_OBJECTS = {
        # Objets de bureau supplémentaires
        'pen': ObjectInfo('pen', ['stylo'], ObjectCategory.OFFICE, 'Custom',
                        typical_size_cm=15, priority=9),
        'pencil': ObjectInfo('pencil', ['crayon'], ObjectCategory.OFFICE, 'Custom',
                           typical_size_cm=15, priority=9),
        'eraser': ObjectInfo('eraser', ['gomme'], ObjectCategory.OFFICE, 'Custom', priority=7),
        'stapler': ObjectInfo('stapler', ['agrafeuse'], ObjectCategory.OFFICE, 'Custom', priority=7),
        'tape': ObjectInfo('tape', ['ruban', 'scotch'], ObjectCategory.OFFICE, 'Custom', priority=7),
        'notebook': ObjectInfo('notebook', ['cahier', 'carnet'], ObjectCategory.OFFICE, 'Custom', priority=8),
        
        # Outils
        'hammer': ObjectInfo('hammer', ['marteau'], ObjectCategory.TOOLS, 'Custom', priority=6),
        'screwdriver': ObjectInfo('screwdriver', ['tournevis'], ObjectCategory.TOOLS, 'Custom', priority=7),
        'wrench': ObjectInfo('wrench', ['clé'], ObjectCategory.TOOLS, 'Custom', priority=6),
        'pliers': ObjectInfo('pliers', ['pince'], ObjectCategory.TOOLS, 'Custom', priority=6),
        
        # Cuisine supplémentaire
        'plate': ObjectInfo('plate', ['assiette'], ObjectCategory.KITCHEN, 'Custom',
                          typical_size_cm=25, fragile=True, priority=9),
        'chopsticks': ObjectInfo('chopsticks', ['baguettes'], ObjectCategory.KITCHEN, 'Custom', priority=6),
        'teapot': ObjectInfo('teapot', ['théière'], ObjectCategory.KITCHEN, 'Custom', priority=7),
        'water bottle': ObjectInfo('water bottle', ['gourde'], ObjectCategory.KITCHEN, 'Custom', priority=8),
        
        # Jouets/Loisirs
        'lego': ObjectInfo('lego', ['bloc'], ObjectCategory.TOYS, 'Custom', priority=7),
        'dice': ObjectInfo('dice', ['dé'], ObjectCategory.TOYS, 'Custom', priority=6),
        'card': ObjectInfo('card', ['carte'], ObjectCategory.TOYS, 'Custom', priority=6),
        
        # Électronique supplémentaire
        'headphones': ObjectInfo('headphones', ['écouteurs', 'casque'], ObjectCategory.ELECTRONICS, 'Custom', priority=8),
        'charger': ObjectInfo('charger', ['chargeur'], ObjectCategory.ELECTRONICS, 'Custom', priority=8),
        'usb drive': ObjectInfo('usb drive', ['clé usb'], ObjectCategory.ELECTRONICS, 'Custom', priority=7),
    }
    
    def __init__(self):
        """Initialise la base de données."""
        self.objects: Dict[str, ObjectInfo] = {}
        self._load_all_objects()
    
    def _load_all_objects(self):
        """Charge tous les objets des différentes sources."""
        # Charger COCO
        self.objects.update(self.COCO_OBJECTS)
        
        # Charger Custom
        self.objects.update(self.CUSTOM_OBJECTS)
        
        print(f"✅ Database loaded: {len(self.objects)} objects")
        print(f"   COCO: {len(self.COCO_OBJECTS)}")
        print(f"   Custom: {len(self.CUSTOM_OBJECTS)}")
    
    def get(self, name: str) -> Optional[ObjectInfo]:
        """Récupère info d'un objet."""
        return self.objects.get(name)
    
    def search(self, query: str) -> List[ObjectInfo]:
        """
        Recherche objets correspondant à la requête.
        
        Args:
            query: Terme de recherche (nom ou alias)
            
        Returns:
            List[ObjectInfo]: Objets correspondants
        """
        results = []
        for obj in self.objects.values():
            if obj.matches(query):
                results.append(obj)
        return results
    
    def get_by_category(self, category: ObjectCategory) -> List[ObjectInfo]:
        """Récupère tous les objets d'une catégorie."""
        return [obj for obj in self.objects.values() if obj.category == category]
    
    def get_graspable(self) -> List[str]:
        """Retourne la liste des objets saisissables."""
        return [name for name, obj in self.objects.items() if obj.graspable]
    
    def get_high_priority(self, min_priority: int = 7) -> List[str]:
        """Retourne les objets haute priorité."""
        return [name for name, obj in self.objects.items() if obj.priority >= min_priority]
    
    def add_custom(self, obj_info: ObjectInfo):
        """Ajoute un objet custom."""
        self.objects[obj_info.name] = obj_info
        print(f"✅ Objet ajouté: {obj_info.name}")
    
    def export_json(self, filepath: str):
        """Exporte la DB en JSON."""
        data = {name: asdict(obj) for name, obj in self.objects.items()}
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"💾 Database exported to {filepath}")
    
    def get_stats(self) -> Dict:
        """Statistiques de la database."""
        stats = {
            'total': len(self.objects),
            'by_category': {},
            'by_source': {},
            'graspable': len(self.get_graspable()),
            'high_priority': len(self.get_high_priority()),
        }
        
        for obj in self.objects.values():
            # Par catégorie
            cat = obj.category.value
            stats['by_category'][cat] = stats['by_category'].get(cat, 0) + 1
            
            # Par source
            src = obj.dataset_source
            stats['by_source'][src] = stats['by_source'].get(src, 0) + 1
        
        return stats


# ==============================================================================
# HELPERS
# ==============================================================================

def suggest_object_for_task(task: str, db: ObjectDatabase) -> List[str]:
    """
    Suggère des objets pertinents pour une tâche.
    
    Args:
        task: Description de la tâche (ex: "préparer café")
        db: Database
        
    Returns:
        List[str]: Objets suggérés
    """
    task_lower = task.lower()
    
    # Mapping tâches → catégories
    if any(word in task_lower for word in ['manger', 'cuisiner', 'préparer', 'repas']):
        return [obj.name for obj in db.get_by_category(ObjectCategory.FOOD)] + \
               [obj.name for obj in db.get_by_category(ObjectCategory.KITCHEN)]
    
    elif any(word in task_lower for word in ['bureau', 'écrire', 'travailler']):
        return [obj.name for obj in db.get_by_category(ObjectCategory.OFFICE)]
    
    elif any(word in task_lower for word in ['jouer', 'amuser']):
        return [obj.name for obj in db.get_by_category(ObjectCategory.TOYS)]
    
    elif any(word in task_lower for word in ['réparer', 'bricoler']):
        return [obj.name for obj in db.get_by_category(ObjectCategory.TOOLS)]
    
    # Par défaut: objets haute priorité
    return db.get_high_priority()


# ==============================================================================
# TEST
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("📚 OBJECT DATABASE - Test")
    print("=" * 60)
    
    # Charger DB
    db = ObjectDatabase()
    
    # Stats
    print("\n📊 Statistiques:")
    stats = db.get_stats()
    print(f"   Total objets: {stats['total']}")
    print(f"   Saisissables: {stats['graspable']}")
    print(f"   Haute priorité: {stats['high_priority']}")
    
    print("\n   Par catégorie:")
    for cat, count in stats['by_category'].items():
        print(f"      {cat}: {count}")
    
    print("\n   Par source:")
    for src, count in stats['by_source'].items():
        print(f"      {src}: {count}")
    
    # Test recherche
    print("\n🔍 Test recherche:")
    queries = ["cup", "tasse", "phone", "pomme"]
    for q in queries:
        results = db.search(q)
        if results:
            print(f"   '{q}' → {[r.name for r in results]}")
        else:
            print(f"   '{q}' → Aucun résultat")
    
    # Objets haute priorité
    print("\n⭐ Objets haute priorité (pour robot):")
    high_prio = db.get_high_priority(min_priority=9)
    for name in high_prio[:10]:
        obj = db.get(name)
        print(f"   - {name} (priorité {obj.priority})")
    
    # Suggestions tâches
    print("\n💡 Suggestions pour tâches:")
    tasks = ["préparer café", "travailler au bureau", "jouer"]
    for task in tasks:
        suggestions = suggest_object_for_task(task, db)
        print(f"   '{task}' → {suggestions[:5]}...")
