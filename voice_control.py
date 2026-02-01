"""
voice_control.py - Voice Command Module using Google Speech Recognition

Uses Google's free Speech-to-Text API for reliable French recognition.
No API key required for basic usage.

Features:
- Wake word activation ("bonjour bras")
- Voice Activity Detection (VAD) - listens until silence
- Fuzzy matching for command recognition
- French-optimized
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Callable
from enum import Enum
import time
from difflib import SequenceMatcher


class CommandAction(Enum):
    """Supported voice command actions."""
    PICK = "pick"
    PLACE = "place"
    MOVE = "move"
    SHOW = "show"
    STOP = "stop"
    HOME = "home"
    OPEN = "open"
    CLOSE = "close"
    UNKNOWN = "unknown"


@dataclass
class VoiceIntent:
    """Parsed voice command intent."""
    action: CommandAction
    target_object: Optional[str] = None
    target_color: Optional[str] = None
    target_position: Optional[str] = None
    raw_text: str = ""
    confidence: float = 0.0


class VoiceController:
    """
    Voice control using Google Speech Recognition.
    Much more reliable than Whisper for real-time use.
    """
    
    # Wake words
    WAKE_WORDS = [
        "bonjour bras", "bonjour brass", "bonjour bra",
        "salut bras", "hey bras", "ok bras",
        "hello arm", "hey arm", "hé bras"
    ]
    
    # Command keywords
    COMMAND_KEYWORDS = {
        CommandAction.PICK: [
            "prends", "prend", "attrape", "saisis", "saisir",
            "récupère", "recupere", "grab", "pick", "take",
            "prendre", "pren", "chope", "choppe"
        ],
        CommandAction.PLACE: [
            "pose", "poser", "dépose", "mets", "met", "place",
            "put", "drop", "lache", "lâche", "repose"
        ],
        CommandAction.MOVE: [
            "va vers", "va au", "va à", "bouge", "bouger",
            "déplace", "deplace", "go to", "move to"
        ],
        CommandAction.SHOW: [
            "montre", "montrer", "pointe", "pointer",
            "show", "point", "regarde", "indique"
        ],
        CommandAction.STOP: [
            "stop", "stoppe", "arrête", "arrete",
            "pause", "halt", "urgence"
        ],
        CommandAction.HOME: [
            "maison", "origine", "home", "repos",
            "retour", "rest", "initial"
        ],
        CommandAction.OPEN: [
            "ouvre", "ouvrir", "ouvert", "open", "écarte"
        ],
        CommandAction.CLOSE: [
            "ferme", "fermer", "fermé", "close", "serre"
        ],
    }
    
    # Colors
    COLOR_KEYWORDS = {
        "red": ["rouge", "roule", "red", "rouges"],
        "blue": ["bleu", "blue", "bleus", "bleue"],
        "green": ["vert", "verte", "green", "verts"],
        "yellow": ["jaune", "yellow", "jaunes"],
        "orange": ["orange", "orangé"],
        "white": ["blanc", "blanche", "white"],
        "black": ["noir", "noire", "black"],
    }
    
    # Objects
    OBJECT_KEYWORDS = {
        "cube": ["cube", "cubes", "cuve", "carré", "boîte", "box"],
        "sphere": ["balle", "boule", "ball", "sphere", "sphère", "ronde"],
        "cylinder": ["cylindre", "canette", "can", "tube"],
        "object": ["objet", "object", "truc", "chose"],
    }
    
    # Positions
    POSITION_KEYWORDS = {
        "left": ["gauche", "left"],
        "right": ["droite", "right"],
        "center": ["centre", "center", "milieu"],
        "front": ["devant", "front", "avant"],
        "back": ["derrière", "back", "arrière"],
    }
    
    def __init__(self, language: str = "fr-FR"):
        """
        Initialize voice controller with Google Speech Recognition.
        
        Args:
            language: Recognition language (fr-FR, en-US)
        """
        self.language = language
        self.recognizer = None
        self.microphone = None
        self.is_active = False
        
        self._init_speech_recognition()
    
    def _init_speech_recognition(self) -> bool:
        """Initialize Google Speech Recognition."""
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 1.5  # Seconds of silence to stop
            
            # Test microphone
            self.microphone = sr.Microphone()
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            print("✅ Google Speech Recognition initialisé")
            return True
            
        except ImportError:
            print("❌ speech_recognition non installé.")
            print("   Run: pip install SpeechRecognition pyaudio")
            return False
        except Exception as e:
            print(f"❌ Erreur micro: {e}")
            return False
    
    def listen(self, timeout: float = 5.0, phrase_time_limit: float = 10.0) -> Tuple[str, float]:
        """
        Listen and transcribe with Google Speech API.
        
        Args:
            timeout: Max seconds to wait for speech to start
            phrase_time_limit: Max seconds for the phrase
            
        Returns:
            (text, confidence) tuple
        """
        if self.recognizer is None:
            return "", 0.0
        
        import speech_recognition as sr
        
        try:
            with self.microphone as source:
                print("🎤 Écoute...", end=" ", flush=True)
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )
            
            print("🔄 Transcription...", end=" ", flush=True)
            
            # Use Google's free API
            text = self.recognizer.recognize_google(audio, language=self.language)
            text = text.lower().strip()
            
            print(f"✅ \"{text}\"")
            return text, 0.95  # Google generally has high confidence
            
        except sr.WaitTimeoutError:
            print("⏱️ Timeout - pas de parole détectée")
            return "", 0.0
        except sr.UnknownValueError:
            print("❓ Pas compris")
            return "", 0.0
        except sr.RequestError as e:
            print(f"❌ Erreur Google API: {e}")
            return "", 0.0
        except Exception as e:
            print(f"❌ Erreur: {e}")
            return "", 0.0
    
    def _find_in_text(self, text: str, keyword_dict: dict) -> Optional[str]:
        """Find matching key from keyword dictionary."""
        text_lower = text.lower()
        
        for key, variations in keyword_dict.items():
            for var in variations:
                if var in text_lower:
                    return key
                # Fuzzy match for longer words
                if len(var) >= 4:
                    for word in text_lower.split():
                        if len(word) >= 3 and SequenceMatcher(None, word, var).ratio() > 0.75:
                            return key
        return None
    
    def check_wake_word(self, text: str) -> bool:
        """Check if wake word is in text."""
        text_lower = text.lower()
        
        for wake_word in self.WAKE_WORDS:
            if wake_word in text_lower:
                return True
            # Check partial match
            words = wake_word.split()
            if all(w in text_lower for w in words):
                return True
        
        return False
    
    def parse_intent(self, text: str) -> VoiceIntent:
        """Parse text into command intent."""
        text_lower = text.lower().strip()
        
        # Find action
        action = CommandAction.UNKNOWN
        
        # Check multi-word phrases first
        for cmd_action, keywords in self.COMMAND_KEYWORDS.items():
            for keyword in keywords:
                if ' ' in keyword and keyword in text_lower:
                    action = cmd_action
                    break
            if action != CommandAction.UNKNOWN:
                break
        
        # Check single words
        if action == CommandAction.UNKNOWN:
            words = text_lower.split()
            for cmd_action, keywords in self.COMMAND_KEYWORDS.items():
                for keyword in keywords:
                    if ' ' not in keyword and keyword in words:
                        action = cmd_action
                        break
                    # Substring for longer keywords
                    if len(keyword) >= 4 and keyword in text_lower:
                        action = cmd_action
                        break
                if action != CommandAction.UNKNOWN:
                    break
        
        # Find color, object, position
        color = self._find_in_text(text_lower, self.COLOR_KEYWORDS)
        target_object = self._find_in_text(text_lower, self.OBJECT_KEYWORDS)
        position = self._find_in_text(text_lower, self.POSITION_KEYWORDS)
        
        # Inference: object/color without action = PICK
        if action == CommandAction.UNKNOWN and (target_object or color):
            action = CommandAction.PICK
        
        # Position without action = MOVE
        if action == CommandAction.UNKNOWN and position:
            action = CommandAction.MOVE
        
        return VoiceIntent(
            action=action,
            target_object=target_object,
            target_color=color,
            target_position=position,
            raw_text=text,
            confidence=0.9 if action != CommandAction.UNKNOWN else 0.3
        )
    
    def listen_command(self) -> Optional[VoiceIntent]:
        """Listen for a command and parse it."""
        text, confidence = self.listen()
        
        if not text:
            return None
        
        # Check for deactivation
        if "au revoir" in text.lower() or "désactive" in text.lower():
            print("👋 Désactivation...")
            self.is_active = False
            return None
        
        intent = self.parse_intent(text)
        return intent
    
    def execute_voice_command(self, intent: VoiceIntent) -> dict:
        """Convert voice intent to robot action."""
        result = {
            'success': False,
            'action': intent.action.value,
            'message': '',
            'params': {}
        }
        
        if intent.action == CommandAction.UNKNOWN:
            result['message'] = f"❓ Commande non reconnue: '{intent.raw_text}'"
            return result
        
        if intent.action == CommandAction.STOP:
            result['success'] = True
            result['message'] = "🛑 ARRÊT D'URGENCE"
            result['params'] = {'emergency': True}
            
        elif intent.action == CommandAction.HOME:
            result['success'] = True
            result['message'] = "🏠 Retour position repos"
            
        elif intent.action == CommandAction.PICK:
            obj = " ".join(filter(None, [intent.target_color, intent.target_object or "objet"]))
            result['success'] = True
            result['message'] = f"🤏 Saisie: {obj}"
            result['params'] = {'object_type': intent.target_object, 'color': intent.target_color}
            
        elif intent.action == CommandAction.PLACE:
            pos = intent.target_position or "ici"
            result['success'] = True
            result['message'] = f"📍 Placement: {pos}"
            result['params'] = {'position': pos}
            
        elif intent.action == CommandAction.OPEN:
            result['success'] = True
            result['message'] = "✋ Ouverture pince"
            result['params'] = {'gripper': 'open'}
            
        elif intent.action == CommandAction.CLOSE:
            result['success'] = True
            result['message'] = "🤏 Fermeture pince"
            result['params'] = {'gripper': 'close'}
            
        elif intent.action == CommandAction.MOVE:
            result['success'] = True
            result['message'] = f"➡️ Déplacement: {intent.target_position or 'avant'}"
            result['params'] = {'direction': intent.target_position}
            
        elif intent.action == CommandAction.SHOW:
            obj = " ".join(filter(None, [intent.target_color, intent.target_object]))
            result['success'] = True
            result['message'] = f"👉 Pointage: {obj}"
        
        return result
    
    def run_continuous(self, command_callback: Callable[[VoiceIntent], None]):
        """
        Run continuous voice control with wake word.
        """
        print("\n" + "=" * 60)
        print("🤖 Mode Commande Vocale (Google Speech)")
        print("=" * 60)
        print("\n📋 Instructions:")
        print("   1. Dis 'Bonjour bras' pour m'activer")
        print("   2. Donne ta commande (prends le cube rouge)")
        print("   3. Dis 'au revoir' pour désactiver")
        print("   4. Ctrl+C pour quitter\n")
        
        try:
            while True:
                # Wait for wake word if not active
                if not self.is_active:
                    print("\n💤 En attente... Dis 'Bonjour bras'")
                    text, _ = self.listen(timeout=10.0)
                    
                    if text and self.check_wake_word(text):
                        print("🟢 ACTIVÉ! Quelle est ta commande?")
                        self.is_active = True
                        
                        # Check if command is in same phrase
                        intent = self.parse_intent(text)
                        if intent.action != CommandAction.UNKNOWN:
                            result = self.execute_voice_command(intent)
                            print(f"   {result['message']}")
                            if result['success'] and command_callback:
                                command_callback(intent)
                        continue
                    continue
                
                # Listen for command
                print("\n🎯 Commande?")
                intent = self.listen_command()
                
                if intent is None:
                    continue
                
                # Execute
                result = self.execute_voice_command(intent)
                print(f"   {result['message']}")
                
                if result['success'] and command_callback:
                    command_callback(intent)
                
                # Stop command exits
                if intent.action == CommandAction.STOP:
                    break
                    
        except KeyboardInterrupt:
            print("\n\n👋 Au revoir!")


# ============================================================================
# TEST
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🎤 Test Google Speech Recognition")
    print("=" * 60)
    
    controller = VoiceController(language="fr-FR")
    
    if controller.recognizer:
        print("\n📋 Test du parsing:\n")
        
        tests = [
            "prends le cube rouge",
            "pose à gauche",
            "ouvre la pince",
            "stop",
            "la balle bleue",  # Should infer PICK
        ]
        
        for cmd in tests:
            intent = controller.parse_intent(cmd)
            result = controller.execute_voice_command(intent)
            print(f"   '{cmd}' → {intent.action.value}: {result['message']}")
        
        print("\n" + "-" * 40)
        print("Test en direct (Ctrl+C pour quitter):\n")
        
        controller.run_continuous(lambda i: print(f"   → Exécuter: {i.action.value}"))
