"""
Natural Language Generation Module for MCSA
Converts AI strategy parameters into human-readable tactical briefings

FREE OPTIONS:
1. Template Mode (Offline, Always Free)
2. Ollama (Local LLM, Free, No Internet Required)
3. Groq API (Free tier: 30 req/min, No credit card required)

Usage:
    from strategy_nlg import StrategyNLG
    
    # Option 1: Templates only (always works)
    nlg = StrategyNLG(mode='template')
    
    # Option 2: Ollama (install ollama locally)
    nlg = StrategyNLG(mode='ollama')
    
    # Option 3: Groq (free API, get key from console.groq.com)
    nlg = StrategyNLG(mode='groq', api_key="gsk_...")
    
    briefing = nlg.generate_briefing(
        predictions=ai_predictions,
        opencv_analysis=opencv_results
    )
"""

import json
import requests
from typing import Dict, Optional, Literal
from dataclasses import dataclass

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class NLGConfig:
    """Configuration for Natural Language Generation"""
    
    # Mode: 'template', 'ollama', or 'groq'
    mode: Literal['template', 'ollama', 'groq'] = 'template'
    
    # Ollama Settings (Local, Free, No Internet)
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "llama3.2"  # or "mistral", "phi3", etc.
    ollama_timeout: int = 30
    
    # Groq Settings (Free API, Requires Internet)
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"  # Fast and free
    groq_url: str = "https://api.groq.com/openai/v1/chat/completions"
    groq_timeout: int = 15
    
    # Strategy Mappings
    STRATEGY_NAMES = [
        'Frontal Assault', 'Pincer Movement', 'Flanking Maneuver', 'Envelopment',
        'Feigned Retreat', 'Ambush', 'Island Hopping', 'Defensive Screen',
        'Hit and Run', 'Concentration of Force', 'Divide and Conquer', 'Crossing the T',
        'Wolf Pack', 'Decoy Operation', 'Breakthrough', 'Double Envelopment',
        'Oblique Approach', 'Echelon Formation', 'Hammer and Anvil', 'Strategic Withdrawal'
    ]
    
    ENGAGEMENT_DISTANCE = ['Long Range', 'Medium Range', 'Close Quarters']
    TEMPO = ['Aggressive', 'Measured', 'Cautious', 'Opportunistic']
    FLANKING = ['No Flank', 'Left Flank', 'Right Flank', 'Both Flanks']
    TARGET_PRIORITY = ['Closest Threats', 'Strongest Vessels', 'Isolated Units', 'Weakest Targets']
    FORMATION = ['Wedge Formation', 'Line Formation', 'Circular Formation', 'Scattered Formation']

# ============================================================================
# Natural Language Generator
# ============================================================================

class StrategyNLG:
    """
    Generates natural language tactical briefings from AI predictions
    Supports: Templates (offline), Ollama (local), Groq (free cloud API)
    """
    
    def __init__(self, mode: str = 'template', api_key: str = ""):
        """
        Initialize NLG module
        
        Args:
            mode: 'template' (offline), 'ollama' (local LLM), or 'groq' (free API)
            api_key: API key for Groq (get free from console.groq.com)
        """
        self.config = NLGConfig(mode=mode, groq_api_key=api_key)
        self._service_available = None
    
    def set_mode(self, mode: str, api_key: str = ""):
        """Change generation mode"""
        self.config.mode = mode
        if api_key:
            self.config.groq_api_key = api_key
        self._service_available = None
    
    def check_service_available(self) -> bool:
        """Check if selected service is available"""
        if self.config.mode == 'template':
            return True
        
        elif self.config.mode == 'ollama':
            try:
                response = requests.get(
                    "http://localhost:11434/api/tags",
                    timeout=3
                )
                self._service_available = response.status_code == 200
                return self._service_available
            except:
                self._service_available = False
                return False
        
        elif self.config.mode == 'groq':
            if not self.config.groq_api_key:
                return False
            try:
                response = requests.get(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {self.config.groq_api_key}"},
                    timeout=3
                )
                self._service_available = response.status_code == 200
                return self._service_available
            except:
                self._service_available = False
                return False
        
        return False
    
    def generate_briefing(self, predictions: Dict, opencv_analysis: Dict = None, 
                         force_template: bool = False) -> Dict:
        """
        Generate natural language briefing from AI predictions
        
        Args:
            predictions: Dictionary from ModelInference.predict()
            opencv_analysis: Optional OpenCV detection results
            force_template: Force template mode regardless of settings
            
        Returns:
            Dictionary with:
                - 'briefing': Full tactical briefing text
                - 'summary': Short 2-3 sentence summary
                - 'mode': Which mode was used
                - 'success': bool
        """
        
        if force_template or self.config.mode == 'template':
            return self._generate_with_templates(predictions, opencv_analysis)
        
        elif self.config.mode == 'ollama':
            if self.check_service_available():
                return self._generate_with_ollama(predictions, opencv_analysis)
            else:
                print("⚠️ Ollama not available, falling back to templates")
                return self._generate_with_templates(predictions, opencv_analysis)
        
        elif self.config.mode == 'groq':
            if self.check_service_available():
                return self._generate_with_groq(predictions, opencv_analysis)
            else:
                print("⚠️ Groq not available, falling back to templates")
                return self._generate_with_templates(predictions, opencv_analysis)
        
        # Default fallback
        return self._generate_with_templates(predictions, opencv_analysis)
    
    # ========================================================================
    # Ollama Generation (Local, Free, Offline)
    # ========================================================================
    
    def _generate_with_ollama(self, predictions: Dict, opencv_analysis: Dict = None) -> Dict:
        """Generate briefing using local Ollama"""
        
        try:
            strategy_name = self.config.STRATEGY_NAMES[predictions['primary_strategy']]
            
            # Prepare context
            prompt = f"""You are an elite naval warfare tactical analyst. Generate a professional 3-paragraph tactical briefing.

BATTLEFIELD INTELLIGENCE:
- Primary Strategy: {strategy_name}
- Confidence: {predictions['strategy_confidence']*100:.0f}%
- Engagement: {self.config.ENGAGEMENT_DISTANCE[predictions['engagement_distance']]}
- Tempo: {self.config.TEMPO[predictions['tempo']]}
- Aggression: {predictions['aggression_level']*100:.0f}%
- Formation: {self.config.FORMATION[predictions['formation_type']]}"""

            if opencv_analysis:
                prompt += f"""
- Allied Ships: {opencv_analysis.get('total_ally_ships', 0)}
- Enemy Ships: {opencv_analysis.get('total_enemy_ships', 0)}
- Terrain Features: {opencv_analysis.get('total_terrain', 0)}"""

            prompt += """

Generate exactly 3 paragraphs:
1. Situation Assessment (2-3 sentences)
2. Strategy Explanation (3-4 sentences) 
3. Tactical Execution (2-3 sentences)

Be professional, clear, and decisive. No jargon."""

            # Call Ollama
            response = requests.post(
                self.config.ollama_url,
                json={
                    "model": self.config.ollama_model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=self.config.ollama_timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                briefing = result['response'].strip()
                
                # Extract summary
                sentences = briefing.split('. ')
                summary = '. '.join(sentences[:2]) + '.'
                
                return {
                    'briefing': briefing,
                    'summary': summary,
                    'mode': 'ollama',
                    'success': True,
                    'model': self.config.ollama_model
                }
            else:
                return self._generate_with_templates(predictions, opencv_analysis)
                
        except Exception as e:
            print(f"Ollama error: {e}")
            return self._generate_with_templates(predictions, opencv_analysis)
    
    # ========================================================================
    # Groq Generation (Free Cloud API)
    # ========================================================================
    
    def _generate_with_groq(self, predictions: Dict, opencv_analysis: Dict = None) -> Dict:
        """Generate briefing using Groq (free API)"""
        
        try:
            strategy_name = self.config.STRATEGY_NAMES[predictions['primary_strategy']]
            
            prompt_data = {
                'primary_strategy': strategy_name,
                'confidence': f"{predictions['strategy_confidence']*100:.0f}%",
                'engagement_distance': self.config.ENGAGEMENT_DISTANCE[predictions['engagement_distance']],
                'tempo': self.config.TEMPO[predictions['tempo']],
                'aggression_level': f"{predictions['aggression_level']*100:.0f}%",
                'formation_type': self.config.FORMATION[predictions['formation_type']]
            }
            
            if opencv_analysis:
                prompt_data['battlefield'] = {
                    'ally_ships': opencv_analysis.get('total_ally_ships', 0),
                    'enemy_ships': opencv_analysis.get('total_enemy_ships', 0),
                    'terrain_features': opencv_analysis.get('total_terrain', 0)
                }
            
            system_prompt = """You are an elite naval warfare tactical analyst. 

Generate a professional 3-paragraph tactical briefing:
1. **Situation Assessment** - Analyze the battlefield
2. **Strategy Recommendation** - Explain the strategy 
3. **Tactical Execution** - Specific actions

Professional tone, clear language, no jargon."""

            user_prompt = f"""Battlefield Intelligence:\n\n{json.dumps(prompt_data, indent=2)}

Generate tactical briefing."""

            # Call Groq API
            response = requests.post(
                self.config.groq_url,
                headers={
                    "Authorization": f"Bearer {self.config.groq_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.config.groq_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 500
                },
                timeout=self.config.groq_timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                briefing = result['choices'][0]['message']['content'].strip()
                
                sentences = briefing.split('. ')
                summary = '. '.join(sentences[:2]) + '.'
                
                return {
                    'briefing': briefing,
                    'summary': summary,
                    'mode': 'groq',
                    'success': True,
                    'model': self.config.groq_model
                }
            else:
                return self._generate_with_templates(predictions, opencv_analysis)
                
        except Exception as e:
            print(f"Groq error: {e}")
            return self._generate_with_templates(predictions, opencv_analysis)
    
    # ========================================================================
    # Template Generation (Always Available, Free)
    # ========================================================================
    
    def _generate_with_templates(self, predictions: Dict, opencv_analysis: Dict = None) -> Dict:
        """Generate briefing using pre-written templates"""
        
        strategy_idx = predictions['primary_strategy']
        strategy_name = self.config.STRATEGY_NAMES[strategy_idx]
        confidence = predictions['strategy_confidence']
        
        # Get battlefield context
        if opencv_analysis:
            ally_count = opencv_analysis.get('total_ally_ships', 0)
            enemy_count = opencv_analysis.get('total_enemy_ships', 0)
            terrain_count = opencv_analysis.get('total_terrain', 0)
            force_ratio = ally_count / max(enemy_count, 1)
        else:
            ally_count = enemy_count = terrain_count = 0
            force_ratio = 1.0
        
        # PARAGRAPH 1: Situation Assessment
        situation = "=== TACTICAL SITUATION ===\n\n"
        
        if opencv_analysis:
            situation += f"Battlefield analysis reveals {ally_count} allied vessel{'s' if ally_count != 1 else ''} "
            situation += f"facing {enemy_count} enemy ship{'s' if enemy_count != 1 else ''}"
            
            if terrain_count > 0:
                situation += f", with {terrain_count} terrain feature{'s' if terrain_count != 1 else ''} present"
            situation += ". "
            
            if force_ratio > 1.5:
                situation += "Our forces possess a significant numerical advantage. "
            elif force_ratio > 1.2:
                situation += "We hold a moderate numerical superiority. "
            elif force_ratio > 0.8:
                situation += "Forces are evenly matched, requiring tactical precision. "
            else:
                situation += "Enemy forces outnumber our vessels, demanding strategic caution. "
        
        # PARAGRAPH 2: Strategy Recommendation
        strategy_templates = {
            'Frontal Assault': "Analysis recommends a direct frontal assault. Concentrate all available firepower along a unified axis of advance. This aggressive approach capitalizes on superior numbers and firepower to overwhelm enemy defenses through sheer force. Maximum aggression is warranted when our forces significantly outnumber the enemy.",
            
            'Pincer Movement': "Intelligence suggests a classic pincer movement. Divide forces into two coordinated groups to attack the enemy from both flanks simultaneously. This maneuver forces the enemy to split their defensive attention, creating vulnerabilities in their formation. This tactic excels when the enemy presents a concentrated formation with exposed flanks.",
            
            'Flanking Maneuver': "Tactical assessment favors a flanking maneuver. Position forces to strike the enemy's vulnerable side, bypassing their strongest defenses. A successful flank attack can collapse enemy cohesion and force a hasty retreat. Flanking proves decisive when enemy forces show poor spatial awareness or rigid formations.",
            
            'Envelopment': "Recommend full envelopment of enemy forces. Maneuver around and behind enemy positions to create a surrounding attack posture. This strategy aims to cut off retreat routes and force enemy capitulation through positional dominance. Envelopment succeeds when we possess superior numbers and maneuverability.",
            
            'Defensive Screen': "Current conditions call for a defensive screen formation. Establish a protective perimeter prioritizing force preservation over aggressive action. Maintain defensive positions while denying enemy advancement. Defense is prudent when outnumbered or when protecting strategic assets.",
            
            'Hit and Run': "Recommend hit-and-run tactics. Execute rapid strikes against exposed enemy units, then withdraw before the enemy can mount an effective response. This guerrilla approach bleeds enemy strength while preserving our forces. This approach works when facing superior enemy numbers but possessing speed advantage.",
            
            'Strategic Withdrawal': "Intelligence recommends strategic withdrawal. Execute an organized retreat to preserve forces for future engagement. This is not defeat but rather a tactical preservation of combat capability. Withdrawal becomes necessary when facing overwhelming enemy superiority.",
            
            'Island Hopping': "Recommend island-hopping strategy. Methodically capture terrain features in sequence, establishing strongpoints while bypassing heavily defended positions. Each captured position becomes a staging area for the next advance. This strategy leverages terrain when multiple islands or land masses are present.",
            
            'Ambush': "Battlefield conditions favor an ambush strategy. Utilize available terrain to conceal forces and engage the enemy from concealment. The element of surprise multiplies our effective combat power significantly. Ambush tactics excel when terrain provides concealment and enemy routes are predictable.",
            
            'Crossing the T': "Recommend the classic 'Crossing the T' maneuver. Position our line perpendicular to the enemy's approach, bringing all guns to bear while they can only return fire from forward batteries. This maximizes our firepower advantage and requires superior positioning and excellent formation discipline.",
            
            'Wolf Pack': "Deploy wolf pack tactics. Coordinate multiple small groups in synchronized attacks from different vectors. Like naval wolfpacks, overwhelm enemy defenses through simultaneous multi-directional pressure. Wolf pack coordination works best with dispersed forces against isolated enemy groups.",
            
            'Hammer and Anvil': "Execute hammer and anvil tactics. Pin enemy forces with a holding force (anvil) while a mobile striking force (hammer) delivers a devastating blow. This classic maneuver requires excellent coordination between two force elements and enemy forces caught between two of our groups.",
            
            'Concentration of Force': "Concentrate all forces at the decisive point. Mass overwhelming combat power against a critical sector of enemy defenses. This Napoleonic principle remains valid: superior force at the key location wins battles. Concentration succeeds when we can identify and exploit a critical weakness."
        }
        
        strategy_desc = strategy_templates.get(strategy_name, 
            f"AI analysis recommends {strategy_name.upper()} as the optimal strategy based on current battlefield conditions.")
        
        strategy_section = f"\n\n=== RECOMMENDED STRATEGY: {strategy_name.upper()} ===\n\n"
        strategy_section += strategy_desc
        strategy_section += f" (AI Confidence: {confidence*100:.0f}%)"
        
        # PARAGRAPH 3: Tactical Execution
        execution = "\n\n=== TACTICAL EXECUTION PARAMETERS ===\n\n"
        
        engagement = self.config.ENGAGEMENT_DISTANCE[predictions['engagement_distance']]
        if engagement == 'Long Range':
            execution += "Engage at LONG RANGE. Maintain maximum standoff distance to exploit superior firepower while minimizing enemy return fire effectiveness. "
        elif engagement == 'Close Quarters':
            execution += "Close to SHORT RANGE. Move to close quarters to maximize hit probability and overcome enemy range advantages. "
        else:
            execution += "Engage at MEDIUM RANGE. Balance firepower effectiveness with survivability at moderate engagement distances. "
        
        tempo = self.config.TEMPO[predictions['tempo']]
        if tempo == 'Aggressive':
            execution += "Adopt an AGGRESSIVE tempo—press the attack relentlessly. "
        elif tempo == 'Cautious':
            execution += "Maintain a CAUTIOUS tempo—advance methodically with security. "
        else:
            execution += f"Proceed with a {tempo.upper()} tempo. "
        
        formation = self.config.FORMATION[predictions['formation_type']]
        execution += f"Deploy in {formation} to optimize tactical positioning. "
        
        target = self.config.TARGET_PRIORITY[predictions['target_priority']]
        execution += f"Prioritize {target.lower()} for engagement. "
        
        if predictions['use_terrain'] == 1 and terrain_count > 0:
            execution += f"\n\nTERRAIN UTILIZATION: Critical. Use available {terrain_count} terrain feature{'s' if terrain_count != 1 else ''} for cover and concealment. "
        
        aggression = predictions['aggression_level']
        retreat = predictions['retreat_threshold']
        
        execution += f"\n\nFORCE MANAGEMENT: Maintain {aggression*100:.0f}% aggression level. "
        
        if retreat > 0.6:
            execution += f"Withdraw if casualties exceed {retreat*100:.0f}%—force preservation is paramount."
        elif retreat < 0.4:
            execution += f"Accept up to {retreat*100:.0f}% casualties to achieve objectives—mission success is critical."
        else:
            execution += f"Retreat threshold set at {retreat*100:.0f}% casualties."
        
        full_briefing = situation + strategy_section + execution
        
        summary = f"{strategy_name} strategy recommended with {confidence*100:.0f}% confidence. "
        if opencv_analysis:
            summary += f"Battlefield: {ally_count} allied vs {enemy_count} enemy vessels. "
        summary += f"Engage at {engagement.lower()} with {tempo.lower()} tempo."
        
        return {
            'briefing': full_briefing,
            'summary': summary,
            'mode': 'template',
            'success': True
        }
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def get_mode_status(self) -> str:
        """Get current mode status string"""
        if self.config.mode == 'template':
            return "📝 Template Mode (Always Available)"
        
        elif self.config.mode == 'ollama':
            if self.check_service_available():
                return f"✅ Ollama Mode ({self.config.ollama_model})"
            else:
                return "⚠️ Ollama Not Running (Install: ollama.com)"
        
        elif self.config.mode == 'groq':
            if self.check_service_available():
                return f"✅ Groq Mode ({self.config.groq_model} - FREE)"
            else:
                return "⚠️ Groq Mode (No Connection)"
        
        return "📝 Template Mode"
    
    def test_service(self) -> Dict:
        """Test current service"""
        if self.config.mode == 'template':
            return {'success': True, 'message': 'Template mode always available'}
        
        elif self.config.mode == 'ollama':
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if response.status_code == 200:
                    models = response.json().get('models', [])
                    return {
                        'success': True,
                        'message': f'Ollama connected ({len(models)} models available)'
                    }
                else:
                    return {'success': False, 'message': f'Ollama returned status {response.status_code}'}
            except Exception as e:
                return {'success': False, 'message': f'Ollama not running: {str(e)}'}
        
        elif self.config.mode == 'groq':
            if not self.config.groq_api_key:
                return {'success': False, 'message': 'No API key configured'}
            try:
                response = requests.get(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {self.config.groq_api_key}"},
                    timeout=5
                )
                if response.status_code == 200:
                    return {'success': True, 'message': 'Groq API connected (FREE tier)'}
                else:
                    return {'success': False, 'message': f'API returned status {response.status_code}'}
            except Exception as e:
                return {'success': False, 'message': f'Connection error: {str(e)}'}
        
        return {'success': False, 'message': 'Unknown mode'}


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("MCSA Natural Language Generator - FREE OPTIONS TEST")
    print("="*70)
    
    example_predictions = {
        'primary_strategy': 1,
        'engagement_distance': 1,
        'tempo': 0,
        'use_terrain': 1,
        'flanking_direction': 3,
        'force_concentration': 0.75,
        'retreat_threshold': 0.4,
        'aggression_level': 0.8,
        'target_priority': 2,
        'formation_type': 1,
        'strategy_confidence': 0.89
    }
    
    example_opencv = {
        'total_ally_ships': 8,
        'total_enemy_ships': 6,
        'total_terrain': 3
    }
    
    # Test Template Mode
    print("\n📝 TEST: Template Mode (Always Free)")
    print("="*70)
    nlg = StrategyNLG(mode='template')
    result = nlg.generate_briefing(example_predictions, example_opencv)
    print(f"Status: {nlg.get_mode_status()}")
    print(f"\n{result['briefing'][:300]}...\n")
    
    # Test Ollama
    print("\n🦙 TEST: Ollama Mode (Local, Free)")
    print("="*70)
    nlg_ollama = StrategyNLG(mode='ollama')
    test_result = nlg_ollama.test_service()
    print(f"Status: {test_result['message']}")
    
    # Test Groq
    print("\n⚡ TEST: Groq Mode (Cloud, Free)")
    print("="*70)
    api_key = input("Enter Groq API key (or press Enter to skip): ").strip()
    if api_key:
        nlg_groq = StrategyNLG(mode='groq', api_key=api_key)
        test_result = nlg_groq.test_service()
        print(f"Status: {test_result['message']}")
        if test_result['success']:
            print("\nGenerating...")
            result = nlg_groq.generate_briefing(example_predictions, example_opencv)
            print(f"\n{result['briefing'][:300]}...")
    
    print("\n" + "="*70)
    print("✅ All FREE options tested!")
    print("="*70)