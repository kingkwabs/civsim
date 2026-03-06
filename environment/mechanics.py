"""
Game mechanics for CivSim.

Handles action execution, combat resolution, resource production, and turn processing.
"""

from __future__ import annotations
import random
from typing import Optional, TYPE_CHECKING

from .map import ResourceType, Terrain
from .actions import (
    Action, ActionType, MoveAction, AttackAction, GatherAction, 
    FoundCityAction, ProduceAction, ProposeTradeAction, AcceptTradeAction,
    RejectTradeAction, DeclareWarAction, ProposeAllianceAction, EndTurnAction
)

if TYPE_CHECKING:
    from .game_state import GameState, Unit, City, Civilization


class GameMechanics:
    """
    Handles all game mechanics and action execution.
    
    Separated from GameState to keep state and logic distinct.
    """
    
    # Production costs (in turns at base production rate)
    PRODUCTION_COSTS = {
        'warrior': 30,
        'archer': 40,
        'worker': 25,
        'settler': 60,
        'scout': 20,
    }
    
    # Unit upkeep costs per turn
    UNIT_UPKEEP = {
        'WARRIOR': {ResourceType.FOOD: 2, ResourceType.MATERIALS: 1},
        'ARCHER': {ResourceType.FOOD: 2, ResourceType.MATERIALS: 1},
        'WORKER': {ResourceType.FOOD: 1},
        'SETTLER': {ResourceType.FOOD: 3},
        'SCOUT': {ResourceType.FOOD: 1},
    }
    
    def __init__(self, game_state: GameState, rng: Optional[random.Random] = None):
        self.state = game_state
        self.rng = rng or random.Random()
    
    def execute_action(self, action: Action) -> dict:
        """
        Execute an action and return the result.
        
        Returns a dict with:
        - success: bool
        - message: str describing what happened
        - effects: dict of side effects (damage dealt, resources gained, etc.)
        """
        if not action.is_valid(self.state):
            return {'success': False, 'message': 'Invalid action', 'effects': {}}
        
        handlers = {
            ActionType.MOVE: self._execute_move,
            ActionType.ATTACK: self._execute_attack,
            ActionType.GATHER: self._execute_gather,
            ActionType.FOUND_CITY: self._execute_found_city,
            ActionType.PRODUCE: self._execute_produce,
            ActionType.PROPOSE_TRADE: self._execute_propose_trade,
            ActionType.ACCEPT_TRADE: self._execute_accept_trade,
            ActionType.REJECT_TRADE: self._execute_reject_trade,
            ActionType.DECLARE_WAR: self._execute_declare_war,
            ActionType.PROPOSE_ALLIANCE: self._execute_propose_alliance,
            ActionType.END_TURN: self._execute_end_turn,
        }
        
        handler = handlers.get(action.action_type)
        if handler:
            return handler(action)
        
        return {'success': False, 'message': f'Unhandled action type: {action.action_type}', 'effects': {}}
    
    def _execute_move(self, action: MoveAction) -> dict:
        """Execute a move action."""
        unit = self.state.units[action.unit_id]
        old_pos = (unit.q, unit.r)
        
        unit.q = action.target_q
        unit.r = action.target_r
        unit.movement_left -= 1
        
        # Check for tile capture
        tile = self.state.map.get_tile(action.target_q, action.target_r)
        captured = False
        if tile and tile.owner != action.civ_id:
            # Only capture if no enemy units present
            enemies = [u for u in self.state.get_units_at(tile.q, tile.r) if u.owner != action.civ_id]
            if not enemies:
                tile.owner = action.civ_id
                captured = True
        
        return {
            'success': True,
            'message': f'Unit moved from {old_pos} to ({action.target_q}, {action.target_r})',
            'effects': {'captured_tile': captured}
        }
    
    def _execute_attack(self, action: AttackAction) -> dict:
        """Execute an attack action with combat resolution."""
        attacker = self.state.units[action.unit_id]
        attacker.has_acted = True
        
        # Find target(s)
        defenders = [
            u for u in self.state.get_units_at(action.target_q, action.target_r)
            if u.owner != action.civ_id
        ]
        target_city = None
        for city in self.state.cities.values():
            if city.q == action.target_q and city.r == action.target_r and city.owner != action.civ_id:
                target_city = city
                break
        
        results = {'units_killed': [], 'city_captured': False, 'attacker_killed': False}
        
        # Combat against units
        if defenders:
            defender = defenders[0]  # Attack first defender
            
            # Simple combat model
            atk_roll = self.rng.randint(1, 20) + attacker.attack
            def_roll = self.rng.randint(1, 20) + defender.defense
            
            damage_to_defender = max(0, (atk_roll - def_roll) * 2)
            damage_to_attacker = max(0, (def_roll - atk_roll))
            
            defender.health -= damage_to_defender
            attacker.health -= damage_to_attacker
            
            if defender.health <= 0:
                results['units_killed'].append(defender.id)
                # Update trust - attacking lowers trust
                self.state.civilizations[defender.owner].update_trust(action.civ_id, -0.2)
            
            if attacker.health <= 0:
                results['attacker_killed'] = True
        
        # Combat against city (if no defenders)
        elif target_city:
            damage_to_city = attacker.attack + self.rng.randint(0, 10)
            target_city.health -= damage_to_city
            
            if target_city.health <= 0:
                # City captured!
                old_owner = target_city.owner
                target_city.owner = action.civ_id
                target_city.health = 100
                
                # Update tile ownership
                tile = self.state.map.get_tile(target_city.q, target_city.r)
                if tile:
                    tile.owner = action.civ_id
                
                results['city_captured'] = True
                
                # Major trust impact
                self.state.civilizations[old_owner].update_trust(action.civ_id, -0.5)
        
        # Check for eliminations
        for civ_id in self.state.civilizations:
            self.state.check_elimination(civ_id)
        
        return {
            'success': True,
            'message': f'Attack executed at ({action.target_q}, {action.target_r})',
            'effects': results
        }
    
    def _execute_gather(self, action: GatherAction) -> dict:
        """Execute a gather action (worker collects resources)."""
        unit = self.state.units[action.unit_id]
        unit.has_acted = True
        
        tile = self.state.map.get_tile(unit.q, unit.r)
        civ = self.state.civilizations[action.civ_id]
        
        # Workers get bonus yield
        gathered = {}
        for rt, base_yield in tile.resource_yields.items():
            amount = base_yield * 1.5  # 50% bonus for manual gathering
            civ.resources[rt] = civ.resources.get(rt, 0) + amount
            gathered[rt.name] = amount
        
        return {
            'success': True,
            'message': f'Worker gathered resources',
            'effects': {'gathered': gathered}
        }
    
    def _execute_found_city(self, action: FoundCityAction) -> dict:
        """Execute founding a new city."""
        unit = self.state.units[action.unit_id]
        civ = self.state.civilizations[action.civ_id]
        
        # Create the city
        city = self.state._create_city(
            action.civ_id, unit.q, unit.r, action.city_name
        )
        
        # Claim nearby tiles
        for tile in self.state.map.get_tiles_in_radius(unit.q, unit.r, 2):
            if tile.owner is None:
                tile.owner = action.civ_id
        
        # Remove the settler
        unit.health = 0  # Mark as dead
        
        return {
            'success': True,
            'message': f'Founded city: {action.city_name}',
            'effects': {'city_id': city.id}
        }
    
    def _execute_produce(self, action: ProduceAction) -> dict:
        """Set city production (actual production happens at turn end)."""
        city = self.state.cities[action.city_id]
        
        # Set production target
        if action.production_type not in city.production_queue:
            city.production_queue = [action.production_type]
            city.production_progress = 0
        
        return {
            'success': True,
            'message': f'City now producing: {action.production_type}',
            'effects': {}
        }
    
    def _execute_propose_trade(self, action: ProposeTradeAction) -> dict:
        """Create a trade offer."""
        from .game_state import TradeOffer
        
        offer = TradeOffer(
            id=self.state._next_trade_id,
            from_civ=action.civ_id,
            to_civ=action.target_civ,
            offer_resources=action.offer_resources,
            request_resources=action.request_resources,
            expires_turn=self.state.current_turn + 5,
        )
        self.state.trade_offers[offer.id] = offer
        self.state._next_trade_id += 1
        
        return {
            'success': True,
            'message': f'Trade offer sent to civ {action.target_civ}',
            'effects': {'offer_id': offer.id}
        }
    
    def _execute_accept_trade(self, action: AcceptTradeAction) -> dict:
        """Accept a trade offer and execute the exchange."""
        offer = self.state.trade_offers[action.trade_id]
        
        from_civ = self.state.civilizations[offer.from_civ]
        to_civ = self.state.civilizations[offer.to_civ]
        
        # Execute the trade
        for rt, amount in offer.offer_resources.items():
            from_civ.resources[rt] -= amount
            to_civ.resources[rt] = to_civ.resources.get(rt, 0) + amount
        
        for rt, amount in offer.request_resources.items():
            to_civ.resources[rt] -= amount
            from_civ.resources[rt] = from_civ.resources.get(rt, 0) + amount
        
        # Improve trust
        from_civ.update_trust(offer.to_civ, 0.1)
        to_civ.update_trust(offer.from_civ, 0.1)
        
        # Remove the offer
        del self.state.trade_offers[action.trade_id]
        
        return {
            'success': True,
            'message': 'Trade accepted and executed',
            'effects': {'trade_completed': True}
        }
    
    def _execute_reject_trade(self, action: RejectTradeAction) -> dict:
        """Reject a trade offer."""
        offer = self.state.trade_offers[action.trade_id]
        
        # Slight trust decrease
        self.state.civilizations[offer.from_civ].update_trust(action.civ_id, -0.05)
        
        del self.state.trade_offers[action.trade_id]
        
        return {
            'success': True,
            'message': 'Trade offer rejected',
            'effects': {}
        }
    
    def _execute_declare_war(self, action: DeclareWarAction) -> dict:
        """Declare war on another civilization."""
        from .game_state import DiplomaticStatus
        
        civ = self.state.civilizations[action.civ_id]
        target = self.state.civilizations[action.target_civ]
        
        # Set war status (mutual)
        civ.relations[action.target_civ] = DiplomaticStatus.AT_WAR
        target.relations[action.civ_id] = DiplomaticStatus.AT_WAR
        
        # Major trust decrease
        target.update_trust(action.civ_id, -0.4)
        
        # If we had an alliance, breaking it is extra bad
        # (already handled by setting to war)
        
        return {
            'success': True,
            'message': f'War declared on {target.name}',
            'effects': {}
        }
    
    def _execute_propose_alliance(self, action: ProposeAllianceAction) -> dict:
        """Propose an alliance (simplified: auto-accept for now)."""
        # In full implementation, this would create a pending proposal
        # For now, simplified to direct alliance if trust is high enough
        
        from .game_state import DiplomaticStatus
        
        civ = self.state.civilizations[action.civ_id]
        target = self.state.civilizations[action.target_civ]
        
        # Check if target trusts us enough
        their_trust = target.trust_scores.get(action.civ_id, 0.5)
        
        if their_trust > 0.6:
            # Alliance accepted
            civ.relations[action.target_civ] = DiplomaticStatus.ALLIED
            target.relations[action.civ_id] = DiplomaticStatus.ALLIED
            
            civ.update_trust(action.target_civ, 0.15)
            target.update_trust(action.civ_id, 0.15)
            
            return {
                'success': True,
                'message': f'Alliance formed with {target.name}',
                'effects': {'alliance_formed': True}
            }
        else:
            return {
                'success': True,
                'message': f'Alliance proposal rejected by {target.name}',
                'effects': {'alliance_formed': False}
            }
    
    def _execute_end_turn(self, action: EndTurnAction) -> dict:
        """End turn for this civilization."""
        # Mark all units as having moved
        for unit in self.state.get_units_for_civ(action.civ_id):
            unit.movement_left = 0
        
        return {
            'success': True,
            'message': 'Turn ended',
            'effects': {}
        }
    
    def process_turn_end(self):
        """
        Process end-of-turn effects for all civilizations.
        
        Called once after all civs have taken their actions.
        """
        for civ_id, civ in self.state.civilizations.items():
            if not civ.is_alive:
                continue
            
            # Collect resources from cities
            for city in self.state.get_cities_for_civ(civ_id):
                yields = city.get_yields(self.state.map)
                for rt, amount in yields.items():
                    civ.resources[rt] = civ.resources.get(rt, 0) + amount
                
                # Process production
                if city.production_queue:
                    current_prod = city.production_queue[0]
                    cost = self.PRODUCTION_COSTS.get(current_prod, 50)
                    
                    # Production rate based on materials
                    prod_rate = 5 + yields.get(ResourceType.MATERIALS, 0)
                    city.production_progress += prod_rate
                    
                    if city.production_progress >= cost:
                        # Complete production
                        self._complete_production(city, current_prod)
                        city.production_queue.pop(0)
                        city.production_progress = 0
            
            # Pay unit upkeep
            for unit in self.state.get_units_for_civ(civ_id):
                upkeep = self.UNIT_UPKEEP.get(unit.unit_type.name, {})
                for rt, cost in upkeep.items():
                    civ.resources[rt] = civ.resources.get(rt, 0) - cost
            
            # Handle negative food (starvation)
            if civ.resources.get(ResourceType.FOOD, 0) < 0:
                # Damage a random unit
                units = self.state.get_units_for_civ(civ_id)
                if units:
                    victim = self.rng.choice(units)
                    victim.health -= 20
                    if victim.health <= 0:
                        pass  # Unit dies
                civ.resources[ResourceType.FOOD] = 0
            
            # Reset units for next turn
            for unit in self.state.get_units_for_civ(civ_id):
                unit.reset_turn()
            
            # Population growth
            for city in self.state.get_cities_for_civ(civ_id):
                food = civ.resources.get(ResourceType.FOOD, 0)
                if food > city.population * 10:
                    city.population += 1
                    civ.resources[ResourceType.FOOD] -= city.population * 5
            
            # Check elimination
            self.state.check_elimination(civ_id)
        
        # Expire old trade offers
        expired = [
            oid for oid, offer in self.state.trade_offers.items()
            if offer.expires_turn <= self.state.current_turn
        ]
        for oid in expired:
            del self.state.trade_offers[oid]
        
        # Advance turn counter
        self.state.current_turn += 1
        
        # Check game end
        self.state.check_game_over()
    
    def _complete_production(self, city: City, production_type: str):
        """Complete a production item in a city."""
        from .game_state import UnitType
        
        type_map = {
            'warrior': UnitType.WARRIOR,
            'archer': UnitType.ARCHER,
            'worker': UnitType.WORKER,
            'settler': UnitType.SETTLER,
            'scout': UnitType.SCOUT,
        }
        
        if production_type in type_map:
            self.state._create_unit(
                city.owner,
                type_map[production_type],
                city.q,
                city.r
            )
