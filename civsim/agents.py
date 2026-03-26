"""Agent interface and baseline implementations for CivSim."""
from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from typing import Optional

from .data_types import (
    Action, ActionType, Build, BuildType, BuyDevCard, DevCardType,
    DivineIntervention, EdgeID, EndTurn, IntersectionID, Observation,
    PlayDevCard, ResourceType, TradeProposal,
)


class Agent(ABC):
    def __init__(self, player_id: int, seed: Optional[int] = None):
        self.player_id = player_id
        self.rng = random.Random(seed)

    @abstractmethod
    def select_action(self, obs: Observation, valid_actions: list[Action]) -> Action:
        ...

    @abstractmethod
    def select_draft_action(self, obs: Observation, valid_positions: list[IntersectionID]) -> IntersectionID:
        ...

    @abstractmethod
    def select_draft_road(self, obs: Observation, valid_edges: list[EdgeID]) -> EdgeID:
        ...

    def respond_to_trade(self, proposal: TradeProposal, obs: Observation) -> bool:
        return False

    def select_target(self, opponents: list[int]) -> int:
        return self.rng.choice(opponents)

    def select_resource_type(self) -> ResourceType:
        return self.rng.choice(list(ResourceType))

    def select_two_resources(self) -> tuple[ResourceType, ResourceType]:
        resources = list(ResourceType)
        return self.rng.choice(resources), self.rng.choice(resources)

    def select_road(self, valid_positions: list[EdgeID]) -> EdgeID:
        return self.rng.choice(valid_positions)


class RandomAgent(Agent):
    def select_action(self, obs: Observation, valid_actions: list[Action]) -> Action:
        return self.rng.choice(valid_actions)

    def select_draft_action(self, obs: Observation, valid_positions: list[IntersectionID]) -> IntersectionID:
        return self.rng.choice(valid_positions)

    def select_draft_road(self, obs: Observation, valid_edges: list[EdgeID]) -> EdgeID:
        return self.rng.choice(valid_edges)

    def respond_to_trade(self, proposal: TradeProposal, obs: Observation) -> bool:
        return self.rng.random() < 0.5


class GreedyAgent(Agent):
    """Heuristic-based agent that scores actions and picks the best."""

    def select_action(self, obs: Observation, valid_actions: list[Action]) -> Action:
        best_score = -float("inf")
        best_action = valid_actions[-1]  # EndTurn fallback

        for action in valid_actions:
            score = self._score_action(action, obs)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _score_action(self, action: Action, obs: Observation) -> float:
        if isinstance(action, Build):
            if action.build_type == BuildType.CITY:
                return 100.0
            elif action.build_type == BuildType.SETTLEMENT:
                return 80.0
            elif action.build_type == BuildType.ROAD:
                return 30.0

        elif isinstance(action, BuyDevCard):
            return 50.0

        elif isinstance(action, PlayDevCard):
            if action.card_type == DevCardType.PLUNDER:
                return 90.0
            elif action.card_type == DevCardType.ESPIONAGE:
                return 70.0
            elif action.card_type == DevCardType.INVENTION:
                return 60.0
            elif action.card_type == DevCardType.EXPANSIONIST:
                return 40.0
            elif action.card_type == DevCardType.MAINTENANCE:
                return 35.0

        elif isinstance(action, DivineIntervention):
            return 20.0

        elif isinstance(action, TradeProposal):
            return 25.0

        elif isinstance(action, EndTurn):
            return -1.0

        return 0.0

    def select_draft_action(self, obs: Observation, valid_positions: list[IntersectionID]) -> IntersectionID:
        # Pick intersection adjacent to tiles with best dice numbers (6, 8, 5, 9)
        best_score = -1.0
        best_pos = valid_positions[0]
        board = obs.board

        for iid in valid_positions:
            score = 0.0
            tiles = board.get_tiles_for_intersection(iid)
            for tile in tiles:
                if tile.dice_number is not None:
                    # Pips: probability weight
                    pips = 6 - abs(7 - tile.dice_number)
                    score += pips
            if score > best_score:
                best_score = score
                best_pos = iid

        return best_pos

    def select_draft_road(self, obs: Observation, valid_edges: list[EdgeID]) -> EdgeID:
        return self.rng.choice(valid_edges)

    def respond_to_trade(self, proposal: TradeProposal, obs: Observation) -> bool:
        # Accept if we're getting more value than giving
        give_total = sum(proposal.requesting.values())
        get_total = sum(proposal.offering.values())
        return get_total >= give_total

    def select_target(self, opponents: list[int]) -> int:
        return opponents[0]  # Target first opponent

    def select_resource_type(self) -> ResourceType:
        # Prefer wheat (useful for cities and dev cards)
        return ResourceType.WHEAT

    def select_two_resources(self) -> tuple[ResourceType, ResourceType]:
        return ResourceType.WHEAT, ResourceType.METAL


# ── MCTS Node ────────────────────────────────────────────────────────────

class MCTSNode:
    """A node in the MCTS search tree."""

    __slots__ = ("state", "player_id", "parent", "action", "children",
                 "untried_actions", "visits", "wins", "rng")

    def __init__(
        self,
        state,  # GameState
        player_id: int,
        parent: Optional[MCTSNode] = None,
        action: Optional[Action] = None,
        valid_actions: Optional[list[Action]] = None,
        rng: Optional[random.Random] = None,
    ):
        self.state = state
        self.player_id = player_id
        self.parent = parent
        self.action = action  # action that led to this node
        self.children: list[MCTSNode] = []
        self.untried_actions: list[Action] = list(valid_actions) if valid_actions else []
        self.visits: int = 0
        self.wins: float = 0.0
        self.rng = rng or random.Random()

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    @property
    def is_terminal(self) -> bool:
        return self.state.game_over

    def ucb1(self, c: float = 1.41) -> float:
        """Upper Confidence Bound for Trees."""
        if self.visits == 0:
            return float("inf")
        exploitation = self.wins / self.visits
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def best_child(self, c: float = 1.41) -> MCTSNode:
        """Select child with highest UCB1 score."""
        return max(self.children, key=lambda child: child.ucb1(c))

    def best_action_child(self) -> MCTSNode:
        """Select most-visited child (used for final move selection)."""
        return max(self.children, key=lambda child: child.visits)


# ── MCTS Agent ───────────────────────────────────────────────────────────

class MCTSAgent(Agent):
    """Monte Carlo Tree Search agent.

    Uses random rollouts to estimate action values. Each call to
    select_action runs `n_simulations` iterations of the MCTS loop:
        1. Selection   — walk down the tree via UCB1
        2. Expansion   — add one untried child
        3. Simulation  — random rollout to terminal state (or depth limit)
        4. Backprop    — propagate result back up the tree

    After all iterations, picks the root's most-visited child.
    """

    def __init__(
        self,
        player_id: int,
        seed: Optional[int] = None,
        n_simulations: int = 100,
        rollout_depth: int = 30,
        exploration_constant: float = 1.41,
    ):
        super().__init__(player_id, seed)
        self.n_simulations = n_simulations
        self.rollout_depth = rollout_depth
        self.exploration_constant = exploration_constant

    def select_action(self, obs: Observation, valid_actions: list[Action]) -> Action:
        # If only one option (just EndTurn), skip search
        if len(valid_actions) == 1:
            return valid_actions[0]

        # Reconstruct state from observation for cloning
        # The observation carries a board reference — we need the full GameState
        # We access it through the board (which is the live board object)
        from .actions import get_valid_actions
        from .action_executors import execute_action

        # Build root node from current state
        # We need to clone the state — obs.board is the live board
        state = self._reconstruct_state(obs)
        root = MCTSNode(
            state=state,
            player_id=self.player_id,
            valid_actions=valid_actions,
            rng=self.rng,
        )

        # Run MCTS iterations
        for _ in range(self.n_simulations):
            node = self._select(root)
            node = self._expand(node)
            result = self._simulate(node)
            self._backpropagate(node, result)

        # Pick the most-visited child
        if not root.children:
            return self.rng.choice(valid_actions)

        best = root.best_action_child()
        return best.action

    # ── MCTS Phases ──────────────────────────────────────────────────

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Phase 1: Walk down the tree using UCB1 until we find a node
        that isn't fully expanded or is terminal."""
        while not node.is_terminal and node.is_fully_expanded:
            if not node.children:
                return node
            node = node.best_child(self.exploration_constant)
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Phase 2: Add one untried child to the node."""
        if node.is_terminal or not node.untried_actions:
            return node

        from .actions import get_valid_actions
        from .action_executors import execute_action

        # Pick a random untried action
        action = node.untried_actions.pop(self.rng.randrange(len(node.untried_actions)))

        # Clone state and apply action
        new_state = node.state.clone()
        pid = new_state.current_player

        if isinstance(action, EndTurn):
            # Simulate end-of-turn: maintenance, advance, start next turn
            new_state.resolve_maintenance(pid)
            new_state.update_progress_points(pid)
            if not new_state.is_game_over():
                new_state.advance_player()
                new_state.roll_dice()
                new_state.produce_resources()
                new_p = new_state.players[new_state.current_player]
                new_p.dev_card_played_this_turn = False
                new_p.maintenance_paid_this_turn = False
        else:
            execute_action(new_state, pid, action)

        # Compute valid actions for the new state's current player
        if new_state.game_over:
            child_actions = []
        else:
            child_pid = new_state.current_player
            child_actions = get_valid_actions(new_state, child_pid)

        child = MCTSNode(
            state=new_state,
            player_id=new_state.current_player,
            parent=node,
            action=action,
            valid_actions=child_actions,
            rng=self.rng,
        )
        node.children.append(child)
        return child

    def _simulate(self, node: MCTSNode) -> float:
        """Phase 3: Random rollout from this node to estimate value.
        Returns 1.0 for win, 0.5 for draw/inconclusive, 0.0 for loss."""
        from .actions import get_valid_actions
        from .action_executors import execute_action

        state = node.state.clone()

        for _ in range(self.rollout_depth):
            if state.game_over:
                break

            pid = state.current_player
            valid = get_valid_actions(state, pid)

            # Random rollout policy (with bias toward EndTurn to keep rollouts short)
            action = self._rollout_policy(valid)

            if isinstance(action, EndTurn):
                state.resolve_maintenance(pid)
                state.update_progress_points(pid)
                if state.is_game_over():
                    break
                state.advance_player()
                state.roll_dice()
                state.produce_resources()
                p = state.players[state.current_player]
                p.dev_card_played_this_turn = False
                p.maintenance_paid_this_turn = False
            else:
                execute_action(state, pid, action)

        return self._evaluate_state(state)

    def _backpropagate(self, node: MCTSNode, result: float) -> None:
        """Phase 4: Walk back to root, updating visit counts and wins."""
        while node is not None:
            node.visits += 1
            # The result is from our agent's perspective.
            # If this node belongs to our player, add the result directly.
            # If it belongs to an opponent, add (1 - result).
            if node.player_id == self.player_id:
                node.wins += result
            else:
                node.wins += (1.0 - result)
            node = node.parent

    # ── Rollout Policy & Evaluation ──────────────────────────────────

    def _rollout_policy(self, valid_actions: list[Action]) -> Action:
        """Biased random policy: higher chance of ending turn to keep
        rollouts moving, but still explores other actions."""
        # 40% chance to pick EndTurn if available, otherwise uniform random
        end_turns = [a for a in valid_actions if isinstance(a, EndTurn)]
        if end_turns and self.rng.random() < 0.4:
            return end_turns[0]
        return self.rng.choice(valid_actions)

    def _evaluate_state(self, state) -> float:
        """Evaluate a terminal or depth-limited state from our perspective.
        Returns value in [0, 1]."""
        if state.game_over:
            if state.winner == self.player_id:
                return 1.0
            elif state.winner is None:
                return 0.5
            else:
                return 0.0

        # Non-terminal: heuristic based on relative progress points
        my_points = state.players[self.player_id].progress_points
        max_opponent_points = max(
            (p.progress_points for pid, p in state.players.items()
             if pid != self.player_id and p.is_active),
            default=0,
        )
        # Normalize to [0, 1] — positive if we're ahead
        from .data_types import WIN_THRESHOLD
        point_diff = my_points - max_opponent_points
        return 0.5 + (point_diff / (2 * WIN_THRESHOLD))

    # ── State Reconstruction ─────────────────────────────────────────

    def _reconstruct_state(self, obs: Observation):
        """Build a cloneable GameState from an observation.

        The observation's board field is actually the live Board object
        from the Environment's GameState. We use it to access the full
        state for cloning. This is a known coupling — in a real
        competition setting, the agent would only have the observation,
        but for self-play training this is acceptable.
        """
        # Walk up from board to find the GameState
        # Since obs.board is the live board, we need the environment to
        # pass us the state. For now, we reconstruct a minimal state.
        from .game_state import GameState
        from .data_types import ResourceType

        state = GameState.__new__(GameState)
        state.board = obs.board.clone()
        state.num_players = len(obs.opponents) + 1
        state.current_player = obs.player_id
        state.current_dice = obs.current_dice
        state.turn_number = obs.turn_number
        state.game_over = False
        state.winner = None
        state.rng = random.Random(self.rng.randint(0, 2**32))
        # Unknown deck contents — approximate with a shuffled full deck
        from .data_types import DEV_CARD_COUNTS
        deck: list = []
        for card_type, count in DEV_CARD_COUNTS.items():
            deck.extend([card_type] * count)
        state.rng.shuffle(deck)
        state.dev_card_deck = deck

        # Reconstruct players
        from .data_types import Player
        state.players = {}

        # Our player — full information
        me = Player(player_id=obs.player_id)
        me.resources = dict(obs.my_resources)
        me.dev_cards = list(obs.my_dev_cards)
        me.progress_points = sum(
            1 if i.building == BuildType.SETTLEMENT else 2
            for i in state.board.intersections.values()
            if i.owner == obs.player_id and i.building is not None
        )
        me.is_active = True
        state.players[obs.player_id] = me

        # Opponents — partial information
        for opp in obs.opponents:
            p = Player(player_id=opp.player_id)
            p.progress_points = opp.progress_points
            p.is_active = opp.is_active
            # Distribute unknown resources evenly
            per_resource = opp.total_resources // max(1, len(ResourceType))
            remainder = opp.total_resources % len(ResourceType)
            for i, r in enumerate(ResourceType):
                p.resources[r] = per_resource + (1 if i < remainder else 0)
            p.dev_cards = []
            state.players[opp.player_id] = p

        # Bank — use vague representation
        state.bank = dict(obs.bank_supply)

        return state

    # ── Draft & Utility Methods (reuse GreedyAgent logic) ────────────

    def select_draft_action(self, obs: Observation, valid_positions: list[IntersectionID]) -> IntersectionID:
        best_score = -1.0
        best_pos = valid_positions[0]
        board = obs.board

        for iid in valid_positions:
            score = 0.0
            tiles = board.get_tiles_for_intersection(iid)
            for tile in tiles:
                if tile.dice_number is not None:
                    pips = 6 - abs(7 - tile.dice_number)
                    score += pips
            if score > best_score:
                best_score = score
                best_pos = iid

        return best_pos

    def select_draft_road(self, obs: Observation, valid_edges: list[EdgeID]) -> EdgeID:
        return self.rng.choice(valid_edges)

    def respond_to_trade(self, proposal: TradeProposal, obs: Observation) -> bool:
        give_total = sum(proposal.requesting.values())
        get_total = sum(proposal.offering.values())
        return get_total >= give_total

    def select_target(self, opponents: list[int]) -> int:
        return opponents[0]

    def select_resource_type(self) -> ResourceType:
        return ResourceType.WHEAT

    def select_two_resources(self) -> tuple[ResourceType, ResourceType]:
        return ResourceType.WHEAT, ResourceType.METAL
