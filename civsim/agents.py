"""Agent interface and baseline implementations for CivSim."""
from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from typing import Optional

from .data_types import (
    Action, ActionType, Build, BuildType, BuyDevCard, DevCardType,
    DivineIntervention, EdgeID, EndTurn, IntersectionID, Observation,
    PlayDevCard, ReestablishBuilding, ResourceType, TradeProposal,
    WIN_THRESHOLD,
)
from .upkeep import (
    buildings_at_risk_from_obs, upkeep_gap_from_obs, upkeep_pressure_from_obs,
)


# Resource weights for trade evaluation. Higher = more valuable to retain.
# Water/Cow are critical for upkeep; Wheat/Metal feed cities and dev cards.
TRADE_VALUE = {
    ResourceType.WATER: 1.6,
    ResourceType.COW:   1.5,
    ResourceType.WHEAT: 1.3,
    ResourceType.METAL: 1.2,
    ResourceType.WOOD:  1.0,
    ResourceType.STONE: 1.0,
}


def _weighted(bundle: dict[ResourceType, int]) -> float:
    return sum(TRADE_VALUE[r] * amt for r, amt in bundle.items())


def _action_key(action: Action) -> tuple:
    """Hashable key uniquely identifying a per-turn action attempt."""
    if isinstance(action, TradeProposal):
        return (
            "trade",
            action.target_player,
            tuple(sorted((r.name, v) for r, v in action.offering.items())),
            tuple(sorted((r.name, v) for r, v in action.requesting.items())),
        )
    if isinstance(action, Build):
        return ("build", action.build_type.name, action.position)
    if isinstance(action, PlayDevCard):
        return ("play_dev", action.card_type.name)
    return (type(action).__name__,)


def _evaluate_trade_for_responder(proposal: TradeProposal, obs: Observation) -> bool:
    """Default heuristic for accepting a p2p trade as the responder.

    Two acceptance paths:
      1. **Net weighted gain** — incoming weighted value strictly exceeds
         outgoing. Always accept if upkeep reserves survive.
      2. **Surplus-for-deficit** — even at a slight weighted loss, accept
         when we hold ≥3 surplus of every requested resource AND the
         incoming offering fills a current gap (we hold ≤1 of it).

    Always reject if accepting would leave water or cow at zero (upkeep).
    """
    my_res = obs.my_resources
    # Affordability
    for r, amt in proposal.requesting.items():
        if my_res.get(r, 0) < amt:
            return False
    # Upkeep guard: don't drain water/cow below 1
    for r, amt in proposal.requesting.items():
        if r in (ResourceType.WATER, ResourceType.COW):
            if my_res.get(r, 0) - amt < 1:
                return False

    incoming = _weighted(proposal.offering)
    outgoing = _weighted(proposal.requesting)
    if incoming > outgoing:
        return True

    # Surplus-for-deficit: helpful even at weighted parity
    surplus = all(my_res.get(r, 0) >= amt + 3 for r, amt in proposal.requesting.items())
    fills_gap = any(my_res.get(r, 0) <= 1 for r in proposal.offering)
    return surplus and fills_gap


class Agent(ABC):
    def __init__(self, player_id: int, seed: Optional[int] = None):
        self.player_id = player_id
        self.rng = random.Random(seed)
        # Per-turn memory: actions the agent already tried this turn that
        # failed. Lets deterministic scorers avoid spinning on the same
        # rejected trade until the action loop's failure cap kicks in.
        self._failed_this_turn: set = set()

    @abstractmethod
    def select_action(self, obs: Observation, valid_actions: list[Action]) -> Action:
        ...

    def on_action_result(self, action: Action, success: bool) -> None:
        """Hook called by the environment after each action result.

        Default behavior: remember failed actions for this turn; clear on
        turn-end-style markers. Agents can override for richer learning.
        """
        if not success:
            self._failed_this_turn.add(_action_key(action))

    def on_turn_start(self) -> None:
        self._failed_this_turn.clear()

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
    """Heuristic-based agent that scores actions and picks the best.

    Two layers of awareness on top of base scoring:
      - **Upkeep-aware** — EndTurn is heavily penalized while buildings
        would fail maintenance; MAINTENANCE dev cards spike to top
        priority; trades for water/cow get a large bonus while short.
      - **Target-aware** — actions are scaled by an *urgency* factor
        derived from progress points vs. WIN_THRESHOLD. As the player
        nears the goal, cities (cheapest +1 PP from an existing
        settlement) dominate; roads/dev cards get de-prioritized;
        trades that fill city ingredients (metal, wheat, cow) get a
        late-game bonus.
    """

    def select_action(self, obs: Observation, valid_actions: list[Action]) -> Action:
        # Cache decision context once per call instead of recomputing per action.
        gap = upkeep_gap_from_obs(obs)
        pressure = upkeep_pressure_from_obs(obs, buffer_turns=1)
        at_risk = buildings_at_risk_from_obs(obs)
        urgency = self._urgency(obs)

        best_score = -float("inf")
        best_action = valid_actions[-1]  # EndTurn fallback

        for action in valid_actions:
            score = self._score_action(action, obs, gap, pressure, at_risk, urgency)
            # Heavy penalty if this exact action already failed this turn —
            # avoids the infinite-retry trap on rejected trades.
            if _action_key(action) in self._failed_this_turn:
                score -= 10_000.0
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _my_pp(self, obs: Observation) -> int:
        """Compute current progress points from the board.

        We read from the board rather than trusting any cached field so
        the value is correct mid-turn even after the agent has built
        something this very action loop.
        """
        from .data_types import CITY_PP, SETTLEMENT_PP
        pp = 0
        for inter in obs.board.intersections.values():
            if inter.owner != obs.player_id or inter.building is None:
                continue
            if inter.building == BuildType.SETTLEMENT:
                pp += SETTLEMENT_PP
            elif inter.building == BuildType.CITY:
                pp += CITY_PP
        return pp

    def _urgency(self, obs: Observation) -> float:
        """Returns a value in [0, 1] reflecting how close we are to winning.

        0.0 = no PP yet, optimize foundation (roads, settlements).
        1.0 = at or above WIN_THRESHOLD, every PP-generating action is
              everything.
        """
        return max(0.0, min(1.0, self._my_pp(obs) / WIN_THRESHOLD))

    def _score_action(
        self,
        action: Action,
        obs: Observation,
        upkeep_gap: dict[ResourceType, int],
        upkeep_pressure: dict[ResourceType, int],
        at_risk: int,
        urgency: float = 0.0,
    ) -> float:
        if isinstance(action, Build):
            if action.build_type == BuildType.CITY:
                # Cheapest +1 PP path when we already own the settlement,
                # but only if we can SUSTAIN it. Without cow buffer the
                # city gets reaped on the next maintenance and the PP
                # gain reverses. Require at least 2 cow on hand (one for
                # the immediate next maintenance, one for the one after).
                my_cow = obs.my_resources.get(ResourceType.COW, 0)
                # After building, cow goes down by 0 (city cost is metal+wheat).
                # But upkeep starts demanding 1 cow per maintenance.
                if my_cow < 2:
                    return 25.0 + urgency * 30.0  # tempting but risky
                return 100.0 + urgency * 150.0
            elif action.build_type == BuildType.SETTLEMENT:
                # +1 PP and opens a future city upgrade. Slight urgency boost.
                # Same sustainability check on water.
                my_water = obs.my_resources.get(ResourceType.WATER, 0)
                if my_water < 2:
                    return 30.0 + urgency * 20.0
                return 80.0 + urgency * 40.0
            elif action.build_type == BuildType.ROAD:
                # 0 PP. As urgency rises, roads become wasted actions.
                return 30.0 * (1.0 - 0.7 * urgency)

        elif isinstance(action, BuyDevCard):
            # Dev card costs 1 metal + 1 wheat + 1 water. Refuse if the
            # water payment would leave us unable to cover upkeep — that
            # guarantees a maintenance failure for a long-shot card draw.
            my_water = obs.my_resources.get(ResourceType.WATER, 0)
            from .upkeep import total_upkeep_cost
            water_needed = total_upkeep_cost(obs.board, obs.player_id).get(
                ResourceType.WATER, 0
            )
            if my_water - 1 < water_needed:
                return -500.0
            # Dev cards lose value as urgency rises — random card draws
            # don't reliably close the gap when the game's nearly won.
            return 50.0 * (1.0 - 0.5 * urgency)

        elif isinstance(action, PlayDevCard):
            if action.card_type == DevCardType.MAINTENANCE and at_risk > 0:
                # Defending against upkeep failure trumps everything else
                return 1000.0 + at_risk * 50.0
            if action.card_type == DevCardType.PLUNDER:
                # Plunder steals all of one resource type — useful for
                # securing city ingredients (metal, wheat, cow) late game.
                return 90.0 + urgency * 30.0
            elif action.card_type == DevCardType.ESPIONAGE:
                return 70.0
            elif action.card_type == DevCardType.INVENTION:
                # Invention pulls 2 from bank — extra valuable with upkeep
                # pressure OR when we need a couple of city ingredients.
                return 60.0 + (40.0 if at_risk > 0 else 0.0) + urgency * 20.0
            elif action.card_type == DevCardType.EXPANSIONIST:
                # Free roads — increasingly useless as urgency rises.
                return 40.0 * (1.0 - 0.6 * urgency)
            elif action.card_type == DevCardType.MAINTENANCE:
                return 35.0

        elif isinstance(action, DivineIntervention):
            # Miracle outcome (5%) grants +2 PP, valuable when very close.
            return 20.0 + urgency * 30.0

        elif isinstance(action, ReestablishBuilding):
            # Re-establishing an abandoned city is a +4 PP instant payoff;
            # a settlement is +2 PP. The premium cost (vs original build) is
            # worth it when close to threshold or when the position is good.
            if action.build_type == BuildType.CITY:
                return 110.0 + urgency * 160.0
            return 85.0 + urgency * 50.0

        elif isinstance(action, TradeProposal):
            return self._score_trade(
                action, obs, upkeep_gap, upkeep_pressure, at_risk, urgency
            )

        elif isinstance(action, EndTurn):
            # Refuse to end the turn if buildings would be lost; the
            # multiplier is large enough to beat every constructive action.
            return -1.0 - 500.0 * at_risk

        return 0.0

    def _score_trade(
        self,
        action: TradeProposal,
        obs: Observation,
        upkeep_gap: dict[ResourceType, int],
        upkeep_pressure: dict[ResourceType, int],
        at_risk: int,
        urgency: float = 0.0,
    ) -> float:
        """Need-based trade scoring with upkeep priority.

        Two levels of upkeep urgency:
          - **gap**: we owe upkeep this turn. Emergency — score 200/unit.
          - **pressure**: we'll owe upkeep next turn given no production.
            Stockpile — score 35/unit. This is the key signal because the
            bank's 4:1 ratio is too slow to react to a same-turn crisis.
        """
        my = obs.my_resources

        # Refuse to surrender resources we already owe for upkeep this turn.
        for r, amt in action.offering.items():
            post = my.get(r, 0) - amt
            if post < 0:
                return -10.0
            still_owed = upkeep_gap.get(r, 0)
            if still_owed > 0 and r in (ResourceType.WATER, ResourceType.COW):
                return -1000.0
            # Also refuse to give away upkeep resources when they're below
            # the stockpile target — would just create a future crisis.
            pressure_for_r = upkeep_pressure.get(r, 0)
            if pressure_for_r > 0 and r in (ResourceType.WATER, ResourceType.COW):
                return -50.0
            if r in (ResourceType.WATER, ResourceType.COW) and post < 1 and at_risk == 0:
                return -10.0

        upkeep_relief = 0.0
        stockpile_bonus = 0.0
        for r, amt in action.requesting.items():
            need = upkeep_gap.get(r, 0)
            if need > 0:
                upkeep_relief += min(need, amt) * 200.0
            # Preemptive stockpile bonus is intentionally small — it should
            # nudge Greedy toward a free water-acquisition trade when no
            # better action exists, not crowd out builds/dev-card buys.
            stock = upkeep_pressure.get(r, 0) - need
            if stock > 0:
                stockpile_bonus += min(stock, amt) * 4.0

        # Target-aware bonus: trades that pull in city-construction
        # ingredients (3 metal, 2 wheat) get more attractive as we close on
        # the win threshold. Settlements also need wood/stone/wheat/cow.
        CITY_INGREDIENTS = (ResourceType.METAL, ResourceType.WHEAT)
        city_focus = 0.0
        if urgency > 0:
            my = obs.my_resources
            for r, amt in action.requesting.items():
                if r in CITY_INGREDIENTS:
                    # How much we lack vs city cost (3 metal, 2 wheat)
                    target = 3 if r == ResourceType.METAL else 2
                    deficit = max(0, target - my.get(r, 0))
                    city_focus += min(deficit, amt) * urgency * 25.0

        if action.target_player is None:
            base = 25.0
        else:
            DESIRED = 2
            gain = 0.0
            for r, amt in action.requesting.items():
                deficit = max(0, DESIRED - my.get(r, 0))
                gain += deficit * TRADE_VALUE[r] * amt
            cost = sum(TRADE_VALUE[r] * amt for r, amt in action.offering.items())
            base = gain * 6.0 - cost

        return base + upkeep_relief + stockpile_bonus + city_focus

    def select_draft_action(self, obs: Observation, valid_positions: list[IntersectionID]) -> IntersectionID:
        return _score_draft_position(obs, valid_positions)

    def select_draft_road(self, obs: Observation, valid_edges: list[EdgeID]) -> EdgeID:
        return self.rng.choice(valid_edges)

    def respond_to_trade(self, proposal: TradeProposal, obs: Observation) -> bool:
        return _evaluate_trade_for_responder(proposal, obs)

    def select_target(self, opponents: list[int]) -> int:
        return opponents[0]  # Target first opponent

    def select_resource_type(self) -> ResourceType:
        # Prefer wheat (useful for cities and dev cards)
        return ResourceType.WHEAT

    def select_two_resources(self) -> tuple[ResourceType, ResourceType]:
        return ResourceType.WHEAT, ResourceType.METAL


def _score_draft_position(
    obs: Observation, valid_positions: list[IntersectionID]
) -> IntersectionID:
    """Shared draft heuristic: combine pip-count with bonuses for upkeep
    resources (water/cow adjacent) and port access.

    Why upkeep matters: settlements cost 1 water/turn to maintain.
    Why ports matter: a 2:1 specific port halves the cost of converting
    surplus into a needed resource — a meaningful lever throughout the
    game. Generic 3:1 ports are weaker but still beat the 4:1 bank.
    """
    best_score = -1.0
    best_pos = valid_positions[0]
    board = obs.board

    for iid in valid_positions:
        score = 0.0
        upkeep_access = 0.0
        for tile in board.get_tiles_for_intersection(iid):
            if tile.dice_number is None:
                continue
            pips = 6 - abs(7 - tile.dice_number)
            score += pips
            if tile.resource == ResourceType.WATER:
                upkeep_access += pips * 1.5
            elif tile.resource == ResourceType.COW:
                upkeep_access += pips * 1.2
        score += upkeep_access
        # Port bonus — sized to make a 2-tile border intersection with a
        # specific port roughly competitive with a 3-tile interior one,
        # without overriding the strongest interior positions.
        from .data_types import PortType
        inter = board.intersections[iid]
        if inter.port is not None:
            score += 7.0 if inter.port != PortType.GENERIC else 3.5
        if score > best_score:
            best_score = score
            best_pos = iid

    return best_pos


# ── MCTS Node ────────────────────────────────────────────────────────────

class MCTSNode:
    """A node in the MCTS search tree.

    Supports both UCB1 (vanilla MCTS) and PUCT (AlphaZero-style) selection.
    PUCT uses a `prior` per child — typically a softmax over heuristic
    action scores — to direct simulation budget toward promising moves.
    Without a prior, PUCT degenerates to UCB1-like behavior.
    """

    __slots__ = ("state", "player_id", "parent", "action", "children",
                 "untried_actions", "visits", "wins", "rng",
                 "priors", "prior")

    def __init__(
        self,
        state,  # GameState
        player_id: int,
        parent: Optional[MCTSNode] = None,
        action: Optional[Action] = None,
        valid_actions: Optional[list[Action]] = None,
        rng: Optional[random.Random] = None,
        priors: Optional[dict] = None,
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
        # `priors` maps action_key -> prior probability for each child this
        # node will eventually expand. The child node receives its own
        # scalar `prior` when created. Both are used for PUCT selection.
        self.priors = priors or {}
        self.prior: float = 0.0

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    @property
    def is_terminal(self) -> bool:
        return self.state.game_over

    def ucb1(self, c: float = 1.41) -> float:
        """Upper Confidence Bound for Trees (vanilla MCTS)."""
        if self.visits == 0:
            return float("inf")
        exploitation = self.wins / self.visits
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def puct(self, c_puct: float = 1.5) -> float:
        """PUCT score (AlphaZero-style).

        Score = Q + c_puct * P * sqrt(parent_visits) / (1 + visits)

        Virtual Q-init: when a child has 0 real visits we use its prior as
        the initial Q estimate instead of 0. This gives MCTS Greedy's
        knowledge from move 1 — important at low sim counts where the
        child may receive only 1-2 visits before MCTS commits.
        """
        if self.visits > 0:
            Q = self.wins / self.visits
        else:
            Q = self.prior  # virtual init: start at heuristic's estimate
        parent_visits = self.parent.visits if self.parent else 1
        exploration = c_puct * self.prior * math.sqrt(max(1, parent_visits)) / (1 + self.visits)
        return Q + exploration

    def best_child(self, c: float = 1.41, use_puct: bool = True) -> MCTSNode:
        """Select child with the highest score (PUCT by default, UCB1 fallback)."""
        if use_puct:
            return max(self.children, key=lambda child: child.puct(c))
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
        rollout_depth: int = 60,
        exploration_constant: float = 1.41,
        use_greedy_rollouts: bool = True,
        action_pruning_k: int = 5,
        use_puct: bool = True,
        puct_temperature: float = 50.0,
        decisive_margin: float = 100.0,
        enable_all_in: bool = True,
        value_network=None,
        policy_network=None,
        policy_prior_temperature: float = 1.0,
    ):
        super().__init__(player_id, seed)
        self.n_simulations = n_simulations
        self.rollout_depth = rollout_depth
        self.exploration_constant = exploration_constant
        # Use a GreedyAgent's heuristic as the rollout policy. Pure random
        # rollouts are useless when the action space is ~60+ per turn —
        # each rollout is too noisy to inform UCB1 selection.
        self.use_greedy_rollouts = use_greedy_rollouts
        self._rollout_agent = GreedyAgent(player_id=player_id, seed=seed) if use_greedy_rollouts else None
        # Action pruning: at each expansion node, keep only the top-K actions
        # by Greedy's heuristic score. Critical for tractability — the raw
        # action space is dominated by p2p trade variants that mostly fail,
        # so MCTS spends all its sim budget on uninformative branches.
        # K=5 keeps ~10 sims per candidate at n_simulations=50 — enough
        # for PUCT to differentiate them reliably.
        self.action_pruning_k = action_pruning_k
        self._prune_agent = GreedyAgent(player_id=player_id, seed=seed)
        # PUCT vs UCB1. With PUCT, action priors (Greedy's softmax) bias
        # the search; sims concentrate on promising branches without
        # needing to visit every child first as plain UCB1 does.
        self.use_puct = use_puct
        # Softmax temperature for converting Greedy scores into priors.
        # Greedy scores span hundreds (city ~250 at high urgency vs
        # EndTurn at -1). Temp ~50 turns that into a non-degenerate
        # distribution — top action gets ~70-90% of probability mass,
        # the rest fan out by score.
        self.puct_temperature = puct_temperature
        # Decisive-move shortcut: when Greedy's top action scores more than
        # `decisive_margin` above #2, skip MCTS and take it. Avoids burning
        # the sim budget on obvious choices (e.g. MAINTENANCE card at risk,
        # affordable city upgrade when ahead). 0 disables.
        self.decisive_margin = decisive_margin
        # All-in detection: if a single Build would push our PP to the
        # win threshold, take it without running MCTS. MCTS at low sim
        # counts can occasionally pick a less-direct move and miss the win.
        self.enable_all_in = enable_all_in
        # Optional trained value network. When set, MCTS uses its value
        # head (sigmoid-squashed) as the depth-limit evaluator instead of
        # the hand-coded "progress points lead" heuristic. The trained
        # head encodes a much richer notion of "is this position winning"
        # than the PP lead alone — it has seen 800+ training episodes of
        # this exact game and weighs resources/buildings/upkeep together.
        self.value_network = value_network
        # Optional trained policy network. When set, PUCT priors come
        # from the policy head's action probabilities instead of from
        # softmaxed Greedy scores. This is the other half of AlphaZero:
        # the policy network has *learned* which actions are typically
        # promising, much richer than any hand-coded heuristic prior.
        # Can be the SAME network as value_network (same PolicyValueNetwork).
        self.policy_network = policy_network
        # Temperature applied to the policy head's log-probs before they
        # become PUCT priors. <1.0 sharpens (trust the top action more);
        # >1.0 flattens (consider more alternatives).
        self.policy_prior_temperature = policy_prior_temperature

    def _find_all_in_action(
        self, obs: Observation, valid_actions: list[Action]
    ) -> Optional[Action]:
        """Return any single action that would push us to WIN_THRESHOLD.

        Considers Build and ReestablishBuilding (which grants the full
        building's PP since we don't already own an unupgraded version).
        Skips actions already known to have failed this turn.
        """
        from .data_types import CITY_PP, SETTLEMENT_PP
        my_pp = 0
        for inter in obs.board.intersections.values():
            if inter.owner != obs.player_id or inter.building is None:
                continue
            if inter.building == BuildType.SETTLEMENT:
                my_pp += SETTLEMENT_PP
            elif inter.building == BuildType.CITY:
                my_pp += CITY_PP
        for a in valid_actions:
            if _action_key(a) in self._failed_this_turn:
                continue
            if isinstance(a, Build):
                if a.build_type == BuildType.SETTLEMENT:
                    gain = SETTLEMENT_PP
                elif a.build_type == BuildType.CITY:
                    gain = CITY_PP - SETTLEMENT_PP
                else:
                    continue
            elif isinstance(a, ReestablishBuilding):
                # Re-establish grants the building's full PP (no prior owner)
                gain = SETTLEMENT_PP if a.build_type == BuildType.SETTLEMENT else CITY_PP
            else:
                continue
            if my_pp + gain >= WIN_THRESHOLD:
                return a
        return None

    def _find_decisive_action(
        self, obs: Observation, valid_actions: list[Action]
    ) -> Optional[Action]:
        """Return Greedy's top action if it scores far above #2, else None.

        Failed actions are filtered out before ranking — the same action that
        was just rejected isn't a "decisive" choice on the next call.
        """
        candidates = [
            a for a in valid_actions
            if _action_key(a) not in self._failed_this_turn
        ]
        if len(candidates) < 2:
            return None
        self._prune_agent.player_id = obs.player_id
        from .upkeep import (
            buildings_at_risk_from_obs,
            upkeep_gap_from_obs, upkeep_pressure_from_obs,
        )
        gap = upkeep_gap_from_obs(obs)
        pressure = upkeep_pressure_from_obs(obs, buffer_turns=1)
        at_risk = buildings_at_risk_from_obs(obs)
        urgency = self._prune_agent._urgency(obs)
        scored: list[tuple[float, Action]] = []
        for a in candidates:
            s = self._prune_agent._score_action(a, obs, gap, pressure, at_risk, urgency)
            scored.append((s, a))
        scored.sort(key=lambda t: t[0], reverse=True)
        if scored[0][0] - scored[1][0] >= self.decisive_margin:
            return scored[0][1]
        return None

    def _compute_priors(
        self, actions: list[Action], obs: Observation
    ) -> dict:
        """Return action_key → prior probability for PUCT.

        If a trained policy_network is configured, use its action
        probabilities directly. Otherwise fall back to a softmax over
        GreedyAgent's per-action heuristic scores.
        """
        if not actions:
            return {}
        if self.policy_network is not None:
            return self._compute_priors_from_policy(actions, obs)

        self._prune_agent.player_id = obs.player_id
        from .upkeep import (
            buildings_at_risk_from_obs,
            upkeep_gap_from_obs, upkeep_pressure_from_obs,
        )
        gap = upkeep_gap_from_obs(obs)
        pressure = upkeep_pressure_from_obs(obs, buffer_turns=1)
        at_risk = buildings_at_risk_from_obs(obs)
        urgency = self._prune_agent._urgency(obs)
        scores = [
            self._prune_agent._score_action(a, obs, gap, pressure, at_risk, urgency)
            for a in actions
        ]
        # Softmax with temperature; subtract max for numerical stability.
        max_s = max(scores)
        exps = [math.exp((s - max_s) / max(1e-6, self.puct_temperature)) for s in scores]
        total = sum(exps) or 1.0
        return {_action_key(a): e / total for a, e in zip(actions, exps)}

    def _compute_priors_from_policy(
        self, actions: list[Action], obs: Observation
    ) -> dict:
        """Query the trained policy head for action probabilities.

        Applies `policy_prior_temperature`: <1.0 sharpens the distribution
        (top action gets more mass — MCTS trusts the policy more),
        >1.0 flattens it (MCTS explores more around the top pick).
        """
        import torch
        from .rl.features import encode_observation, encode_action
        state_vec = torch.tensor(encode_observation(obs), dtype=torch.float32)
        action_vecs = torch.tensor(
            [encode_action(a) for a in actions], dtype=torch.float32
        )
        mask = torch.ones(len(actions), dtype=torch.float32)
        with torch.no_grad():
            log_probs, _ = self.policy_network(state_vec, action_vecs, mask)
            T = max(1e-6, self.policy_prior_temperature)
            if T != 1.0:
                scaled = log_probs / T
                # Re-normalize via log-softmax for numerical stability
                scaled = scaled - torch.logsumexp(scaled, dim=-1, keepdim=True)
                probs = torch.exp(scaled).tolist()
            else:
                probs = torch.exp(log_probs).tolist()
        return {_action_key(a): float(p) for a, p in zip(actions, probs)}

    def _prune_actions(
        self, valid_actions: list[Action], obs: Observation
    ) -> list[Action]:
        """Return the top-K actions by Greedy's heuristic score.

        EndTurn is always kept so MCTS can choose to terminate the turn.
        Pruning shrinks ~60-action turns down to a handful of *plausibly
        good* candidates, letting MCTS's simulation budget go further.
        """
        if not self.action_pruning_k or len(valid_actions) <= self.action_pruning_k:
            return list(valid_actions)

        self._prune_agent.player_id = obs.player_id
        from .upkeep import (
            buildings_at_risk_from_obs,
            upkeep_gap_from_obs, upkeep_pressure_from_obs,
        )
        gap = upkeep_gap_from_obs(obs)
        pressure = upkeep_pressure_from_obs(obs, buffer_turns=1)
        at_risk = buildings_at_risk_from_obs(obs)
        urgency = self._prune_agent._urgency(obs)

        scored: list[tuple[float, Action]] = []
        for a in valid_actions:
            s = self._prune_agent._score_action(a, obs, gap, pressure, at_risk, urgency)
            scored.append((s, a))
        scored.sort(key=lambda t: t[0], reverse=True)
        top = [a for _, a in scored[: self.action_pruning_k]]
        # Always include EndTurn so MCTS can stop a turn even if Greedy
        # scores it dead last (which it usually does).
        for a in valid_actions:
            if isinstance(a, EndTurn) and a not in top:
                top.append(a)
                break
        return top

    def select_action(self, obs: Observation, valid_actions: list[Action]) -> Action:
        # If only one option (just EndTurn), skip search
        if len(valid_actions) == 1:
            return valid_actions[0]

        # Reconstruct state from observation for cloning
        # The observation carries a board reference — we need the full GameState
        # We access it through the board (which is the live board object)
        from .actions import get_valid_actions
        from .action_executors import execute_action

        # ── Cheap shortcut paths (cost-neutral; avoid burning sims) ──

        # 1) All-in detection: take any Build that wins immediately.
        if self.enable_all_in:
            winning = self._find_all_in_action(obs, valid_actions)
            if winning is not None:
                return winning

        # 2) Decisive-move shortcut: if Greedy strongly prefers one move,
        #    trust it instead of running MCTS.
        if self.decisive_margin > 0:
            decisive = self._find_decisive_action(obs, valid_actions)
            if decisive is not None:
                return decisive

        # Prune action space to top-K candidates before tree search
        pruned_actions = self._prune_actions(valid_actions, obs)
        priors = self._compute_priors(pruned_actions, obs) if self.use_puct else {}

        # Build root node from current state
        state = self._reconstruct_state(obs)
        root = MCTSNode(
            state=state,
            player_id=self.player_id,
            valid_actions=pruned_actions,
            rng=self.rng,
            priors=priors,
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
        """Phase 1: Walk down the tree using UCB1 / PUCT until we find a
        node that isn't fully expanded or is terminal."""
        while not node.is_terminal and node.is_fully_expanded:
            if not node.children:
                return node
            node = node.best_child(self.exploration_constant, use_puct=self.use_puct)
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
            # Use the shared turn-transition helper so MCTS rollouts apply
            # the same maintenance-interval, rain, and barn-day rules as
            # the live environment. Previously these diverged and made
            # MCTS dramatically pessimistic.
            new_state.end_player_turn(pid)
            if not new_state.is_game_over():
                new_state.advance_player()
                new_state.start_player_turn()
        else:
            execute_action(new_state, pid, action)

        # Compute valid actions for the new state's current player, prune
        # to top-K, and compute priors for PUCT selection at the child.
        child_priors: dict = {}
        if new_state.game_over:
            child_actions: list[Action] = []
        else:
            child_pid = new_state.current_player
            child_actions = get_valid_actions(new_state, child_pid)
            child_obs = new_state.get_observation(child_pid)
            if self.action_pruning_k and len(child_actions) > self.action_pruning_k:
                child_actions = self._prune_actions(child_actions, child_obs)
            if self.use_puct:
                child_priors = self._compute_priors(child_actions, child_obs)

        child = MCTSNode(
            state=new_state,
            player_id=new_state.current_player,
            parent=node,
            action=action,
            valid_actions=child_actions,
            rng=self.rng,
            priors=child_priors,
        )
        # Set the new child's prior (from PARENT's priors over this action)
        child.prior = node.priors.get(_action_key(action), 1.0 / max(1, len(node.priors) or 1))
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
            action = self._rollout_policy(valid, state)

            if isinstance(action, EndTurn):
                state.end_player_turn(pid)
                if state.is_game_over():
                    break
                state.advance_player()
                state.start_player_turn()
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

    def _rollout_policy(self, valid_actions: list[Action], state=None) -> Action:
        """Rollout policy. With use_greedy_rollouts=True (default) the rollouts
        use a fresh GreedyAgent's scoring — far more informative than uniform
        random when the action space is in the dozens. Falls back to biased
        random when state is None or greedy rollouts are disabled.
        """
        if self._rollout_agent is not None and state is not None:
            # Build an obs for the current rollout player. The rollout agent
            # uses scoring (not learning), so any player_id alias works.
            pid = state.current_player
            obs = state.get_observation(pid)
            # Temporarily point the rollout agent at this player so its
            # internal player_id-based logic (upkeep, target awareness) works.
            self._rollout_agent.player_id = pid
            return self._rollout_agent.select_action(obs, valid_actions)

        non_end = [a for a in valid_actions if not isinstance(a, EndTurn)]
        end_turns = [a for a in valid_actions if isinstance(a, EndTurn)]
        if state is not None and non_end and end_turns:
            from .upkeep import buildings_at_risk
            player = state.players[state.current_player]
            if buildings_at_risk(player, state.board) > 0:
                return self.rng.choice(non_end)
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

        # Prefer the trained RL value head when provided — it encodes a
        # richer notion of "winning" than raw PP lead.
        if self.value_network is not None:
            return self._value_from_network(state)

        # Fallback: hand-coded heuristic based on relative progress points
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

    def _value_from_network(self, state) -> float:
        """Query the trained value head for a state-value estimate.

        The RL network's value head was trained to predict discounted
        returns. Raw values can be roughly [-5, 10] (the win/loss
        bonuses). Sigmoid squashes to (0, 1) for compatibility with
        MCTS's win-rate-scale Q values.
        """
        import torch
        from .rl.features import encode_observation
        obs = state.get_observation(self.player_id)
        state_vec = torch.tensor(encode_observation(obs), dtype=torch.float32)
        with torch.no_grad():
            trunk = self.value_network.state_trunk(state_vec.unsqueeze(0))
            value = self.value_network.value_head(trunk).squeeze()
            return torch.sigmoid(value).item()

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
        return _score_draft_position(obs, valid_positions)

    def select_draft_road(self, obs: Observation, valid_edges: list[EdgeID]) -> EdgeID:
        return self.rng.choice(valid_edges)

    def respond_to_trade(self, proposal: TradeProposal, obs: Observation) -> bool:
        return _evaluate_trade_for_responder(proposal, obs)

    def select_target(self, opponents: list[int]) -> int:
        return opponents[0]

    def select_resource_type(self) -> ResourceType:
        return ResourceType.WHEAT

    def select_two_resources(self) -> tuple[ResourceType, ResourceType]:
        return ResourceType.WHEAT, ResourceType.METAL
