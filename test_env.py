"""
Test script for CivSim environment.

Run this to verify the environment and agents work correctly.
"""

import sys
sys.path.insert(0, '/home/claude/civsim')

from environment import CivSimEnv, make_env
from agents import RandomAgent, HeuristicAgent, AgentPersonality


def test_environment_creation():
    """Test that we can create an environment."""
    print("Testing environment creation...")
    env = make_env(num_civilizations=4, map_width=15, map_height=15, seed=42)
    obs = env.reset()
    
    assert env.state is not None, "Game state should be initialized"
    assert len(obs) == 4, "Should have observations for 4 civs"
    assert env.state.current_turn == 0, "Should start at turn 0"
    
    print(f"  ✓ Created environment with {len(env.state.civilizations)} civilizations")
    print(f"  ✓ Map size: {env.map_width}x{env.map_height}")
    print(f"  ✓ Initial observations generated for all civs")
    
    return env


def test_valid_actions():
    """Test that valid actions are generated correctly."""
    print("\nTesting valid action generation...")
    env = make_env(num_civilizations=4, seed=42)
    env.reset()
    
    for civ_id in range(4):
        actions = env.get_valid_actions(civ_id)
        print(f"  Civ {civ_id}: {len(actions)} valid actions")
        
        # Should have at least end_turn action
        assert len(actions) > 0, f"Civ {civ_id} should have at least one valid action"

        # Check action types
        action_types = {type(a).__name__ for a in actions}
        print(f"    Action types: {action_types}")
    
    print("  ✓ All civs have valid actions")


def test_random_agent():
    """Test that random agents can play a game."""
    print("\nTesting random agents...")
    env = make_env(num_civilizations=4, map_width=15, map_height=15, max_turns=50, seed=42)
    obs = env.reset()
    
    agents = {i: RandomAgent(i, seed=i) for i in range(4)}
    
    turns_played = 0
    while not env.is_done() and turns_played < 100:
        for civ_id in env.get_active_civs():
            valid_actions = env.get_valid_actions(civ_id)
            action = agents[civ_id].select_action(obs[civ_id], valid_actions)
            result = env.step(civ_id, action)
            obs = result.observations
            
            if result.terminated:
                break
        
        turns_played += 1
    
    print(f"  ✓ Game ran for {env.state.current_turn} turns")
    print(f"  ✓ Game over: {env.is_done()}")
    if env.get_winner() is not None:
        print(f"  ✓ Winner: Civ {env.get_winner()}")
    
    scores = env.get_scores()
    for civ_id, score in scores.items():
        print(f"    Civ {civ_id}: score = {score}")


def test_heuristic_agents():
    """Test that heuristic agents can play against each other."""
    print("\nTesting heuristic agents...")
    env = make_env(num_civilizations=4, map_width=15, map_height=15, max_turns=50, seed=42)
    obs = env.reset()

    # Create agents with different personalities
    agents = {
        0: HeuristicAgent(0, AgentPersonality.EXPANSIONIST, seed=0),
        1: HeuristicAgent(1, AgentPersonality.ECONOMIST, seed=1),
        2: HeuristicAgent(2, AgentPersonality.DIPLOMAT, seed=2),
        3: HeuristicAgent(3, AgentPersonality.BALANCED, seed=3),
    }
    
    print(f"  Agents: {[a.name for a in agents.values()]}")
    
    turns_played = 0
    while not env.is_done() and turns_played < 200:
        for civ_id in env.get_active_civs():
            valid_actions = env.get_valid_actions(civ_id)
            action = agents[civ_id].select_action(obs[civ_id], valid_actions)
            result = env.step(civ_id, action)
            obs = result.observations
            
            if result.terminated:
                break
        
        turns_played += 1
    
    print(f"  ✓ Game ran for {env.state.current_turn} turns")
    
    scores = env.get_scores()
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print("  Final standings:")
    for civ_id, score in sorted_scores:
        civ_name = env.state.civilizations[civ_id].name
        agent_type = agents[civ_id].personality.name
        status = "ALIVE" if env.state.civilizations[civ_id].is_alive else "DEAD"
        print(f"    {civ_name} ({agent_type}): {score} - {status}")


def test_ascii_render():
    """Test ASCII rendering."""
    print("\nTesting ASCII render...")
    env = make_env(num_civilizations=4, map_width=15, map_height=15, seed=42)
    env.reset()
    
    ascii_map = env.render_ascii()
    print(ascii_map)
    print("  ✓ ASCII render working")


def test_observation_structure():
    """Test the structure of observations."""
    print("\nTesting observation structure...")
    env = make_env(num_civilizations=4, seed=42)
    obs = env.reset()
    
    sample_obs = obs[0]
    expected_keys = [
        'turn', 'civ_id', 'resources', 'technologies',
        'visible_tiles', 'visible_units', 'visible_cities',
        'own_units', 'own_cities', 'relations', 'trust_scores',
        'incoming_offers', 'outgoing_offers'
    ]
    
    for key in expected_keys:
        assert key in sample_obs, f"Observation missing key: {key}"
        print(f"  ✓ {key}: {type(sample_obs[key]).__name__}")
    
    print(f"  Resources: {sample_obs['resources']}")
    print(f"  Own units: {len(sample_obs['own_units'])}")
    print(f"  Own cities: {len(sample_obs['own_cities'])}")
    print(f"  Visible tiles: {len(sample_obs['visible_tiles'])}")


def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("CivSim Environment Tests")
    print("=" * 50)
    
    test_environment_creation()
    test_valid_actions()
    test_observation_structure()
    test_random_agent()
    test_heuristic_agents()
    test_ascii_render()
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)


if __name__ == "__main__":
    run_all_tests()
