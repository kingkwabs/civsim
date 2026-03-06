"""
Example: Play a peaceful CivSim game

This demonstrates the game with no war/combat mechanics.
Civilizations compete through expansion, resource gathering, and diplomacy.
"""

from environment import CivSimEnv
from agents import HeuristicAgent, AgentPersonality

# Create environment
env = CivSimEnv(
    num_civilizations=4,
    map_width=20,
    map_height=20,
    max_turns=100,
    seed=42
)

# Reset to start
obs = env.reset()

# Create agents with different peaceful strategies
agents = {
    0: HeuristicAgent(0, AgentPersonality.EXPANSIONIST, name="Explorer Empire"),
    1: HeuristicAgent(1, AgentPersonality.ECONOMIST, name="Trade Federation"),
    2: HeuristicAgent(2, AgentPersonality.DIPLOMAT, name="Alliance League"),
    3: HeuristicAgent(3, AgentPersonality.BALANCED, name="Balanced Kingdom"),
}

print("=" * 60)
print("🌍 PEACEFUL CIVILIZATION SIMULATION 🌍")
print("=" * 60)
print("\nCivilizations:")
for i, civ in env.state.civilizations.items():
    agent = agents[i]
    print(f"  {i}. {civ.name} - Strategy: {agent.personality.name}")
    print(f"     Starting units: Worker, Scout, Settler")
    print(f"     Starting resources: 50 Food, 50 Materials, 0 Tech")

print("\n" + "=" * 60)
print("Game Rules:")
print("  • No combat or war - purely peaceful competition")
print("  • Win by expanding territory and growing cities")
print("  • Form alliances and trade resources")
print("  • Highest score after 100 turns wins")
print("=" * 60)

# Game loop
turn = 0
while not env.is_done():
    # Each civ takes actions until they end their turn
    for civ_id in env.get_active_civs():
        valid_actions = env.get_valid_actions(civ_id)
        action = agents[civ_id].select_action(obs[civ_id], valid_actions)

        result = env.step(civ_id, action)
        obs = result.observations

        if result.terminated:
            break

    # Print progress every 20 turns
    if env.state.current_turn % 20 == 0 and env.state.current_turn > 0:
        print(f"\n📊 Turn {env.state.current_turn} Update:")
        scores = env.get_scores()
        sorted_civs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        for rank, (civ_id, score) in enumerate(sorted_civs, 1):
            civ = env.state.civilizations[civ_id]
            agent = agents[civ_id]
            num_cities = len(env.state.get_cities_for_civ(civ_id))
            num_units = len(env.state.get_units_for_civ(civ_id))

            status = "🏆" if rank == 1 else f"#{rank}"
            print(f"  {status} {civ.name} ({agent.personality.name})")
            print(f"     Score: {score} | Cities: {num_cities} | Units: {num_units}")
            print(f"     Resources: {int(civ.resources.get('FOOD', 0))} Food, "
                  f"{int(civ.resources.get('MATERIALS', 0))} Materials, "
                  f"{int(civ.resources.get('TECHNOLOGY', 0))} Tech")

# Game over!
print("\n" + "=" * 60)
print("🎮 GAME OVER! 🎮")
print("=" * 60)

scores = env.get_scores()
sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

print("\n🏆 Final Rankings:")
for rank, (civ_id, score) in enumerate(sorted_scores, 1):
    civ = env.state.civilizations[civ_id]
    agent = agents[civ_id]
    num_cities = len(env.state.get_cities_for_civ(civ_id))
    num_units = len(env.state.get_units_for_civ(civ_id))

    medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
    status = "WINNER!" if civ_id == env.get_winner() else "ALIVE" if civ.is_alive else "ELIMINATED"

    print(f"\n{medal} {rank}. {civ.name} - {status}")
    print(f"   Strategy: {agent.personality.name}")
    print(f"   Final Score: {score}")
    print(f"   Cities Founded: {num_cities}")
    print(f"   Units: {num_units}")

    # Show what contributed to their score
    city_score = num_cities * 100
    territory = sum(1 for t in env.state.map if t.owner == civ_id)
    territory_score = territory * 5
    print(f"   Territory Owned: {territory} tiles ({territory_score} pts)")
    print(f"   City Score: {city_score} pts")

print("\n" + "=" * 60)
print(f"Total Turns Played: {env.state.current_turn}")
print("=" * 60)

# Show final map
print("\n📍 Final Map:")
print(env.render_ascii())

print("\n✨ Game completed successfully with peaceful mechanics! ✨")
