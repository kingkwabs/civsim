# 🏘️ New Settlement-Based Game Design

## Major Changes from Original

### ❌ **REMOVED:**
- All units (Worker, Scout, Settler)
- Movement system
- Combat system
- Fog of war
- Unit upkeep/starvation
- Production queues
- Gathering actions

### ✅ **NEW CORE MECHANICS:**

---

## 1. **Resources (Catan-Style)**

### **6 Resource Types:**
```python
WOOD    🌲  # From forests
STONE   🪨  # From mountains
CATTLE  🐄  # From plains
WHEAT   🌾  # From fertile lands
WATER   💧  # From water tiles
METAL   ⚒️  # From mountains/special tiles
```

### **Resource Generation (Percentage-Based):**

Each tile has a **percentage chance** each turn to generate resources:

```python
Terrain.PLAINS:     60% cattle, 30% wheat
Terrain.FOREST:     70% wood, 20% water
Terrain.MOUNTAINS:  70% stone, 40% metal
Terrain.WATER:      80% water
Terrain.FERTILE:    80% wheat, 30% water
Terrain.DESERT:     40% stone, 30% metal
```

**Each turn:**
- Roll random number (0-1) for each tile
- If roll < percentage, generate 1 of that resource
- Collected by settlements touching that tile

---

## 2. **Corner-Based Hex Positioning**

### **Hexagon Corners:**
```
Each hex has 6 corners (numbered 0-5):

         0
       /   \
      5     1
      |  🏘️  |  ← Settlement goes AT CORNER
      4     2
       \   /
         3
```

### **HexCorner Class:**
```python
HexCorner(
    q=5,           # Reference hex q coordinate
    r=3,           # Reference hex r coordinate
    corner_idx=2   # Which corner (0-5)
)
```

### **Key Property:**
- Each corner touches **3 hexes**
- Settlement at corner collects from all 3 tiles

**Example:**
```
Settlement at corner (q=5, r=3, idx=2):
  → Touches tiles: (5,3), (6,3), (5,4)
  → If tiles are [Forest, Plains, Mountain]:
     - 70% chance: 1 wood (from forest)
     - 60% chance: 1 cattle (from plains)
     - 70% chance: 1 stone (from mountain)
     - 40% chance: 1 metal (from mountain)
```

---

## 3. **Settlement Tiers**

### **Tier 1: Settlement**
- Basic building
- Collects resources at 1× multiplier
- Worth **1 victory point**

### **Tier 2: City**
- Upgraded from settlement
- Collects resources at **2× multiplier**
- Worth **3 victory points**

**Upgrade Example:**
```
Settlement on [Forest, Plains, Mountain]:
  → Each turn: ~70% wood, ~60% cattle, ~70% stone
  → Upgrade to City
  → Each turn: ~70% chance of 2 wood, ~60% chance of 2 cattle, etc.
```

---

## 4. **Building Costs**

### **Settlement:**
```python
BUILD_SETTLEMENT = {
    WOOD: 2,
    STONE: 1,
    WHEAT: 1,
    CATTLE: 1
}
```

### **City (Upgrade):**
```python
UPGRADE_TO_CITY = {
    STONE: 3,
    METAL: 2,
    WHEAT: 2
}
```

### **Road:**
```python
BUILD_ROAD = {
    WOOD: 1,
    STONE: 1
}
```

---

## 5. **Roads System**

### **Purpose:**
Roads connect settlements and are required for starting positions.

### **Road Structure:**
```python
Road(
    id=0,
    owner=0,
    corner1=HexCorner(q=5, r=3, idx=2),
    corner2=HexCorner(q=5, r=3, idx=3)  # Adjacent corner
)
```

### **Starting Setup:**
Each civilization starts with:
- **2 settlements** (pre-placed at good locations)
- **2 roads** (one from each settlement)
- **10 of each resource**

---

## 6. **Maintenance System**

### **Individual Timers:**
Each settlement tracks its own maintenance schedule:

```python
Settlement(
    founded_turn=0,
    next_maintenance_turn=10  # Due on turn 10
)
```

**Every 10 turns** after founding, maintenance is due.

### **Maintenance Costs:**

**Settlement:**
```python
SETTLEMENT_MAINTENANCE = {
    WHEAT: 1,
    CATTLE: 1,
    WOOD: 1
}
```

**City:**
```python
CITY_MAINTENANCE = {
    WHEAT: 2,
    CATTLE: 2,
    WOOD: 1,
    STONE: 1
}
```

### **Abandonment Mechanic:**

**If you cannot pay maintenance:**
1. Settlement becomes `is_abandoned = True`
2. You lose ownership
3. Settlement stays on map (visible to all)
4. Other players can see it's abandoned

**To Reclaim:**
```python
RECLAIM_COST = {
    WOOD: 3,
    STONE: 2,
    WHEAT: 1
}

Total cost = MAINTENANCE + RECLAIM_COST
```

---

## 7. **New Actions**

### **FoundSettlementAction:**
```python
FoundSettlementAction(
    civ_id=0,
    corner=HexCorner(q=5, r=3, idx=2),
    name="New Settlement"
)
```
- Costs: 2 wood, 1 stone, 1 wheat, 1 cattle
- Places settlement at corner
- Immediately starts collecting resources

### **UpgradeCityAction:**
```python
UpgradeCityAction(
    civ_id=0,
    settlement_id=3
)
```
- Costs: 3 stone, 2 metal, 2 wheat
- Upgrades settlement → city
- Doubles resource collection

### **BuildRoadAction:**
```python
BuildRoadAction(
    civ_id=0,
    corner1=HexCorner(...),
    corner2=HexCorner(...)  # Must be adjacent
)
```
- Costs: 1 wood, 1 stone
- Connects two corners
- Currently optional (for future expansion mechanics)

### **PayMaintenanceAction:**
```python
PayMaintenanceAction(
    civ_id=0,
    settlement_id=3
)
```
- Costs: maintenance resources
- Resets maintenance timer
- Prevents abandonment

### **ReclaimSettlementAction:**
```python
ReclaimSettlementAction(
    civ_id=0,
    settlement_id=5  # Abandoned settlement
)
```
- Costs: maintenance + reclaim fee
- Takes ownership of abandoned settlement
- Resets maintenance timer

### **Trade/Diplomacy Actions:**
- `ProposeTradeAction` (unchanged)
- `AcceptTradeAction` (unchanged)
- `ProposeAllianceAction` (unchanged)
- `EndTurnAction` (unchanged)

---

## 8. **Turn Flow**

### **Each Turn:**

1. **Resource Generation Phase:**
   ```python
   for settlement in all_settlements:
       if not settlement.is_abandoned:
           tiles = settlement.get_adjacent_tiles()
           for tile in tiles:
               for resource, chance in tile.resource_yields.items():
                   if random() < chance:
                       amount = 1 * settlement.get_resource_multiplier()
                       civ.resources[resource] += amount
   ```

2. **Maintenance Check Phase:**
   ```python
   for settlement in civ.settlements:
       if current_turn >= settlement.next_maintenance_turn:
           if civ.can_afford(maintenance_cost):
               # Player must decide to pay or abandon
               pass
           else:
               settlement.is_abandoned = True
               settlement.owner = None
   ```

3. **Action Phase:**
   - Players take actions (found settlements, build roads, trade, etc.)
   - End turn when done

4. **Scoring Update:**
   ```python
   score = (num_settlements × 1) + (num_cities × 3)
   score += territory_tiles × 5
   score += sum(resources × 0.5)
   ```

---

## 9. **Victory Conditions**

### **Score Components:**
| Component | Points |
|-----------|--------|
| Settlement | 1 pt |
| City | 3 pts |
| Territory tile | 5 pts |
| Each resource | 0.5 pts |
| Alliance | Bonus pts |

### **Win Conditions:**
- Highest score after max_turns
- First to reach score threshold (e.g., 50 pts)

---

## 10. **Example Gameplay**

### **Turn 1:**
```
Player starts with:
  - 2 settlements
  - 2 roads
  - 10 wood, 10 stone, 10 cattle, 10 wheat, 10 water, 10 metal

Settlement 1 touches: [Forest, Plains, Mountain]
Settlement 2 touches: [Fertile, Fertile, Water]

Resource generation:
  - Forest (70% wood): Roll 0.65 → SUCCESS: +1 wood
  - Plains (60% cattle): Roll 0.75 → FAIL: +0 cattle
  - Mountain (70% stone): Roll 0.45 → SUCCESS: +1 stone
  - Fertile (80% wheat): Roll 0.30 → SUCCESS: +1 wheat
  - Fertile (80% wheat): Roll 0.50 → SUCCESS: +1 wheat
  - Water (80% water): Roll 0.20 → SUCCESS: +1 water

Total collected: +1 wood, +1 stone, +2 wheat, +1 water
New resources: 11 wood, 11 stone, 10 cattle, 12 wheat, 11 water, 10 metal
```

### **Turn 5:**
```
Resources: 15 wood, 13 stone, 12 cattle, 14 wheat, 13 water, 11 metal

Action: Found new settlement
Costs: -2 wood, -1 stone, -1 wheat, -1 cattle
New resources: 13 wood, 12 stone, 11 cattle, 13 wheat, 13 water, 11 metal

Now have 3 settlements → 3 points
```

### **Turn 10:**
```
Maintenance due for Settlement 1 and 2!

Pay for Settlement 1: -1 wheat, -1 cattle, -1 wood
Pay for Settlement 2: -1 wheat, -1 cattle, -1 wood

Resources after maintenance: 11 wood, 12 stone, 9 cattle, 11 wheat, 13 water, 11 metal
```

### **Turn 15:**
```
Resources: 18 wood, 15 stone, 12 cattle, 14 wheat, 16 water, 14 metal

Action: Upgrade Settlement 3 to City
Costs: -3 stone, -2 metal, -2 wheat

Settlement 3 now collects 2× resources from its 3 tiles!
```

---

## 11. **Key Differences from Original**

| Feature | Old (Unit-Based) | New (Settlement-Based) |
|---------|-----------------|----------------------|
| Core Entity | Units + Cities | Settlements only |
| Movement | Yes | No |
| Combat | Yes | No |
| Resources | 3 types (Food/Materials/Tech) | 6 types (Wood/Stone/etc.) |
| Generation | Fixed yields | Percentage-based |
| Placement | Hex centers | Hex corners |
| Upkeep | Unit food consumption | Settlement maintenance |
| Expansion | Settlers move & found | Direct placement |
| Actions/turn | 15-30 | 3-5 |
| Complexity | High | Low |

---

## 12. **Implementation Status**

✅ **Completed:**
- New 6-resource system defined
- HexCorner class for corner positioning
- Settlement & City classes
- Road class
- Maintenance timer system
- Resource multipliers

🚧 **In Progress:**
- Removing old unit code
- New action classes
- Game initialization with 2 settlements + 2 roads
- Resource generation implementation
- Maintenance payment/abandonment logic

⏳ **TODO:**
- Update agents for new mechanics
- Update tests
- Balance resource percentages
- Add placement restrictions (if any)

---

This is a **complete redesign** turning CivSim into a **Catan-inspired settlement management game**!
