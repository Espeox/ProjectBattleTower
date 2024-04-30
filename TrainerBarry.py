import numpy as np
from gym.spaces import Space, Box
from poke_env.player import Gen9EnvSinglePlayer
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.side_condition import STACKABLE_CONDITIONS, SideCondition
from poke_env.environment.battle import Battle
from poke_env.player.battle_order import BattleOrder, ForfeitBattleOrder

#Creates the Gen9 Battle Bot, Barry
class Barry(Gen9EnvSinglePlayer):
    _ACTION_SPACE = list(range(2 * 4 + 6))
    _DEFAULT_BATTLE_FORMAT = "gen9ou"

    def action_to_move(self, action: int, battle: Battle) -> BattleOrder:  # pyre-ignore
        """Converts actions to move orders.
        The conversion is done as follows

        action = -1:
            The battle will be forfeited.
        0 <= action < 4:
            The actionth available move in battle.available_moves is executed.
        4 <= action < 8:
            The action - 16th available move in battle.available_moves is executed,
            while terastallizing.
        8 <= action < 14
            The action - 20th available switch in battle.available_switches is executed.

        If the proposed action is illegal, a random legal move is performed.
        """
        if action == -1:
            return ForfeitBattleOrder()
        elif (
            action < 4
            and action < len(battle.available_moves)
            and not battle.force_switch
        ):
            return self.agent.create_order(battle.available_moves[action])
        elif (
            battle.can_terastallize
            and 0 <= action - 4 < len(battle.available_moves)
            and not battle.force_switch
        ):
            return self.agent.create_order(
                battle.available_moves[action - 4], terastallize=True
            )
        elif 0 <= action - 8 < 6:
            switch_mon = battle.team[list(battle.team.keys())[action-8]]
            if switch_mon.fainted and not switch_mon.active:
                return self.agent.choose_random_move(battle)
            else:
                return self.agent.create_order(switch_mon)
        else:
            return self.agent.choose_random_move(battle)


    def capture_replay(self, capture: bool = True):
        self.agent._save_replays = capture
        
    def calc_reward(self, last_battle, current_battle) -> float: #RECREATE
        global battle_tester
        battle_tester = current_battle
        return self.reward_computing_helper(
            current_battle, fainted_value=4.0, hp_value=1.0, status_value=3.0, hazard_value=2.0, victory_value=20.0
        )
    def reward_computing_helper(
        self,
        battle: AbstractBattle,
        *,
        fainted_value: float = 0.0,
        hp_value: float = 0.0,
        number_of_pokemons: int = 6,
        starting_value: float = 0.0,
        status_value: float = 0.0,
        victory_value: float = 1.0,
        hazard_value: float = 1.0
    ) -> float:
        """A helper function to compute rewards.

        :param battle: The battle for which to compute rewards.
        :type battle: AbstractBattle
        :param fainted_value: The reward weight for fainted pokemons. Defaults to 0.
        :type fainted_value: float
        :param hp_value: The reward weight for hp per pokemon. Defaults to 0.
        :type hp_value: float
        :param number_of_pokemons: The number of pokemons per team. Defaults to 6.
        :type number_of_pokemons: int
        :param starting_value: The default reference value evaluation. Defaults to 0.
        :type starting_value: float
        :param status_value: The reward value per non-fainted status. Defaults to 0.
        :type status_value: float
        :param victory_value: The reward value for winning. Defaults to 1.
        :type victory_value: float
        :return: The reward.
        :rtype: float
        """
        if battle not in self._reward_buffer:
            self._reward_buffer[battle] = starting_value
        current_value = 0

        for mon in battle.team.values():
            current_value += mon.current_hp_fraction * hp_value
            if mon.fainted:
                current_value -= fainted_value
            elif mon.status is not None:
                current_value -= status_value

        current_value += (number_of_pokemons - len(battle.team)) * hp_value

        for mon in battle.opponent_team.values():
            current_value -= mon.current_hp_fraction * hp_value
            if mon.fainted:
                current_value += fainted_value
            elif mon.status is not None:
                current_value += status_value

        current_value -= (number_of_pokemons - len(battle.opponent_team)) * hp_value

        current_value += sum(list(battle.side_conditions.values())) * hazard_value
        current_value -= sum(list(battle.opponent_side_conditions.values())) * hazard_value

        if battle.won:
            current_value += victory_value
        elif battle.lost:
            current_value -= victory_value

        to_return = current_value - self._reward_buffer[battle]
        self._reward_buffer[battle] = current_value

        return to_return
    
    def encode_status(self, status):
        if status == None:
            return [0, 0, 0, 0, 0]
        encoded = []
        for s in ("FNT", "BRN", "PSN", "SLP", "TOX"): #PAR and FRZ not included for this bot
            if status.name == s:
                encoded.append(1)
            else:
                encoded.append(0)
        return encoded

    def encode_hazards(self, hazard_dict):
        encoded = [0]
        if hazard_dict.get(SideCondition(16)):
            encoded[0] = 1
        for h in (15, 19):
            if hazard_dict.get(SideCondition(h)) != None:
                encoded.append(hazard_dict.get(SideCondition(h)))
            else:
                encoded.append(0)
        return encoded
    
    def team_preview_embedding(self): 
        battle = []
        #Active Field
        battle += [0]*14
        battle += [0, 0, 0, 0, 0, 1, 1, 0]*6
        battle += battle
        return np.float32(battle) #size 124

    def embed_battle(self, battle: AbstractBattle):
        #global battle_tester Used for Testing
        #battle_tester = battle
        
        if battle._in_team_preview:
            return self.team_preview_embedding()
        
        active_field = []
        # 0:5 - which pokemon is active
        # 6:10 - Modifiers
        # 11:13 - Hazard

        for k in battle.team.keys():
            if battle.team[k].active:
                active_field.append(1)
            else:
                active_field.append(0)
        #active_field[list(battle.team).index("p1: " + battle.active_pokemon.species.capitalize())] = 1
        active_field.append(battle.active_pokemon.boosts.get("atk"))
        active_field.append(battle.active_pokemon.boosts.get("def"))
        active_field.append(battle.active_pokemon.boosts.get("spa"))
        active_field.append(battle.active_pokemon.boosts.get("spd"))
        active_field.append(battle.active_pokemon.boosts.get("spe")) #11
        active_field = active_field + self.encode_hazards(battle.side_conditions) #Len 14

        pokemon_team = [] #np.ones(42)
        # For all 6 Pokemon
        # 0:4 - Status
        # 5 - HP
        # 6 - has_item
        # 7 - Terastralized
        
        for p in list(battle.team):
            poke = battle.team[p]
            poke_info = self.encode_status(poke.status)# Len 5
            poke_info.append(poke.current_hp_fraction) # Len 1
            poke_info.append(int(poke.item != "")) # Len 1
            poke_info.append(int(poke.terastallized)) # Len 1
            pokemon_team = pokemon_team + poke_info # Len 48

        opponent_active_field = []
        # 0:5 - which pokemon is active
        # 6:10 - Modifiers
        # 11:13 - Hazard
        for tpo in list(battle._teampreview_opponent_team):
            opponent_active_field.append(int(tpo.species == battle.opponent_active_pokemon.species))
            
        opponent_active_field.append(battle.opponent_active_pokemon.boosts.get("atk"))
        opponent_active_field.append(battle.opponent_active_pokemon.boosts.get("def"))
        opponent_active_field.append(battle.opponent_active_pokemon.boosts.get("spa"))
        opponent_active_field.append(battle.opponent_active_pokemon.boosts.get("spd"))
        opponent_active_field.append(battle.opponent_active_pokemon.boosts.get("spe"))
        opponent_active_field = opponent_active_field + self.encode_hazards(battle.opponent_side_conditions)

        opponent_pokemon_team = []
        # For all 6 Pokemon
        # 0:3 - Status
        # 4 - HP
        # 5 - has_item
        # 6 - Terastralized
        for p in list(battle._teampreview_opponent_team):
            if ("p2: " + p.species.capitalize()) in battle.opponent_team:
                opponent_pokemon = battle.opponent_team["p2: " + p.species.capitalize()]
                op_poke_info = self.encode_status(opponent_pokemon.status)
                op_poke_info.append(opponent_pokemon.current_hp_fraction)
                op_poke_info.append(int(opponent_pokemon.item != ""))
                op_poke_info.append(int(opponent_pokemon.terastallized))
            #opponent_pokemon = battle.opponent_team["p2: " + p.species.capitalize()]
            #if opponent_pokemon == None:
            #    op_poke_info = [0, 0, 0, 0, 0, 100, 1, 0]
            else:
                op_poke_info = [0, 0, 0, 0, 0, 100, 1, 0]
                
            opponent_pokemon_team = opponent_pokemon_team + op_poke_info
        # print(len(active_field))
        # print(len(pokemon_team))
        # print(len(opponent_active_field))
        # print(len(opponent_pokemon_team))
        # Final vector
        final_vector = active_field + pokemon_team + opponent_active_field + opponent_pokemon_team
        return np.float32(final_vector)

    def describe_embedding(self) -> Space: #Still need to complete
        low = []
        high = []
        
        #Active Pokemon
        low += [0, 0, 0, 0, 0, 0]
        high += [1, 1, 1, 1, 1, 1]
        
        #Modifier
        low += [0, 0, 0, 0, 0]
        high += [4, 4, 4, 4, 4]

        #Hazards
        low += [0, 0, 0] 
        high += [1, 3, 3] #CHECK SPIKES

        #Each Pokemon
        poke_low = []
        poke_high = []
        #Status
        poke_low += [0, 0, 0, 0, 0]
        poke_high += [1, 1, 1, 1, 1]
        #HP
        poke_low += [0]
        poke_high += [1] 
        #Has_Item
        poke_low += [0]
        poke_high += [1]
        #Terastralized
        poke_low += [0]
        poke_high += [1]
        #Add to embedding
        low += poke_low*6
        high += poke_high*6

        #Repeat for opponent side
        low += low
        high += high
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )