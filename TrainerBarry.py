import numpy as np
from gym.spaces import Space, Box
from poke_env.player import Gen8EnvSinglePlayer, Gen9EnvSinglePlayer
from gym.utils.env_checker import check_env
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.side_condition import STACKABLE_CONDITIONS, SideCondition
from poke_env.player import RandomPlayer, wrap_for_old_gym_api
from poke_env.player import RandomPlayer
from poke_env.player import background_evaluate_player

class Barry(Gen9EnvSinglePlayer):
    def calc_reward(self, last_battle, current_battle) -> float:
        global battle_tester
        battle_tester = current_battle
        if current_battle.won:
            print("Game Won")
        if current_battle.lost:
            print("Game Lost")
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )
    
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
        battle += [0, 0, 0, 0, 0, 100, 1, 0]*6
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
            poke_info.append(poke.current_hp_fraction*100) # Len 1
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
                op_poke_info.append(opponent_pokemon.current_hp_fraction*100) #Check if needs to be fraction?
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
        poke_high += [100] 
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