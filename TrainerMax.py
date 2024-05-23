import numpy as np
import random
from poke_env.player import Player


class MaxDamagePlayer(Player):
    def choose_move(self, battle, epsilon = 0.3):
        # If the player can attack, it will
        if battle.available_moves and random.random() <= epsilon:
            # Finds the best move among available ones
            best_move = None
            highest_move_dmg = -1
            mon_a = battle.active_pokemon
            mon_b = battle.opponent_active_pokemon

            for move in battle.available_moves:
                print(f'Checking Move Type {move.type}, and Base Power {move.base_power}')
                move_dmg = move.base_power * move.type.damage_multiplier(mon_b.type_1, mon_b.type_2, type_chart=battle._data.type_chart)
                
                if move_dmg > highest_move_dmg:
                    best_move = move
                    highest_move_dmg = move_dmg
            
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)

    def teampreview(self, battle):
        mon_performance = {}

        # For each of our pokemons
        for i, mon in enumerate(battle.team.values()):
            # We store their average performance against the opponent team
            mon_performance[i] = np.mean(
                [
                    teampreview_performance(mon, opp, battle._data.type_chart)
                    for opp in battle.opponent_team.values()
                ]
            )

        # We sort our mons by performance
        ordered_mons = sorted(mon_performance, key=lambda k: -mon_performance[k])

        # We start with the one we consider best overall
        # We use i + 1 as python indexes start from 0
        #  but showdown's indexes start from 1
        return "/team " + "".join([str(i + 1) for i in ordered_mons])


def teampreview_performance(mon_a, mon_b, type_chart):
    # We evaluate the performance on mon_a against mon_b as its type advantage
    a_on_b = b_on_a = -np.inf
    for type_ in mon_a.types:
        if type_:
            a_on_b = max(a_on_b, type_.damage_multiplier(mon_b.type_1, mon_b.type_2, type_chart=type_chart))
    # We do the same for mon_b over mon_a
    for type_ in mon_b.types:
        if type_:
            b_on_a = max(b_on_a, type_.damage_multiplier(mon_a.type_1, mon_a.type_2, type_chart=type_chart))
    # Our performance metric is the different between the two
    return a_on_b - b_on_a