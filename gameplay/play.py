#!/usr/bin/env python


import argparse
import re

from gameplay.engine import GameEngine


def main():
    parser = argparse.ArgumentParser(
        description='Play the CodeNames game.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', type=str, default='CHCH',
                        help='Config <spy1><team1><spy2><team2> using C,H.')
    parser.add_argument('-x', '--expert', action='store_true',
                        help='Expert clues. For now implements \'unlimited\' only.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible games.')
    parser.add_argument('--init', type=str, default=None,
                        help='Initialize words ASSASSIN;TEAM1;TEAM2;NEUTRAL')
    args = parser.parse_args()

    if not re.match('^[CH]{4}$', args.config):
        print('Invalid configuration. Try HHHH or CHCH.')
        return -1

    d = dict(H='human', C='computer')
    spy1 = d[args.config[0]]
    team1 = d[args.config[1]]
    spy2 = d[args.config[2]]
    team2 = d[args.config[3]]

    e = GameEngine(seed=args.seed, expert=args.expert)
    e.play_game(spy1, team1, spy2, team2, init=args.init)


if __name__ == '__main__':
    main()
