import os
from time import sleep, time
from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.random_agent import RandomAgent

from epsilon_profile import EpsilonProfile
from agent.qagent import QAgent

import logging


def main():
    game = SpaceInvaders(display=True)

    # TEST TRAINED AGENT WITH LEARNED Q VALUES
    gamma = 0.95
    alpha = 1
    eps_profile = EpsilonProfile(0.7, 0.05)
    max_steps = 500
    n_episodes = 180000

    controller = QAgent(game, eps_profile, gamma, alpha)
    controller.loadQFromFile(
        os.path.join(os.path.abspath("LearnedQ"),
                     "Q_SXY_E180000_S500_G0.95_I0.7_F0.05.npy"))

    state = game.reset()
    while True:
        action = controller.select_greedy_action(state)
        state, reward, is_done = game.step(action)
        print("\r#> Score: {}  ".format(game.score_val), end=" ")
        sleep(0.0001)

    # -------------------------------------------------------------------------
    # TRAIN NEW AGENT

    # gamma = 0.95
    # alpha = 1
    # eps_profile = EpsilonProfile(0.7, 0.05)
    # max_steps = 500
    # n_episodes = 180000

    # fileName = "Q_{}_E{}_S{}_G{}_I{}_F{}".format("SXY", n_episodes, max_steps, gamma,
    #                                         eps_profile.initial,
    #                                         eps_profile.final)

    # controller = QAgent(game, eps_profile, gamma, alpha, fileName)

    # startTime = time()
    # controller.learn(game, n_episodes, max_steps)
    # endTime = time()
    # controller.saveQToFile(os.path.join("LearnedQ", fileName))

    # print()
    # print(
    #     "############################################################################"
    # )
    # print("FINISHED LEARNING")
    # print("    n_episodes: ", n_episodes)
    # print("    max_steps: ", max_steps)
    # print("    gamma: ", gamma)
    # print("    eps_profile (initial, final, dec_episode, dec_step): ",
    #     eps_profile.initial, eps_profile.final, eps_profile.dec_episode,
    #     eps_profile.dec_step)
    # print("    time learning: ", endTime - startTime)
    # print(
    #     "############################################################################"
    # )


if __name__ == '__main__':
    main()
