import os
from time import sleep, time
from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.random_agent import RandomAgent

from epsilon_profile import EpsilonProfile
from agent.qagent import QAgent

import logging


def main():
    game = SpaceInvaders(display=False)

    # for eps_i in [0.7]: #0.6
    #     for eps_f in [0.05]:
    #         n_episodes = 200000
    #         max_steps = 500
    #         gamma = 0.95
    #         alpha = 1
    #         eps_profile = EpsilonProfile(eps_i, eps_f)
    #         fileName = "Q_{}_E{}_S{}_G{}_I{}_F{}".format("SXY", n_episodes, max_steps,
    #                                                     gamma, eps_profile.initial,
    #                                                     eps_profile.final)
    #         controller = QAgent(game, eps_profile, gamma, alpha, fileName)

    #         controller.loadQFromFile(
    #             os.path.abspath(
    #                 os.path.join("LearnedQ/",
    #                             fileName + ".npy")))

    #         ### PLAY GAME ###
    #         state = game.reset()
    #         # startTime = time()
    #         # while time() < startTime + 10:
    #         while True:
    #             action = controller.select_greedy_action(state)
    #             state, reward, is_done = game.step(action)
    #             # print(state)
    #             # sleep(0.0001)

    #         game.reset()

    # -------------------------------------------------------------------------
    # TEST (ALL) PREVIOUSLY LEARNED AGENTS

    # gamma = 0.95
    # alpha = 1
    # eps_profile = EpsilonProfile(0, 0)

    # for file in os.listdir(os.path.abspath("LearnedQ")):
    #     if file == ".DS_Store":
    #         continue

    #     if not file in ["Q_SXY_E50000_S500_G0.95_I0.6_F0.05.npy",
    #         "Q_SXY_E50000_S500_G0.95_I0.6_F0.1.npy",
    #         "Q_SXY_E50000_S500_G0.95_I0.7_F0.05.npy",
    #         "Q_SXY_E100000_S500_G0.95_I0.7_F0.05.npy"
    #         ]:
    #         continue
        
    #     controller = QAgent(game, eps_profile, gamma, alpha)
    #     controller.loadQFromFile(os.path.join(os.path.abspath("LearnedQ"), file))

    #     ### PLAY GAME ###
    #     state = game.reset()
    #     i = 0
    #     while i < 20000:
    #     # while True:
    #         i += 1
    #         action = controller.select_greedy_action(state)
    #         state, reward, is_done = game.step(action)
    #         # print(state)
    #         # sleep(0.0001)

    #     print(file, "    score: ", game.score_val)

    # -------------------------------------------------------------------------
    # TEST EVOLUTION OF LEARNED AGENT

    gamma = 0.95
    alpha = 1
    eps_profile = EpsilonProfile(0, 0)

    directory = "QEvolution_Q_SXY_E200000_S500_G0.95_I0.6_F0.05"
    logging.basicConfig(filename=directory+'.log', encoding='utf-8', level=logging.DEBUG)

    file = 0
    while file < 200000:
        
        controller = QAgent(game, eps_profile, gamma, alpha)
        controller.loadQFromFile(os.path.join(os.path.abspath(directory), str(file) + ".npy"))

        ### PLAY GAME ###
        state = game.reset()
        i = 0
        while i < 20000:
        # while True:
            i += 1
            action = controller.select_greedy_action(state)
            state, reward, is_done = game.step(action)
            # print(state)
            # sleep(0.0001)

        logging.debug("{:06d}".format(file) + ", " + str(game.score_val))
        file += 1000

    # -------------------------------------------------------------------------
    # LEARN

    # for eps_i in [0.6]:
    #     for eps_f in [0.05]:
    #         for max_steps in [500]:
    #             n_episodes = 200000
    #             gamma = 0.95
    #             alpha = 1
    #             eps_profile = EpsilonProfile(eps_i, eps_f)
    #             fileName = "Q_{}_E{}_S{}_G{}_I{}_F{}".format("SXY", n_episodes, max_steps, gamma,
    #                                                     eps_profile.initial,
    #                                                     eps_profile.final)
    #             controller = QAgent(game, eps_profile, gamma, alpha, fileName)

    #             startTime = time()
    #             controller.learn(game, n_episodes, max_steps)
    #             endTime = time()
    #             controller.saveQToFile(os.path.join("LearnedQ", fileName))

    #             print()
    #             print(
    #                 "############################################################################"
    #             )
    #             print("FINISHED LEARNING")
    #             print("    n_episodes: ", n_episodes)
    #             print("    max_steps: ", max_steps)
    #             print("    gamma: ", gamma)
    #             print("    eps_profile (initial, final, dec_episode, dec_step): ",
    #                 eps_profile.initial, eps_profile.final, eps_profile.dec_episode,
    #                 eps_profile.dec_step)
    #             print("    time learning: ", endTime - startTime)
    #             print(
    #                 "############################################################################"
    #             )

if __name__ == '__main__':
    main()
