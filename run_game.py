import os
from time import sleep, time
from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.random_agent import RandomAgent

from epsilon_profile import EpsilonProfile
from agent.qagent import QAgent


def main():
    game = SpaceInvaders(display=False)

    # for eps_i in [0.95, 0.5, 0.1]:
    #     for eps_f in [0.1, 0.0, 0.01]:
    #         print("eps_i: ", eps_i, "    eps_f: ", eps_f)

    #         n_episodes = 50000
    #         max_steps = 1000
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
    #         startTime = time()
    #         while time() < startTime + 10:
    #             action = controller.select_action(state)
    #             state, reward, is_done = game.step(action)
    #             # print(state)
    #             # sleep(0.0001)

    #         game.reset()

    # -------------------------------------------------------------------------
    for eps_i in [0.6, 0.8, 0.7]:
        for eps_f in [0.1]:
            for max_steps in [500]:
                n_episodes = 50000
                # max_steps = 3000
                gamma = 0.95
                alpha = 1
                eps_profile = EpsilonProfile(eps_i, eps_f)
                fileName = "Q_{}_E{}_S{}_G{}_I{}_F{}".format("SXY", n_episodes, max_steps, gamma,
                                                        eps_profile.initial,
                                                        eps_profile.final)
                controller = QAgent(game, eps_profile, gamma, alpha, fileName)

                startTime = time()
                controller.learn(game, n_episodes, max_steps)
                endTime = time()
                controller.saveQToFile(os.path.join("LearnedQ", fileName))

                print()
                print(
                    "############################################################################"
                )
                print("FINISHED LEARNING")
                print("    n_episodes: ", n_episodes)
                print("    max_steps: ", max_steps)
                print("    gamma: ", gamma)
                print("    eps_profile (initial, final, dec_episode, dec_step): ",
                    eps_profile.initial, eps_profile.final, eps_profile.dec_episode,
                    eps_profile.dec_step)
                print("    time learning: ", endTime - startTime)
                print(
                    "############################################################################"
                )

    # -------------------------------------------------------------------------
    # n_episodes = 2000
    # max_steps = 700
    # gamma = 0.95
    # alpha = 1
    # eps_profile = EpsilonProfile(0.9, 0.1)
    # fileName = "Q_{}_E{}_S{}_G{}_I{}_F{}".format("SXY", n_episodes, max_steps, gamma,
    #                                           eps_profile.initial,
    #                                           eps_profile.final)
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
    #       eps_profile.initial, eps_profile.final, eps_profile.dec_episode,
    #       eps_profile.dec_step)
    # print("    time learning: ", endTime - startTime)
    # print(
    #     "############################################################################"
    # )

if __name__ == '__main__':
    main()
