import os
from time import sleep, time
from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.random_agent import RandomAgent

from epsilon_profile import EpsilonProfile
from agent.qagent import QAgent


def main():
    game = SpaceInvaders(display=False)

    # -------------------------------------------------------------------------
    n_episodes = 10000
    max_steps = 3000
    gamma = 0.95
    alpha = 1
    eps_profile = EpsilonProfile(1.0, 0.1)
    fileName = "Q_{}_E{}_S{}_G{}_I{}_F{}".format("SS", n_episodes, max_steps, gamma,
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
    n_episodes = 10000
    max_steps = 5000
    gamma = 0.95
    alpha = 1
    eps_profile = EpsilonProfile(1.0, 0.1)
    fileName = "Q_{}_E{}_S{}_G{}_I{}_F{}".format("SS", n_episodes, max_steps, gamma,
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
    n_episodes = 10000
    max_steps = 5000
    gamma = 0.999
    alpha = 1
    eps_profile = EpsilonProfile(1.0, 0.1)
    fileName = "Q_{}_E{}_S{}_G{}_I{}_F{}".format("SS", n_episodes, max_steps, gamma,
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
    n_episodes = 20000
    max_steps = 5000
    gamma = 0.95
    alpha = 1
    eps_profile = EpsilonProfile(1.0, 0.5)
    fileName = "Q_{}_E{}_S{}_G{}_I{}_F{}".format("SS", n_episodes, max_steps, gamma,
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



if __name__ == '__main__':
    main()
