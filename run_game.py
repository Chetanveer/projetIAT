from time import sleep, time
from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.random_agent import RandomAgent

from epsilon_profile import EpsilonProfile
from agent.qagent import QAgent


def main():
    ### AGENT LEARN ###
    n_episodes = 50  # *************** 200 3000
    max_steps = 20000  # *************** 5060
    gamma = 0.95  # big: end steps are equally important, small: only current success is important
    # Shouldn't really make difference because current and future score are directly correlated

    alpha = 1  # MUST ALWAYS STAY
    eps_profile = EpsilonProfile(
        1.0, 0.1
    )  # Can make decrease slower (e.g. 0.01) to explore more and be less gready

    game = SpaceInvaders(display=False)

    # controller = KeyboardController()
    # controller = RandomAgent(game.na)

    controller = QAgent(game, eps_profile, gamma, alpha)

    startTime = time()
    controller.learn(game, n_episodes, max_steps)
    endTime = time()

    controller.saveQToFile()
    # controller.loadQFromFile("LearnedQ100E50000S.npy")

    print()
    print("############################################################################")
    print("FINISHED LEARNING")
    print("    n_episodes: ", n_episodes)
    print("    max_steps: ", max_steps)
    print("    gamma: ", gamma)
    print("    eps_profile (initial, final, dec_episode, dec_step): ",
          eps_profile.initial, eps_profile.final, eps_profile.dec_episode,
          eps_profile.dec_step)
    print("    time learning: ", endTime - startTime)
    print("############################################################################")

    ### PLAY GAME ###
    # state = game.reset()
    # while True:
    #     action = controller.select_action(state)
    #     state, reward, is_done = game.step(action)

    #     # sleep(0.0001)


if __name__ == '__main__':
    main()
