from time import sleep
from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.random_agent import RandomAgent

from epsilon_profile import EpsilonProfile
from agent.qagent import QAgent


def main():
    ### AGENT LEARN ###
    n_episodes = 10  # *************** 200 3000
    max_steps = 500    # *************** 5060
    gamma = 0.95 # big: end steps are equally important, small: only current success is important
    # Shouldn't really make difference because current and future score are directly correlated

    alpha = 1 # MUST ALWAYS STAY 
    eps_profile = EpsilonProfile(1.0 , 0.1) # Can make decrease slower (e.g. 0.01) to explore more and be less gready

    game = SpaceInvaders(display=False)

    # controller = KeyboardController()
    # controller = RandomAgent(game.na)

    controller = QAgent(game, eps_profile, gamma, alpha)
    controller.learn(game, n_episodes, max_steps)
    controller.saveQToFile()
    # controller.loadQFromFile("LearnedQ100E50000S.npy")

    ### PLAY GAME ###
    # state = game.reset()
    # while True:
    #     action = controller.select_action(state)
    #     state, reward, is_done = game.step(action)

    #     # sleep(0.0001)


if __name__ == '__main__':
    main()
