from time import sleep
from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.random_agent import RandomAgent


def main():

    game = SpaceInvaders(display=True)
    controller = KeyboardController()
    # controller = RandomAgent(game.na)

    state = game.reset()
    while True:
        action = controller.select_action(state)
        state, reward, is_done = game.step(action)
        print("position of X"+str(game.get_state()))
        # print("position of Y" + str(game.get_player_Y())
        # print("position of X_invader" + str(game.get_invaders_X())
        # print("position of Y_invader" + str(game.get_invaders_Y())

        sleep(0.0001)


if __name__ == '__main__':
    main()
