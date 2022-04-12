import os
import numpy as np
from copy import deepcopy
import pandas as pd

# from world.maze import Maze
from game import SpaceInvaders

X_MIN = 0
X_MAX = 75
Y_MIN = 0
Y_MAX = 25


class VIAgent():
    """ 
    Un agent capable de résoudre un labyrinthe donné grâce à l'algorithme d'itération 
    sur les valeurs (VI = Value Iteration).
    """
    def __init__(self, spaceInvaders: SpaceInvaders, gamma: float):
        """"À LIRE
        Ce constructeur initialise une nouvelle instance de la classe ValueIteration.
        Il stocke les différents paramètres nécessaires au fonctionnement de l'algorithme et initialise à 0 la 
        fonction de valeur d'état, notée V.

        :attribut V: La fonction de valeur d'états
        :type V: un tableau de dimension : ny x nx 

        :attribut maze: Le modèle du labyrinthe. Il permet de récupérer la fonction de transition (maze.dynamics) et la récompense (maze.reward)
        :type maze: DeterministicMazeModel

        :attribut gamma: le facteur d'atténuation
        :type gamma: float
        :requirement: 0 <= gamma <= 1

        - Visualisation des données
        :attribut mazeValues: la fonction de valeur stockée qui sera écrite dans un fichier de log après la résolution complète
        :type mazeValues: data frame pandas
        :penser à bien stocker aussi la taille du labyrinthe (nx,ny)
        """
        self.gamma = gamma
        self.spaceInvaders = spaceInvaders

        self.nxp = X_MAX - X_MIN + 1
        self.nxi = X_MAX - X_MIN + 1
        self.nyi = Y_MAX - Y_MIN + 1
        self.nb = 1 + 1

        self.states = [(a, b, c, d) for a in range(self.nxp)
                       for b in range(self.nxi) for c in range(self.nyi)
                       for d in range(self.nb)]
        
        self.numActions = 4

        self.V = np.zeros([self.nxp, self.nxi, self.nyi, self.nb])
        # self.V[0,0,0,0] = 1
        # self.V[0,0,0,1] = 2
        # self.V[0,0,1,0] = 3
        # self.V[0,1,0,0] = 4
        # self.V[1,0,0,0] = 5

        # print(self.V)

        # Visualisation des données (vous n'avez pas besoin de comprendre cette partie)
        # self.mazeValues = pd.DataFrame(data={'nx': maze.nx, 'ny': [maze.ny]})

    def solve(self, error: float):
        """
        Cette méthode résoud le problème avec une tolérance donnée.
        Elle doit proposer l'option de stockage de la fonction de valeur dans un fichier de log (logV.csv)
        """
        n_iteration = 0
        V_copy = np.zeros([self.nxp, self.nxi, self.nyi, self.nb])
        while ((n_iteration == 0) or not self.done(self.V, V_copy, error)):
            n_iteration += 1
            self.V = deepcopy(V_copy)
            for state in self.states():
                # if (not self.maze.maze[state]): #TODO
                V_copy[state] = self.bellman_operator(state)

        #     # Sauvegarde les valeurs intermédiaires
        #     self.mazeValues = self.mazeValues.append(
        #         {
        #             'episode':
        #             n_iteration,
        #             'value':
        #             np.reshape(self.V, (1, self.maze.ny * self.maze.nx))[0]
        #         },
        #         ignore_index=True)
        # self.mazeValues.to_csv(
        #     os.path.abspath('TP1/partie_2/visualisation/logV.csv'))

    def done(self, V, V_copy, error) -> bool:
        """À COMPLÉTER!
        Cette méthode retourne vraie si la condition d'arrêt de 
        l'algorithme est vérifiée. Sinon elle retourne faux. 
        Pour garantie la convergence en tout état, il est préférable 
        d'utiliser la norme infini comme critère d'arrêt.
        """

        if ((V - V_copy).max() < error):
            print("Done -> True")
            return True

        print("Done -> False")
        return False

        # raise NotImplementedError("VI NotImplementedError at function done.")

    def bellman_operator(self, state: 'Tuple[int, int]') -> float:
        """À COMPLÉTER!
        Cette méthode calcul l'opérateur de mise à jour de bellman pour un état s. 

        :param state: Un état quelconque
        :return: La valeur de mise à jour de la fonction de valeur
        """
        # Retourne une exception si l'état n'est pas valide
        # ... this function doesn't seem to be called in this case
        max_value = -np.infty
        for a in range(self.numActions):
            q_s_a = 0.
            for next_state in self.states():
                # Compléter ici votre équation de Bellman
                # Note: On utilisera la fonction de récompense (self.maze.getReward) et la fonction de transition (self.maze.getDynamics).
                q_s_a += self.maze.getDynamics(state, a, next_state) * (
                    self.maze.getReward(state, a) +
                    self.gamma * self.V[next_state[0], next_state[1]])

                # raise NotImplementedError("Value Iteration NotImplementedError at Function bellman_operator.")
            if (q_s_a > max_value):
                max_value = q_s_a
        print(max_value)
        return max_value

    def select_action(self, state: 'Tuple[int, int]') -> int:
        """À COMPLÉTER!
        Cette méthode retourne l'action optimale.

        :param state: L'état courant
        :return: L'action optimale

        doit retourner une exception si l'état n'est pas valide
        """
        max_value = -np.infty
        amax = 0
        for a in range(self.maze.na):
            q_s_a = 0.
            for next_state in self.maze.getStates():
                # Compléter ici votre équation de Bellman
                # Note: On utilisera la fonction de récompense (self.maze.getReward) et la fonction de transition (self.maze.getDynamics).
                q_s_a += self.maze.getDynamics(state, a, next_state) * (
                    self.maze.getReward(state, a) +
                    self.gamma * self.V[next_state[0], next_state[1]])

                # raise NotImplementedError("Value Iteration NotImplementedError at Function select_action")
            if (q_s_a > max_value):
                max_value = q_s_a
                amax = a
        return amax