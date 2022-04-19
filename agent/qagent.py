import os
import numpy as np
from epsilon_profile import EpsilonProfile

# Suppress future warnings from pandas
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd

from game import SpaceInvaders

X_MIN = 0
X_MAX = 76  # TODO
Y_MIN = 0
Y_MAX = 10

NUMER_ACTIONS = 4


class QAgent():
    """ 
    Cette classe d'agent représente un agent utilisant la méthode du Q-learning 
    pour mettre à jour sa politique d'action.
    """
    def __init__(self,
                 spaceInvaders: SpaceInvaders,
                 eps_profile: EpsilonProfile,
                 gamma: float,
                 alpha: float,
                 fileLog="logQ"):
        """A LIRE
        Ce constructeur initialise une nouvelle instance de la classe QAgent.
        Il doit stocker les différents paramètres nécessaires au fonctionnement de l'algorithme et initialiser la 
        fonction de valeur d'action, notée Q.

        :param maze: Le labyrinthe à résoudre 
        :type maze: Maze

        :param eps_profile: Le profil du paramètre d'exploration epsilon 
        :type eps_profile: EpsilonProfile
        
        :param gamma: Le discount factor 
        :type gamma: float
        
        :param alpha: Le learning rate 
        :type alpha: float

        - Visualisation des données
        :attribut mazeValues: la fonction de valeur stockée qui sera écrite dans un fichier de log après la résolution complète
        :type mazeValues: data frame pandas
        :penser à bien stocker aussi la taille du labyrinthe (nx,ny)

        :attribut qvalues: la Q-valeur stockée qui sera écrite dans un fichier de log après la résolution complète
        :type mazeValues: data frame pandas
        """

        self.na = NUMER_ACTIONS

        # Initialise la fonction de valeur Q
        self.Q = np.zeros([X_MAX + 1, Y_MAX + 1, 1 + 1, self.na])

        self.spaceInvaders = spaceInvaders

        # Paramètres de l'algorithme
        self.gamma = gamma
        self.alpha = alpha

        self.eps_profile = eps_profile
        self.epsilon = self.eps_profile.initial

        # Visualisation des données (vous n'avez pas besoin de comprendre cette partie)
        self.qvalues = pd.DataFrame(data={'episode': [], 'score': [], 'Q_sum': []})

        self.fileLog = fileLog

    def getQ(self, state, action):
        return self.Q[state[0]][state[1]][state[2]][action]

    def setQ(self, state, action, value):
        self.Q[state[0]][state[1]][state[2]][action] = value

    def saveQToFile(self,
                    file=os.path.join(os.path.dirname(__file__),
                                      '../LearnedQ/LearnedQ.npy')):
        np.save(file, self.Q)

    def loadQFromFile(self,
                      file=os.path.join(os.path.dirname(__file__),
                                        '../LearnedQ/LearnedQ.npy')):
        self.Q = np.load(file)

    def learn(self, env: SpaceInvaders, n_episodes, max_steps):
        """Cette méthode exécute l'algorithme de q-learning. 
        Il n'y a pas besoin de la modifier. Simplement la comprendre et faire le parallèle avec le cours.

        :param env: L'environnement 
        :type env: gym.Envselect_action
        :param num_episodes: Le nombre d'épisode
        :type num_episodes: int
        :param max_num_steps: Le nombre maximum d'étape par épisode
        :type max_num_steps: int

        # Visualisation des données
        Elle doit proposer l'option de stockage de (i) la fonction de valeur & (ii) la Q-valeur 
        dans un fichier de log
        """
        n_steps = np.zeros(n_episodes) + max_steps

        # Execute N episodes
        for episode in range(n_episodes):
            # Reinitialise l'environnement
            state = env.reset()  # <=
            # Execute K steps
            for step in range(max_steps):
                # Selectionne une action
                action = self.select_action(state)
                # Echantillonne l'état suivant et la récompense
                next_state, reward, terminal = env.step(action)
                # Mets à jour la fonction de valeur Q
                self.updateQ(state, action, reward, next_state)

                if terminal:
                    n_steps[episode] = step + 1
                    break

                state = next_state
            # Mets à jour la valeur du epsilon
            self.epsilon = max(
                self.epsilon - self.eps_profile.dec_episode /
                (n_episodes - 1.), self.eps_profile.final)

            # Sauvegarde et affiche les données d'apprentissage
            if n_episodes >= 0:
                print(
                    "\r#> Ep.: {}/{}    Sum(Q): {}    Curr. Score: {}  ".
                    format(episode, n_episodes - 1, np.sum(self.Q),
                           self.spaceInvaders.score_val),
                    end=" ")
                self.save_log(env, episode)
                state = env.reset()

            # Save Q for for further analysis
            if episode % 1000 == 0:
                self.saveQToFile(
                    os.path.join(
                        os.path.dirname(__file__),
                        '../QEvolution_Q_SXY_E100000_S500_G0.95_I0.7_F0.05',
                        str(episode) + '.npy'))

        self.qvalues.to_csv(
            os.path.join(os.path.dirname(__file__), '../visualisation',
                         self.fileLog + '.csv'))

    def updateQ(self, state, action, reward, next_state):
        """À COMPLÉTER!
        Cette méthode utilise une transition pour mettre à jour la fonction de valeur Q de l'agent. 
        Une transition est définie comme un tuple (état, action récompense, état suivant).
        :param state: L'état origine
        :param action: L'action
        :param reward: La récompense perçue
        :param next_state: L'état suivant
        """

        # If invader reached boarder its y-position is set to a too small value. This one has to be
        # increased to make it at least Y_MAX
        if next_state[1] < Y_MIN:
            next_state[1] = Y_MIN

        val = (1. - self.alpha) * self.getQ(state, action) + self.alpha * (
            reward + self.gamma * np.max(self.Q[next_state]))  # TODO maybe

        # self.Q[state][action] = val
        # self.Q[state[0]][state[1]][state[2]][state[3]][action] = val
        self.setQ(state, action, val)

    def select_action(self, state: int):
        """À COMPLÉTER!
        Cette méthode retourne une action échantilloner selon le processus d'exploration (ici epsilon-greedy).
        :param state: L'état courant
        :return: L'action 
        """
        if np.random.rand() < self.epsilon:
            a = np.random.randint(self.na)  # random action
        else:
            a = self.select_greedy_action(state)

        return a

    def select_greedy_action(self, state: 'Tuple[int, int]'):
        """
        Cette méthode retourne l'action gourmande.

        :param state: L'état courant
        :return: L'action gourmande
        """

        # If invader reached boarder its y-position is set to a too small value. This one has to be
        # increased to make it at least Y_MAX
        if state[1] < Y_MIN:
            state[1] = Y_MIN

        mx = np.max(self.Q[state])
        # greedy action with random tie break
        return np.random.choice(np.where(self.Q[state] == mx)[0])

    def save_log(self, env, episode):
        """Sauvegarde les données d'apprentissage.
        :warning: Vous n'avez pas besoin de comprendre cette méthode
        """

        self.qvalues = self.qvalues.append(
            {
                'episode': episode,
                'score': self.spaceInvaders.score_val,
                'Q_sum': np.sum(self.Q)
            },
            ignore_index=True)
