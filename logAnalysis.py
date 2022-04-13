import os
import plotly.express as px
import pandas as pd


class logAnalysis:
    def __init__(self, file: str):
        self.file = file
        self.values = []

    def printCurves(self):
        df = pd.read_csv(self.file)
        fig = px.scatter(x=df["episode"], y=df["score"])
        fig.show()


if __name__ == '__main__':
    log = logAnalysis(
<<<<<<< HEAD
        os.path.join(os.path.dirname(__file__), 'visualisation/Q_E1000_S5000_G0.95_I1.0_F0.1.csv'))
=======
        os.path.join(os.path.dirname(__file__), 'visualisation/logQ.csv'))
>>>>>>> Significantly shrink state space by removing invader y coordinate from state
    log.printCurves()
