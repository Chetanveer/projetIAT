import os
import plotly.express as px
import pandas as pd


class logAnalysis:
    def __init__(self, file: str):
        self.file = file
        self.values = []

    def printCurves(self):
        df = pd.read_csv(self.file)
        fig1 = px.scatter(x=df["episode"], y=df["score"])
        fig2 = px.scatter(x=df["episode"], y=df["Q_sum"])
        fig1.show()
        fig2.show()


if __name__ == '__main__':
    log = logAnalysis(
        os.path.join(os.path.dirname(__file__), 'visualisation/Q_SXY_E200000_S500_G0.95_I0.7_F0.05.csv'))
    log.printCurves()
