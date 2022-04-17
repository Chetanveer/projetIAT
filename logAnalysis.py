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
        os.path.join(os.path.dirname(__file__), 'visualisation/Q_SXY_E50000_S500_G0.95_I0.95_F0.1_Alpha0.8.csv'))
    log.printCurves()
