import os
import plotly.express as px
import pandas as pd


class logAnalysis:

	def __init__(self, file : str):
		self.file = file
		self.values = []

	def printCurves(self):
		df = pd.read_csv(os.path.join(os.path.dirname(__file__),
                         'visualisation/logQ.csv'))
		fig = px.scatter(x=df["episode"], y=df["value"])
		fig.show()

if __name__ == '__main__':
	log = logAnalysis("")
	log.printCurves()