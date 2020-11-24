# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 14:26:16 2020

@author: charl
"""

import plotly.graph_objects as go
import pandas as pd

impax1 = pd.read_csv("impala_x1/training_data_IMPALAx1.csv")
impax2 = pd.read_csv("impala_x2/training_data_IMPALAx2.csv")
impax4 = pd.read_csv("impala_x4/training_data_IMPALAx4.csv")
nature = pd.read_csv("nature_dqn/training_data_IMPALAx1.csv")

fig = go.Figure()
fig.add_trace(go.Scatter(x=impax1["steps"], y=impax1["rewards"],
                    mode='lines',
                    name='IMPALA x1'))
fig.add_trace(go.Scatter(x=impax2["steps"], y=impax2["rewards"],
                    mode='lines',
                    name='IMPALA x2'))
fig.add_trace(go.Scatter(x=impax4["steps"], y=impax4["rewards"],
                    mode='lines',
                    name='IMPALA x4'))
fig.add_trace(go.Scatter(x=impax1["steps"], y=impax1["rewards"],
                    mode='lines',
                    name='Nature CNN'))

fig.show()