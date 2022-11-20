import pandas as pd
from numpy import zeros, float64, size

def initialize(states, actions):
    global q_table
    q_table = pd.DataFrame((zeros((states,size(actions)))),
				columns=actions, dtype=float64)