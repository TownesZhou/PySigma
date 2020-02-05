"""
    Sigmaboard, resembling the concept of Tensorboard from Tensorflow, a web-based interactive visualization tool for
        Sigma graphical backend.
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from Cognitive import *
from Graphical import *


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


def render(sigma):
    """
        Run the Dash server locally

        :param sigma:   The sigma program to be displayed
    """
    assert type(sigma) is Sigma, "First argument must be a Sigma program"

    # Dash init
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    # Global style
    style = {
        'title': {
            'textAlign': 'center',
        }
    }

    # HTML elements and layout
    app.layout = html.Div([
        # Title
        html.H1('Sigmaboard: Interactive Visualization tool for Sigma Graphical Backend',
                style=style['title']),

        # Main canvas for displaying factor graph
        dcc.Graph(
            id='graph',
            figure={
                'data': None,
                'layout': {
                    'clickmode': 'event+select'
                }
            },
        )

    ])


    app.run_server(debug=True)