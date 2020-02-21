"""
    Sigmaboard, resembling the concept of Tensorboard from Tensorflow, a web-based interactive visualization tool for
        Sigma graphical backend.
"""

import networkx as nx
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from textwrap import dedent as d
import json
from . import Sigma
from .graphical._nodes import *

from dash.dependencies import Input, Output

# External CSS
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


# Node colors for display
# When finding correct node colors, first find if node type is specified, otherwise refer to parent type color
def node2color(node):
    assert isinstance(node, Node), "Argument must be of Node type or one of its children types"

    nodetype2colors = {
        VariableNode: "Brown",
        FactorNode: "DarkSlateBlue",
        WMFN: "Aqua",
        LTMFN: "DodgerBlue",
        WMVN: "Crimson",
        PBFN: "DarkViolet"
    }

    t = node.__class__
    while t not in nodetype2colors.keys():
        t = t.__bases__[0]
    return nodetype2colors[t]


def render(sigma):
    """
        Run the Dash server locally

        :param sigma:   The sigma program to be displayed
    """
    assert type(sigma) is Sigma, "First argument must be a Sigma program"

    # Compute Graph layout and scatter plot trace
    # Testing with different layouts

    # TODO: Find a better node layout algorithm
    # pos = nx.circular_layout(sigma.G)
    pos = nx.kamada_kawai_layout(sigma.G)
    # pos = nx.planar_layout(sigma.G)
    # pos = nx.random_layout(sigma.G)
    # pos = nx.shell_layout(sigma.G)
    # pos = nx.spring_layout(sigma.G)
    # pos = nx.spectral_layout(sigma.G)
    # pos = nx.spiral_layout(sigma.G)

    fig = go.Figure()

    # node scatter trace. Factor node is 0, Variable node is 1
    for node in sigma.G.nodes:
        x, y = pos[node]
        node_type = isinstance(node, VariableNode)

        # hover test
        hovertext = ""
        for key, value in node.pretty_log.items():
            hovertext += str(key) + ":  " + str(value) + "<br>"     # "<br>" for line break in hover text

        fig.add_trace(go.Scatter(
            mode='markers+text',
            x=[x],
            y=[y],
            marker=dict(size=40,
                        line=dict(width=2, color="DarkSlateGrey"),
                        # color="crimson" if node_type == 0 else "LightGreen",
                        color=node2color(node),
                        symbol="square" if node_type == 0 else "circle"),
            hoverinfo="text",
            hovertext=hovertext,
            text=[node.name],                   # Node name
            textposition="bottom center"
        ))

    # edge scatter trace, with invincible middle points
    edge_x, edge_y = [], []
    mid_x, mid_y = [], []
    for edge in sigma.G.edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

        # invincible middle point for each edge to enable hover effect on edge
        xi = (x0 + x1) / 2
        yi = (y0 + y1) / 2
        mid_x.append(xi)
        mid_y.append(yi)

    edge_trace = go.Scatter(x=edge_x, y=edge_y,
                            line=dict(width=0.5, color='black'),
                            hoverinfo='none',
                            mode='lines')
    mid_trace = go.Scatter(x=mid_x, y=mid_y,
                           mode='markers',
                           hoverinfo='text',
                           hovertext='edge',
                           marker=dict(size=200, color="LightSkyBlue"),  # invincible marker
                           opacity=0)

    fig.add_trace(edge_trace)
    fig.add_trace(mid_trace)

    # Arrows indicating edge direction to be insert in layout's annotation
    edge_arrow = [dict(
        ax=(pos[edge[0]][0] + pos[edge[1]][0]) / 2,
        ay=(pos[edge[0]][1] + pos[edge[1]][1]) / 2,
        axref='x',
        ayref='y',
        x=(pos[edge[0]][0] + pos[edge[1]][0] * 5) / 6,
        y=(pos[edge[0]][1] + pos[edge[1]][1] * 5) / 6,
        xref='x',
        yref='y',
        showarrow=True,
        arrowhead=3,
        arrowsize=4,
        arrowwidth=1
        # opacity=1,
    ) for edge in sigma.G.edges]

    # Set scatter plot layout. Use annotation to display arrowhead
    fig.update_layout(go.Layout(
        showlegend=False,
        hovermode='closest',
        margin={'b': 40, 'l': 40, 'r': 40, 't': 40},
        xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
        yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
        height=800,
        clickmode='event+select',
        annotations=edge_arrow
    ))

    # Dash init
    app = dash.Dash(__name__,
                    external_stylesheets=external_stylesheets)

    app.title = "SigmaBoard"

    # Global style
    # styles = {
    #     'pre': {
    #         'border': 'thin lightgrey solid',
    #         'overflowX': 'scroll'
    #     }
    # }

    # HTML elements and layout
    app.layout = html.Div(children=[
        # Title
        html.Div(id="header", children=[
            html.H1('Sigmaboard')
        ]),


        # Main Div
        html.Div(className="row", children=[

            # Left side console section
            html.Div(className="four columns", children=[
                # Top left search and filter section
                html.Div(className="sections", id='search-section', children=[
                    dcc.Markdown(d("""
                            ### Search Node
                    """)),
                    dcc.Input(type="text", className="search-term", id="node_name_search", placeholder="Node Name Contains... (eg ALPHA)", value=""),
                    dcc.Input(type="text", className="search-term", id="variable_search", placeholder="Contains Variable Name... (eg arg_1)", value=""),
                    dcc.Checklist(id="node-type",
                        options=[
                            {'label': 'Variable Node', 'value': 'VN'},
                            {'label': 'Function Node', 'value': 'FN'},
                            {'label': 'Predicate', 'value': 'PRED'},
                            {'label': 'Conditional', 'value': 'COND'},
                        ],
                        value=['VN', 'FN', 'PRED', 'COND']
                    )
                ]),
                # Middle left console display section
                html.Div(className="sections", id="display-section", children=[
                    dcc.Markdown(d("""
                            ### Click nodes to display attributes
                        """)),
                    html.Pre(id='click-data', className="pop_up", children="None")
                ]),
                # Bottom left link message memory
                html.Div(className="sections", id='message-section', children=[
                    dcc.Markdown(d("""
                            ### link message memory
                        """))
                ])
            ]),

            # Main canvas for displaying factor graph
            html.Div(className="eight columns", children=[
                dcc.Graph(
                    className="col-9",
                    id='graph',
                    figure=fig,
                    # style={"height": '80vh'}
                )
            ])
        ])

    ])

    @app.callback(
        Output('click-data', 'children'),
        [Input('graph', 'clickData')])
    def display_click_data(clickData):
        if clickData is not None and 'text' in clickData['points'][0]:
            node_name = clickData["points"][0]["text"]
            info = ""
            for n in sigma.G.nodes:
                if str(n) == node_name:
                    # hover test
                    for key, value in node.pretty_log.items():
                        info += str(key) + ":  " + str(value) + "\n"
            return info


    @app.callback(
        Output('graph', 'figure'),
        [Input('node_name_search', 'value'),
         Input('variable_search', 'value'),
         Input('node-type', 'value')])
    def update_figure(node_name, variables, node_type):
        if node_name is not None and variables is not None:
            for n in fig['data']:
                if n.text is None:
                    continue
                # Here to define the search
                node_type_graph = 'FN' if n['marker']['symbol'] == 'square' else 'VN'
                pred_or_cond = n.text[0][:4]

                v_list = variables.split(',')

                node_info = n.hovertext.split('<br>')
                for attri in node_info:
                    a = attri.split("  ")
                    if 'variable names' in a[0] or 'all variables' in a[0]:
                        variable_name_list = eval(a[1])

                if node_name.upper() in n.text[0].upper() and \
                        v_list[0] == '' or any(v in variable_name_list for v in v_list) and \
                        node_type_graph in node_type and \
                        pred_or_cond in node_type:
                    n['opacity'] = 1
                else:
                    n['opacity'] = 0.2
            return fig
        return fig

    app.run_server(debug=True)
