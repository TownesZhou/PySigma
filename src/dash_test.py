"""
    Dash web-based visualization test
"""


import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import networkx as nx


def test_1():

    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    colors = {
        'background': '#111111',
        'text': '#7FDBFF'
    }

    app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
        html.H1(children='Hello World!',
                style={
                    'textAlign': 'center',
                    'color': colors['text']
                }),

        html.Div(children='''
            Dash: A web application framework for Python.
        ''',
                style={
                    'textAlign': 'center',
                    'color': colors['text']
                }),

        dcc.Graph(
            id='example-graph-2',
            figure={
                'data': [
                    {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                    {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
                ],
                'layout': {
                    'title': 'Dash Data Visualization',
                    'plot_bgcolor': colors['background'],
                    'paper_bgcolor': colors['background'],
                    'font': {
                        'color': colors['text']
                        }
                }
            }
        )
    ])

    app.run_server(debug=True)


def test_2():

    # Sample directed bipartite graph
    G = nx.DiGraph()
    G.add_nodes_from([1, 2, 3, 4], bipartite=0)
    G.add_nodes_from(['a', 'b', 'c'], bipartite=1)
    G.add_edges_from([(1, 'a'), (1, 'b'), (2, 'b'), (2, 'c'), (3, 'c'), (4, 'a')])

    # layout position
    pos = nx.planar_layout(G)

    # Create figure
    fig = go.Figure()

    # node and edge trace
    for node in G.nodes:
        x, y = pos[node]
        # shape= go.layout.Shape(
        #         type='rect' if G.nodes[node]['bipartite'] == 0 else 'circle',
        #         x0=x-0.05,
        #         y0=y-0.05,
        #         x1=x+0.05,
        #         y1=y+0.05,
        #         fillcolor="crimson" if G.nodes[node]['bipartite'] == 0 else 'LightGreen',
        #         hoverinfo=node
        #     )
        # fig.add_shape(shape)
        node_type = G.nodes[node]['bipartite']
        if node_type == 0:
            fig.add_trace(go.Scatter(
                mode='lines',
                x=[x-0.05, x-0.05, x+0.05, x+0.05, x-0.05],
                y=[y-0.05, y+0.05, y+0.05, y-0.05, y-0.05],
                fill='toself',
                fillcolor="crimson",
                marker=dict(color="crimson"),
                hoverinfo="text",
                hovertext="Factor Node"
            ))
        else:
            fig.add_trace(go.Scatter(
                mode='markers',
                x=[x],
                y=[y],
                marker=dict(size=40, color="LightGreen"),
                hoverinfo="text",
                hovertext="Variable Node"
            ))


    edge_x, edge_y = [], []
    for edge in G.edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(x=edge_x, y=edge_y,
                            line=dict(width=0.5, color='#888'),
                            hoverinfo='none',
                            mode='lines')
    fig.add_trace(edge_trace)


    # Dash init
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

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
        html.Div([
            dcc.Graph(
                id='graph',
                figure=fig,
                style={"height": '80vh'}

            )
        ])


    ])

    app.run_server(debug=True)


if __name__ == '__main__':
    # test_1()
    test_2()
