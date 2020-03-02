"""
    Dash web-based visualization test
"""


import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from textwrap import dedent as d
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

    # node scatter trace
    for node in G.nodes:
        x, y = pos[node]
        node_type = G.nodes[node]['bipartite']

        # Using scatter instead of layout.shapes for support for hover effect
        # if node_type == 0:
        #     fig.add_shape(go.layout.Shape(
        #         type='rect',
        #         x0=x-0.05,
        #         y0=y-0.05,
        #         x1=x+0.05,
        #         y1=y+0.05,
        #         line=dict(color="crimson"),
        #         fillcolor="crimson"
        #     ))
        #     # fig.add_trace(go.Scatter(
        #     #     mode='lines',
        #     #     x=[x-0.05, x-0.05, x+0.05, x+0.05, x-0.05],
        #     #     y=[y-0.05, y+0.05, y+0.05, y-0.05, y-0.05],
        #     #     fill='toself',
        #     #     fillcolor="crimson",
        #     #     marker=dict(color="crimson"),
        #     #     hoverinfo="text",
        #     #     hovertext="Factor Node"
        #     # ))
        # else:
        #     fig.add_trace(go.Scatter(
        #         mode='markers',
        #         x=[x],
        #         y=[y],
        #         marker=dict(size=40, color="LightGreen", symbol="circle"),
        #         hoverinfo="text",
        #         hovertext="Variable Node"
        #     ))

        fig.add_trace(go.Scatter(
            mode='markers+text',
            x=[x],
            y=[y],
            marker=dict(size=40,
                        line=dict(width=2, color="DarkSlateGrey"),
                        color="crimson" if node_type == 0 else "LightGreen",
                        symbol="square" if node_type == 0 else "circle"),
            hoverinfo="text",
            hovertext="Factor Node" if node_type == 0 else "Variable Node",
            text=[("Factor_" if node_type == 0 else "Variable_") + str(node)],
            textposition="bottom center"
        ))

    # edge scatter trace, with invincible middle points
    edge_x, edge_y = [], []
    mid_x, mid_y = [], []
    for edge in G.edges:
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
                           marker=dict(size=200, color="LightSkyBlue"),     # invincible marker
                           opacity=0)

    fig.add_trace(edge_trace)
    fig.add_trace(mid_trace)

    # Arrows indicating edge direction to be insert in layout's annotation
    edge_arrow = [dict(
        ax=(pos[edge[0]][0] + pos[edge[1]][0]) / 2,
        ay=(pos[edge[0]][1] + pos[edge[1]][1]) / 2,
        axref='x',
        ayref='y',
        x=(pos[edge[0]][0] * 5 + pos[edge[1]][0]) / 6,
        y=(pos[edge[0]][1] * 5 + pos[edge[1]][1]) / 6,
        xref='x',
        yref='y',
        showarrow=True,
        arrowhead=3,
        arrowsize=4,
        arrowwidth=1,
        opacity=1
    ) for edge in G.edges]

    # Set scatter plot layout
    fig.update_layout(go.Layout(
        showlegend=False,
        hovermode='closest',
        margin={'b': 40, 'l': 40, 'r': 40, 't': 40},
        xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
        yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
        height=600,
        clickmode='event+select',
        annotations=edge_arrow
    ))

    # Dash init
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    # Using bootstrap css
    # external_stylesheets = [dict(
    #     href="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css",
    #     rel="stylesheet",
    #     integrity="sha384-Vkoo8x4CGsO3+Hhxv8T/Q5PaXtkKtu6ug5TOeNV6gBiFeWPGFN9MuhOf23Q9Ifjh",
    #     crossorigin="anonymous"
    # )]

    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    # Global style
    styles = {
        'title': {
            'textAlign': 'center',
        },
        'pre': {
            'border': 'thin lightgrey solid',
            'overflowX': 'scroll'
        }
    }

    # HTML elements and layout
    app.layout = html.Div(children=[
        # Title
        html.H1('Sigmaboard',
                style=styles['title']),

        # Main Div
        html.Div(className="row", children=[

            # Left side console section
            html.Div(className="three columns", children=[
                # Top left search and filter section
                html.Div(className="twelve columns", children=[

                ]),
                # Bottom left console display section
                html.Div(className="twelve columns", children=[
                    dcc.Markdown(d("""
                        ### Click nodes to display attributes
                    """)),
                    html.Pre(id='click-data', style=styles['pre'], children="None")
                ])
            ]),

            # Main canvas for displaying factor graph
            html.Div(className="nine columns", children=[
                dcc.Graph(
                    className="col-9",
                    id='graph',
                    figure=fig,
                    style={"height": '80vh'}

                )
            ])
        ])


    ])

    app.run_server(debug=True)


if __name__ == '__main__':
    # test_1()
    test_2()
