"""
    Sigmaboard, resembling the concept of Tensorboard from Tensorflow, a web-based interactive visualization tool for
        Sigma graphical backend.
"""

import networkx as nx
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go

from graphical import LTMFN, PBFN, WMFN
from . import Sigma
from .graphical.basic_nodes import *

from dash.dependencies import Input, Output


class SigmaBoard:
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css',
                                 'https://code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css']

    external_scripts = ["https://code.jquery.com/jquery-1.12.4.js",
                             "https://code.jquery.com/ui/1.12.1/jquery-ui.js"]

    def __init__(self, sigma):
        """
            Initialize SigmaBoard

            :param sigma:   The sigma program to be displayed
        """
        assert type(sigma) is Sigma, "First argument must be a Sigma program"
        self.click_data_info = ""
        self.variable_node_list = []
        self.factor_node_list = []
        self.PBFN_list = []
        self.LTMFN_list = []
        self.WMFN_list = []
        self.ACFN_list = []
        self.FFN_list = []
        self.NLFN_list = []
        self.ADFN_list = []
        self.ATFN_list = []
        self.BJFN_list = []
        self.GFFN_list = []
        self.WMVN_list = []
        self.G = sigma.G
        self.fig = None
        self.html_layout = None
        self.__init_fig()
        self.__init_html_layout()
        self.__init_node_list()


    # Node colors for display
    # When finding correct node colors, first find if node type is specified, otherwise refer to parent type color
    def __node2color(self, node):
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

    def __init_fig(self):
        """
            Create the scattered plot representing the factor node

        """
        # Compute Graph layout and scatter plot trace
        # Testing with different layouts

        # TODO: Find a better node layout algorithm
        # pos = nx.circular_layout(self.G)
        pos = nx.kamada_kawai_layout(self.G)
        # pos = nx.planar_layout(self.G)
        # pos = nx.random_layout(self.G)
        # pos = nx.shell_layout(self.G)
        # pos = nx.spring_layout(self.G)
        # pos = nx.spectral_layout(self.G)
        # pos = nx.spiral_layout(self.G)

        fig = go.Figure()

        # edge scatter trace, with invincible middle points
        edge_x, edge_y = [], []
        mid_x, mid_y = [], []
        for edge in self.G.edges:
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
        ) for edge in self.G.edges]

        # node scatter trace. Factor node is 0, Variable node is 1
        for node in self.G.nodes:
            x, y = pos[node]
            node_type = isinstance(node, VariableNode)

            # hover test
            hovertext = ""
            for key, value in node.pretty_log.items():
                hovertext += str(key) + ":  " + str(value) + "<br>"  # "<br>" for line break in hover text

            fig.add_trace(go.Scatter(
                mode='markers+text',
                x=[x],
                y=[y],
                marker=dict(size=40,
                            line=dict(width=2, color="DarkSlateGrey"),
                            # color="crimson" if node_type == 0 else "LightGreen",
                            color=self.__node2color(node),
                            symbol="square" if node_type == 0 else "circle"),
                hoverinfo="text",
                hovertext=hovertext,
                text=[node.name],  # Node name
                textposition="bottom center"
            ))

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

        self.fig = fig

    def __init_html_layout(self):
        """
            Initialize HTML layout

        """
        # HTML elements and layout
        self.html_layout = html.Div(children=[
            # Title
            html.Div(id="header", children=[
                html.H1('Sigmaboard')
            ]),

            # Main Div
            html.Div(className="row", children=[

                # Left side console section
                html.Div(className="four columns column", children=[
                    # Top left search and filter section
                    html.Div(className="portlet sections", id='search-section', children=[
                        html.Div(className="portlet-header", children=[
                            html.H3("Search Node")
                        ]),
                        # dcc.Markdown(d("""
                        #         ### Search Node
                        # """)),
                        html.Div(className="portlet-content", children=[
                            html.Div(className="wrap-search", children=[
                                html.Div(className="wrap-search-bar", children=[
                                    dcc.Input(type="text", className="search-term", id="node_name_search",
                                              placeholder="Node Name Contains... (eg ALPHA)", value=""),
                                    dcc.Input(type="text", className="search-term", id="variable_search",
                                              placeholder="Contains Variable Name... (eg arg_1)", value=""),
                                ]),
                                html.Div(className="wrap-checklist", children=[
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
                                dcc.Dropdown(
                                    options=self.variable_node_list,
                                    multi=True,
                                    placeholder="select a variable node",
                                ),
                                dcc.Dropdown(
                                    options=self.factor_node_list,
                                    multi=True,
                                    placeholder="select a factor node",
                                ),
                                dcc.Dropdown(
                                    options=self.PBFN_list,
                                    multi=True,
                                    placeholder="select a PBFN node",
                                ),
                                dcc.Dropdown(
                                    options=self.LTMFN_list,
                                    multi=True,
                                    placeholder="select a LTMF node",
                                ),
                                dcc.Dropdown(
                                    options=self.WMFN_list,
                                    multi=True,
                                    placeholder="select a WMFN node",
                                ),
                                dcc.Dropdown(
                                    options=self.ACFN_list,
                                    multi=True,
                                    placeholder="select a ACFN node",
                                ),
                                dcc.Dropdown(
                                    options=self.FFN_list,
                                    multi=True,
                                    placeholder="select a FFN node",
                                ),
                            ]),
                            html.Div(className="clear-float")
                        ]),
                    ]),
                    # Middle left console display section
                    html.Div(className="sections portlet", id="display-section", children=[
                        html.Div(className="portlet-header", children=[
                            html.H3("Click nodes to display attributes")
                        ]),
                        # dcc.Markdown(d("""
                        #         ### Click nodes to display attributes
                        #     """)),
                        html.Div(className="portlet-content", id='click-data-wrap', children=[
                            html.Pre(id='click-data', className="pop_up", children="None")
                        ])
                    ]),
                    # Bottom left link message memory
                    # html.Div(className="sections portlet", id='message-section', children=[
                    #     dcc.Markdown(d("""
                    #             ### link message memory
                    #         """))
                    # ])
                ]),

                # Main canvas for displaying factor graph
                html.Div(className="eight columns", children=[
                    dcc.Graph(
                        className="col-9",
                        id='graph',
                        figure=self.fig,
                        # style={"height": '80vh'}
                    )
                ])
            ])

        ])

    def __init_node_list(self):
        """
            Classify Sigma Nodes into categories

        """
        for node in self.G.nodes:
            if isinstance(node, VariableNode):
                self.variable_node_list.append({'label': node.name, 'value': node.name})
            else:
                self.factor_node_list.append({'label': node.name, 'value': node.name})
            if isinstance(node, PBFN):
                self.PBFN_list.append({'label': node.name, 'value': node.name})
            if isinstance(node, LTMFN):
                self.LTMFN_list.append({'label': node.name, 'value': node.name})
            if isinstance(node, WMFN):
                self.WMFN_list.append({'label': node.name, 'value': node.name})
            if isinstance(node, ACFN):
                self.ACFN_list.append({'label': node.name, 'value': node.name})
            if isinstance(node, FFN):
                self.FFN_list.append({'label': node.name, 'value': node.name})
            if isinstance(node, NLFN):
                self.NLFN_list.append({'label': node.name, 'value': node.name})
            if isinstance(node, ADFN):
                self.ADFN_list.append({'label': node.name, 'value': node.name})
            if isinstance(node, ATFN):
                self.ATFN_list.append({'label': node.name, 'value': node.name})
            if isinstance(node, BJFN):
                self.BJFN_list.append({'label': node.name, 'value': node.name})
            if isinstance(node, GFFN):
                self.GFFN_list.append({'label': node.name, 'value': node.name})
            if isinstance(node, WMVN):
                self.WMVN_list.append({'label': node.name, 'value': node.name})


    def render(self):
        """
            Run the Dash server locally

        """
        app = dash.Dash(__name__,
                        external_scripts=self.external_scripts,
                        external_stylesheets=self.external_stylesheets)

        app.title = "SigmaBoard"
        app.layout = self.html_layout

        @app.callback(
            Output('click-data', 'children'),
            [Input('graph', 'clickData')])
        def display_click_data(clickData):
            if clickData is not None and 'text' in clickData['points'][0]:
                node_name = clickData["points"][0]["text"]
                info = node_name + "\n"
                for n in self.G.nodes:
                    if str(n) == node_name:
                        for key, value in n.pretty_log.items():
                            info += "\t" + str(key) + ":  " + str(value) + "\n"
                        self.click_data_info += info + "\n"
                        return self.click_data_info
            return self.click_data_info


        @app.callback(
            Output('graph', 'figure'),
            [Input('node_name_search', 'value'),
             Input('variable_search', 'value'),
             Input('node-type', 'value')])
        def update_figure(node_name, variables, node_type):
            if node_name is not None and variables is not None:
                for n in self.fig['data']:
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
                            (v_list[0] == '' or any(v in variable_name_list for v in v_list)) and \
                            node_type_graph in node_type and \
                            pred_or_cond in node_type:
                        n['opacity'] = 1
                    else:
                        n['opacity'] = 0.2
                return self.fig
            return self.fig

        app.run_server(debug=True)
