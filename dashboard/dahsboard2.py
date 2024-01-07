import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc  # 需要先安装: pip install dash-bootstrap-components
import plotly.graph_objs as go
import pandas as pd
import numpy as np

# 假设的投资组合数据
dates = pd.date_range(start="2015-01-01", periods=60, freq='M')
returns = np.random.normal(0.02, 0.05, size=len(dates))
cumulative_returns = np.cumprod(1 + returns) - 1

# 初始化Dash应用，添加Bootstrap样式
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# 定义应用布局
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1('Markowitz Portfolio Optimizer'), width={'size': 6, 'offset': 3}, className="mb-4 mt-4")),

    dbc.Row([
        dbc.Col([
            html.Label('User name:'),
            dcc.Input(id='username', value='', type='text', className="mb-2"),

            html.Label('Amount (EUR):'),
            dcc.Input(id='amount', value='1000', type='number', className="mb-2"),

            html.Label('Risk Profile:'),
            dcc.Slider(id='risk-profile', min=0, max=1, step=0.1, value=0.5,
                       marks={i / 10: str(i / 10) for i in range(0, 11)}, className="mb-2"),

            html.Label('Asset Mix:'),
            dcc.Dropdown(id='asset-mix', options=[
                {'label': 'Aggressive', 'value': 'AGG'},
                {'label': 'Balanced', 'value': 'BAL'},
                {'label': 'Conservative', 'value': 'CON'}
            ], value='BAL', className="mb-2"),

            # 更多的控件可以在这里添加
        ], width=4),

        dbc.Col([
            dcc.Graph(id='portfolio-graph')
        ], width=8),
    ])
], fluid=True)


# 定义回调函数更新图表
@app.callback(
    Output('portfolio-graph', 'figure'),
    [Input('amount', 'value'), Input('risk-profile', 'value'), Input('asset-mix', 'value')]
)
def update_graph(amount, risk_profile, asset_mix):
    # 根据输入值更新图表
    if amount:
        amount = float(amount)
    else:
        amount = 1000

    # 这里可以根据risk_profile和asset_mix添加逻辑来调整回报率和风险

    # 更新图表
    figure = go.Figure(data=[
        go.Scatter(x=dates, y=amount * (1 + cumulative_returns), mode='lines+markers')
    ])
    figure.update_layout(title='Portfolio Returns Over Time')
    return figure


# 运行服务器
if __name__ == '__main__':
    app.run_server(debug=True)

