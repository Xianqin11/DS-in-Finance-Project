from dash import Dash, dcc, html, dash_table, Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np

# 假设的投资组合数据
dates = pd.date_range(start="2015-01-01", periods=60, freq='M')
returns = np.random.normal(0.02, 0.05, size=len(dates))
cumulative_returns = np.cumprod(1 + returns) - 1

# 创建一个包含数据的DataFrame
df = pd.DataFrame({
    'Date': dates,
    'Cumulative Returns': cumulative_returns
})

# 初始化Dash应用
app = Dash(__name__)

# 定义应用布局
app.layout = html.Div([
    html.H1('Markowitz Portfolio Optimizer', style={'textAlign': 'center'}),

    # 控件面板
    html.Div([
        html.Label('User name:'),
        dcc.Input(id='username', value='', type='text'),

        html.Label('Amount (EUR):'),
        dcc.Input(id='amount', value='1000', type='number'),

        html.Label('Date'),
        dcc.Input(id='date', value='', type='number'),

        # 更多的控件可以在这里添加
    ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),

    # 图表和表格容器
    html.Div([
        # 图表
        dcc.Graph(id='portfolio-graph', style={'display': 'inline-block', 'width': '70%'}),

        # 表格
        dash_table.DataTable(
            id='portfolio-table',
            columns=[{"name": i, "id": i} for i in df.columns],
            data=df.to_dict('records'),
            style_table={'height': '300px', 'overflowY': 'auto'},
            style_cell={'textAlign': 'left'},
            style_as_list_view=True,
        )
    ], style={'width': '70%', 'display': 'inline-block', 'verticalAlign': 'top'}),
])


# 定义回调函数更新图表
@app.callback(
    Output('portfolio-graph', 'figure'),
    [Input('amount', 'value')]
)
def update_graph(amount):
    # 根据输入值更新图表
    if amount:
        amount = float(amount)
    else:
        amount = 1000

    # 更新图表
    figure = {
        'data': [
            go.Scatter(
                x=df['Date'],
                y=amount * (1 + df['Cumulative Returns']),
                mode='lines+markers'
            )
        ],
        'layout': {
            'title': 'Portfolio Returns Over Time'
        }
    }
    return figure


# 运行服务器
if __name__ == '__main__':
    app.run_server(debug=True)
