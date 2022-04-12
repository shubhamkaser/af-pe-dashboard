import plotly.express as px
from misc import reverse_list
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


config = {'colors' : ["#002244", "#ff0066", "#66cccc", "#ff9933", "#337788",
          "#429e79", "#474747", "#f7d126", "#ee5eab", "#b8b8b8"], # Color palette for visualizations
'color_axis' : '#d62728', # Color for axis on residuals chart and scatter plot
'waterfall_digits' : 2}

def multiple_line_plot(df_act_for, value_vars, id_vars):
    df_melt = df_act_for.melt(id_vars=id_vars, value_vars=value_vars)
    fig_multi_line = px.line(df_melt, x=id_vars , y='value' , color='variable', title='Actual vs Forecast')
    fig_multi_line.update_layout(height = 600, width = 1400,title={
        'text': "Actual vs Forecast",
        'y':0.9,
        'x':0.5,
        'font': {'size':22},
        'xanchor': 'center',
        'yanchor': 'top'})
    return(fig_multi_line)

def plot_forecasts_vs_truth(df_act_for, metric):
    """Creates a plotly line plot showing forecasts and actual values on evaluation period.
    Returns
    -------
    go.Figure
        Plotly line plot showing forecasts and actual values on evaluation period.
    """
    fig = px.line(
            df_act_for,
            x="date",
            y=["act_"+metric, metric],
            color_discrete_sequence=config["colors"][1:],
            hover_data={"variable": True, "value": ":.2f", "date": False},
        )
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all"),
                ]
            )
        ),
    )
    fig.update_layout(
        yaxis_title=metric,
        legend_title_text="",
        height=500,
        width=1400,
        title_text="Forecast vs Actuals",
        title_x=0.5,
        title_y=1,
        hovermode="x unified",
    )
    return(fig)

def plot_error_rate(df_act_for, metric):
    df_act_for['error'] = df_act_for['act_'+metric] - df_act_for[metric]
    fig = px.line(
        df_act_for,
        x="date",
        y="error",
        color_discrete_sequence=config["colors"][3:],
        hover_data={"error": ":.2f", "date": False},
    )
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all"),
                ]
            )
        ),
    )
    fig.update_layout(
        yaxis_title=metric,
        legend_title_text="",
        height=500,
        width=1400,
        title_text="Error (Actual - Forecast)",
        title_x=0.5,
        title_y=1,
        hovermode="x unified",
    )
    return(fig)