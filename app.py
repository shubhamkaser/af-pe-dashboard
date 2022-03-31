from audioop import mul
from re import A
from statistics import mode
# from turtle import width
from matplotlib.backends.backend_agg import RendererAgg
import streamlit as st
import numpy as np
import pandas as pd
from pandas import json_normalize
import urllib.request
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image
# from streamlit_lottie import st_lottie
import requests
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import plotly.express as px
from plotter import multiple_line_plot
from plotter import plot_forecasts_vs_truth
from plotter import plot_error_rate
import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


st.set_page_config(page_title="Model Performance Evaluation",layout="wide")
st.write("""
# Model Performance Evaluation Dashboard
""")
st.markdown('''
## Overview
''')

forecast_all = pd.read_csv('delivered_aggregated.csv')
actuals_all = pd.read_csv('Cable One Hist Data Combined 2022-03-15.csv')

# Wrangling forecast
forecast_all.date = pd.to_datetime(forecast_all.date)
forecast_all = forecast_all.sort_values(by=['date','series_id']).reset_index(drop=True)
# Wrangling actuals
actuals_all['Tactic'] = actuals_all['Tactic'].replace('NonBrand','Nonbrand')
actuals_all['series_id'] = actuals_all['Brand'] + '|' + actuals_all['Channel'] + '|' + actuals_all['Tactic']
actuals_all = actuals_all[['Date','series_id','Cost','Consolidated_Orders','Impressions','Clicks']]
actuals_all.Date = pd.to_datetime(actuals_all.Date)
actuals_all = actuals_all.groupby(by=['Date','series_id']).sum().reset_index()
actuals_all.columns = ['act_'+x for x in forecast_all.columns.tolist()]
actuals_all.act_date = pd.to_datetime(actuals_all.act_date)
actuals_all = actuals_all.sort_values(by=['act_date','act_series_id']).reset_index(drop=True)

#merge for the final data
df_act_for_all = forecast_all.merge(actuals_all[actuals_all['act_date'].isin(forecast_all.date)], left_on = ['date','series_id'],
             right_on=['act_date','act_series_id'], how='left')
df_act_for_all.drop(columns=['act_date','act_series_id'], inplace=True)
#Confirm with Linda
df_act_for_all=df_act_for_all[~df_act_for_all['series_id'].str.endswith('FP')]

series_id = st.selectbox('Select Series ID', df_act_for_all.series_id.unique())
df_act_for = df_act_for_all[df_act_for_all['series_id'] == series_id]

#Plot Actual vs Forecast

fig_act_vs_forecast = multiple_line_plot(df_act_for, value_vars=['cost','act_cost'], id_vars='date')
st.plotly_chart(fig_act_vs_forecast)

# Global Performance

st.write('''
### Monthly Performance
''')

def get_eval_df(metric):
  eval_dict = {}
  for i in df_act_for.series_id.unique():
    series_dict = {}
    for month in range(df_act_for.date.dt.month.min(), df_act_for.date.dt.month.max()+1):
        series_df = df_act_for[df_act_for['series_id'] == i]
        series_df = series_df[series_df['date'].dt.month == month]
        series_df = series_df.dropna()
        if series_df.shape[0]!=0:
            y_true = series_df['act_'+metric]
            y_pred = series_df[metric]
            series_eval_dict= {}
            series_eval_dict['r2'] = r2_score(y_true, y_pred)
            series_eval_dict['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            series_eval_dict['mse'] = mean_squared_error(y_true, y_pred)
            series_eval_dict['mae'] = mean_absolute_error(y_true, y_pred)
            month = dt.date(1900, month, 1).strftime('%B')
            series_dict[month] = series_eval_dict
        else:
            pass
    eval_dict[i] = series_dict
  reform = {(outerKey, innerKey): values for outerKey, innerDict in eval_dict.items() for innerKey, values in innerDict.items()}
  eval_df = pd.DataFrame(reform).T
  return(eval_df)

metric_df = get_eval_df('cost')

metric_df = metric_df.reset_index()
metric_df.columns = ['series_id','month','r2','rmse','mse','mae']

config = {'colors' : ["#002244", "#ff0066", "#66cccc", "#ff9933", "#337788",
          "#429e79", "#474747", "#f7d126", "#ee5eab", "#b8b8b8"], # Color palette for visualizations
'color_axis' : '#d62728', # Color for axis on residuals chart and scatter plot
'waterfall_digits' : 2}

eval_all = {'metrics': ['R2','RMSE', 'MSE', 'MAE']}

month = st.selectbox('Select Month', metric_df.month.unique())
metric_df_filtered = metric_df[(metric_df['series_id'] == series_id) &
          (metric_df['month'] == month)]

col1, col2, col3, col4 = st.columns(4)
col1.markdown(
    f"<p style='color: {config['colors'][1]}; "
    f"font-weight: bold; font-size: 20px;'> {eval_all['metrics'][0]}</p>",
    unsafe_allow_html=True,
)
col1.markdown(metric_df_filtered[(metric_df_filtered['series_id'] == series_id) &
          (metric_df_filtered['month'] == month)]['r2'].values[0])
col2.markdown(
    f"<p style='color: {config['colors'][1]}; "
    f"font-weight: bold; font-size: 20px;'> {eval_all['metrics'][1]}</p>",
    unsafe_allow_html=True,
)
col2.markdown(metric_df_filtered[(metric_df_filtered['series_id'] == series_id) &
          (metric_df_filtered['month'] == month)]['rmse'].values[0])
col3.markdown(
    f"<p style='color: {config['colors'][1]}; "
    f"font-weight: bold; font-size: 20px;'> {eval_all['metrics'][2]}</p>",
    unsafe_allow_html=True,
)
col3.markdown(metric_df_filtered[(metric_df_filtered['series_id'] == series_id) &
          (metric_df_filtered['month'] == month)]['mse'].values[0])
col4.markdown(
    f"<p style='color: {config['colors'][1]}; "
    f"font-weight: bold; font-size: 20px;'> {eval_all['metrics'][3]}</p>",
    unsafe_allow_html=True,
)
col4.markdown(metric_df_filtered[(metric_df_filtered['series_id'] == series_id) &
          (metric_df_filtered['month'] == month)]['mae'].values[0])


st.write('''
### Deep Dive
''')
filtered_metric = metric_df[metric_df['series_id'] == series_id]
metrics = ['r2','rmse','mse','mae']
if len(metrics) > 0:
    fig = make_subplots(
        rows=len(metrics) // 2 + len(metrics) % 2, cols=2, subplot_titles=metrics
    )
    for i, metric in enumerate(metrics):
        colors = (
            [config["colors"][i % len(config["colors"])]]
            * 12
        )
        fig_metric = go.Bar(
            x=filtered_metric['month'], y=filtered_metric[metric], marker_color=colors
        )
        fig.append_trace(fig_metric, row=i // 2 + 1, col=i % 2 + 1)
    fig.update_layout(
        height=300 * (len(metrics) // 2 + len(metrics) % 2),
        width=1400,
        showlegend=False,
    )
st.plotly_chart(fig)

st.markdown('''
### Error Analysis
''')

forecasts_vs_truth = plot_forecasts_vs_truth(df_act_for)
st.plotly_chart(forecasts_vs_truth)

error_rate = plot_error_rate(df_act_for)
st.plotly_chart(error_rate)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 