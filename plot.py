'''
CSE 163 Final Project
Alex Hayes
Sedona Munguia
Creates data visualizations
'''


import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from Clean_Data import data


def resample_data(data):
    """
    Takes in data and returns the data grouped by city
    resampled by month and averaged
    """

    data.index = pd.to_datetime(data['Date'])
    df = data.groupby('city')
    cities = list(df.groups.keys())
    df = df.resample('M').mean()
    df['Vol Per Capita'] = df['Total Volume'] / df['population']

    return (df, cities)


def plotting(df, cities, col, scale):
    """
    Takes in dataframe, cities, column name, and scale to plot and
    creates a bubble map that changes with time, with bubble
    size correlating to the column given divided by scale
    """

    length = len(df.loc[cities[0]])
    dates = pd.period_range('2015-01-31', periods=length, freq='M')
    df['text'] = col + ': ' + (df[col]).astype(str)
    limits = [(i, i+1) for i in range(length)]

    fig = go.Figure()

    for i in range(len(limits)):
        lim = limits[i]
        df_sub = df.loc[cities[0]][lim[0]:lim[1]]

        for city in cities[1:]:
            df_sub = df_sub.append(df.loc[city][lim[0]:lim[1]])

        fig.add_trace(go.Scattergeo(
            locationmode='USA-states',
            lon=df_sub['lng'],
            lat=df_sub['lat'],
            text=df_sub['text'],
            marker=dict(
                size=df_sub[col]/scale,
                color="lightseagreen",
                line_color='rgb(40,40,40)',
                line_width=0.5,
                sizemode='area'),
            name='{0} - {1}'.format(lim[0], lim[1])))

    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": "Date: " + str(dates[i])}], )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(active=10, currentvalue={"prefix": "Date: "},
               pad={"t": 50}, steps=steps)]

    fig.update_layout(
            sliders=sliders,
            title_text='Avocado ' + col +
                       '<br>(Click legend to toggle traces)',
            geo=dict(scope='usa', landcolor='rgb(217, 217, 217)',))

    return fig


def vol_map(data):
    """
    Takes in the tuple of dataframe and list of cities from
    the resample data and plots a map using Volume per Capita
    """

    fig = plotting(data[0], data[1], 'Vol Per Capita', 0.0002)
    fig.write_html("vol_map.html")


def price_map(data):
    """
    Takes in the tuple of dataframe and list of cities from
    the resample data and plots a map using Average Price
    """

    fig = plotting(data[0], data[1], 'AveragePrice', 0.003)
    fig.write_html("price_map.html")


def vol_plot(data):
    """
    Takes in tuple from resample data and creates a plot of
    total volume over time
    """

    df = data[0]
    for i in data[1]:
        x = df['Total Volume'].loc[i]
        x.plot()

    plt.title('Avocado Volume over Time')
    plt.legend(data[1], loc='upper center', bbox_to_anchor=(1.2, 1))
    plt.savefig('vol_plot.png')


def vpc_plot(data):
    """
    Takes in tuple from resample data and creates a plot of
    volume per capita over time
    """

    df = data[0]
    for i in data[1]:
        x = df['Total Volume'].loc[i]
        x.plot()

    plt.title('Avocado Volume per Capita over Time')
    plt.legend(data[1], loc='upper center', bbox_to_anchor=(1.2, 1))
    plt.savefig('vpc_plot.png')


def price_plot(data):
    """
    Takes in tuple from resample data and creates a plot of
    average price over time
    """

    df = data[0]
    for i in data[1]:
        x = df['AveragePrice'].loc[i]
        x.plot()

    plt.title('Avocado Average Price over Time')
    plt.legend(data[1], loc='upper center', bbox_to_anchor=(1.2, 1))
    plt.savefig('price_plot.png')


def main():
    '''
    Executes program.
    '''
    result = resample_data(data)
    vol_plot(result)
    vpc_plot(result)
    price_plot(result)
    vol_map(result)
    price_map(result)


if __name__ == '__main__':
    main()
