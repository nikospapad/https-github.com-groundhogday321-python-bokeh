https://github.com/groundhogday321/python-bokeh.git

Bokeh is an interactive visualization library.
Currently, Bokeh has two main interfaces:

bokeh.models - A low-level interface that provides the most flexibility to application developers
bokeh.plotting - A higher-level interface centered around composing visual glyphs
Bokeh can also be used with the HoloViews library.

bokeh.charts is deprecated.

Install¶
Go to python package index at python.org and see install instructions for bokeh.

Documentation
In [55]:
from IPython.display import IFrame
documentation = IFrame(src='https://bokeh.pydata.org/en/latest/', width=1000, height=450)
display(documentation)
Imports
Some imports can be imported in different ways and appear to do the same thing (i.e.-bokeh.io.show, bokeh.plotting.show)

In [56]:
# standard bokeh imports
from bokeh.io import output_notebook, show, reset_output

# other bokeh imports
import bokeh
from bokeh.plotting import figure
# more imports in cells below as needed

# other imports
import numpy as np
import pandas as pd
from vega_datasets import data as vds
Troubleshooting
reset_output(), then output_notebook() to keep from opening new tabs and display plots in notebook
from bokeh.charts import 'Plot Type' is deprecated, use from bokeh.plotting import figure instead
if something is not working, try to update to the latest version of bokeh
Sample Data Sets
In [57]:
from bokeh.sampledata import iris
# sample data set (dataframe)
iris_dataset = iris.flowers
iris_dataset.head()
Out[57]:
sepal_length	sepal_width	petal_length	petal_width	species
0	5.1	3.5	1.4	0.2	setosa
1	4.9	3.0	1.4	0.2	setosa
2	4.7	3.2	1.3	0.2	setosa
3	4.6	3.1	1.5	0.2	setosa
4	5.0	3.6	1.4	0.2	setosa
Display Plots In Notebook
In [58]:
# configure the default output state to generate output in notebook cells when show() is called
# in colab, output_notebook() is called in each cell (this is not always the case)
output_notebook()
Loading BokehJS ...
In [59]:
# output_notebook?
Save Plots
In [60]:
# if not using a notebook environment, use this to save/open your bokeh plots
# save your plot to html file or click save icon next to plot
# output_file()

'''
comment out output_file(), run reset_output(), then run output_notebook() to keep from opening new tabs 
and display plots in notebook
'''
Out[60]:
'\ncomment out output_file(), run reset_output(), then run output_notebook() to keep from opening new tabs \nand display plots in notebook\n'
Steps to Create Plots
create figure - used to create/house plot

call plot (glyph) method (types = line, bar, scatter, etc.)

show figure plot

ColumnDataSource
The ColumnDataSource is a data source used throughout Bokeh. Bokeh often creates the ColumnDataSource automatically, however there are times when it is useful to create them explicitly.

The ColumnDataSource is a (dictionary) mapping of column names (strings) to sequences of values. The mapping is provided by passing a dictionary with string keys and lists (or similar data structures) as values.

In [61]:
from bokeh.models import ColumnDataSource

column_data_source = ColumnDataSource({'A': [1, 2, 3, 4, 5],
                                       'B': [5, 4, 3, 2, 1],
                                       'C': [1, 3, 5, 1, 2]})

column_data_source.data
Out[61]:
{'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1], 'C': [1, 3, 5, 1, 2]}
Line Plot
In [62]:
# line plot

from bokeh.models import HoverTool

# data
x_line = np.arange(10)
y_line = np.random.rand(10)

# line plot
line_plot = figure(plot_width=500, plot_height=325, title='Line Plot', x_axis_label='x', y_axis_label='y')
line_plot.line(x_line, y_line, legend='line', line_width=2)

# add hover tool
line_plot.add_tools(HoverTool())

# another way to set axis labels
# line_plot.xaxis.axis_label = 'x-axis'
# line_plot.yaxis.axis_label = 'y-axis'

show(line_plot)
Plot Tools = Documentation, Pan, Box Zoom, Wheel Zoom, Save, Reset, Hover, Bokeh Plot Tools Information, Additional Tools Can Be Added

In [63]:
# line plot - multiple lines

output_notebook()

# data
multi_line_x = np.arange(10)
multi_line_y1 = np.random.rand(10)
multi_line_y2 = np.random.rand(10)
multi_line_y3 = np.random.rand(10)

# plot 
multi_line_plot = figure(plot_width=500, plot_height=300, toolbar_location='below')
multi_line_plot.line(multi_line_x, multi_line_y1, color='red', line_width=3)
multi_line_plot.line(multi_line_x, multi_line_y2, color='blue', line_width=3)
multi_line_plot.line(multi_line_x, multi_line_y3, color='yellow', line_width=3)
show(multi_line_plot)
Loading BokehJS ...
Bar Charts
In [64]:
# bar chart

# data
x_bar = ['category1', 'category2', 'category3', 'category4', 'category5']
y_bar = np.random.rand(5)*10

# sort data (sort x by its cooresponding y)
sorted_categories = sorted(x_bar, key=lambda x: y_bar[x_bar.index(x)], reverse=True)

# plot
bar_chart = figure(x_range=sorted_categories, title='Bar Plot', x_axis_label='x', y_axis_label='y', plot_height=300)
bar_chart.vbar(x_bar, top=y_bar, color='blue', width=0.5)
bar_chart.y_range.start = 0
show(bar_chart)

# use hbar for horizontal bar chart
In [65]:
# stacked bar chart
# note - appears to work fine with dataframe or column data source converted dataframe
# if dataframe does not work, use column data source

stacked_bar_df = pd.DataFrame({'y': [1, 2, 3, 4, 5],
                               'x1': [1, 2, 4, 3, 4],
                               'x2': [1, 4, 2, 2, 3]})

cds_stacked_bar_df = ColumnDataSource(stacked_bar_df)

stacked_bar_chart = figure(plot_width=600, plot_height=300, title='stacked bar chart')

stacked_bar_chart.hbar_stack(['x1', 'x2'], 
                             y='y', 
                             height=0.8, 
                             color=('grey', 'lightgrey'), 
                             source=cds_stacked_bar_df)

show(stacked_bar_chart)
In [66]:
# grouped bar chart

from bokeh.core.properties import value
from bokeh.transform import dodge

# data
categories = ['category1', 'category2', 'category3']
grouped_bar_df = pd.DataFrame({'categories' : categories,
                               '2015': [2, 1, 4],
                               '2016': [5, 3, 3],
                               '2017': [3, 2, 4]})

# plot
grouped_bar = figure(x_range=categories, y_range=(0, 10), plot_height=250)

# offsets bars / bar locations on axis
dodge1 = dodge('categories', -0.25, range=grouped_bar.x_range)
dodge2 = dodge('categories',  0.0,  range=grouped_bar.x_range)
dodge3 = dodge('categories',  0.25, range=grouped_bar.x_range)

grouped_bar.vbar(x=dodge1, top='2015', width=0.2, source=grouped_bar_df, color='gray', legend=value('2015'))
grouped_bar.vbar(x=dodge2, top='2016', width=0.2, source=grouped_bar_df, color='blue', legend=value('2016'))
grouped_bar.vbar(x=dodge3, top='2017', width=0.2, source=grouped_bar_df, color='green', legend=value('2017'))

# format legend
grouped_bar.legend.location = 'top_left'
grouped_bar.legend.orientation = 'horizontal'

show(grouped_bar)
Stacked Area Chart
In [67]:
stacked_area_df = pd.DataFrame({'x': [1, 2, 3, 4, 5],
                                'y1': [1, 2, 4, 3, 4],
                                'y2': [1, 4, 2, 2, 3]})

stacked_area_plot = figure(plot_width=600, plot_height=300)

stacked_area_plot.varea_stack(['y1', 'y2'],
                              x='x',
                              color=('green', 'lightgreen'),
                              source=stacked_area_df)

show(stacked_area_plot)
In [68]:
# stacked_area_plot.varea_stack?
Scatter Plots
In [69]:
# vega data sets cars data
cars = vds.cars()
cars.tail()
Out[69]:
Acceleration	Cylinders	Displacement	Horsepower	Miles_per_Gallon	Name	Origin	Weight_in_lbs	Year
401	15.6	4	140.0	86.0	27.0	ford mustang gl	USA	2790	1982-01-01
402	24.6	4	97.0	52.0	44.0	vw pickup	Europe	2130	1982-01-01
403	11.6	4	135.0	84.0	32.0	dodge rampage	USA	2295	1982-01-01
404	18.6	4	120.0	79.0	28.0	ford ranger	USA	2625	1982-01-01
405	19.4	4	119.0	82.0	31.0	chevy s-10	USA	2720	1982-01-01
In [70]:
# scatter plot

# data
x_scatter = cars.Weight_in_lbs
y_scatter = cars.Miles_per_Gallon

# plot 
scatter_plot = figure(plot_width=500, plot_height=300, x_axis_label='Weight_in_lbs', y_axis_label='Miles_per_Gallon')
scatter_plot.circle(x_scatter, y_scatter, size=15, line_color='navy', fill_color='orange', fill_alpha=0.5)
show(scatter_plot)
Other scatter plot variations include: cross, x, diamond, diamond_cross, circle_x, circle_cross, triangle, inverted_triangle, square, square_x, square_cross, asterisk

In [71]:
# vega data sets iris data
iris = vds.iris()
iris.tail()
Out[71]:
petalLength	petalWidth	sepalLength	sepalWidth	species
145	5.2	2.3	6.7	3.0	virginica
146	5.0	1.9	6.3	2.5	virginica
147	5.2	2.0	6.5	3.0	virginica
148	5.4	2.3	6.2	3.4	virginica
149	5.1	1.8	5.9	3.0	virginica
In [72]:
# scatter plot subgroups using iris data

from bokeh.transform import factor_cmap, factor_mark

# data
# use vega_datasets iris data

# plot 
species = ['setosa', 'versicolor', 'virginica']
markers = ['hex', 'cross', 'triangle']

scatter_plot_subgroups = figure(plot_width=600, 
                                plot_height=400, 
                                title ='Iris', 
                                x_axis_label='petalLength', 
                                y_axis_label='petalWidth')

scatter_plot_subgroups.scatter(x='petalLength',
                               y='petalWidth',
                               source=iris,
                               legend='species',
                               fill_alpha=0.5,
                               size=15,
                               color=factor_cmap(field_name='species', palette='Dark2_3', factors=species),
                               marker=factor_mark('species', markers, species)
                              )

# move legend
scatter_plot_subgroups.legend.location = 'top_left'
show(scatter_plot_subgroups)
In [73]:
# scatter_plot_subgroups.scatter?
Subplots
In [74]:
from bokeh.layouts import gridplot

output_notebook()

# data
subplot_x1 = cars['Acceleration']; subplot_y1 = cars['Miles_per_Gallon']
subplot_x2 = cars['Cylinders']; subplot_y2 = cars['Miles_per_Gallon']
subplot_x3 = cars['Horsepower']; subplot_y3 = cars['Miles_per_Gallon']
subplot_x4 = cars['Weight_in_lbs']; subplot_y4 = cars['Miles_per_Gallon']

# figures
subplot1 = figure(plot_width=300, plot_height=300)
subplot2 = figure(plot_width=300, plot_height=300)
subplot3 = figure(plot_width=300, plot_height=300)
subplot4 = figure(plot_width=300, plot_height=300)

# plots
subplot1.circle(subplot_x1, subplot_y1)
subplot2.circle(subplot_x2, subplot_y2)
subplot3.circle(subplot_x3, subplot_y3)
subplot4.circle(subplot_x4, subplot_y4)

# subplots gridplot
grid = gridplot([subplot1, subplot2, subplot3, subplot4], ncols=2)

# show
show(grid)
Loading BokehJS ...
Link Plots
In [75]:
from bokeh.layouts import gridplot

linked_data_x = np.arange(10)
linked_data_y = np.random.rand(10)

# linked plot 1
linked_plot1 = figure(width=250, height=250)
linked_plot1.circle(linked_data_x, linked_data_y)

# create new plots and share both ranges
linked_plot2 = figure(width=250, height=250, x_range=linked_plot1.x_range, y_range=linked_plot1.y_range)
linked_plot2.line(linked_data_x, linked_data_y)

linked_plot3 = figure(width=250, height=250, x_range=linked_plot1.x_range, y_range=linked_plot1.y_range)
linked_plot3.vbar(linked_data_x, top=linked_data_y, width=0.5)

# the subplots in a gridplot
linked_gridplot = gridplot([[linked_plot1, linked_plot2, linked_plot3]])

# show the results
show(linked_gridplot)
Linked Selection - Box Select, Lasso Select
In [76]:
# data
seattle_weather = vds.seattle_weather()
seattle_weather.tail()
Out[76]:
date	precipitation	temp_max	temp_min	wind	weather
1456	2015-12-27	8.6	4.4	1.7	2.9	fog
1457	2015-12-28	1.5	5.0	1.7	1.3	fog
1458	2015-12-29	0.0	7.2	0.6	2.6	fog
1459	2015-12-30	0.0	5.6	-1.0	3.4	sun
1460	2015-12-31	0.0	5.6	-2.1	3.5	sun
In [77]:
from bokeh.transform import factor_cmap, factor_mark

TOOLS = 'box_select, lasso_select, reset, wheel_zoom, pan'

weather_types = ['drizzle', 'rain', 'sun', 'snow', 'fog']
weather_markers = ['hex', 'cross', 'triangle', 'square', 'circle_x']

# use ColumnDataSource for linking interactions
seattle_weather_source = ColumnDataSource(seattle_weather)

# scatter plot 1
weather_scatter = figure(plot_width=900, plot_height=300, y_axis_label='Temp', x_axis_type='datetime', tools=TOOLS)
weather_scatter.circle('date', 'temp_max', size=15, fill_alpha=0.1, source=seattle_weather_source)

# scatter plot 2
weather_scatter_zoom = figure(plot_width=900, plot_height=500, x_axis_type='datetime', tools=TOOLS)
weather_scatter_zoom.scatter('date', 
                             'temp_max', 
                             size=15, 
                             fill_alpha=0.1,
                             color=factor_cmap(field_name='weather', palette='Dark2_5', factors=weather_types),
                             marker=factor_mark('weather', weather_markers, weather_types),
                             legend='weather',
                             source=seattle_weather_source
                             )

# shared data between plots helps the linked selection to work

# format legend
weather_scatter_zoom.legend.location = 'top_left'
weather_scatter_zoom.legend.orientation = 'horizontal'

weather_grid = gridplot([[weather_scatter], [weather_scatter_zoom]])
show(weather_grid)
Labels and Annotations
In [78]:
from bokeh.models.annotations import Label, LabelSet, Arrow
from bokeh.models.arrow_heads import NormalHead

output_notebook()

# data
fig_with_label_data = ColumnDataSource({'x': np.arange(10), 
                                        'y': [4, 7, 5, 5, 9, 2, 3, 4, 3, 4]})

# plot
fig_with_label = figure()
fig_with_label.line(x='x', y='y', source=fig_with_label_data)

# add label
label = Label(x=4, y=9, x_offset=10, text='Higest Point', text_baseline='middle')
fig_with_label.add_layout(label)

# add multiple labels
labels = LabelSet(x='x', y='y', text='y', level='glyph', source=fig_with_label_data)
fig_with_label.add_layout(labels)

# arrow annotation
fig_with_label.add_layout(Arrow(end=NormalHead(fill_color='orange'), x_start=5, y_start=7.5, x_end=4.5, y_end=8.8))

show(fig_with_label)
Loading BokehJS ...
Color Bar
In [79]:
cars.head()
Out[79]:
Acceleration	Cylinders	Displacement	Horsepower	Miles_per_Gallon	Name	Origin	Weight_in_lbs	Year
0	12.0	8	307.0	130.0	18.0	chevrolet chevelle malibu	USA	3504	1970-01-01
1	11.5	8	350.0	165.0	15.0	buick skylark 320	USA	3693	1970-01-01
2	11.0	8	318.0	150.0	18.0	plymouth satellite	USA	3436	1970-01-01
3	12.0	8	304.0	150.0	16.0	amc rebel sst	USA	3433	1970-01-01
4	10.5	8	302.0	140.0	17.0	ford torino	USA	3449	1970-01-01
In [80]:
from bokeh.models import LinearColorMapper, ColorBar
from bokeh.transform import transform

output_notebook()

# data
# use vega_datasets cars data

# map numbers in a range low, high - linearly into a sequence of colors (a palette)
color_mapper = LinearColorMapper(palette='Viridis256', low=cars.Weight_in_lbs.min(), high=cars.Weight_in_lbs.max())

# plot
colorbar_fig = figure(plot_width=600, plot_height=400, x_axis_label='Horsepower', y_axis_label='Miles_per_Gallon')
colorbar_fig.circle(x='Horsepower', 
                    y='Miles_per_Gallon',
                    source=cars,
                    color=transform('Weight_in_lbs', color_mapper), 
                    size=15, 
                    alpha=0.5)

# render a color bar based on a color mapper
color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12, location=(0,0), title='Weight')
colorbar_fig.add_layout(color_bar, 'right')

show(colorbar_fig)
Loading BokehJS ...
Map
In [81]:
# convert longitude, latitude to mercator coordinates
# example - Detroit Michigan 42.334197, -83.047752

from pyproj import Proj, transform

def create_coordinates(long_arg,lat_arg):
    in_wgs = Proj(init='epsg:4326')
    out_mercator = Proj(init='epsg:3857')
    long, lat = long_arg, lat_arg
    mercator_x, mercator_y = transform(in_wgs, out_mercator, long, lat)
    print(mercator_x, mercator_y)
    
# Detroit
create_coordinates(-83.047752,42.334197)
-9244833.464166068 5211172.739903524
In [82]:
# Cleveland
create_coordinates(-81.694703,41.499437)
-9094212.73846772 5086289.969305812
In [83]:
# Chicago 
create_coordinates(-87.629849,41.878111)
-9754910.168971453 5142738.513793045
In [84]:
from bokeh.tile_providers import get_provider, Vendors

tile_provider = get_provider(Vendors.CARTODBPOSITRON)
# tile_provider = get_provider(Vendors.STAMEN_TONER_BACKGROUND)

# range bounds supplied in web mercator coordinates
m = figure(plot_width=800, 
           plot_height=400,
           x_range=(-12000000, 9000000), 
           y_range=(-1000000, 7000000), 
           x_axis_type='mercator', 
           y_axis_type='mercator')

m.add_tile(tile_provider)

m.circle(x=-9244833, y=5211172, size=10, color='red')
m.circle(x=-9094212, y=5086289, size=10, color='blue')
m.circle(x=-9754910, y=5142738, size=10, color='orange')

show(m)
Interactive Widgets
In [91]:
# change size of scatter plot circles
from bokeh.layouts import column
from bokeh.models import Slider

# create figure and plot
change_plot_size = figure(plot_width=600, plot_height=300)
change_plot_size_r = change_plot_size.circle([1,2,3,4,5], [3,2,5,6,4], radius=0.1, alpha=0.5)

# create widget and link
slider = Slider(start=0.1, end=1, step=0.01, value=0.2)
slider.js_link('value', change_plot_size_r.glyph, 'radius')

show(column(change_plot_size, slider))
In [86]:
from sklearn import linear_model
from bokeh.layouts import layout
from bokeh.models import Toggle
import numpy as np

output_notebook()

# data
x = [1,2,3,4,5,6,7,8,9,10]
X = np.array(x).reshape(-1, 1)
y = [2,2,4,1,5,6,8,2,3,7]
Y = np.array(y).reshape(-1, 1)

# linear regression object
regr = linear_model.LinearRegression()

# fit linear model
regr.fit(X, Y)

# make predictions
pred = regr.predict(X)

# plot with regression line
regr_plot = figure(plot_width=500, plot_height=300)
regr_plot.scatter(x, y, size=10)
regr_line = regr_plot.line(x, pred.flatten(), line_color='red')

toggle_button = Toggle(label='line of best fit', button_type='success', active=True)
toggle_button.js_link('active', regr_line, 'visible')

show(layout([regr_plot], [toggle_button]))
Loading BokehJS ...
In [87]:
# slider.js_link?
Interactive Widgets with ipywidgets
In [88]:
seattle_weather['year'] = pd.DatetimeIndex(seattle_weather['date']).year
seattle_weather.tail()
Out[88]:
date	precipitation	temp_max	temp_min	wind	weather	year
1456	2015-12-27	8.6	4.4	1.7	2.9	fog	2015
1457	2015-12-28	1.5	5.0	1.7	1.3	fog	2015
1458	2015-12-29	0.0	7.2	0.6	2.6	fog	2015
1459	2015-12-30	0.0	5.6	-1.0	3.4	sun	2015
1460	2015-12-31	0.0	5.6	-2.1	3.5	sun	2015
In [89]:
import ipywidgets
from bokeh.io import push_notebook
from bokeh.models import Range1d

sw = seattle_weather.copy()

# widget
drop_down = ipywidgets.Dropdown(options=[2012, 2013, 2014, 2015],
                                value=2012,
                                description='years:',
                                disabled=False)

# data
x_bar_data_ipyw = ['precipitation', 'temp_max', 'temp_min', 'wind']
y_bar_data_ipyw = [sw[sw.year==2012]['precipitation'].mean(), 
                   sw[sw.year==2012]['temp_max'].mean(), 
                   sw[sw.year==2012]['temp_min'].mean(), 
                   sw[sw.year==2012]['wind'].mean()]
    
# figure and plot
bar_chart_interactive = figure(x_range=x_bar_data_ipyw, plot_height=300)
bar_ipyw = bar_chart_interactive.vbar(x_bar_data_ipyw, top=y_bar_data_ipyw, color='green', width=0.5)
bar_chart_interactive.y_range=Range1d(0, 18)

# function - bar chart
def weather_averages(year):
    if year == 2012: 
        bar_ipyw.data_source.data['top'] = [sw[sw.year==2012]['precipitation'].mean(), 
                                            sw[sw.year==2012]['temp_max'].mean(), 
                                            sw[sw.year==2012]['temp_min'].mean(), 
                                            sw[sw.year==2012]['wind'].mean()]
    elif year == 2013: 
        bar_ipyw.data_source.data['top'] = [sw[sw.year==2013]['precipitation'].mean(), 
                                            sw[sw.year==2013]['temp_max'].mean(), 
                                            sw[sw.year==2013]['temp_min'].mean(), 
                                            sw[sw.year==2013]['wind'].mean()]
    elif year == 2014: 
        bar_ipyw.data_source.data['top'] = [sw[sw.year==2014]['precipitation'].mean(), 
                                            sw[sw.year==2014]['temp_max'].mean(), 
                                            sw[sw.year==2014]['temp_min'].mean(), 
                                            sw[sw.year==2014]['wind'].mean()]
    elif year == 2015: 
        bar_ipyw.data_source.data['top'] = [sw[sw.year==2015]['precipitation'].mean(), 
                                            sw[sw.year==2015]['temp_max'].mean(), 
                                            sw[sw.year==2015]['temp_min'].mean(), 
                                            sw[sw.year==2015]['wind'].mean()]
    push_notebook()

show(bar_chart_interactive, notebook_handle=True)
        
# interaction
ipywidgets.interact(weather_averages, year=drop_down)
interactive(children=(Dropdown(description='years:', options=(2012, 2013, 2014, 2015), value=2012), Output()),…
Out[89]:
<function __main__.weather_averages(year)>
In [90]:
sw.groupby('year').mean().T
Out[90]:
year	2012	2013	2014	2015
precipitation	3.349727	2.268493	3.377534	3.121096
temp_max	15.276776	16.058904	16.995890	17.427945
temp_min	7.289617	8.153973	8.662466	8.835616
wind	3.400820	3.015890	3.387671	3.159726
More
Embed images, etc. in tooltip example (Configuring Plot Tools - Mouse over the dots)
More examples in documentation
Next
In the next tutorial (or very near future) we will go over Holoviews which works with Bokeh.
