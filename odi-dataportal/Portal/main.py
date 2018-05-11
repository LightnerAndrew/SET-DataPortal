###########################################################################
##########         SET - Data Portal                         ##############
############################################################################

# Author: Andrew Lightner
# Date: 09/05/2018
# Email: lightnera1@gmail.com


############################################################################
##############              Load Packages                  #################
############################################################################

# Bokeh Packages
from bokeh.io import curdoc
from bokeh.models import Legend, Button, RadioButtonGroup, LegendItem, BasicTicker, ColorBar, LinearColorMapper, PrintfTickFormatter, ColumnDataSource,MultiSelect, Span, Panel, Tabs, LabelSet, Select, Div, RangeSlider, Slider, TextInput, CategoricalColorMapper, HoverTool, CustomJS, FactorRange, TableColumn, DataTable, Label
from bokeh.plotting import figure
from bokeh.layouts import layout, widgetbox, row, column
from bokeh.transform import factor_cmap, dodge
from bokeh.palettes import Category20b, Viridis
from bokeh.core.properties import value


# Other packages
import os
import pandas as pd
from os.path import dirname, join
import numpy as np
import sqlalchemy
from sqlalchemy import Table, MetaData, select, and_, or_
from sqlalchemy.exc import NoSuchTableError
from math import pi
from scipy.interpolate import spline
import json
import more_itertools as mit
from textwrap import wrap

############################################################################
##############              Load Data                      #################
############################################################################

# Connect with SQL Server
def connect(user, password, db, host='localhost', port=5432):
    '''Returns a connection and a metadata object'''
    # We connect with the help of the PostgreSQL URL
    # postgresql://federer:grandestslam@localhost:5432/tennis
    url = 'postgresql://{}:{}@{}:{}/{}'
    url = url.format(user, password, host, port, db)

    # The return value of create_engine() is our connection object
    engine = sqlalchemy.create_engine(url, client_encoding='utf8')

    ## Commections
    connection = engine.connect()

    # We then bind the connection to MetaData()
    meta = sqlalchemy.MetaData(bind=engine, reflect=True)

    return engine, connection, meta

#collect key variables to access data using connect function
engine, connection, meta = connect()



# Create Data Table to load
# The except function exists for the first upload of the app, as the data is not uploaded until after the first uploadself.
# This is the result of an unclean process of uploading the data to the SQL Server
try:
    data_table = Table('odi-portal-april2018', MetaData(), autoload=True, autoload_with=engine)
except NoSuchTableError:
    pd.DataFrame(columns=names).to_sql('odi-portal-april2018', con=engine, schema='public')
    data_table = Table('odi-portal-april2018', MetaData(), autoload=True, autoload_with=engine)

##### Create Main selection from SQL Table
stmt_main = select([data_table])



############################################################################
##############              List of Vars                   #################
############################################################################

# List of variables located in the jsom file
# Also includes lables and sources.
axis_map_notes = json.load(open(join(dirname(__file__), 'TextFiles', 'vars_json.txt')))


# Generate group variable dictionary for the box and whisker plot
# The World Bank Income variable is currnetly unreliable - need to revisitself.
# Return - 'World Bank Income': 'Var189' when fixed.
GroupVars = {'SET Fragility Index': 'Var188', 'Within World Bank Index': 'Var190', 'Within OECD Index': 'Var187'}


#### Generate indicator list for multi-variable selections
indicators = [(axis_map_notes[i][0],i) for i in axis_map_notes]




#########################################################
########## Generate CountryChoice Widget Early  #########
#########################################################

# Each broad subject heading must have its own country selection:
# This is because each have its own callback whch refers to a different update function,
# When they point to multiple update functions, the app freezes.

CountryList = [(i, i) for i in ['Outside OECD Index', 'Within OECD Index', 'Active Conflict', 'At Risk of Conflict', 'Limited Conflict', 'Subnational Conflict', 'Transition From Conflict', 'Outside WB Index', 'Within WB Index', 'Upper middle income', 'Lower middle income', 'Low income', 'Afghanistan', 'Albania', 'Algeria', 'Angola', 'Antigua and Barbuda', 'Argentina', 'Armenia', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belize', 'Benin', 'Bhutan', 'Bolivia', 'Bosnia and Herzegovina', 'Botswana', 'Brazil', 'Brunei Darussalam', 'Burkina Faso', 'Burundi', 'Cabo Verde', 'Cambodia', 'Cameroon', 'Central African Republic', 'Chad', 'Chile', 'China', 'Colombia', 'Comoros', 'Congo Republic', 'Costa Rica', "Cote d'Ivoire", 'Cuba', 'DR Congo', 'Djibouti', 'Dominica', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Ethiopia', 'Fiji', 'Gabon', 'Gambia', 'Georgia', 'Ghana', 'Grenada', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras', 'Hong Kong', 'India', 'Indonesia', 'Iran', 'Iraq', 'Jamaica', 'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati', 'Kosovo', 'Kuwait', 'Kyrgyz Republic', 'Laos', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Macao', 'Macedonia', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Marshall Islands', 'Mauritania', 'Mauritius', 'Mexico', 'Micronesia, Fed. Sts.', 'Moldova', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nauru', 'Nepal', 'Nicaragua', 'Niger', 'Nigeria', 'North Korea', 'Oman', 'Pakistan', 'Palau', 'West Bank and Gaza', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Qatar', 'Russia', 'Rwanda', 'Samoa', 'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone', 'Solomon Islands', 'Somalia', 'South Africa', 'South Korea', 'South Sudan', 'Sri Lanka', 'Sudan', 'Suriname', 'Swaziland', 'Syria', 'Tajikistan', 'Tanzania', 'Thailand', 'Timor-Leste', 'Togo', 'Tonga', 'Trinidad and Tobago', 'Tunisia', 'Turkey', 'Turkmenistan', 'Tuvalu', 'Uganda', 'Ukraine', 'United Arab Emirates', 'Uruguay', 'Uzbekistan', 'Vanuatu', 'Venezuela', 'Vietnam', 'Yemen', 'Zambia', 'Zimbabwe']]

# Single Country selections for each subject category
CountrySTRUC = Select(title="Country or Group Category", value="Malawi", options=CountryList)
CountryLP = Select(title="Country or Group Category", options=CountryList, value="Malawi")
CountryFIRM = Select(title="Country or Group Category", options=CountryList, value="Malawi")
CountryTRADE = Select(title="Country or Group Category", options=CountryList, value="Malawi")
CountryCROSS = Select(title="Highlighted Country", options=CountryList, value="Malawi")
CountryTAX = Select(title="Highlighted Country", options=CountryList, value="Malawi")
CountryEMP = Select(title="Highlighted Country", options=CountryList, value="Malawi")


# Multiple Country selections where necessary
### countries refers to CROSS category
countries = MultiSelect(title="Country Selection", value=['Outside OECD Index', 'Within OECD Index'],
                           options=CountryList, size=10)
LPcountries = MultiSelect(title="Country Selection", value=['Outside OECD Index', 'Within OECD Index'],
                           options=CountryList, size=10)

# Group Selection for the
GroupSelect = MultiSelect(title="Country Selection", value=['Within OECD Index'],
                           options=[('Within OECD Index', 'OECD Fragility Index'), ('SET Fragility Index', 'SET Fragility Index'), ('Within World Bank Index', 'World Bank Fragility Index')], size=5)




#########################################################
########## Number Widget Options Dictionaries   #########
#########################################################


####### Generate Year choices
axis_year =  {str(i): (str(i)) for i in range(1991,2018)}

############ Generate Axis Map for Font Size Choices
axis_font = {str(i): (str(i)) for i in range(12, 32)}

############ Generate Axis Map for Round Choices
axis_round = {str(i): (str(i)) for i in range(0, 4)}

############ Generate Axis Map for Font Size Choices
axis_barsize = {str(i): (str(i)) for i in range(1, 10)}

############ Generate Axis Map for Cutoff
axis_cutoff = {str(i): (str(i)) for i in range(10, 100)}

############ Generate Axis Map for Group Year Choices
axis_groupyear = {str(i): (str(i)) for i in range(1, 10)}



###############################################
#####       SET Style Options      ############
###############################################

# Formatting Options


# Label Options for the Footnotes of the Tabel
label_opts = dict(
        x=0, y=0,
        x_units='screen', y_units='screen',
        text_font_size='10pt', text_font = 'arial'
    )

###############################################
#####          Legend Options      ############
###############################################


axis_legend = {'Bottom Right': 'bottom_right',
                'Bottom Left': 'bottom_left',
                'Top Right' : 'top_right',
                'Top Left': 'top_left'}

axis_legend_orientation = {'Horizontal': 'horizontal',
                'Vertical': 'vertical'}



####################################################
############  Define Palettes       ################
####################################################

# SET Palette based on
SET_palette = ['#361c7f', '#b35900', '#990000', '#016450', '#02818a', '#3690c0', '#67a9cf', '#a6bddb', '#d0d1e6', '#fed976']
color_blender = ['#4d004b', '#810f7c', '#88419d', '#8c6bb1', '#8c96c6', '#9ebcda', '#bfd3e6', '#e0ecf4', '#f7fcfd']
large_color = Category20b[20]


####################################################
############  Define Tool Options   ################
####################################################

tools = "pan,wheel_zoom,box_zoom,reset,save"




####################################################
############  Useful Functions      ################
####################################################


def obs_footnote(country_vals):
    obs = []
    for i in country_vals:
        end = ', '
        if len(country_vals)==n:
            end='. '

        obs+= i +': '+str(list(consecutive(data[data['countryname']==i]['year'].as_matrix()))).strip('[]')+end
        n+=1
    return obs


#####################################################################
#####################################################################
###    HeatMap - Cross Sectional Comparison                    ####
#####################################################################
#####################################################################

# General Widgets for HeatMap
title_heatmap = TextInput(title="Title", value="Box Plot - Cross Sectional")
T_choice = Select(title="Success Threshold", options=sorted(axis_cutoff.keys()), value="75")
minyear_heatmap = Select(title="Start Year", options=sorted(axis_year.keys()), value="1991")
maxyear_heatmap = Select(title="End Year", options=sorted(axis_year.keys()), value="2016")
font_heatmap = Select(title="Title Font Size", options=sorted(axis_font.keys()), value="24")

# Generate Key Indicator Options
# (Future editions may want to make the selection of these variables as options)
ID = json.load(open(join(dirname(__file__), 'TextFiles', 'ID_json.txt')))
print(ID)
indicator_options = Select(title="Indicator Selection", options=sorted(ID.keys()), value='Productivity')
success = Select(title='Number of Consecutive Years which Defines Success', options=sorted(axis_groupyear.keys()), value='4')

# List of variables which need to be changed to a percentage change
scales = ID['Economic Foundations']+ID['Trade']

### Generate variable which can set the order of the visuals according to the SET Categories
order = {'Active Conflict':[4],  'Transition From Conflict':[3], 'Subnational Conflict':[2],'At Risk of Conflict':[1], 'Limited Conflict':[0]}
order_df =pd.melt(pd.DataFrame(order), value_vars=list(pd.DataFrame(order).columns)).rename(columns={'variable': 'SET'})
order_df['SET'] = order_df['SET'].str.strip()
order_df = order_df.rename(columns={'value': 'order'})


# Select the Observations - Needs significant work to make more efficient.
def select_obs_heatmap():
    # choice of indicators
    grouping = ID[indicator_options.value]

    #######################
    # Data Selection
    #######################
    # Define statemetn with country selection
    stmt = stmt_main.where(
        data_table.columns.year.between(int(minyear_heatmap.value),int(maxyear_heatmap.value)))

    # select observations using sqlalchemy
    dictionary = {'countryname': [],
             'year':[],
             'SET': []}
    for i in grouping:
        dictionary[i] = []

    for result in connection.execute(stmt):
        dictionary['countryname'].append(result.countryname)
        dictionary['year'].append(result.year)
        dictionary['SET'].append(result.Var188)
        for i in grouping:
            dictionary[i].append(result[axis_map_notes[i][0]])


    # Generate dataframe of results
    selected = pd.DataFrame(dictionary)
    selected.sort_values(['countryname', 'year'], inplace=True)

    # Generate change variables if necessary:
    if (indicator_options.value =='Economic Foundations') or (indicator_options.value =='Trade'):
        for i in grouping:
            selected[i] = selected.groupby('countryname')[i].pct_change(fill_method=None, limit=0)*100

    # Cleaning
    selected.replace([np.inf, -np.inf], np.nan, inplace=True)
    selected.reset_index(inplace=True)




    ############################
    # Generate threshold dataset
    ############################
    threshold_dict = {i: [np.round(np.percentile(selected[i].dropna(), int(T_choice.value)), 2)] for i in grouping}
    # Reshape threshold dataset to be most useful
    print('this is the problem')
    threshold =pd.melt(pd.DataFrame(threshold_dict), value_vars=list(pd.DataFrame(threshold_dict).columns[:-1]))
    # Small amount of cleaning
    print('it made it here')
    threshold['variable'] = threshold['variable'].str.strip()


    ############################
    # List of Fragile States for Ordering
    ############################
    data_fragile = selected.groupby(['countryname','SET']).mean().reset_index()[['countryname', 'SET']]
    ### order_df is insered here.
    data_fragile = pd.merge(data_fragile, order_df, on='SET')





    # 1 - observations which meet the threshold
    data = pd.DataFrame()

    # interate over each variable of interest
    for var in grouping:
        # Generate an empty dictionary to evenutually merge with main
        country_meetscon = {}
        coutnry_counts = {}

        # iterate over each country in the dataset
        df = selected[['countryname', 'year', var, 'SET']]

        # Choose only those observations which meet the standard
        df = df[df[var]>threshold_dict[var]]

        # Cleaning
        df['year'] = df['year'].astype(int)
        df.sort_values(['countryname', 'year'], inplace=True)


        # Generate list for number of years which the threshold is met
        total_num = [0]
        totals = df.groupby('countryname').count()[var].reset_index().rename(columns={var: 'count'})



        # iterate over each country in the dataset (left in the dataset)
        for i in list(selected['countryname'].unique()):

            # get results of algorithm which finds all consecutive observations
            a = [b for b in list(consecutive(df[df['countryname']==i]['year']))]

            # Only those with multiple consecutive years. (they are tubples as opposed to int)
            final = [b for b in a if type(b)==tuple]

            # Generate list of the number of consecutive years for each
            largest_time = [(c[1]-c[0]+1) for c in final if type(c)==tuple]

            # replaces longest time with at least for observations which met the requirement once
            # they will not show up in the largest time since they will record as an 'int' value.
            if largest_time ==[]:
                largest_time=[1]
            # append final results to the meets_threshold dictionary.
            country_meetscon[i+'| '+var] = [final, max(largest_time), a]


        # Empty df
        df = {}

        # Take results from loop above
        df['countries'] = list(country_meetscon.keys())
        df['Results'] = list(country_meetscon.values())
        df = pd.DataFrame(df)

        # Split variables
        ids = df['countries'].str.split('|', expand=True)
        df['countryname']= ids[0]
        df['variable']= ids[1]
        df[['GP','LP', 'Allyears']] = pd.DataFrame(df.Results.values.tolist(), index= df.index)

        # Merge in fragile categories and thresholds
        df = pd.merge(df, data_fragile, on='countryname', how='left')
        df = df[['countryname', 'SET', 'variable', 'GP','LP','Allyears',  'order']]
        df['variable'] = df['variable'].str.strip()
        df['SET'] = df['SET'].str.strip()
        df = pd.merge(df, threshold, on='variable')
        df = pd.merge(totals, df, on=['countryname'])


        # Place Final Dictionary within
        data = data.append(df)

    # Change [] to None
    data['GP'] = data['GP'].apply(lambda y: 'None' if len(y)==0 else y).apply(lambda x: x[0:]).astype(str)

    # Choose only variables of interest
    data = data[['countryname', 'SET','count', 'variable', 'GP','LP','Allyears' , 'value', 'order']]

    # Generate order in terms of fragility category
    # Re-index with df order
    data.dropna(inplace=True)
    data.sort_values(['order', 'countryname'], inplace=True)
    data.drop('order', axis=1, inplace=True)
    data.set_index('SET', inplace=True)

    return data




def create_figure_heatmap():
    # Get data
    data = select_obs_heatmap()

    # Generate the color mapper
    #### Choose the variable to color by (longest consecutive (LP) or the total number (GP))
    Val_of_int = 'LP'
    colors = Viridis[11]
    #linear mapper, choose number of options by number of colors, high and low based on choice of value
    mapper = LinearColorMapper(palette=colors, low=data[Val_of_int].min(), high=data[Val_of_int].max())


    # List of countries for the plot
    countries = list(data['countryname'].unique())

    # Generate SourceData
    source = ColumnDataSource(data)


    # list of tools for plot
    TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"

    # Generate the plot options
    p = figure(plot_width=900, plot_height=2500, title=None,
               x_range=list(data.variable.unique()), y_range=countries,
               x_axis_location="above",
               tools=TOOLS, toolbar_location='below',
               background_fill_color="#1a001a")

    # Generate the heatmap
    p.rect(x="variable", y="countryname", width=1, height=1,
           source=source,
           fill_color={'field': Val_of_int, 'transform': mapper},
           line_color=None)

    # Generate color code
    color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="15pt",
                         ticker=BasicTicker(desired_num_ticks=len(colors)),
                         label_standoff=6, border_line_color=None, location=(0, 0))

    # Format axis, etc.
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.xaxis.major_label_text_font_size = "12pt"
    p.yaxis.major_label_text_font_size = "9pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = pi / 2.5

    # Add color bar to the right of the plot
    p.add_layout(color_bar, 'right')

    # Add hover tool
    p.select_one(HoverTool).tooltips = [
        ('Country', '@countryname'),
        ('Category', '@SET'),
        ('Variable', '@variable'),
        ('Threshold', '@value{0.0}%'),
        ('Amount of Years', '@count'),
        ('Growth Periods', '@GP'),
         ('Longest', '@LP')
    ]
    return p




# Generate data for the download
def update_data_heatmap():
    # Access Data
    data = select_obs_heatmap()

    # Cleaning
    data.reset_index(inplace=True)
    data['GP'] = data['GP'].str.replace(',', '-')
    data.sort_values(['SET','countryname', 'LP', 'variable'], inplace=True)
    data.drop('Allyears', axis=1, inplace=True)

    # Set the threshold for placement in the tables
    data = data[data['LP']>=(int(success.value)+1)]

    # Generate ColumnsDatasource
    source = ColumnDataSource(data)


    return source







#####################################################################
#####################################################################
###    Box PLot  - Cross Sectional Comparison                    ####
#####################################################################
#####################################################################


title_boxplot = TextInput(title="Title", value="Box Plot - Cross Sectional")
boxplot_var = Select(title="Variable of Interest", options=sorted(axis_map_notes.keys()), value="Proportion of Employment (%): Agriculture")
minyear_boxplot = Select(title="Start Year", options=sorted(axis_year.keys()), value="2008")
maxyear_boxplot = Select(title="End Year", options=sorted(axis_year.keys()), value="2016")
font_boxplot = Select(title="Title Font Size", options=sorted(axis_font.keys()), value="24")
note_boxplot = TextInput(title="Additional Note Content", value="")
legend_location_boxplot = Select(title="Legend Location", options=sorted(axis_legend.keys()), value="Bottom Right")
legend_location_ori_boxplot = Select(title="Legend Orientation", options=sorted(axis_legend_orientation.keys()), value="Vertical")
boxplot_options = RadioButtonGroup(labels=["Group Analysis", 'Country Analysis'], active=0, width=400)
round_boxplot = Select(title="Number of Decimal Points", options=sorted(axis_round.keys()), value="3")
bar_width = Slider(title='Width of Bars', start=0.05, end=.4, value=0.2, step=.04)
group_years = Select(title="Year Groupings", options=sorted(axis_groupyear.keys()), value="3")
bar_order = RadioButtonGroup(labels=["Ascending", "Descending"], active=0, width=250)


def select_obs_boxplot():
    country_vals = countries.value
    Var_Interest = axis_map_notes[boxplot_var.value][0]
    group_val = GroupSelect.value[0]

    # Select the Correct Observations

    if boxplot_options.active ==1:
        # Define statemetn with country selection
        stmt = stmt_main.where(and_(
            data_table.columns.countryname.in_(country_vals),
            data_table.columns.year.between(int(minyear_boxplot.value),int(maxyear_boxplot.value))))

        dictionary = {'countryname': [],
                 'year':[],
                 Var_Interest: []}

        for result in connection.execute(stmt):
            dictionary['countryname'].append(result.countryname)
            #dictionary['countrycode'].append(result.countrycode)
            dictionary['year'].append(result.year)
            dictionary[Var_Interest].append(result[Var_Interest])

    if boxplot_options.active ==0:
        # Define statemetn without country selection for group analysis
        stmt = stmt_main.where(
            data_table.columns.year.between(int(minyear_boxplot.value),int(maxyear_boxplot.value)))

        dictionary = {'countryname': [],
                 'year':[],
                 group_val: [],
                 Var_Interest: []}

        for result in connection.execute(stmt):
            dictionary['countryname'].append(result.countryname)
            dictionary['year'].append(result.year)
            dictionary[GroupSelect.value[0]].append(result[GroupVars[group_val]])
            dictionary[Var_Interest].append(result[Var_Interest])

    selected = pd.DataFrame(dictionary)


    return selected

import more_itertools as mit
from textwrap import wrap

## GEt year observations
def consecutive(iterable):
    """Yield range of consecutive numbers."""
    for group in mit.consecutive_groups(iterable):
        group = list(group)
        if len(group) == 1:
            yield group[0]
        else:
            yield group[0], group[-1]


def create_figure_boxplot():
    country_vals = countries.value
    group_val = GroupSelect.value[0]
    no_data = []

    # Choose variable of interest
    Var_Interest = axis_map_notes[boxplot_var.value][0]

    # Choose the order of the plots
    order = True
    if bar_order.active ==1:
        order=False

    # Get Data
    data = select_obs_boxplot()

    # Drop missing values
    data.dropna(inplace=True)

    # Ensure adequate data exists.
    if boxplot_options.active ==1:
        no_data= list(set(country_vals) - set(list(data.countryname.unique())))


    data.sort_values(['countryname', 'year'], inplace=True)
    # If there is adequate data, then generate the plot
    if len(no_data)==0:

        # Generate observation list
        obs = ''
        n = 1

        if boxplot_options.active ==1:
            for i in country_vals:
                end = ', '
                if len(country_vals)==n:
                    end='. '

                obs+= i +': '+str(list(consecutive(data[data['countryname']==i]['year'].as_matrix()))).strip('[]')+end
                n+=1

            # break text into several lines
            obs = obs.replace(',', ' -').replace(') -', '),')

            # add additional notes to the list.
        obs = obs+ note_boxplot.value


        # Rename Variable
        data.rename(columns={Var_Interest: 'VarInt'}, inplace=True)
        # # Group by countryname
        types = 2
        if boxplot_options.active ==1:
            group = data.groupby('countryname')
            groups = group.describe()
            groups.drop('year',axis=1, inplace=True)
            groups.columns = groups.columns.droplevel(0)
            groups.sort_values('mean', inplace=True, ascending =order)

        if boxplot_options.active ==0:
            group = data.groupby(group_val)
            groups = group.describe()
            groups.drop('year',axis=1, inplace=True)
            groups.columns = groups.columns.droplevel(0)
            groups.sort_values('mean', inplace=True, ascending =order)


        q1 = groups['25%'].to_frame().rename(columns={'25%': 'VarInt'})
        q2 = groups['50%'].to_frame().rename(columns={'50%': 'VarInt'})
        q3 = groups['75%'].to_frame().rename(columns={'75%': 'VarInt'})
        qmean = groups['mean'].to_frame().rename(columns={'mean': 'VarInt'})
        iqr = q3 - q1
        upper = q3 + 1.5*iqr
        lower = q1 - 1.5*iqr


        #q2.sort_values(['VarInt'], ascending=order)
        cats = list(groups.index)


        # Prep the outliers for plotting
        # find the outliers for each category

        # isolate the outliers
        def outliers(group):
            cat = group.name
            return group[(group.VarInt > upper.loc[cat]['VarInt']) | (group.VarInt < lower.loc[cat]['VarInt'])]

        # place all outliers within the group dataframe
        group = group.apply(outliers).dropna().reset_index().drop('level_1', axis=1)

        # if there are outliers, generate a ColumnDataSource out of the frame
        if len(group)>0:
            source = ColumnDataSource(group)


        # make the plot
        p = figure(tools="save", background_fill_color="#eeeafa", title="", x_range=cats, plot_width = 800, plot_height=800)

        # if no outliers, shrink lengths of stems to be no longer than the minimums or maximums
        qmin = groups['min'].to_frame().rename(columns={'min': 'VarInt'})
        qmax = groups['max'].to_frame().rename(columns={'max': 'VarInt'})
        upper.VarInt = [min([x,y]) for (x,y) in zip(list(qmax.loc[:,'VarInt']),upper.VarInt)]
        lower.VarInt = [max([x,y]) for (x,y) in zip(list(qmin.loc[:,'VarInt']),lower.VarInt)]

        # stems
        p.segment(cats, upper.VarInt, cats, q3.VarInt, line_color="black")
        p.segment(cats, lower.VarInt, cats, q1.VarInt, line_color="black")

        # boxes
        p.vbar(cats, 0.7, q2.VarInt, q3.VarInt, fill_color="#361c7f", line_color="black")
        p.vbar(cats, 0.7, q1.VarInt, q2.VarInt, fill_color="#b35900", line_color="black")

        # whiskers (almost-0 height rects simpler than segments)
        p.rect(cats, lower.VarInt, 0.2, 0.01, line_color="black")
        p.rect(cats, upper.VarInt, 0.2, 0.01, line_color="black")

        # small edits to style
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = "white"
        p.grid.grid_line_width = 2
        p.xaxis.major_label_text_font_size="12pt"

        # If outliers exist, plot them with hover tool to explore
        if len(group)>0:
            if boxplot_options.active ==1:
                ID = 'countryname'
            else:
                ID = group_val


            line = p.circle(x = ID, y='VarInt', color ="#361c7f", source=source, size=15)

            hover = HoverTool(
                    renderers=[line],
                    tooltips=[
                        ('Country', '@countryname'),
                        ( 'Year',   '@year'),
                        ( boxplot_var.value,   '@VarInt')
                    ]
            )

            p.add_tools(hover)

        if (len(country_vals)>3):
            p.xaxis.major_label_orientation = pi/2
        if group_val == 'SET Fragility Index':
            p.xaxis.major_label_orientation = pi/2

        # Generate Notes for the observations and additional notes
        obs = "\n".join(wrap(obs, 95)).split('\n')

        d = {0:"", 1:"", 2:"", 3:""}
        for i in range(0, len(obs)):
            d[i] = obs[i]


        msg1 = 'Source: '+axis_map_notes[boxplot_var.value][1]+'.'
        caption1 = Label(text=msg1, **label_opts, text_color='#999999')
        p.add_layout(caption1, 'below')
        if boxplot_options.active ==0:
            msg2 = d[0]
            caption2 = Label(text=msg2, **label_opts, text_color='#999999')
            p.add_layout(caption2, 'below')
        if boxplot_options.active ==1:
            msg2 ='Observations: '+ d[0]
            caption2 = Label(text=msg2, **label_opts, text_color='#999999')
            p.add_layout(caption2, 'below')

        msg3 = d[1]
        caption2 = Label(text=msg3, **label_opts, text_color='#999999')
        p.add_layout(caption2, 'below')

        msg4 = d[2]
        caption2 = Label(text=msg4, **label_opts, text_color='#999999')
        p.add_layout(caption2, 'below')

        msg4 = d[3]
        caption2 = Label(text=msg4, **label_opts, text_color='#999999')
        p.add_layout(caption2, 'below')
        #### SETTINGS TO MATCH Set
        p.xgrid.visible = False
        p.title.text_color = '#361c7f'
        p.title.text_font = "arial"
        p.title.text_font_style = "bold"
        p.grid.grid_line_color='#CBCBCB'
        p.grid.grid_line_width=2.0
        fontvalue = font_boxplot.value+'pt'
        p.title.text_font_size = fontvalue
        p.xaxis.axis_label=''
        p.yaxis.axis_label=boxplot_var.value
        p.axis.axis_label_text_font = 'arial'
        p.axis.axis_label_text_color = '#999999'
        p.axis.axis_label_text_font_style = 'normal'
        p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
        p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
        p.axis.axis_line_color=None
        p.axis.major_tick_line_color=None
        p.axis.minor_tick_line_color=None
        p.axis.major_label_text_font_size = '16pt'
        p.axis.major_label_text_color='#999999'
        p.outline_line_color = None
        p.title.text = title_boxplot.value
        p.axis.axis_label_text_font_size = '12pt'

    if len(no_data) >0:

        p = figure(plot_width=900, plot_height=600, background_fill_color="#eeeafa")


        #### SETTINGS TO MATCH Set
        p.xgrid.visible = False
        p.title.text_color = '#361c7f'
        p.title.text_font = "arial"
        p.title.text_font_size = "20pt"
        p.title.text_font_style = "bold"
        p.grid.grid_line_color='#CBCBCB'
        p.grid.grid_line_width=2.0
        p.xaxis.axis_label='Year'
        p.yaxis.axis_label=line_var.value
        p.axis.axis_label_text_font = 'arial'
        p.axis.axis_label_text_color = '#999999'
        p.axis.axis_label_text_font_style = 'normal'
        p.axis.axis_label_text_font_size = '16pt'
        p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
        p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
        p.axis.axis_line_color=None
        p.axis.major_tick_line_color=None
        p.axis.minor_tick_line_color=None
        p.axis.major_label_text_font_size = '16pt'
        p.axis.major_label_text_color='#999999'
        p.outline_line_color = None
        p.legend.click_policy="mute"
        label = Label(x=1.1, y=18, text='Insufficient Data', text_font_size='70pt', text_color='#016450')
        p.add_layout(label)
        p.xaxis.axis_label = ''
        p.yaxis.axis_label = ''
        p.title.text = 'Insufficient Data : '+str(no_data)



    return p






#####################################################################
#####################################################################
###    Bar Chart - Cross Sectional Comparison                    ####
#####################################################################
#####################################################################


title_bar_cross = TextInput(title="Title", value="Bar Chart - Cross Sectional")
bar_cross_var = Select(title="Variable of Interest", options=sorted(axis_map_notes.keys()), value="Proportion of Employment (%): Agriculture")
minyear_bar_cross = Select(title="Start Year", options=sorted(axis_year.keys()), value="2008")
maxyear_bar_cross = Select(title="End Year", options=sorted(axis_year.keys()), value="2016")
font_bar_cross = Select(title="Title Font Size", options=sorted(axis_font.keys()), value="24")
note_bar_cross = TextInput(title="Additional Note Content", value="")
legend_location_bar_cross = Select(title="Legend Location", options=sorted(axis_legend.keys()), value="Bottom Right")
legend_location_ori_bar_cross = Select(title="Legend Orientation", options=sorted(axis_legend_orientation.keys()), value="Vertical")
bar_plot_options = RadioButtonGroup(
        labels=["Averages", "By Country/Year", "By Year/Country"], active=0, width=400)
round_bar_cross = Select(title="Number of Decimal Points", options=sorted(axis_round.keys()), value="3")
bar_width = Slider(title='Width of Bars', start=0.05, end=.4, value=0.2, step=.04)
group_years = Select(title="Year Groupings", options=sorted(axis_groupyear.keys()), value="3")
bar_order = RadioButtonGroup(
        labels=["Ascending", "Descending"], active=0, width=250)
bar_height = Select(title="Plot Height", options=sorted(axis_groupyear.keys()), value="7")




def select_obs_bar_cross():
    country_vals = countries.value
    Var_Interest = axis_map_notes[bar_cross_var.value][0]
    group_val = GroupSelect.value[0]

    # Select the Correct Observations

    # Define statemetn with country selection
    stmt = stmt_main.where(and_(
        data_table.columns.countryname.in_(country_vals),
        data_table.columns.year.between(int(minyear_bar_cross.value),int(maxyear_bar_cross.value))))

    dictionary = {'countryname': [],
             'year':[],
             Var_Interest: []}

    for result in connection.execute(stmt):
        dictionary['countryname'].append(result.countryname)
        #dictionary['countrycode'].append(result.countrycode)
        dictionary['year'].append(result.year)
        dictionary[Var_Interest].append(result[Var_Interest])



    selected = pd.DataFrame(dictionary)


    return selected



## GEt year observations
def consecutive(iterable):
    """Yield range of consecutive numbers."""
    for group in mit.consecutive_groups(iterable):
        group = list(group)
        if len(group) == 1:
            yield group[0]
        else:
            yield group[0], group[-1]


def create_figure_bar_cross():
    country_vals = countries.value
    group_val = GroupSelect.value[0]
    no_data = []

    # Choose variable of interest
    Var_Interest = axis_map_notes[bar_cross_var.value][0]

    # Choose the order of the plots
    order = True
    if bar_order.active ==1:
        order=False

    # Get Data
    data = select_obs_bar_cross()

    # Drop missing values
    data.dropna(inplace=True)

    # Ensure adequate data exists.

    no_data= list(set(country_vals) - set(list(data.countryname.unique())))



    # If there is adequate data, then generate the plot
    try:

        # Generate observation list
        obs = ''
        n = 1


        for i in country_vals:
            end = ', '
            if len(country_vals)==n:
                end='. '
            obs+= i +': '+str(list(consecutive(data[data['countryname']==i]['year'].as_matrix()))[0])+end
            n+=1

        # break text into several lines
        obs = obs.replace(',', ' -').replace(') -', '),')

            # add additional notes to the list.
        obs = obs+ note_bar_cross.value



        # Average Chart
        if bar_plot_options.active ==0:
            # Generate source for bar chart
            #df = data
            data = data.groupby('countryname').mean()
            data.reset_index(inplace=True)
            data[Var_Interest] = data[Var_Interest].round(int(round_bar_cross.value))
            data.sort_values(Var_Interest, ascending=order, inplace=True)
            C_list = list(data['countryname'].unique())
            source = ColumnDataSource(data=dict(country=C_list, counts=list(data[Var_Interest])))
            #ource_cicle = ColumnDataSource(data=dict(country=df['countryname'],counts = df[Var_Interest]))
            p = figure(x_range=C_list, plot_height=(int(bar_height.value)*100), plot_width=900, background_fill_color="#eeeafa")
            colors = SET_palette[0:len(C_list)]
            if len(C_list) >7:
                colors = Category20b[len(C_list)]

            # Generate Plot
            p.vbar(x='country', top='counts', width=0.9, source=source, alpha=0.6,
                   line_color='white', fill_color=factor_cmap('country', palette=colors, factors=C_list))
            #p.circle(x='country', y='counts', source =source_cicle, size=30, fill_color=factor_cmap('country', palette=colors, factors=C_list))
            ### Add labels to bars
            labels = LabelSet(x='country', y='counts', text='counts', level='glyph',
            x_offset=-13.5, y_offset=0, source=source, render_mode='canvas')
            p.add_layout(labels)


            p.yaxis.axis_label = bar_cross_var.value + ' (Mean from '+minyear_bar_cross.value+ '-'+maxyear_bar_cross.value + ')'
            p.yaxis.axis_label = bar_cross_var.value
            #Generate Hover
            var_hover = '@'+Var_Interest+'{0.00 a}'
            hover = HoverTool(
                    tooltips=[
                        ('Country', '@country'),
                        ( bar_cross_var.value,   '@counts')
                    ]
            )
            p.add_tools(hover)


            # Generate notes for under the visual.
            obs = "\n".join(wrap(obs, 150)).split('\n')

            d = {0:"", 1:"", 2:"", 3:""}
            for i in range(0, len(obs)):
                d[i] = obs[i]

            msg1 = 'Source: '+axis_map_notes[bar_cross_var.value][1]+'.'
            caption1 = Label(text=msg1, **label_opts, text_color='#999999')
            p.add_layout(caption1, 'below')

            msg2 = 'Observations: ' +d[0]
            caption2 = Label(text=msg2, **label_opts, text_color='#999999')
            p.add_layout(caption2, 'below')

            msg3 = d[1]
            caption2 = Label(text=msg3, **label_opts, text_color='#999999')
            p.add_layout(caption2, 'below')

            msg4 = d[2]
            caption2 = Label(text=msg4, **label_opts, text_color='#999999')
            p.add_layout(caption2, 'below')

            msg4 = d[3]
            caption2 = Label(text=msg4, **label_opts, text_color='#999999')
            p.add_layout(caption2, 'below')


        # Country/Year Chart
        if bar_plot_options.active ==1:
            C_list  = list(data['countryname'].unique())

            # Chosen group year value
            groups = int(group_years.value)

            ##### Generate the group vars
            data[Var_Interest] = data.groupby(['countryname'])[Var_Interest].apply(pd.rolling_mean, groups, min_periods=groups)
            ### Drop the early years without previous records for means.
            data['year'] = data['year'].astype(int)
            data.dropna(inplace=True)
            data.sort_values(['countryname', 'year'], inplace=True)


            # Generate basic years list
            years1 = list(data['year'].unique())

            # list to fill proper year vars with
            years= []

            # Generate range of years if needed
            if groups == 1:
                for i in range(0, len(years1)):
                    years += [str(int(years1[i]))]

            if groups > 1:
                for y in years1:
                    years += [str(y-(groups-1))+'-'+str(y)]

            data['year'] = years*len(C_list)

            # Generate dictionary for source selected
            dataset = pd.DataFrame(columns = ['countryname', 'year', Var_Interest])
            # loop over the countries selected
            for c in C_list:
                #selected the country
                a = data[data['countryname']==c]

                # choose every _ observation based on widget selection
                a = a.iloc[::groups, :]
                # Change the years to proper intervals
                # choose every _ observation based on widget selection
                # loop through the countries to add to dictionary in proper format for plot
                dataset = dataset.append(a)

            years = list(dataset['year'].unique())


            dictionary = {'country': C_list}
            for y in years:
                    dictionary[y] = data[data['year']==y][Var_Interest].tolist()



            source = ColumnDataSource(data=dictionary)


            colors = SET_palette[0:len(years)]
            ### Generate (janky) spacing algorithm


            if len(years)>2:
                b = -.28-(.8/len(years))
                spacing = []
                for i in range(0,len(years)):
                    b = b+(.8/len(years))
                    spacing = spacing+[b]
            if len(years)==2:
                spacing = [-.15, .15]
            if len(years)==1:
                spacing = [0]



            p = figure(x_range=C_list, plot_height=600, plot_width=900, background_fill_color="#eeeafa")

            for y in range(0,len(years)):
                p.vbar(x=dodge('country', spacing[y], range=p.x_range), alpha=0.6,top=str(years[y]), width=bar_width.value, source=source,
                   color=colors[y], legend=value(str(years[y])))

            p.yaxis.axis_label = bar_cross_var.value
            #Generate Hover
            var_hover = '@'+Var_Interest+'{0.00 a}'
            hover = HoverTool(
                    tooltips=[
                        ( 'Year',   '@country'            )
                    ]
            )
            p.add_tools(hover)


        # Year/Country
        if bar_plot_options.active ==2:
            C_list = list(data['countryname'].unique())
            # Chosen group year value
            groups = int(group_years.value)

            ##### Generate the group vars
            data[Var_Interest] = data.groupby(['countryname'])[Var_Interest].apply(pd.rolling_mean, groups, min_periods=groups)
            ### Drop the early years without previous records for means.
            data['year'] = data['year'].astype(int)
            data.dropna(inplace=True)
            data.sort_values(['countryname', 'year'], inplace=True)

            # Generate basic years list
            years1 = list(data['year'].unique())

            # list to fill proper year vars with
            years= []

            # Generate range of years if needed
            if groups == 1:
                for i in range(0, len(years1)):
                    years += [str(int(years1[i]))]

            if groups > 1:
                for y in years1:
                    years += [str(y-(groups-1))+'-'+str(y)]

            # Generate dictionary for source selected
            dictionary = {'years': []}

            # loop over the countries selected
            for c in C_list:
                #selected the country
                a = data[data['countryname']==c]
                # Change the years to proper intervals
                a['year'] = years
                # choose every _ observation based on widget selection
                a = a.iloc[::groups, :]

                # Please proper years list in dictionary
                dictionary['years'] = list(a['year'].unique())

                # loop through the countries to add to dictionary in proper format for plot
                for y in years:
                    dictionary[c] = a[Var_Interest].tolist()


            source = ColumnDataSource(data=dictionary)
            colors = SET_palette[0:len(C_list)]
            if len(C_list) >7:
                colors = Category20b[len(C_list)]


            ### Generate (janky) spacing algorithm
            if len(C_list)>2:
                b = -.28-(.8/len(C_list))
                spacing = []
                for i in range(0,len(C_list)):
                    b = b+(.8/len(C_list))
                    spacing = spacing+[b]
            if len(C_list)==2:
                spacing = [-.15, .15]
            if len(C_list)==1:
                spacing = [0]


            p = figure(x_range=dictionary['years'], plot_height=550, plot_width=900, background_fill_color="#eeeafa")

            for y in range(0,len(C_list)):
                p.vbar(x=dodge('years', spacing[y], range=p.x_range), alpha=0.6, top=C_list[y], width=bar_width.value, source=source,
                   color=colors[y], legend=value(C_list[y]))
            p.yaxis.axis_label = bar_cross_var.value


            #Generate Hover
            var_hover = '@'+Var_Interest+'{0.00 a}'
            hover = HoverTool(
                    tooltips=[
                        ( 'Year',   '@years')
                    ]
            )
            p.add_tools(hover)
        ## Sizing settings
        if len(country_vals)>3:
            p.xaxis.major_label_orientation = pi/2


        #### SETTINGS TO MATCH Set
        p.xgrid.visible = False
        p.title.text_color = '#361c7f'
        p.title.text_font = "arial"
        p.title.text_font_style = "bold"
        p.title.text = title_bar_cross.value
        p.grid.grid_line_color='#CBCBCB'
        p.grid.grid_line_width=2.0
        p.xaxis.axis_label=''
        p.yaxis.axis_label=bar_cross_var.value
        p.axis.axis_label_text_font = 'arial'
        p.axis.axis_label_text_color = '#999999'
        p.axis.axis_label_text_font_style = 'normal'
        p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
        p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
        p.axis.axis_line_color=None
        p.axis.major_tick_line_color=None
        p.axis.minor_tick_line_color=None
        fontvalue = font_boxplot.value+'pt'
        p.title.text_font_size = fontvalue
        p.axis.major_label_text_font_size = '16pt'
        p.axis.major_label_text_color='#999999'
        p.outline_line_color = None
        p.axis.axis_label_text_font_size = '12pt'

        p.legend.orientation = axis_legend_orientation[legend_location_ori_bar_cross.value]
        '''
        msg2 = note_bar_cross2.value
        caption2 = Label(text=msg2, **label_opts, text_color='#999999')
        p.add_layout(caption2, 'below')

        msg3 = note_bar_cross3.value
        caption2 = Label(text=msg3, **label_opts, text_color='#999999')
        p.add_layout(caption2, 'below')

        msg4 = note_bar_cross4.value
        caption2 = Label(text=msg4, **label_opts, text_color='#999999')
        p.add_layout(caption2, 'below')
        '''
        p.legend.location = axis_legend[legend_location_bar_cross.value]

    except IndexError:

        p = figure(plot_width=900, plot_height=550, background_fill_color="#eeeafa")


        #### SETTINGS TO MATCH Set
        p.xgrid.visible = False
        p.title.text_color = '#361c7f'
        p.title.text_font = "arial"
        p.title.text_font_size = "20pt"
        p.title.text_font_style = "bold"
        p.grid.grid_line_color='#CBCBCB'
        p.grid.grid_line_width=2.0
        p.xaxis.axis_label='Year'
        p.yaxis.axis_label=line_var.value
        p.axis.axis_label_text_font = 'arial'
        p.axis.axis_label_text_color = '#999999'
        p.axis.axis_label_text_font_style = 'normal'
        p.axis.axis_label_text_font_size = '16pt'
        p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
        p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
        p.axis.axis_line_color=None
        p.axis.major_tick_line_color=None
        p.axis.minor_tick_line_color=None
        p.axis.major_label_text_font_size = '16pt'
        p.axis.major_label_text_color='#999999'
        p.outline_line_color = None
        p.legend.click_policy="mute"
        label = Label(x=1.1, y=18, text='Insufficient Data', text_font_size='70pt', text_color='#016450')
        p.add_layout(label)

        p.xaxis.axis_label = ''
        p.yaxis.axis_label = ''
        p.title.text = 'Insufficient Data : '+str(no_data)



    return p



#####################################################################
#####################################################################
###            Line Chart - Over Time Comparison                 ####
#####################################################################
#####################################################################


title_linecross = TextInput(title="Title", value="Line Chart Over Time")
line_var = Select(title="Variable of Interest", options=sorted(axis_map_notes.keys()), value="Proportion of Employment (%): Agriculture")
minyear_linecross = Select(title="Start Year", options=sorted(axis_year.keys()), value="1991")
maxyear_linecross = Select(title="End Year", options=sorted(axis_year.keys()), value="2016")
font_linecross = Select(title="Title Font Size", options=sorted(axis_font.keys()), value="24")
note_linecross = TextInput(title="Additional Note Content", value="")
note_linecross2 = TextInput(title="Additional Note Content - Line 2", value="")
note_linecross3 = TextInput(title="Additional Note Content - Line 3", value="")
note_linecross4 = TextInput(title="Additional Note Content - Line  4", value="")
legend_location_linecross = Select(title="Legend Location", options=sorted(axis_legend.keys()), value="Bottom Left")
legend_location_ori_linecross = Select(title="Legend Orientation", options=sorted(axis_legend_orientation.keys()), value="Vertical")
linecross_scatter = RadioButtonGroup(
        labels=["Line", "Line with scatter"], active=0, width=250)
rolling_linecross = Slider(title='Rolling Mean - Years', value=1, start=1, end=5)

def select_obs_linecross():
    country_vals = countries.value
    Var_Interest = axis_map_notes[line_var.value][0]
    # Select the Right Variables
    #selected = data[['countrycode', 'countryname' , 'year', Var_Interest]]

    minyear = int(minyear_linecross.value)
    maxyear = int(maxyear_linecross.value)




    # Select the Correct Observations
    stmt = stmt_main.where(and_(
        data_table.columns.countryname.in_(country_vals),
        data_table.columns.year.between(minyear,maxyear)))

    dictionary = {'countryname': [],
             #'countrycode': [],
             'year':[],
             Var_Interest: []}
    for result in connection.execute(stmt):
        dictionary['countryname'].append(result.countryname)
        #dictionary['countrycode'].append(result.countrycode)
        dictionary['year'].append(result.year)
        dictionary[Var_Interest].append(result[Var_Interest])

    # Generate the moving average
    selected = pd.DataFrame(dictionary).sort_values('year')
    selected[Var_Interest] = selected.groupby(['countryname'])[Var_Interest].apply(pd.rolling_mean, rolling_linecross.value, min_periods=rolling_linecross.value)

    #selected[Var_Interest] = pd.rolling_mean(selected[Var_Interest], 5, min_periods=5, center=False)
    #selected.reset_index(inplace=True)


    dict2 = {'a': [1]}

    # Generate dataset for plotting scatter with line charts
    if linecross_scatter.active ==1:
        # Select the Correct Observations
        stmt2 = stmt_main.where(
            data_table.columns.year.between(minyear,maxyear))

        dict2 = {'countryname': [],
                 'year':[],
                 Var_Interest: []}

        for result in connection.execute(stmt2):
            dict2['countryname'].append(result.countryname)
            #dict2['countrycode'].append(result.countrycode)
            dict2['year'].append(result.year)
            dict2[Var_Interest].append(result[Var_Interest])
        dict2 = pd.DataFrame(dict2)
        dict2[Var_Interest] = dict2.groupby(['countryname'])[Var_Interest].apply(pd.rolling_mean, rolling_linecross.value, min_periods=rolling_linecross.value)


    source_scatter = ColumnDataSource(dict2)

    legends = {}

    for c in country_vals:
        legends[c] = selected[selected['countryname']==c]['countryname'].reset_index(drop=True)[0]
    #selected.drop('countryname', axis=1, inplace=True)
    # Soruces for Cirlces in the line graph
    sources = {}
    for c in country_vals:
        df = selected[selected['countryname']==c]
        sources[c] = ColumnDataSource(df)

    # min and max for the years
    m = min(selected['year'])
    M = max(selected['year'])

    # min and max for the years
    m = min(selected['year'])
    M = max(selected['year'])

    ##########################################
    # Generate the curned lines for the charts


    no_data = []
    lines = {}

    i = 0
    for c in country_vals:

        # Generate country specfic dataset - cleaner
        country_df = pd.DataFrame()
        country_df['year']= selected[selected['countryname']==c]['year']
        country_df[Var_Interest] = selected[selected['countryname']==c][Var_Interest]
        country_df = country_df.interpolate().dropna()
        if len(country_df)!=0:
            # Generate the areas and x2s for each country
            df = pd.DataFrame()
            df['year'] = np.linspace(country_df['year'].min(), country_df['year'].max(), 180)

            df[Var_Interest] = spline(country_df['year'], country_df[Var_Interest], df['year'])
            lines[c] = ColumnDataSource(df)
        else:
            no_data.append(c)

    selected.columns = ['countryname', 'year', 'Var_Interest']




    return {'sources': sources, 'lines': lines, 'm':m, 'M':M, 'legends': legends, 'source_scatter':source_scatter, 'no_data': no_data}

def create_figure_linecross():
    country_vals = countries.value
    Var_Interest = axis_map_notes[line_var.value][0]
    data = select_obs_linecross()
    no_data = data['no_data']

    if len(no_data) == False:
        lines = data['lines']
        sources = data['sources']
        legends = data['legends']
        source_scatter = data['source_scatter']
        fontvalue = font_linecross.value+'pt'
        num = len(countries.value)
        colors = SET_palette[0:num]



        p = figure(plot_width=900, plot_height=650, background_fill_color="#eeeafa")
        a = 1
        if linecross_scatter.active ==1:
            p = figure(plot_width=1000, plot_height=600, background_fill_color="#eeeafa")
            p.circle('year', Var_Interest, source=source_scatter, color='#999999', alpha=.25)

        l=0
        for n in country_vals:
            p.line('year', Var_Interest, legend=legends[n], source=lines[n], line_width=4, color=colors[l], muted_color=colors[l], muted_alpha=0.3)
            l+=1
        l=0
        for n in country_vals:
            p.circle('year', Var_Interest,legend=legends[n], name=n, source=sources[n], color=colors[l], size=10, muted_color=colors[l], muted_alpha=0.3)
            l+=1

        p.legend.location='bottom_left'
        # Adjust the Labels of the Plot
        p.title.text = title_linecross.value
        p.title.text_font_size = fontvalue
        p.xaxis.axis_label = 'Year'
        p.yaxis.axis_label = line_var.value
        var_hover = '@'+axis_map_notes[line_var.value][0]+'{0.00 a}'
        hover = HoverTool(
            names=country_vals,
            tooltips=[
                ('Country', '@countryname'),
                ( 'Year',   '@year'            ),
                ( line_var.value, var_hover  ),
            ],
            # display a tooltip whenever the cursor is vertically in line with a glyph
            mode='vline'
        )
        p.add_tools(hover)
        msg1 = 'Source: '+axis_map_notes[line_var.value][1]+'. '+note_linecross.value
        caption1 = Label(text=msg1, **label_opts, text_color='#999999')
        p.add_layout(caption1, 'below')

        msg2 = note_linecross2.value
        caption2 = Label(text=msg2, **label_opts, text_color='#999999')
        p.add_layout(caption2, 'below')

        msg3 = note_linecross3.value
        caption2 = Label(text=msg3, **label_opts, text_color='#999999')
        p.add_layout(caption2, 'below')

        msg4 = note_linecross4.value
        caption2 = Label(text=msg4, **label_opts, text_color='#999999')
        p.add_layout(caption2, 'below')

        #### SETTINGS TO MATCH Set
        p.xgrid.visible = False
        p.title.text_color = '#361c7f'
        p.title.text_font = "arial"
        p.title.text_font_style = "bold"
        p.grid.grid_line_color='#CBCBCB'
        p.grid.grid_line_width=2.0
        p.xaxis.axis_label='Year'
        p.yaxis.axis_label=line_var.value
        p.axis.axis_label_text_font = 'arial'
        p.axis.axis_label_text_color = '#999999'
        p.axis.axis_label_text_font_style = 'normal'
        p.axis.axis_label_text_font_size = '12pt'
        p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
        p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
        p.axis.axis_line_color=None
        p.axis.major_tick_line_color=None
        p.axis.minor_tick_line_color=None
        p.axis.major_label_text_font_size = '16pt'
        p.axis.major_label_text_color='#999999'
        p.outline_line_color = None
        p.legend.orientation = axis_legend_orientation[legend_location_ori_linecross.value]
        p.legend.location = axis_legend[legend_location_linecross.value]
        p.legend.background_fill_alpha = 0.3
        p.legend.click_policy="mute"

    # Generate option if there is not enough data
    if len(no_data) >0:

        p = figure(plot_width=900, plot_height=550, background_fill_color="#eeeafa")


        #### SETTINGS TO MATCH Set
        p.xgrid.visible = False
        p.title.text_color = '#361c7f'
        p.title.text_font = "arial"
        p.title.text_font_size = "20pt"
        p.title.text_font_style = "bold"
        p.grid.grid_line_color='#CBCBCB'
        p.grid.grid_line_width=2.0
        p.xaxis.axis_label='Year'
        p.yaxis.axis_label=line_var.value
        p.axis.axis_label_text_font = 'arial'
        p.axis.axis_label_text_color = '#999999'
        p.axis.axis_label_text_font_style = 'normal'
        p.axis.axis_label_text_font_size = '16pt'
        p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
        p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
        p.axis.axis_line_color=None
        p.axis.major_tick_line_color=None
        p.axis.minor_tick_line_color=None
        p.axis.major_label_text_font_size = '16pt'
        p.axis.major_label_text_color='#999999'
        p.outline_line_color = None
        p.legend.click_policy="mute"
        label = Label(x=1.1, y=18, text='Insufficient Data', text_font_size='70pt', text_color='#016450')
        p.add_layout(label)
        p.xaxis.axis_label = ''
        p.yaxis.axis_label = ''
        p.title.text = 'Insufficient Data : '+str(no_data)


    return p

def update_linecrossdata():
    source = ColumnDataSource(data=dict())
    country_vals = countries.value
    Var_Interest = axis_map_notes[line_var.value][0]
    minyear = int(minyear_linecross.value)
    maxyear = int(maxyear_linecross.value)


    #selected = data.loc[data['countrycode'].isin(country_vals)]
    # Select the Right Variables
    #selected = selected[['countryname', 'countrycode', 'year', Var_Interest]]


    # Select the Correct Observations
    stmt = stmt_main.where(and_(
        data_table.columns.countryname.in_(country_vals),
        data_table.columns.year.between(minyear,maxyear)))

    dictionary = {'countryname': [],
             #'countrycode': [],
             'year':[],
             Var_Interest: []}
    for result in connection.execute(stmt):
        dictionary['countryname'].append(result.countryname)
        #dictionary['countrycode'].append(result.countrycode)
        dictionary['year'].append(result.year)
        dictionary[Var_Interest].append(result[Var_Interest])


    selected = pd.DataFrame(dictionary)
    selected[Var_Interest] = selected.groupby(['countryname'])[Var_Interest].apply(pd.rolling_mean, rolling_linecross.value, min_periods=rolling_linecross.value)



    # Soruces for line graph
    source.data = {
		'Country'             : selected.countryname,
		'Year'           : selected.year,
		'Var_Interest' : selected[Var_Interest]
        }

    return source



columns_linescross = [
    TableColumn(field="Country", title="Country"),
    TableColumn(field="Year", title="Year"),
    TableColumn(field="Var_Interest", title='Variable of Interest'),

]

#####################################################################
#####################################################################
###            Area Chart - Composition of Trade                 ####
#####################################################################
#####################################################################
# Seperate Data
#xport = ['countryname', 'year','exports_food_pctsum', 'exports_oresmet_pctsum', 'exports_fuel_pctsum', 'exports_manu_pctsum', 'exports_agraw_pctsum']
#export_data = data[export]


# Generate Widgets
exportarea_title = TextInput(title="Title", value="Export Composition Over Time")
min_yearTRADE = Select(title="Start Year", options=sorted(axis_year.keys()), value="1991")
max_yearTRADE = Select(title="End Year", options=sorted(axis_year.keys()), value="2016")
font_TRADE = Select(title="Title Font Size", options=sorted(axis_font.keys()), value="14")
note_TRADE = TextInput(title="Note Content", value="X-Axis: Proportion of Total Merchandise Exports. Y-axis: Year. Source: World Bank Development Indicators.")
note_TRADE = TextInput(title="Additional Note Content", value="")
note_TRADE2 = TextInput(title="Additional Note Content - Line 2", value="")
note_TRADE3 = TextInput(title="Additional Note Content - Line 3", value="")
note_TRADE4 = TextInput(title="Additional Note Content - Line  4", value="")



def select_obs_trade():
    country_val = CountryTRADE.value

    #selected = export_data.loc[export_data['countryname']==country_val]
    minyear = int(min_yearTRADE.value)
    maxyear = int(max_yearTRADE.value)
    #selected = selected[
    #    ( selected.year >= minyear) &
    #    ( selected.year <= maxyear)
    #]
    #m = min(selected['year'])
    #M = max(selected['year'])


    # Select the Correct Observations
    stmt = stmt_main.where(and_(
        data_table.columns.countryname==country_val,
        data_table.columns.year.between(minyear,maxyear)))

    dictionary = {'countryname': [],
             'year':[],
             'exports_food_pctsum':[],
             'exports_oresmet_pctsum':[],
             'exports_fuel_pctsum':[],
             'exports_manu_pctsum':[],
             'exports_agraw_pctsum':[],
             }
    for result in connection.execute(stmt):
        dictionary['countryname'].append(result.countryname)
        dictionary['exports_food_pctsum'].append(result.Var184)
        dictionary['exports_fuel_pctsum'].append(result.Var183)
        dictionary['exports_oresmet_pctsum'].append(result.Var185)
        dictionary['exports_manu_pctsum'].append(result.Var186)
        dictionary['exports_agraw_pctsum'].append(result.Var182)
        dictionary['year'].append(result.year)
        #dictionary[Var_Interest].append(result[Var_Interest])


    selected = pd.DataFrame(dictionary).dropna()

    return selected

def create_figure_trade():
    selected = select_obs_trade()
    m = min(selected['year'])
    M = max(selected['year'])
    selected.set_index('year', inplace=True, drop=True)
    selected.drop(['countryname'], axis=1, inplace=True)
    selected.columns = ['yy0', 'yy1','yy2','yy3','yy4']
    source = ColumnDataSource(selected)

    def  stacked(selected):
        df_top = selected.cumsum(axis=1)
        df_bottom = df_top.shift(axis=1).fillna({'yy0': 0})[::-1]
        df_stack = pd.concat([df_bottom, df_top], ignore_index=True)
        return df_stack
    names = ['Food', 'Fuel', 'Ores', 'Manu.', 'Ag. Raw']
    areas = stacked(selected)
    colors = color_blender[0:5]
    x2 = np.hstack((selected.index[::-1], selected.index))
    p = figure(x_range=(m, M), y_range=(0, 100), plot_height=600, plot_width=750)
    p.grid.minor_grid_line_color = '#eeeeee'

    p.patches([x2] * areas.shape[1], [areas[c].values for c in areas],
              color=colors, alpha=0.8, line_color=None)
              #legend=["%s Sector" % c for c in sectors]

    p.line(x='year', y='yy0',source=source, color='#440154', line_width=.2)


    ### Add labels
    labels = []
    for i, area in enumerate(areas):
        r = p.patch(x2, areas[area], color=colors[i], alpha=0.8, line_color=None)
        labels.append(LegendItem(label=dict(value=names[i]), renderers=[r]))

    legend = Legend(items=labels, location=(0, -30))
    p.add_layout(legend, 'right')

    msg1 = 'Source: World Bank Development Indicators. '+note_TRADE.value
    caption1 = Label(text=msg1, **label_opts, text_color='#999999')
    p.add_layout(caption1, 'below')

    msg2 = note_TRADE2.value
    caption2 = Label(text=msg2, **label_opts, text_color='#999999')
    p.add_layout(caption2, 'below')

    msg3 = note_TRADE3.value
    caption2 = Label(text=msg3, **label_opts, text_color='#999999')
    p.add_layout(caption2, 'below')

    msg4 = note_TRADE4.value
    caption2 = Label(text=msg4, **label_opts, text_color='#999999')
    p.add_layout(caption2, 'below')
    msg1 = note_TRADE.value

    # Adjust the Labels of the Plot
    fontvalue = font_TRADE.value+'pt'
    p.title.text = exportarea_title.value
    p.title.text_font_size = fontvalue

    # Add the HoverTool to the plot
    p.add_tools(hover_exports)

    #### SETTINGS TO MATCH Set
    p.xgrid.visible = False
    p.title.text_color = '#361c7f'
    p.title.text_font = "arial"
    p.title.text_font_style = "bold"
    p.grid.grid_line_color='#CBCBCB'
    p.grid.grid_line_width=2.0
    p.xaxis.axis_label='Year'
    p.axis.axis_label_text_font_size = '16pt'
    p.yaxis.axis_label='Proportion of Total Merchandise Exports (%)'
    p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
    p.axis.axis_line_color=None
    p.axis.major_tick_line_color=None
    p.axis.minor_tick_line_color=None
    p.axis.axis_label_text_font = 'arial'
    p.axis.axis_label_text_color = '#999999'
    p.axis.axis_label_text_font_style = 'normal'
    p.axis.major_label_text_font_size = '16pt'
    p.axis.major_label_text_color='#999999'
    p.outline_line_color = None
    return p


# Generate the HoverTool
hover_exports = HoverTool(

    tooltips=[
        ( 'Year',   '@year'            ),
        ( 'Ores and Metals', '@yy4{0.0 a}%' ),
        ( 'Manufactured Goods', '@yy3{0.0 a}%'     ),
	( 'Fuel', '@yy2{0.0 a}%'),
	( 'Food Products', '@yy1{0.0 a}%'       ),
	( 'Raw Agriculture',  '@yy0{0.0 a}%'       )
    ],
    # display a tooltip whenever the cursor is vertically in line with a glyph
    mode='vline'
)

def update_tradedata():
    source = ColumnDataSource(data=dict())
    country_val = CountryTRADE.value
    selected = select_obs_trade()

    # Soruces for line graph
    source.data = {
		'Country'             : selected.countryname,
		'Year'           : selected.year,
		'Food Exports' : selected.exports_food_pctsum,
        'Ores and Metals' : selected.exports_oresmet_pctsum,
        'Fuel Exports' : selected.exports_fuel_pctsum,
        'Manufactured Goods' : selected.exports_manu_pctsum,
        'Raw Agriculture Products': selected.exports_agraw_pctsum
        }

    return source



columns_tradedata = [
    TableColumn(field="Country", title="Country"),
    TableColumn(field="Year", title="Year"),
    TableColumn(field='Food Exports', title='Food Exports'),
    TableColumn(field='Ores and Metals', title='Ores and Metals'),
    TableColumn(field='Fuel Exports', title='Fuel Exports'),
    TableColumn(field='Manufactured Goods', title='Manufactured Goods'),
    TableColumn(field='Raw Agriculture Products', title='Raw Agriculture Products'),
]

#####################################################################
#####################################################################
###      Area Chart - Composition of Tax Rev (OIL RICH)          ####
#####################################################################
#####################################################################

# Generate Widgets

tax_rc_title = TextInput(title="Title - Resources Disaggregated", value="Tax Composition Over Time - Resource Revenue Disaggregated")
min_year_taxrc = Select(title="Start Year", options=sorted(axis_year.keys()), value="1991")
max_year_taxrc = Select(title="End Year", options=sorted(axis_year.keys()), value="2016")
fontrc = Select(title="Title Font Size", options=sorted(axis_font.keys()), value="14")
note_taxrc = TextInput(title="Note Content", value="X-Axis: Proportion of Tax Revenue - Accounting for Resource Revenue. Y-axis: Year. Source: International Centre for Tax and Development.")

# Because I put this within the SET category, there is no need to create the year vars


tax_vars= ['tot_resource_revpct', 'direct_inc_sc_ex_resource_revpct', 'tax_g_spct', 'tax_int_trade_transpct', 'other_rc']



# Generate Call-backs
def select_obs_tax_rc():
    country_val = CountryTAX.value
    #selected = tax_data.loc[tax_data['countrycode']==country_val]
    minyear = int(min_year_taxrc.value)
    maxyear = int(max_year_taxrc.value)
    # Select the Correct Observations
    stmt = stmt_main.where(and_(
        data_table.columns.countryname==country_val,
        data_table.columns.year.between(minyear,maxyear)))

    dictionary = {'countryname': [],
             'year':[],
             'tot_resource_revpct':[],
             'direct_inc_sc_ex_resource_revpct':[],
             'tax_g_spct':[],
             'tax_int_trade_transpct':[],
             'other_rc':[],
             }
    for result in connection.execute(stmt):
        dictionary['countryname'].append(result.countryname)
        dictionary['tot_resource_revpct'].append(result.Var13)
        dictionary['direct_inc_sc_ex_resource_revpct'].append(result.Var21)
        dictionary['tax_g_spct'].append(result.Var165)
        dictionary['tax_int_trade_transpct'].append(result.Var133)
        dictionary['other_rc'].append(result.Var155)
        dictionary['year'].append(result.year)
        #dictionary[Var_Interest].append(result[Var_Interest])

    selected = pd.DataFrame(dictionary).dropna()


    m = min(selected['year'])
    M = max(selected['year'])
    #selected = selected.dropna()

    df = pd.DataFrame()
    df['year'] = np.linspace(selected['year'].min(), selected['year'].max(), 180)

    for v in tax_vars:
        df[v]= spline(selected['year'], selected[v], df['year'])

    df.set_index('year', inplace=True, drop=True)
    df.columns = ['yy0', 'yy1','yy2','yy3', 'yy4']



    return {'df': df, 'selected': selected, 'm': m, 'M':M}




def create_figure_tax_rc():
    choice = select_obs_tax_rc()
    df = choice['df']
    source = choice['selected']
    m = choice['m']
    M = choice['M']
    fontvalue = fontrc.value+'pt'
    def  stacked(df):
        df_top = df.cumsum(axis=1)
        df_bottom = df_top.shift(axis=1).fillna({'yy0': 0})[::-1]
        df_stack = pd.concat([df_bottom, df_top], ignore_index=True)
        return {'df_stack': df_stack, 'df_top':df_top}


    stackedvals = stacked(df)
    areas = stackedvals['df_stack']

    source = ColumnDataSource(source)
    colors = all_palettes['Viridis'][5]
    x2 = np.hstack((df.index[::-1], df.index))
    p = figure(x_range=(m, M), y_range=(0, 1), plot_height=800, plot_width=900)
    p.grid.minor_grid_line_color = '#eeeeee'

    p.patches([x2] * areas.shape[1], [areas[c].values for c in areas],
              color=colors, alpha=0.8, line_color=None)

    p.circle(x='year', y='tot_resource_revpct',source=source, color='#440154', size=10)

    ### Add labels
    labels = []
    names = ['Recouces', 'Income/Capital','Goods and Services','Trade', 'Other']
    for i, area in enumerate(areas):
        r = p.patch(x2, areas[area], color=colors[i], alpha=0.8, line_color=None)
        labels.append(LegendItem(label=dict(value=names[i]), renderers=[r]))

    legend = Legend(items=labels, location=(0, -30))
    p.add_layout(legend, 'right')

    # Add the HoverTool to the plot
    p.add_tools(hover_tax)
    msg1 = note_taxrc.value
    caption1 = Label(text=msg1, **label_opts, text_color='#999999')
    p.add_layout(caption1, 'below')
    # Adjust the Labels of the Plot
    p.title.text = tax_rc_title.value
    p.title.text_font_size = fontvalue
    p.xaxis.axis_label = 'Year'
    p.yaxis.axis_label = 'Share of Tax Revenue (%)'

        #### SETTINGS TO MATCH Set
    p.xgrid.visible = False
    p.title.text_color = '#361c7f'
    p.title.text_font = "arial"
    p.title.text_font_style = "bold"
    p.grid.grid_line_color='#CBCBCB'
    p.grid.grid_line_width=2.0
    p.axis.axis_label_text_font = 'arial'
    p.axis.axis_label_text_color = '#999999'
    p.axis.axis_label_text_font_size = '16pt'
    p.axis.axis_label_text_font_style = 'normal'
    p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
    p.axis.axis_line_color=None
    p.axis.major_tick_line_color=None
    p.axis.minor_tick_line_color=None
    p.axis.major_label_text_font_size = '16pt'
    p.axis.major_label_text_color='#999999'
    p.outline_line_color = None


    return p



hover_tax = HoverTool(
    tooltips=[
        ( 'Year',   '@year'            ),
        ( 'Resources',  '@tot_resource_revpct{0.00 a}' ), # use @{ } for field names with spaces
        ( 'Income Taxes', '@direct_inc_sc_ex_resource_revpct{0.00 a}'      ),
	    ( 'Goods and Services', '@tax_g_spct{0.00 a}'),
	    ( 'Trade Taxes', '@tax_int_trade_transpct{0.00 a}'       ),
	    ( 'Other Sources', '@other_rc{0.00 a}'     )
    ],

    # display a tooltip whenever the cursor is vertically in line with a glyph
    mode='vline'
)



#####################################################################
#####################################################################
###      Area Chart - Composition of Tax Rev (non oil )          ####
#####################################################################
#####################################################################



tax_nonrc_title = TextInput(title="Title - Resources Within", value="Tax Composition Over Time")

def select_obs_tax_nonrc():
    country_val = CountryTAX.value
    #selected = data_nonrc.loc[data_nonrc['countryname']==country_val]
    minyear = int(min_year_taxrc.value)
    maxyear = int(max_year_taxrc.value)

    stmt = stmt_main.where(and_(
        data_table.columns.countryname==country_val,
        data_table.columns.year.between(minyear,maxyear)))

    dictionary = {'countryname': [],
             'year':[],
             'direct_inc_scpct':[],
             'tax_g_spct':[],
             'tax_int_trade_transpct':[],
             'other_nonrc':[],
             }
    for result in connection.execute(stmt):
        dictionary['countryname'].append(result.countryname)
        dictionary['direct_inc_scpct'].append(result.Var170)
        dictionary['tax_g_spct'].append(result.Var165)
        dictionary['tax_int_trade_transpct'].append(result.Var133)
        dictionary['other_nonrc'].append(result.Var33)
        dictionary['year'].append(result.year)
        #dictionary[Var_Interest].append(result[Var_Interest])

    selected = pd.DataFrame(dictionary).dropna()


    m = min(selected['year'])
    M = max(selected['year'])
    return {'selected': selected, 'm': m, 'M':M}

def create_figure_tax_nonrc():
    choice = select_obs_tax_nonrc()
    df = choice['selected']
    df.set_index('year', inplace=True, drop=True)
    df.drop(['countryname'], axis=1, inplace=True)

    df.columns = ['yy0', 'yy1','yy2','yy3']

    m = choice['m']
    M = choice['M']
    fontvalue = str(fontrc.value)+'pt'
    def  stacked(df):
        df_top = df.cumsum(axis=1)
        df_bottom = df_top.shift(axis=1).fillna({'yy0': 0})[::-1]
        df_stack = pd.concat([df_bottom, df_top], ignore_index=True)
        return {'df_stack': df_stack, 'df_top':df_top}


    stackedvals = stacked(df)
    areas = stackedvals['df_stack']

    source = ColumnDataSource(df)
    colors = all_palettes['Viridis'][4]
    x2 = np.hstack((df.index[::-1], df.index))
    p = figure(x_range=(m, M), y_range=(0, 1), plot_height=800, plot_width=900)
    p.grid.minor_grid_line_color = '#eeeeee'
    vars_list = ['r', 'a', 't', 't']
    p.patches([x2] * areas.shape[1], [areas[c].values for c in areas],
              color=colors, alpha=0.8, line_color=None)

    p.line(x='year', y='yy0',source=source, color='#440154', line_width=.2)

    names = ['Income/Capital','Goods and Services','Trade', 'Other']
    ### Add labels
    labels = []
    for i, area in enumerate(areas):
        r = p.patch(x2, areas[area], color=colors[i], alpha=0.8, line_color=None)
        labels.append(LegendItem(label=dict(value=names[i]), renderers=[r]))

    legend = Legend(items=labels, location=(0, -30))
    p.add_layout(legend, 'right')


    # Add the HoverTool to the plot
    p.add_tools(hover_taxnon)

    # Adjust the Labels of the Plot
    p.title.text = tax_nonrc_title.value
    p.title.text_font_size = fontvalue
    p.xaxis.axis_label = 'Year'
    p.yaxis.axis_label = 'Share of Tax Revenue'

    #### SETTINGS TO MATCH Set
    p.xgrid.visible = False
    p.title.text_color = '#361c7f'
    p.title.text_font = "arial"
    p.title.text_font_style = "bold"
    p.grid.grid_line_color='#CBCBCB'
    p.grid.grid_line_width=2.0
    p.axis.axis_label_text_font = 'arial'
    p.axis.axis_label_text_color = '#999999'
    p.axis.axis_label_text_font_style = 'normal'
    p.axis.axis_label_text_font_size = '16pt'
    p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
    p.axis.axis_line_color=None
    p.axis.major_tick_line_color=None
    p.axis.minor_tick_line_color=None
    p.axis.major_label_text_font_size = '16pt'
    p.axis.major_label_text_color='#999999'
    p.outline_line_color = None
    p.legend.orientation = "horizontal"

    return p



hover_taxnon = HoverTool(
    tooltips=[
        ( 'Year',   '@year'            ),
        ( 'Trade Taxes', '@yy3{0.00 a}%'      ),
	( 'Goods and Services Taxes', '@yy2{0.00 a}%'),
	( 'Other Taxes', '@yy1{0.00 a}%'       ),
	( 'Income Taxes', '@yy0{0.00 a}%'       )
    ],

    # display a tooltip whenever the cursor is vertically in line with a glyph
    mode='vline'
)
################# Tax Tables

def update_taxdatarc():
    source = ColumnDataSource(data=dict())
    selected = select_obs_tax_rc()['selected']
    #selected = data.loc[data['countryname']==country_val]

    # Soruces for line graph
    source.data = {
		'Country'             : selected.countryname,
		'Year'           : selected.year,
		'Resource' : selected.tot_resource_revpct,
        'Income' : selected.direct_inc_sc_ex_resource_revpct,
        'GandS' : selected.tax_g_spct,
        'Trade' : selected.tax_int_trade_transpct,
        'Other': selected.other_rc
        }

    return source


columns_taxdatarc = [
    TableColumn(field="Country", title="Country"),
    TableColumn(field="Year", title="Year"),
    TableColumn(field='Resource', title='Resource Revenue'),
    TableColumn(field='Income', title='Income and Capital Tax Revenue'),
    TableColumn(field='GandS', title='Goods and Service Tax Revenue'),
    TableColumn(field='Trade', title='Trade Tax Revenue'),
    TableColumn(field='Other', title='Other Revenue Sources'),
]

############ Tax table non-rc
def update_taxdata_nonrc():
    source = ColumnDataSource(data=dict())
    selected = select_obs_tax_nonrc()['selected']


    # Soruces for line graph
    source.data = {
		'Country'             : selected.countryname,
		'Year'           : selected.year,
        'Income' : selected.direct_inc_scpct,
        'GandS' : selected.tax_g_spct,
        'Trade' : selected.tax_int_trade_transpct,
        'Other': selected.other_nonrc
        }

    return source

columns_tax_nonrc = [
    TableColumn(field="Country", title="Country"),
    TableColumn(field="Year", title="Year"),
    TableColumn(field='Income', title='Income and Capital Tax Revenue'),
    TableColumn(field='GandS', title='Goods and Service Tax Revenue'),
    TableColumn(field='Trade', title='Trade Tax Revenue'),
    TableColumn(field='Other', title='Other Revenue Sources'),
]




#####################################################################
#####################################################################
###              Plot - Economic Transformation                  ####
#####################################################################
#####################################################################



# Generate Widgets
min_yearSET = Select(title="Start Year", options=sorted(axis_year.keys()), value="1991")
max_yearSET = Select(title="End Year", options=sorted(axis_year.keys()), value="2016")
fontSET = Select(title="Title Font Size", options=sorted(axis_font.keys()), value="14")
title_name_emp = TextInput(title="Title - Employment Share", value="Employment Share Over Time 1990-2016")



def select_obs_emp():
    country_val = CountryEMP.value
    #selected = data.loc[data['countryname']==country_val]
    #filter_col = [col for col in data if col.startswith(('employ_share', 'year', 'countryname'))]
    #selected = selected[filter_col]
    minyear = int(min_yearSET.value)
    maxyear = int(max_yearSET.value)


    # Select the Correct Observations
    stmt = stmt_main.where(and_(
        data_table.columns.countryname ==country_val,
        data_table.columns.year.between(minyear,maxyear)))

    dictionary = {'countryname': [],
                'employ_share_ag': [],
                'employ_share_manu': [],
                'employ_share_trans': [],
                'employ_share_retail': [],
                'employ_share_constr': [],
                'employ_share_mining': [],
                'employ_share_other': [],
                'year':[]}
    for result in connection.execute(stmt):
        dictionary['countryname'].append(result.countryname)
        #dictionary['countrycode'].append(result.countrycode)
        dictionary['year'].append(result.year)
        dictionary['employ_share_ag'].append(result.Var73)
        dictionary['employ_share_manu'].append(result.Var92)
        dictionary['employ_share_trans'].append(result.Var118)
        dictionary['employ_share_retail'].append(result.Var3)
        dictionary['employ_share_constr'].append(result.Var171)
        dictionary['employ_share_mining'].append(result.Var142)
        dictionary['employ_share_other'].append(result.Var36)


    selected = pd.DataFrame(dictionary).sort_values('year')

    m = min(selected['year'])
    M = max(selected['year'])

    selected.set_index('year', inplace=True, drop=True)
    selected.drop(['countryname'], axis=1, inplace=True)
    selected = selected.dropna()

    #### Fix observations which do not add to 100 (or close to it)
    variables = [col for col in selected if col.startswith('employ')]
    num = selected[variables].sum(axis=1).to_frame()

    if 99.5 >= num[0][m] or num[0][m] >= 100.5:

        selected['total'] = selected[variables].sum(axis=1)
        for i in variables:
            selected[i] = selected[i]/selected['total']*100

        selected.drop('total', inplace=True, axis=1)

    selected.columns = ['yy0', 'yy1','yy2','yy3','yy4','yy5','yy6']

    return {'selected': selected, 'm': m, 'M':M}

def create_figure_emp():
    choice = select_obs_emp()
    df = choice['selected']
    m = choice['m']
    M = choice['M']
    source = ColumnDataSource(df)

    def  stacked(df):
        df_top = df.cumsum(axis=1)
        df_bottom = df_top.shift(axis=1).fillna({'yy0': 0})[::-1]
        df_stack = pd.concat([df_bottom, df_top], ignore_index=True)
        return df_stack


    areas = stacked(df)
    tools = "pan,wheel_zoom,box_zoom,reset,save"
    x2 = np.hstack((df.index[::-1], df.index))

    p= figure(x_range=(m, M), y_range=(0, 100), plot_height=800, plot_width=950, tools=tools,)
    p.grid.minor_grid_line_color = '#eeeeee'

    p.patches([x2] * areas.shape[1], [areas[c].values for c in areas],
              color=color_blender[0:7], alpha=0.8, line_color=None)
    p.line(x='year', y='yy0',source=source, color='#016450', line_width=.2)

    names = ['Agiculture','Contruction', 'Manufacturing', 'Mining',  'Other', 'Retail', 'Transportation']
    ### Add labels
    labels = []
    for i, area in enumerate(areas):
        r = p.patch(x2, areas[area], color=color_blender[i], alpha=0.8, line_color=None)
        labels.append(LegendItem(label=dict(value=names[i]), renderers=[r]))

    legend = Legend(items=labels, location=(0, -30))
    p.add_layout(legend, 'right')

    # Adjust the Labels of the Plot
    fontvalue = fontSET.value+'pt'
    p.title.text = title_name_emp.value+': '+CountryEMP.value
    p.title.text_font_size = fontvalue
    p.xaxis.axis_label = 'Year'
    p.yaxis.axis_label = 'Employment Share (%)'
    # Add the HoverTool to the plot
    p.add_tools(hover_emp)
    #### SETTINGS TO MATCH Set
    p.xgrid.visible = False
    p.title.text_color = '#361c7f'
    p.title.text_font = "arial"
    p.title.text_font_style = "bold"
    p.grid.grid_line_color='#CBCBCB'
    p.grid.grid_line_width=2.0
    p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
    p.axis.axis_line_color=None
    p.axis.major_tick_line_color=None
    p.axis.minor_tick_line_color=None
    p.axis.axis_label_text_font = 'arial'
    p.axis.axis_label_text_color = '#999999'
    p.axis.axis_label_text_font_size = '16pt'
    p.axis.axis_label_text_font_style = 'bold'
    p.title.text_font_style = "bold"
    p.axis.major_label_text_font_size = '16pt'
    p.axis.major_label_text_color='#999999'
    p.outline_line_color = None


    return p




hover_emp = HoverTool(
    tooltips=[
        ( 'Year',   '@year'            ),
        ( 'Transportation', '@yy6{0.0 a}%'      ),
	( 'Retail', '@yy5{0.0 a}%' ),
	( 'Other', '@yy4{0.0 a}%'       ),
	( 'Mining', '@yy3{0.0 a}%'       ),
        ( 'Manufacturing', '@yy2{0.0 a}%'        ),
        ( 'Contruction', '@yy1{0.0 a}%'      ),
        ( 'Agriculture', '@yy0{0.0 a}%'       )
    ],

    # display a tooltip whenever the cursor is vertically in line with a glyph
    mode='vline'
)


################# GVA app
title_name_gva = TextInput(title="Title - GVA Share", value="GVA Percentage Over Time 1990-2016")


def select_obs_gva():
    country_val = CountryEMP.value
    #selected = data.loc[data['countryname']==country_val]
    #filter_col = [col for col in data if col.startswith(('GVA_shar', 'year', 'countryname'))]
    #selected = selected[filter_col]
    minyear = int(min_yearSET.value)
    maxyear = int(max_yearSET.value)

    # Select the Correct Observations
    stmt = stmt_main.where(and_(
        data_table.columns.countryname ==country_val,
        data_table.columns.year.between(minyear,maxyear)))

    dictionary = {'countryname': [],
                'GVA_share_ag': [],
                'GVA_share_manu': [],
                'GVA_share_trans': [],
                'GVA_share_retail': [],
                'GVA_share_constr': [],
                'GVA_shar_mining': [],
                'GVA_share_other': [],
                'year':[]}
    for result in connection.execute(stmt):
        dictionary['countryname'].append(result.countryname)
        #dictionary['countrycode'].append(result.countrycode)
        dictionary['year'].append(result.year)
        dictionary['GVA_share_ag'].append(result.Var42)
        dictionary['GVA_share_manu'].append(result.Var129)
        dictionary['GVA_share_trans'].append(result.Var91)
        dictionary['GVA_share_retail'].append(result.Var50)
        dictionary['GVA_share_constr'].append(result.Var136)
        dictionary['GVA_shar_mining'].append(result.Var76)
        dictionary['GVA_share_other'].append(result.Var34)



    selected = pd.DataFrame(dictionary).sort_values('year')



    selected.drop(['countryname'], axis=1, inplace=True)
    selected = selected.dropna()
    m = min(selected['year'])
    M = max(selected['year'])
    selected.set_index('year', inplace=True, drop=True)


    #### Fix observations which do not add to 100 (or close to it)
    variables = [col for col in selected if col.startswith('GVA')]

    num = selected[variables].sum(axis=1).to_frame()

    if 99.5 >= num[0][m] or num[0][m] >= 100.5:

        selected['total'] = selected[variables].sum(axis=1)
        for i in variables:
            selected[i] = selected[i]/selected['total']*100

        selected.drop('total', inplace=True, axis=1)
    selected = selected[[ 'GVA_share_ag', 'GVA_share_constr', 'GVA_share_manu','GVA_shar_mining',
       'GVA_share_other', 'GVA_share_retail', 'GVA_share_trans']]
    selected.columns = ['yy0', 'yy1','yy2','yy3','yy4','yy5','yy6']

    return {'selected': selected, 'm': m, 'M':M}



def create_figure_gva():
    choice = select_obs_gva()
    df = choice['selected']
    m = choice['m']
    M = choice['M']

    source = ColumnDataSource(df)

    def  stacked(df):
        df_top = df.cumsum(axis=1)

        df_bottom = df_top.shift(axis=1).fillna({'yy0': 0})[::-1]
        df_stack = pd.concat([df_bottom, df_top], ignore_index=True)
        return df_stack

    areas = stacked(df)

    x2 = np.hstack((df.index[::-1], df.index))

    p = figure(x_range=(m, M), y_range=(0, 100), plot_height=800, plot_width=950)
    p.grid.minor_grid_line_color = '#eeeeee'

    p.patches([x2] * areas.shape[1], [areas[c].values for c in areas],
              color=color_blender[0:7], alpha=0.8, line_color=None)

    p.line(x='year', y='yy0',source=source, color='#016450', line_width=.2)


    names = ['Agiculture','Contruction', 'Manufacturing', 'Mining',  'Other', 'Retail', 'Transportation']
    ### Add labels
    labels = []
    for i, area in enumerate(areas):
        r = p.patch(x2, areas[area], color=color_blender[i], alpha=0.8, line_color=None)
        labels.append(LegendItem(label=dict(value=names[i]), renderers=[r]))

    legend = Legend(items=labels, location=(0, -30))
    p.add_layout(legend, 'right')
    fontvalue = fontSET.value+'pt'
    p.title.text = title_name_gva.value+': '+CountryEMP.value
    p.title.text_font_size = fontvalue
    p.xaxis.axis_label = 'Year'
    p.yaxis.axis_label = 'Gross Value Added (%)'
    # Add the HoverTool to the plot
    p.add_tools(hover_gva)

    #### SETTINGS TO MATCH Set
    p.xgrid.visible = False
    p.title.text_color = '#361c7f'
    p.title.text_font = "arial"
    p.title.text_font_style = "bold"
    p.grid.grid_line_color='#CBCBCB'
    p.grid.grid_line_width=2.0
    p.axis.axis_label_text_font_size = '16pt'
    p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
    p.axis.axis_line_color=None
    p.axis.major_tick_line_color=None
    p.axis.minor_tick_line_color=None
    p.axis.axis_label_text_font = 'arial'
    p.axis.axis_label_text_color = '#999999'
    p.axis.axis_label_text_font_style = 'bold'
    p.axis.major_label_text_font_size='16pt'
    p.axis.major_label_text_color='#999999'
    p.outline_line_color = None


    return p

hover_gva = HoverTool(
    tooltips=[
        ( 'Year',   '@year'            ),
        ( 'Transportation', '@yy6{0.0 a}%'      ),
	( 'Retail', '@yy5{0.0 a}%' ),
	( 'Other', '@yy4{0.0 a}%'       ),
	( 'Mining', '@yy3{0.0 a}%'       ),
        ( 'Manufacturing', '@yy2{0.0 a}%'        ),
        ( 'Contruction', '@yy1{0.0 a}%'      ),
        ( 'Agriculture', '@yy0{0.0 a}%'       )
    ],
    # display a tooltip whenever the cursor is vertically in line with a glyph
    mode='vline'
)




#####################################################################
#####################################################################
###              Plot - Economic Transformation                  ####
#####################################################################
#####################################################################


###########################################
# Add Widgets

minavyear = Select(title="Change in Employment Share Mean - Start Year", options=sorted(axis_year.keys()), value="2008")
maxavyear = Select(title="Change in Employment Share Mean - End Year", options=sorted(axis_year.keys()), value="2016")
font_relemp = Select(title="Title Font Size", options=sorted(axis_font.keys()), value="14")
note_relemp = TextInput(title="Additional Note Content", value="")
note_relemp2 = TextInput(title="Additional Note Content - Line 2", value="")
note_relemp3 = TextInput(title="Additional Note Content - Line 3", value="")
note_relemp4 = TextInput(title="Additional Note Content - Line  4", value="")

# Choose the title of the relemp visual
title_relemp = TextInput(title="Title - Employment Share", value="Changes in Employment Share - Over Time")
# Choose the Size of the Font
# Generate HoverTool
hover_relemp = HoverTool(name='circle',
                         tooltips=
                         [('Country', '@countryname'),
                          ('Industry', '@Industry'),
                          ('Change in Employment Share', '@empave{0.0 a}%'),
                          ('Relative Labour Productivity', '@relLP{0.0 a}'),
                          ('Share of Total Employment', '@ES{0.0 a}%')])

mining = RadioButtonGroup(
        labels=["Mining", "Without Mining"], active=0, width=250)
##################################################
#### Add the Callbacks


def select_obs_relemp():
    #saves the value chosen
    country_val = CountrySTRUC.value

    # Selected Country:
    #selected = df_empave.loc[df_empave['countryname']==country_val]
    #only selects years chosen in the average emp choice.
    minyear = int(minavyear.value)
    maxyear = int(maxavyear.value)

    # Select the Correct Observations
    stmt = stmt_main.where(and_(or_(
        data_table.columns.year==minyear,
        data_table.columns.year==maxyear),
        data_table.columns.countryname ==country_val))



    dictionary = {'countryname': [],
                'ES_Agriculture': [],
                'ES_Mining': [],
                'ES_Retail': [],
                'ES_Other': [],
                'ES_Construction': [],
                'ES_Transportation': [],
                'ES_Manufacturing': [],
                'relLP_Agriculture': [],
                'relLP_Mining': [],
                'relLP_Retail': [],
                'relLP_Other': [],
                'relLP_Construction': [],
                'relLP_Transportation': [],
                'relLP_Manufacturing': [],
                'year':[]}
    for result in connection.execute(stmt):
        dictionary['countryname'].append(result.countryname)
        #dictionary['countrycode'].append(result.countrycode)
        dictionary['year'].append(result.year)
        dictionary['ES_Agriculture'].append(result.Var73)
        dictionary['ES_Manufacturing'].append(result.Var92)
        dictionary['ES_Transportation'].append(result.Var118)
        dictionary['ES_Retail'].append(result.Var3)
        dictionary['ES_Construction'].append(result.Var171)
        dictionary['ES_Mining'].append(result.Var142)
        dictionary['ES_Other'].append(result.Var36)

        dictionary['relLP_Agriculture'].append(result.Var11)
        dictionary['relLP_Manufacturing'].append(result.Var107)
        dictionary['relLP_Transportation'].append(result.Var108)
        dictionary['relLP_Retail'].append(result.Var44)
        dictionary['relLP_Construction'].append(result.Var20)
        dictionary['relLP_Mining'].append(result.Var105)
        dictionary['relLP_Other'].append(result.Var51)


    selected = pd.DataFrame(dictionary).sort_values('year')

    # Set index to countrycode
    selected.set_index('countryname', inplace=True, drop=True)

    #create average change in employment var
    # I can imagine there is a much simpler way to accomplish this task.
    # But here is one way that works.

    types = ['Agriculture', 'Mining', 'Manufacturing' , 'Construction', 'Retail', 'Transportation', 'Other']

    # Loop through the types of dissagregates
    for t in types:
        #create a blank
        df_empavea = pd.DataFrame()

        # generate a column with the the only non-NaN value as the minavyear value of ES
        df_empavea['minyear'] = (selected[selected['year']==minyear]['ES_'+t])
        # same for maxavyear value
        df_empavea['maxyear'] = (selected[selected['year']==maxyear]['ES_'+t])

        # fill NAs with zero to allow for the sum argument to work
        df_empavea['minyear'] = df_empavea['minyear'].fillna(0)
        df_empavea['maxyear'] = df_empavea['maxyear'].fillna(0)



        # Group by countrycode and sum together the
        df_empavea1 = df_empavea.groupby('countryname').agg({'minyear' : 'sum'})
        df_empavea2 = df_empavea.groupby('countryname').agg({'maxyear' : 'sum'})


        # Join the datasets together
        growth = df_empavea1.join(df_empavea2, how='outer')
        # Replace zero options back with NaNs
        growth = growth.replace(to_replace=0, value=np.NaN)


        # Generate the empave_var and add to selected dataframe.
        # This value should be same for all years since a year disaggregate is not mentioned
        selected['empave_'+t] = ((growth['maxyear']-growth['minyear'])/growth['minyear'])*100



    # Drop all observations except the year chosen for the
    selected = selected[(selected.year == maxyear)]

    # Drop all CES vars, the visual focuses on averages
    filter_col = [col for col in selected if col.startswith(('empave', 'ES', 'relLP'))]
    selected = selected[filter_col]

    # Split columns between empave, ES and relLP and the types
    # This creates multi-level column names which can be shifted
    selected.columns = selected.columns.str.split('_', expand=True)

    # Transform the dataframe where:
    # the rows are only empave,
    selected = selected.stack(level=1)

    # Generate ES_scale to scale the circle sizes relative to their prevalance in society
    selected['ES_scale'] = selected['ES']*5
    selected['id']= selected.index
    d = selected['id'].apply(pd.Series)
    selected['countryname'] = d[0]
    selected['Industry']= d[1]
    if mining.active == 1:
        selected = selected[selected['Industry']!='Mining']

    source = ColumnDataSource(selected)
    return source


#########################################
#### Make the Plot

def create_figure_relemp():
    source = select_obs_relemp()
    a = source.data

    # Generate the Plot
    # Color Settings
    color_mapper = CategoricalColorMapper(factors=source.data['index'], palette=SET_palette)



    p = figure(plot_height=700, plot_width=850, title="",  background_fill_color="#eeeafa")


    regression1 = np.polyfit(source.data['empave'], source.data['relLP'], 1)

    x_1, y_1 = zip(*((i, i*regression1[0] + regression1[1]) for i in range(int(min(source.data['empave'])), int(max(source.data['empave'])))))
    #x2_1, y2_1 = zip(*((i, i*(regression2[0]**2)+i*regression2[1]+regression2[2]) for i in range(int(min(selected[x_name])), int(max(selected[x_name])))))

    p.line(x=x_1, y=y_1, color='#999999', line_width=5, name='regline')

    p.circle(x= 'empave', y= 'relLP', fill_alpha=0.6, source=source, size = 'ES_scale',
            color=dict(field='index', transform=color_mapper), legend=None, name='circle')

    maxval = np.amax(source.data['relLP'])+1
    # Add Line at x = 0 to make positive and negative growth clear
    #p.line([0,0], [-1, maxval], line_width=1, color='black')

    # Add the HoverTool to the plot
    p.add_tools(hover_relemp)

    # Adjust the Labels of the Plot
    fontvalue = font_relemp.value+'pt'
    p.title.text_font_size = fontvalue
    p.title.text = title_relemp.value
    p.xaxis.axis_label = 'Change in Employment Share (%): '+minavyear.value+' - '+ maxavyear.value
    p.yaxis.axis_label = 'Relative Labour Productivity ('+maxavyear.value+')'

    # Make lables for the table
    labels = LabelSet(x='empave', y='relLP', text='Industry', level='glyph',
                  x_offset=5, y_offset=5, source=source, render_mode='canvas')
    p.add_layout(labels)
    zero_line = Span(location=0,
                                  dimension='height', line_color='#999999', line_width=4)
    p.add_layout(zero_line)
    ### add footnotes
    msg1 = 'Source: United Nations Statistics Division. '+note_relemp.value
    caption1 = Label(text=msg1, **label_opts, text_color='#999999')
    p.add_layout(caption1, 'below')

    msg2 = note_relemp2.value
    caption2 = Label(text=msg2, **label_opts, text_color='#999999')
    p.add_layout(caption2, 'below')

    msg3 = note_relemp3.value
    caption2 = Label(text=msg3, **label_opts, text_color='#999999')
    p.add_layout(caption2, 'below')

    msg4 = note_relemp4.value
    caption2 = Label(text=msg4, **label_opts, text_color='#999999')
    p.add_layout(caption2, 'below')

    #### SETTINGS TO MATCH Set


    #p.xgrid.visible = False
    p.title.text_color = '#361c7f'
    p.title.text_font = "arial"
    p.title.text_font_style = "bold"
    p.grid.grid_line_color='#CBCBCB'
    p.grid.grid_line_width=2.0
    p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
    p.axis.axis_line_color=None
    p.axis.major_tick_line_color=None
    p.axis.minor_tick_line_color=None
    p.axis.axis_label_text_font = 'arial'
    p.axis.axis_label_text_color = '#999999'
    p.axis.axis_label_text_font_style = 'normal'
    p.axis.major_label_text_font_size = '16pt'
    p.axis.major_label_text_color='#999999'
    p.outline_line_color = None
    p.axis.axis_label_text_font_size = '16pt'
    p.legend.orientation = "vertical"

    return p


columns_relemp = [
    TableColumn(field="countryname", title="Country"),
    TableColumn(field="Industry", title="Year"),
    TableColumn(field="empave", title='Change in Employment Over Time-Period Chosen'),
    TableColumn(field="relLP", title='Relative Labour Productivity at End Year')

]


#####################################################################
#####################################################################
###          Plot - Relative LP and Employment Share             ####
#####################################################################
#####################################################################


minavyear = Select(title="Start Year", options=sorted(axis_year.keys()), value="2008")
maxavyear = Select(title="End Year", options=sorted(axis_year.keys()), value="2016")
font_relbar = Select(title="Title Font Size", options=sorted(axis_font.keys()), value="14")
note_relbar = TextInput(title="Additional Note Content", value="")

# Choose the title of the relbar visual
title_relbar = TextInput(title="Title", value="Relative Labour Productivity and Employment")
# Choose the Size of the Font
# Generate HoverTool


def select_obs_relbar():
    #saves the value chosen
    country_val = CountrySTRUC.value

    # Selected Country:
    #selected = df_empave.loc[df_empave['countryname']==country_val]
    #only selects years chosen in the average emp choice.
    minyear = int(minavyear.value)
    maxyear = int(maxavyear.value)

    # Select the Correct Observations
    stmt = stmt_main.where(and_(
        data_table.columns.year==maxyear,
        data_table.columns.countryname ==country_val))



    ES = {
                'ES_Agriculture': [],
                'ES_Mining': [],
                'ES_Retail': [],
                'ES_Other': [],
                'ES_Construction': [],
                'ES_Transportation': [],
                'ES_Manufacturing': []}
    REL = {
                'relLP_Agriculture': [],
                'relLP_Mining': [],
                'relLP_Retail': [],
                'relLP_Other': [],
                'relLP_Construction': [],
                'relLP_Transportation': [],
                'relLP_Manufacturing': []}
    for result in connection.execute(stmt):
        #ES['countrycode'].append(result.countrycode)
        #ES['year'].append(result.year)
        ES['ES_Agriculture'].append(result.Var73)
        ES['ES_Manufacturing'].append(result.Var92)
        ES['ES_Transportation'].append(result.Var118)
        ES['ES_Retail'].append(result.Var3)
        ES['ES_Construction'].append(result.Var171)
        ES['ES_Mining'].append(result.Var142)
        ES['ES_Other'].append(result.Var36)

        REL['relLP_Agriculture'].append(result.Var11)
        REL['relLP_Manufacturing'].append(result.Var107)
        REL['relLP_Transportation'].append(result.Var108)
        REL['relLP_Retail'].append(result.Var44)
        REL['relLP_Construction'].append(result.Var20)
        REL['relLP_Mining'].append(result.Var105)
        REL['relLP_Other'].append(result.Var51)



    types = ['Agriculture','Construction', 'Manufacturing' ,'Mining',   'Retail', 'Transportation', 'Other']

    data = []
    for row in pd.DataFrame(ES).iterrows():
        index, d = row
        data.append(d.tolist())


    top =[]
    for row in pd.DataFrame(REL).iterrows():
        index, d = row
        top.append(d.tolist())

    dictionary = {'top': top[0],
                'data': data[0],
                'names': types}

    dataset = pd.DataFrame(dictionary)
    dataset.sort_values('top', inplace=True)

    if mining.active ==1:
        dataset = dataset[dataset.names != 'Mining']


    return dataset


#########################################
#### Make the Plot

def create_figure_relbar():
    dataset = select_obs_relbar()
    # Generate the Plot
    # Color Settings

    top = dataset['top'].as_matrix()
    data = dataset['data'].as_matrix()
    names = dataset['names'].as_matrix()


    cutoffs = [0]
    for i in range(0, len(top)):
        a = data[i]+cutoffs[i]
        cutoffs.append(a)

    bottom = [0] * len(top)
    left=cutoffs[0:len(top)]
    right=cutoffs[1:(len(top)+1)]

    labels = []

    p = figure(plot_height=700, plot_width=850, title="",  background_fill_color="#eeeafa")


    for i in range(0,len(top)):
        r = p.quad(top=top[i], bottom=bottom[i], left=left[i],
               right=right[i], color=SET_palette[i], name =names[i]
              )
        labels.append(LegendItem(label=dict(value=names[i]), renderers=[r]))


    legend = Legend(items=labels, location=(0, -30))
    p.add_layout(legend, 'right')

    # Adjust the Labels of the Plot
    fontvalue = font_relbar.value+'pt'
    p.title.text_font_size = fontvalue
    p.title.text = title_relbar.value
    p.xaxis.axis_label = 'Employment Share (%)'
    p.yaxis.axis_label = 'Relative Labour Productivity'

    ### add footnotes
    msg1 = 'Source: United Nations Statistics Division. '+note_relbar.value
    caption1 = Label(text=msg1, **label_opts, text_color='#999999')
    p.add_layout(caption1, 'below')


    #### SETTINGS TO MATCH Se


    #p.xgrid.visible = False
    p.title.text_color = '#361c7f'
    p.title.text_font = "arial"
    p.title.text_font_style = "bold"
    p.grid.grid_line_color='#CBCBCB'
    p.grid.grid_line_width=2.0
    p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
    p.axis.axis_line_color=None
    p.axis.major_tick_line_color=None
    p.axis.minor_tick_line_color=None
    p.axis.axis_label_text_font = 'arial'
    p.axis.axis_label_text_color = '#999999'
    p.axis.axis_label_text_font_style = 'normal'
    p.axis.major_label_text_font_size = '16pt'
    p.axis.major_label_text_color='#999999'
    p.outline_line_color = None
    p.axis.axis_label_text_font_size = '16pt'
    p.legend.orientation = "vertical"

    return p


#columns_relbar = [
#    TableColumn(field="countryname", title="Country"),
#    TableColumn(field="Industry", title="Year"),
#    TableColumn(field="empave", title='Average Employment Over Time-Period Chosen'),
#    TableColumn(field="relLP", title='Relative Labour Productivity at End Year')

#]

#####################################################################
#####################################################################
#               Labour Productivity Between and Within              #
#####################################################################
#####################################################################

# Generate Widgets
sector_list = ['Agriculture', 'Manufacturing', 'Transportation', 'Retail', 'Construction', 'Mining', 'Other', 'Total']

LPList = []
for i in sector_list:
    LPList.append((i, i))


axis_map_withinbtw1 = {
    "Agriculture": ["Var75", "Var2"],
    "Mining": ["Var82", "Var24"],
    "Manufacturing": ["Var119", "Var10"],
    "Construction": ["Var102", "Var153"],
    "Retail": ["Var181", "Var166"],
    "Transportation": ["Var88", "Var154"],
    "Other": ["Var53", "Var151"]
    }
axis_map_withinbtw2 = {
    "Agriculture": ["Var140", "Var180"],
    "Mining": ["Var47", "Var167"],
    "Manufacturing": ["Var85", "Var83"],
    "Construction": ["Var86", "Var174"],
    "Retail": ["Var22", "Var72"],
    "Transportation": ["Var70", "Var69"],
    "Other": ["Var7", "Var66"],
    'Total': [["Var140","Var47", "Var85", "Var86","Var22", "Var70", "Var7"],  ["Var180","Var167", "Var83", "Var174", "Var72",  "Var69","Var66" ]]
    }

title_LP = TextInput(title="Title", value="Examine Labour Productivity Growth")
LP_var = Select(title="Variable of Interest", options=sorted(axis_map_withinbtw2.keys()), value="Agriculture")
minyear_LP = Select(title="Start Year", options=sorted(axis_year.keys()), value="2002")
maxyear_LP = Select(title="End Year", options=sorted(axis_year.keys()), value="2016")
font_LP = Select(title="Title Font Size", options=sorted(axis_font.keys()), value="24")
note_LP = TextInput(title="Additional Note Content - Line 1", value="")
note_LP2 = TextInput(title="Additional Note Content - Line 2", value="")
note_LP3 = TextInput(title="Additional Note Content - Line 3", value="")
note_LP4 = TextInput(title="Additional Note Content - Line  4", value="")
order_barLP = RadioButtonGroup(
        labels=["Ascending", "Descending"], active=0, width=250)
bar_widthLP = Slider(title='Width of Bars', start=0.05, end=.5, value=0.2, step=.025)
group_yearsLP = Select(title="Year Groupings", options=sorted(axis_groupyear.keys()), value="5")
LP_variables = MultiSelect(title="Sector Selections", value=sector_list,
                           options=LPList)
order_barLP = RadioButtonGroup(
        labels=["Ascending", "Descending"], active=0, width=250)


def select_obs_withinLP():
    country_vals = LPcountries.value

    # Select the Correct Observations
    stmt = stmt_main.where(and_(
        data_table.columns.countryname.in_(country_vals),
        data_table.columns.year.between(int(minyear_LP.value),int(maxyear_LP.value))))


    LP = axis_map_withinbtw2[LP_var.value]

    if LP_var.value=='Total':
        dictionary = {'countryname':[],
                    'year': []}
        for i in LP:
            for g in i:
                dictionary[g] = []


        LPS = LP[0]+LP[1]

        for result in connection.execute(stmt):
            dictionary['countryname'].append(result.countryname)
            dictionary['year'].append(result.year)
            for i in range(0, len(LPS)):

                dictionary[LPS[i]].append(result[LPS[i]])

        data = pd.DataFrame(dictionary)


        data[0] = data[LP[0]].sum(axis=1)
        data[1] = data[LP[1]].sum(axis=1)
        data = data[['countryname', 'year', 0, 1]]
        data.sort_values(['countryname', 'year'], inplace=True)


    else:
        dictionary = {'countryname': [],
                    #'countryname': [],
                    LP[0]: [],
                    LP[1]: [],
                    'year':[]}
        for result in connection.execute(stmt):
            #dictionary['countrycode'].append(result.countrycode)
            dictionary['countryname'].append(result.countryname)
            dictionary['year'].append(result.year)
            dictionary[LP[0]].append(result[LP[0]])
            dictionary[LP[1]].append(result[LP[1]])

        data = pd.DataFrame(dictionary)

    return data



def create_figure_withinbtw():
    data = select_obs_withinLP()

    C_list = list(data['countryname'].unique())
    LP = axis_map_withinbtw2[LP_var.value]
    if LP_var.value=='Total':
        LP = [0,1]

    # Chosen group year value
    groups =  int(group_yearsLP.value)

    ##### Generate the group vars
    data[LP[0]] = data.groupby(['countryname'])[LP[0]].apply(pd.rolling_mean, groups, min_periods=groups)
    data[LP[1]] = data.groupby(['countryname'])[LP[1]].apply(pd.rolling_mean, groups, min_periods=groups)

    ### Drop the early years without previous records for means.
    data['year'] = data['year'].astype(int)
    data.dropna(inplace=True)


    # Generate basic years list
    years1 = list(data['year'].unique())

    # list to fill proper year vars with
    years= []

    # Generate range of years if needed
    if groups == 1:
        for i in range(0, len(years1)):
            years += [str(int(years1[i]))]

    if groups > 1:
        for y in years1:
            years += [str(y-(groups-1))+'-'+str(y)]

    dictionary = {}
    # loop over the countries data
    for c in C_list:
        d = {}
        #data the country
        a = data[data['countryname']==c]
        # Change the years to proper intervals
        a['year'] = years
        # choose every _ observation based on widget selection
        a = a.iloc[::groups, :]

        # Please proper years list in dictionary
        d['years'] = list(a['year'].unique())

        # loop through the countries to add to dictionary in proper format for plot
        for y in years:
            d['Within'] = a[LP[0]].tolist()
            d['Between'] = a[LP[1]].tolist()
            d['Country'] = np.repeat([c], len(list(a['year'].unique())))

        dictionary[c]=d

    # Explore wet

    SET_palette = ['#361c7f','#9467bd', '#b35900', '#990000'] + Category20[20]
    sources  = {}
    for i, d in dictionary.items():
        sources[i] = ColumnDataSource(data=d)

    p = figure(x_range=dictionary[C_list[0]]['years'], title="Attempts at ", plot_height=800, plot_width=900,  background_fill_color="#eeeafa")

    legend_it = []


    ### Generate (janky) spacing algorithm
    if len(C_list)>3:
        b = -.28-(.8/len(C_list))
        spacing = []
        for i in range(0,len(C_list)):
            b = b+(.8/len(C_list))
            spacing = spacing+[b]
    if len(C_list)==3:
        spacing = [-.25, 0, .25]
    if len(C_list)==2:
        spacing = [-.15, .15]
    if len(C_list)==1:
        spacing = [0]


    n = 0
    names = []


    for i in range(0, len(C_list)):

        #Create the within plot
        c = p.vbar(x=dodge('years', spacing[i], range=p.x_range), top='Within', width=bar_widthLP.value, source=sources[C_list[i]], fill_alpha=0.65,
               color=SET_palette[n], name="Within " +LP_var.value+' Growth: '+C_list[i])
        legend_it.append(("Within " +LP_var.value+' Growth: '+C_list[i], [c]))
        names.append("Within " +LP_var.value+' Growth: '+C_list[i])
        n +=1

        # Plot Between on top of the within plot
        c = p.vbar(x=dodge('years',  spacing[i],  range=p.x_range), top='Between', width=bar_widthLP.value, source=sources[C_list[i]], fill_alpha=0.65,
               color=SET_palette[n], name= "Between " +LP_var.value+' Growth: '+C_list[i])
        n +=1
        legend_it.append(("Between " +LP_var.value+' Growth: '+C_list[i], [c]))
        names.append("Between " +LP_var.value+' Growth: '+C_list[i])


    legend3 = Legend(items=legend_it, location=(0, 0))
    p.x_range.range_padding = 0.1
    p.xgrid.grid_line_color = None
    p.add_layout(legend3, 'below')

    #Generate Hover
    hover = HoverTool(
            names = names,
            tooltips=[
                ('Country', '@Country'),
                ( 'Year',   '@years'),
                ('Within', '@Within{0.00 a}'),
                ('Between', '@Between{0.00 a}'),

            ]
    )

    p.add_tools(hover)

    p.x_range.range_padding = 0.01
    p.xgrid.grid_line_color = None
    fontvalue = font_LP.value+'pt'
    p.title.text_font_size = fontvalue
    p.title.text = title_LP.value

    msg1 = 'Source: United Nations Statistics Division. '+note_LP.value
    caption1 = Label(text=msg1, **label_opts, text_color='#999999')
    p.add_layout(caption1, 'below')

    msg2 = note_LP2.value
    caption2 = Label(text=msg2, **label_opts, text_color='#999999')
    p.add_layout(caption2, 'below')

    msg3 = note_LP3.value
    caption2 = Label(text=msg3, **label_opts, text_color='#999999')
    p.add_layout(caption2, 'below')

    msg4 = note_LP4.value
    caption2 = Label(text=msg4, **label_opts, text_color='#999999')
    p.add_layout(caption2, 'below')
    #### SETTINGS TO MATCH Set
    p.xgrid.visible = False
    p.title.text_color = '#361c7f'
    p.title.text_font = "arial"
    p.title.text_font_style = "bold"
    p.grid.grid_line_color='#CBCBCB'
    p.grid.grid_line_width=2.0
    p.xaxis.axis_label=''
    p.yaxis.axis_label='Change in Labour Productivity (%)'
    p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
    p.axis.axis_line_color=None
    p.axis.axis_label_text_font = 'arial'
    p.axis.axis_label_text_color = '#999999'
    p.axis.axis_label_text_font_style = 'normal'
    p.axis.major_tick_line_color=None
    p.axis.minor_tick_line_color=None
    p.axis.major_label_text_font_size = '16pt'
    p.axis.major_label_text_color='#999999'
    p.outline_line_color = None
    p.axis.axis_label_text_font_size = '16pt'
    p.legend.orientation = "vertical"

    return p


#####################################################################
#####################################################################
###    Labour Productivity Plots - Annualised Growth             ####
#####################################################################
#####################################################################




# ## Actual Plot Create Plot
def select_obs_annualLP():
    country_vals = CountryLP.value

    # Select the Correct Observations
    stmt = stmt_main.where(and_(
        data_table.columns.countryname==country_vals,
        data_table.columns.year.between(int(minyear_LP.value),int(maxyear_LP.value))))


    dictionary = {#'countrycode': [],
                  'countryname': [],
                'Agriculture': [],
                'Mining': [],
                'Retail': [],
                'Other': [],
                'Construction': [],
                'Transportation': [],
                'Manufacturing': [],
                'Total': [],
                'year':[]}

    for result in connection.execute(stmt):
        #dictionary['countrycode'].append(result.countrycode)
        dictionary['countryname'].append(result.countryname)
        dictionary['Agriculture'].append(result.Var124)
        dictionary['Manufacturing'].append(result.Var37)
        dictionary['Transportation'].append(result.Var0)
        dictionary['Retail'].append(result.Var104)
        dictionary['Construction'].append(result.Var40)
        dictionary['Mining'].append(result.Var52)
        dictionary['Other'].append(result.Var145)
        dictionary['Total'].append(result.Var38)
        dictionary['year'].append(result.year)

    data = pd.DataFrame(dictionary)
    data = data[LP_variables.value + ['countryname', 'year']]
    return data

def create_figure_annualLP():
    data = select_obs_annualLP()
    var_list = LP_variables.value
    # Chosen group year value

    # Multiple Bar Plot
    a = 0
    if a ==0:

        groups = int(group_yearsLP.value)

        ##### Generate the group vars
        for var in var_list:
            data[var]= data.groupby(['countryname'])[var].apply(pd.rolling_mean, groups, min_periods=groups)


        ### Drop the early years without previous records for means.
        data['year'] = data['year'].astype(int)
        data.dropna(inplace=True)
        # Generate basic years list
        years1 = list(data['year'].unique())

        # list to fill proper year vars with
        years= []

        # Generate range of years if needed
        if groups == 1:
            for i in range(0, len(years1)):
                years += [str(int(years1[i]))]

        if groups > 1:
            for y in years1:
                years += [str(y-(groups-1))+'-'+str(y)]

        # Change the years to proper intervals
        data['year'] = years
        # choose every _ observation based on widget selection
        data = data.iloc[::groups, :]



        SET_palette = ['#361c7f','#9467bd', '#b35900', '#990000'] + Category20[20]


        sourceLP = ColumnDataSource(data)

        p = figure(x_range=list(data['year'].as_matrix()), plot_width=900, plot_height=700,  background_fill_color="#eeeafa")

        legend_it = []


        ### Generate (janky) spacing algorithm
        if len(var_list)>3:
            b = -.28-(.8/len(var_list ))
            spacing = []
            for i in range(0,len(var_list )):
                b = b+(.8/len(var_list ))
                spacing = spacing+[b]
        if len(var_list )==3:
            spacing = [-.25, 0, .25]
        if len(var_list )==2:
            spacing = [-.25, .25]
        if len(var_list )==1:
            spacing = [0]
        ORDER = True
        if order_barLP.active==1:
            ORDER = False

        # Order the plot in ascending or descending order depending on choices
        var_list = list(data[var_list].mean().sort_values(ascending=ORDER).index)

        n = 0
        names = []

        for i in range(0, len(var_list)):
            c = p.vbar(x=dodge('year', spacing[i], range=p.x_range), top=var_list[i], width=bar_widthLP.value, source=sourceLP, fill_alpha=0.65,
                   color=SET_palette[n], name = var_list[i])
            legend_it.append((var_list[i], [c]))
            names.append(var_list[i])

            n +=1


        legend = Legend(items=legend_it, location=(0,0), orientation='horizontal')
        p.x_range.range_padding = 0.1
        p.xgrid.grid_line_color = None
        p.legend.orientation = "horizontal"
        p.add_layout(legend, 'below')

        tuples = []
        for i in var_list:
            tuples.append((i+' Sector', '@'+i+'{0.00 a}'))
        tuples = [('Country', '@countryname'),('Year(s)' , '@year')] + tuples

        #Generate Hover
        hover = HoverTool(
                names = names,
                tooltips=tuples,
        )

        p.add_tools(hover)


    #### SETTINGS TO MATCH Set

    p.title.text = title_LP.value
    p.title.text_font_size = font_LP.value + 'pt'
    p.grid.grid_line_alpha=0.3
    p.xgrid.visible = False
    p.title.text_color = '#361c7f'
    p.title.text_font = "arial"
    p.title.text_font_style = "bold"
    p.axis.axis_label_text_font_size = '16pt'
    p.grid.grid_line_color='#CBCBCB'
    p.grid.grid_line_width=2.0
    p.xaxis.axis_label=''
    p.yaxis.axis_label=''
    p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
    p.axis.axis_line_color=None
    p.axis.major_tick_line_color=None
    p.axis.minor_tick_line_color=None
    p.axis.major_label_text_font_size='16px'
    p.axis.major_label_text_color='#999999'
    p.outline_line_color = None

    msg1 = 'Source: UN Statistics Division'+note_LP.value
    caption1 = Label(text=msg1, **label_opts, text_color='#999999')
    p.add_layout(caption1, 'below')

    msg2 = note_LP2.value
    caption2 = Label(text=msg2, **label_opts, text_color='#999999')
    p.add_layout(caption2, 'below')


    return p



#####################################################################
#####################################################################
#               Scatter Plot - Explore Policies                     #
#####################################################################
#####################################################################


# Generate the Widgets
GDP = Slider(title="Maximum GDP Per Capita (Thousands)", value=40000, start=0, end=40000, step=2000)
minyear = Select(title="Start Year", options=sorted(axis_year.keys()), value="1991")
maxyear = Select(title="End Year", options=sorted(axis_year.keys()), value="2016")
font_scatter = Select(title="Title Font Size", options=sorted(axis_font.keys()), value="14")
#Country_sc = TextInput(title="Country")
x_axis  = Select(title="X Axis", options=sorted(axis_map_notes.keys()), value="Proportion of Employment (%): Agriculture")
y_axis = Select(title="Y Axis", options=sorted(axis_map_notes.keys()), value="GDP per capita (constant 2010 US$) (thousands)")
title_name = TextInput(title="Title", value="Scatter Plot")
note_scatter = TextInput(title="Additional Note Content", value='')
### Adding notes to Scatter acts very weird.  Address at different point.


# Generate HoverTool
hover_scatter = HoverTool(tooltips=[('Country', '@countryname'),
                           ('Year', '@year'),
			   ('X Value', '@xx{0.00 a}'),
			   ('Y Value', '@yy{0.00 a}')])

# Generate the Blank Source
source = ColumnDataSource(data=dict(countryname = [] , xx= [], yy= [], year= [], color=[], OECD_norm=[]))
source1 = ColumnDataSource(data=dict(rx = [], ry =[]))
#source2 = ColumnDataSource(data=dict(r2x = [], r2y =[]))

# Create Button which Downloads CSV file
button = Button(label="Download Data", button_type="success")
button.callback = CustomJS(args=dict(source=source),
                           code=open(join(dirname(__file__), 'models', 'download.js')).read())


# Generate the Plot

t = figure(plot_height=900, plot_width=900, background_fill_color="#eeeafa")
t.circle(x= 'xx', y= 'yy', fill_alpha=0.5, source=source,
        color='color', size = 10, legend='OECD_norm', name='scatter')

t.line(x='rx', y='ry', color='#999999', line_width=5, source=source1,name='regline')
#t.line(x='r2x', y='r2y', color='#999999', line_width=5, source=source2)


# Adjust the Labels of the Plot
    #### SETTINGS TO MATCH Set
t.title.text_color = '#361c7f'
t.title.text_font = "arial"
t.title.text_font_style = "bold"
t.grid.grid_line_color='#CBCBCB'
t.grid.grid_line_width=1.0
t.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
t.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
t.axis.axis_line_color=None
t.axis.major_tick_line_color=None
t.axis.minor_tick_line_color=None
t.axis.major_label_text_font_size = '16pt'
t.axis.major_label_text_color='#999999'
t.outline_line_color = None
t.axis.axis_label_text_font = 'arial'
t.axis.axis_label_text_color = '#999999'
t.axis.axis_label_text_font_size = '16pt'
t.axis.axis_label_text_font_style = 'normal'


# Add the Callbacks Functions


def update_sc():

    minyeara = int(minyear.value)
    maxyeara = int(maxyear.value)
    x_name = axis_map_notes[x_axis.value][0]
    y_name = axis_map_notes[y_axis.value][0]

    stmt = stmt_main.where(and_(
         data_table.columns.Var63 <= GDP.value,
         data_table.columns.year.between(minyeara,maxyeara)))

    dictionary = {'countryname': [],
                  #'countryname': [],
                  x_name : [],
                  y_name : [],
                  'year' : [],
                  'OECD_fragile': []
                  }

    for result in connection.execute(stmt):
         #dictionary['countrycode'].append(result.countrycode)
         dictionary['countryname'].append(result.countryname)
         dictionary['year'].append(result.year)
         if x_name != 'year':
             dictionary[x_name].append(result[x_name])
         dictionary[y_name].append(result[y_name])
         dictionary['OECD_fragile'].append(result.Var187)


    selected= pd.DataFrame(dictionary)

    selected["color"] = np.where(selected["OECD_fragile"] == 'Within OECD Index', '#b35900', '#361c7f' )


    t.title.text = title_name.value #+": %d observations " % len(df)
    #p.title.text = "%d Observations Selected" % len(df)
    fontvalue = font_scatter.value+'pt'
    t.title.text_font_size = fontvalue
    t.xaxis.axis_label = x_axis.value
    t.yaxis.axis_label = y_axis.value
    t.legend.background_fill_alpha = 0.3
    t.legend.label_text_font_size = '14pt'
    selected.dropna(inplace=True)
    source.data = dict(
        #countrycode = selected['countrycode'],
        countryname = selected['countryname'],
        xx = selected[x_name],
        yy = selected[y_name],
        year = selected["year"],
        color = selected['color'],
        OECD_norm = selected['OECD_fragile'])

    regression1 = np.polyfit(selected[x_name], selected[y_name], 1)
    #regression2 = np.polyfit(selected[x_name], selected[y_name], 2)

    x_1, y_1 = zip(*((i, i*regression1[0] + regression1[1]) for i in range(int(min(selected[x_name])), int(max(selected[x_name])))))
    #x2_1, y2_1 = zip(*((i, i*(regression2[0]**2)+i*regression2[1]+regression2[2]) for i in range(int(min(selected[x_name])), int(max(selected[x_name])))))


    source1.data = dict(
        rx = x_1,
        ry = y_1
    )



# Add the Tool to the plot
t.add_tools(hover_scatter)

#Load the Initial Plot
update_sc()

#####################################################################
#####################################################################
#               Empty Plot                           #
#####################################################################
#####################################################################



empty = figure(plot_height=600, plot_width=700, title="")
label = Label(x=1.1, y=18, text='Insufficient Data', text_font_size='70pt', text_color='#016450')
empty.add_layout(label)



#####################################################################
#####################################################################
#               Text Documents                             #
#####################################################################
#####################################################################


intro_portal = Div(text=open(join(dirname(__file__), 'tops', 'intro_portal.html')).read(), width=900)


'''
intro_areaempgva = Div(text=open(join(dirname(__file__), 'tops', 'intro_areaempgva.html')).read(), width=800)
intro_relemp = Div(text=open(join(dirname(__file__), 'tops', 'intro_relemp.html')).read(), width=800)
intro_LP = Div(text=open(join(dirname(__file__), 'tops', 'intro_LP.html')).read(), width=800)
intro_TRADE = Div(text=open(join(dirname(__file__), 'tops', 'intro_TRADE.html')).read(), width=800)
intro_areataxrc = Div(text=open(join(dirname(__file__), 'tops', 'intro_areataxrc.html')).read(), width=800)
intro_scatter = Div(text=open(join(dirname(__file__), 'tops', 'intro_scatter.html')).read(), width=800)
'''
exit = Div(text=open(join(dirname(__file__), 'tops', 'exit.html')).read(), width=900)





##########################################################################
##########################################################################
#                    Generate plot choice section                        #
##########################################################################
##########################################################################

####### Choose the Section of plots:
axis_map_subject = {
        "Structural Economic Change": "STR",
        #'Taxation and Resource Mobilisation': 'TAX',
        'Employment and Value-Added Composition': 'EMP',
        "Labour Productivity": "LP",
	"Trade": 'TRADE',
	#"Firm-Level Analysis": 'firms',
	"Country/Group Comparison": 'cross'
	}
Subject_choice = Select(title='Choose Economic Transformation Subject', options=sorted(axis_map_subject.keys()), value="Country/Group Comparison")


###########################################################
########### Create different first choices
###########################################################


#### Structual Transformation
axis_map_EMP = { "Area Chart - Employment and GVA Composition": "empgva"}
		     #"Scatter - Changes in Employment Composition and Relative LP": "empave",
		#'Tax Revenue Composition': 'tax_area'}
Plot_EMP = Select(title='Type of Plot', options=sorted(axis_map_EMP.keys()), value="Area Chart - Employment and GVA Composition")


#### Labour Productivity Charts :
axis_map_LP = {"Between/Within Labour Productivity": "Bar - Between/Within Labour Productivity",
	       "Changes in Labour Productivity (Annualised)": "Bar - Between/Within Labour Productivity",}
Plot_LP = Select(title='Type of Plot', options=sorted(axis_map_LP.keys()), value='Between/Within Labour Productivity')


#### Trade Charts
axis_map_trade = {"Area Chart - Composition of Exports" : 'none'}
Plot_TRADE = Select(title='Type of Plot', options=sorted(axis_map_trade.keys()), value="Area Chart - Composition of Exports")


#### Firm Charts
axis_map_firm = {"Currently no visuals are available" : 'none'}
Plot_FIRM = Select(title='Type of Plot', options=sorted(axis_map_firm.keys()), value='Currently no visuals are available')


#### Cross-sectional Charts Charts :
axis_map_cross = {"Scatter" : 'scatter',
                'Line Chart - Time Series': 'Line',
                'Box-and-Whisker Plot - Aggregate and Country': 'Whisker',
                'Bar Chart - Comparison': 'Bar',
                'Growth Heatmap - Finding Successful Cases': 'heatmap'}
Plot_CROSS = Select(title='Type of Plot', options=sorted(axis_map_cross.keys()), value='Line Chart - Time Series')

#### Structual Transformation
axis_map_STRUC = {
		     "Scatter - Changes in Employment Composition and Relative LP": "empave",
             "Bar - Employment Composition and Relative LP": "empave"}
		#'Tax Revenue Composition': 'tax_area'}
Plot_STRUC = Select(title='Type of Plot', options=sorted(axis_map_STRUC.keys()), value="Scatter - Changes in Employment Composition and Relative LP")


#### Structual Transformation
axis_map_TAX = { 'Tax Revenue Composition': 'tax_area'}
Plot_TAX = Select(title='Type of Plot', options=sorted(axis_map_TAX.keys()), value="Tax Revenue Composition")





###############################################################
########             Choice Widgets             ###############
###############################################################

# generat the first widget so we don't get an error in the first layout
First_choices = widgetbox()




###########################################
# Generate the inital layout to be altered.

layout = layout([[intro_portal],
		[],
		[],
		[],
		[],
		[exit]])


UPDATE= Button(label="Update", button_type="success")
# Update the heading choices when the subject widgets changes
def update_start():
	if Subject_choice.value=="Employment and Value-Added Composition":
		# generate the widgets within the update()
		# does not work otherwise
		layout.children[1] = widgetbox(Subject_choice, Plot_EMP)
		update_plot_EMP()
	elif Subject_choice.value=="Labour Productivity":
		layout.children[1] = widgetbox( Subject_choice, Plot_LP)
		update_plot_LP()
	elif Subject_choice.value=="Trade":
		layout.children[1] = widgetbox( Subject_choice, Plot_TRADE)
		update_plot_TRADE()
	elif Subject_choice.value=="Firm-Level Analysis":
		layout.children[1] = widgetbox( Subject_choice,Plot_FIRM)
		update_plot_FIRM()
	elif Subject_choice.value=="Country/Group Comparison":
		layout.children[1] = widgetbox( Subject_choice, Plot_CROSS)
		update_plot_CROSS()
	elif Subject_choice.value=="Taxation and Resource Mobilisation":
		layout.children[1] = widgetbox( Subject_choice, Plot_TAX)
		update_plot_TAX()
	elif Subject_choice.value=="Structural Economic Change":
		layout.children[1] = widgetbox( Subject_choice, Plot_STRUC)
		update_plot_STRUC()

	else:
		print('there is something wrong')


###############################################################################
################## Generate the callbacks which change the plots
###############################################################################

def update_plot_EMP():
	# Generate the Area Chart - Employment/GVA
	print('EMPGVA')
	if Plot_EMP.value=="Area Chart - Employment and GVA Composition":
		print('Employ')
		# Text heading
		layout.children[2] = Div(text=open(join(dirname(__file__), 'tops', 'intro_areaempgva.html')).read(), width=900)
		# Widgets
		layout.children[3] = row(widgetbox(CountryEMP, title_name_emp, title_name_gva), widgetbox(fontSET, min_yearSET, max_yearSET))
		# Plot
		layout.children[4] = Tabs(tabs=[Panel(child=create_figure_emp(), title='Employment'), Panel(child=create_figure_gva(), title='Gross Value Added')])


def update_plot_TAX():
	# Generate the Area Chart - Tax Revenue
	print('here - tax')
	if Plot_TAX.value=="Tax Revenue Composition":
		# Text heading
		print('Tax')
		layout.children[2] = row(Div(text=open(join(dirname(__file__), 'tops', 'intro_areataxrc.html')).read(), width=900))
		# Widgets
		layout.children[3] = column(Div(text=open(join(dirname(__file__), 'tops', 'nodata.html')).read(), width=900), widgetbox(CountrySTRUC))
		layout.children[4] = empty
		layout.children[3] = Tabs(tabs=[Panel(child=create_figure_tax_nonrc(), title='Resource Revenue Within'), Panel(child=create_figure_tax_rc(), title='Resource Revenue Disaggregated')])
		# if data exist, replace the empty plot
		layout.children[4] = column(widgetbox(CountryTAX, tax_nonrc_title, tax_rc_title, min_year_taxrc, max_year_taxrc, fontrc), widgetbox(Button(label="Download - Resource Dissagregated", button_type="success", callback = CustomJS(args=dict(source=update_taxdatarc()), code=open(join(dirname(__file__), 'models', "download_taxdatarc.js")).read())), DataTable(source=update_taxdatarc(), columns=columns_taxdatarc, width=900, fit_columns=False)), widgetbox(Button(label="Download - Resource Revenue Within", button_type="success", callback = CustomJS(args=dict(source=update_taxdata_nonrc()), code=open(join(dirname(__file__), 'models', "download_taxdata_nonrc.js")).read())), DataTable(source=update_taxdata_nonrc(), columns=columns_tax_nonrc, width=900, fit_columns=False)))
	else:
		print('why')


def update_plot_FIRM():
	# no plots
	if Plot_FIRM.value=='Currently no visuals are available':

		# Appologies for not plot, else blanks
		layout.children[2] = Div(text=open(join(dirname(__file__), 'tops', 'intro_sorry.html')).read(), width=900)
		layout.children[3] = widgetbox()
		layout.children[4] = widgetbox()

############################
### Trade


def update_plot_TRADE():
	# Default is the no plots
	if Plot_TRADE.value=="Area Chart - Composition of Exports":
		# Appologies for not plot, else blanks
		layout.children[2] = Div(text=open(join(dirname(__file__), 'tops', 'intro_TRADE.html')).read(), width=900)
		layout.children[3] = row(widgetbox(CountryTRADE,min_yearTRADE, max_yearTRADE, exportarea_title, font_TRADE, note_TRADE, note_TRADE2, note_TRADE3, note_TRADE4), create_figure_trade())
		layout.children[4] = row(widgetbox(Button(label="Download Data", button_type="success", callback = CustomJS(args=dict(source=update_tradedata()), code=open(join(dirname(__file__), 'models', "download_tradedata.js")).read())), DataTable(source=update_tradedata(), columns=columns_tradedata, width=900, fit_columns=False)))

	else:
		print('why')


def update_plot_LP():
	# Generate the Area Chart when selected
	if Plot_LP.value=='Between/Within Labour Productivity':
		# Text heading
		layout.children[2] = Div(text=open(join(dirname(__file__), 'tops', 'intro_LP.html')).read(), width=900)	# Widgets
		layout.children[3] = row(widgetbox(LPcountries, LP_var, minyear_LP, maxyear_LP, group_yearsLP, bar_widthLP, title_LP, font_LP, note_LP, note_LP2), create_figure_withinbtw())
		# Plot
		layout.children[4] =  widgetbox() #row(widgetbox(LP_var, minyear_LP, maxyear_LP, group_years, bar_width, order_bar, title_LP, font_LP, note_LP, note_LP2, legend_location_LP, legend_location_ori_LP), widgetbox(button_withbtwLP, DataTable(source=update_table_withbtw(), columns=columns, width=800, fit_columns=False)))
		# The heading stays the same
	elif Plot_LP.value=='Changes in Labour Productivity (Annualised)':

		layout.children[2] = Div(text=open(join(dirname(__file__), 'tops', 'intro_LP_annualised.html')).read(), width=900)		# Plot and Widgits (row)
		layout.children[3] = row(widgetbox(CountryLP, LP_variables, minyear_LP, maxyear_LP, group_yearsLP, order_barLP, title_LP, font_LP, bar_widthLP,  note_LP, note_LP2), create_figure_annualLP())
		# Blank widgit box
		layout.children[4] = widgetbox() #row(widgetbox(button_annualLP, DataTable(source=update_table_annual(), columns=columns, width=650, fit_columns=False)))

def update_plot_STRUC():

	print('STRUC')
	if  Plot_STRUC.value=="Scatter - Changes in Employment Composition and Relative LP":
		# The heading stays the same

		layout.children[2] = row(Div(text=open(join(dirname(__file__), 'tops', 'intro_relemp.html')).read(), width=900))
		# Plot and Widgits (row)
		layout.children[3] = row(widgetbox(CountrySTRUC, minavyear, maxavyear, mining, title_relemp, font_relemp, note_relemp), create_figure_relemp())
		# Blank widgit box
		layout.children[4] = row(widgetbox(Button(label="Download Data", button_type="success", callback = CustomJS(args=dict(source=select_obs_relemp()), code=open(join(dirname(__file__), 'models', "download_relemp.js")).read())), DataTable(source=select_obs_relemp(), columns=columns_relemp, width=900, fit_columns=False)))

	elif  Plot_STRUC.value=="Bar - Employment Composition and Relative LP":
		# The heading stays the same

		layout.children[2] = row(Div(text=open(join(dirname(__file__), 'tops', 'intro_relemp.html')).read(), width=900))
		# Plot and Widgits (row)
		layout.children[3] = row(widgetbox(CountrySTRUC, minavyear, maxavyear,mining, title_relbar, font_relbar, note_relbar), create_figure_relbar())
		# Blank widgit box
		layout.children[4] = widgetbox() #row(widgetbox(Button(label="Download Data", button_type="success", callback = CustomJS(args=dict(source=select_obs_relemp()), code=open(join(dirname(__file__), 'models', "download_relemp.js")).read())), DataTable(source=select_obs_relemp(), columns=columns_relemp, width=900, fit_columns=False)))



	else:
		print('I wonder why')


#############################
### Cross-sectional

def update_plot_CROSS():
	# Default is the Scatter
	if Plot_CROSS.value=='Scatter':
		# Scatter heading
		layout.children[2] = Div(text=open(join(dirname(__file__), 'tops', 'intro_scatter.html')).read(), width=900)
		# Widgit box and scatter plot
		layout.children[3] = column(widgetbox(x_axis, y_axis), row(widgetbox(minyear, maxyear, GDP, title_name, font_scatter, button), t))
		# blank
		layout.children[4] = widgetbox()
	elif Plot_CROSS.value=='Line Chart - Time Series':
		# Scatter heading
		layout.children[2] = Div(text=open(join(dirname(__file__), 'tops', 'intro_line.html')).read(), width=900)
		# Widgit box and scatter plot
		layout.children[3] = column(widgetbox(line_var), row(widgetbox(linecross_scatter, countries, minyear_linecross, maxyear_linecross, rolling_linecross, title_linecross, font_linecross,legend_location_linecross, legend_location_ori_linecross, note_linecross ), create_figure_linecross()))
		# blank
		layout.children[4] = widgetbox(Button(label="Download Data", button_type="success", callback = CustomJS(args=dict(source=update_linecrossdata()),
                                   code=open(join(dirname(__file__), 'models', "download_linecross.js")).read())), DataTable(source=update_linecrossdata(), columns=columns_linescross, width=900, fit_columns=False))

	elif Plot_CROSS.value=='Box-and-Whisker Plot - Aggregate and Country':
		# Scatter heading
		layout.children[2] = Div(text=open(join(dirname(__file__), 'tops', 'intro_arealine.html')).read(), width=900)
		# Widgit box and scatter plot
		if boxplot_options.active==0:
		          layout.children[3] = column(boxplot_var, row(widgetbox(boxplot_options, GroupSelect, minyear_boxplot, maxyear_boxplot, bar_order, title_boxplot, font_boxplot, note_boxplot), create_figure_boxplot()))
		elif boxplot_options.active==1:
		        layout.children[3] = column(boxplot_var, row(widgetbox(boxplot_options, countries, minyear_boxplot, maxyear_boxplot, bar_order, title_boxplot, font_boxplot, note_boxplot), create_figure_boxplot()))
		layout.children[4] = widgetbox()

        #layout.children[4] = widgetbox(Button(label="Download Data", button_type="success", callback = CustomJS(args=dict(source=update_areacrossdata()),
                            #code=open(join(dirname(__file__), 'models', "download_linecross.js")).read())), DataTable(source=update_areacrossdata(), columns=columns_areacross, width=900, fit_columns=False))
	elif Plot_CROSS.value=='Bar Chart - Comparison':
		# Scatter heading
		layout.children[2] = Div(text=open(join(dirname(__file__), 'tops', 'intro_bar_chart.html')).read(), width=900)
		# Widgit box and scatter plot
		if bar_plot_options.active==0:
			layout.children[3] = column(bar_cross_var, row(widgetbox(bar_plot_options, countries, bar_order, minyear_bar_cross, maxyear_bar_cross, title_bar_cross, font_bar_cross, bar_height, round_bar_cross, note_bar_cross ), create_figure_bar_cross()))
		elif bar_plot_options.active==1:
			layout.children[3] = column(bar_cross_var, row(widgetbox(bar_plot_options, countries, group_years, minyear_bar_cross, maxyear_bar_cross, title_bar_cross, font_bar_cross, bar_width, legend_location_ori_bar_cross, legend_location_bar_cross, note_bar_cross), create_figure_bar_cross()))
		elif bar_plot_options.active==2:
			layout.children[3] = column(bar_cross_var, row(widgetbox(bar_plot_options, countries, group_years, minyear_bar_cross, maxyear_bar_cross, title_bar_cross, font_bar_cross, bar_width, legend_location_ori_bar_cross, legend_location_bar_cross, note_bar_cross), create_figure_bar_cross()))

		layout.children[4] = widgetbox()


	elif Plot_CROSS.value=='Growth Heatmap - Finding Successful Cases':
		# Scatter heading
		layout.children[2] = Div(text=open(join(dirname(__file__), 'tops', 'intro_success.html')).read(), width=900)
		# Widgit box and scatter plot
		layout.children[3] = widgetbox()
		layout.children[4] = widgetbox()

		layout.children[3] = row(create_figure_heatmap(), column(widgetbox(controls_heatmap), Button(label="Download Successful Cases", button_type="success", callback=CustomJS(args=dict(source=update_data_heatmap()),
                                   code=open(join(dirname(__file__), 'models', 'download_heatmap.js')).read()))))

	else:
		print('What the hell')



#######################
# initiate the initial choice widgets
update_start()

UPDATE.on_click(update_start())

#######################################################################
###########3       Generate the callbacks:
#######################################################################

########## Callbacks to change the initial choice variables
Subject_choice.on_change('value', lambda attr, old, new: update_start())
Plot_STRUC.on_change('value', lambda attr, old, new: update_plot_STRUC())
Plot_LP.on_change('value', lambda attr, old, new: update_plot_LP())
Plot_TRADE.on_change('value', lambda attr, old, new: update_plot_TRADE())
Plot_FIRM.on_change('value', lambda attr, old, new: update_plot_FIRM())
Plot_CROSS.on_change('value', lambda attr, old, new: update_plot_CROSS())
Plot_TAX.on_change('value', lambda attr, old, new: update_plot_TAX())
Plot_EMP.on_change('value', lambda attr, old, new: update_plot_EMP())



########### Generate the on.change updates for the plot widgets
########### These widgets recall the update_plot functions above.

CountrySTRUC.on_change('value', lambda attr, old, new: update_plot_STRUC())
CountryLP.on_change('value', lambda attr, old, new: update_plot_LP())
CountryFIRM.on_change('value', lambda attr, old, new: update_plot_FIRM())
CountryTRADE.on_change('value', lambda attr, old, new: update_plot_TRADE())
CountryTAX.on_change('value', lambda attr, old, new: update_plot_TAX())
CountryEMP.on_change('value', lambda attr, old, new: update_plot_EMP())


#### STRUC plots
# RELEMP
controls_relemp = [minavyear, maxavyear, title_relemp, font_relemp, note_relemp, note_relemp2, note_relemp3, note_relemp4]
for control in controls_relemp:
    control.on_change('value', lambda attr, old, new: update_plot_STRUC())

mining.on_change('active',  lambda attr, old, new: update_plot_STRUC())

# RELEMP
controls_relbar = [minavyear, maxavyear, title_relbar, font_relbar, note_relbar]
for control in controls_relbar:
    control.on_change('value', lambda attr, old, new: update_plot_STRUC())

# Area chart
controls_SET = [title_name_emp, title_name_gva, min_yearSET, max_yearSET, fontSET]
for control in controls_SET:
    control.on_change('value', lambda attr, old, new: update_plot_EMP())
# Generate the on.change updates()
controls_tax_rc = [tax_nonrc_title, tax_rc_title, min_year_taxrc, max_year_taxrc, fontrc]
for control in controls_tax_rc:
    control.on_change('value', lambda attr, old, new: update_plot_TAX())


#### LP Plots
controls_LP = [LPcountries, LP_var, minyear_LP, LP_variables, maxyear_LP, group_yearsLP, bar_widthLP, title_LP, font_LP, note_LP, note_LP2]
for control in controls_LP:
    control.on_change('value', lambda attr, old, new: update_plot_LP())
order_barLP.on_change('active', lambda attr, old, new: update_plot_LP())

#### TRADE Plots
controls_TRADE = [min_yearTRADE, max_yearTRADE, exportarea_title, font_TRADE, note_TRADE, note_TRADE2, note_TRADE3, note_TRADE4]
for control in controls_TRADE:
    control.on_change('value',lambda attr, old, new: update_plot_TRADE())


#### FIRM plots

#### Scatter
# Generate callbacks for the scatter plot
controls_scatter = [x_axis, y_axis, minyear, maxyear, GDP, title_name, font_scatter ]
for control in controls_scatter:
    control.on_change('value', lambda attr, old, new: update_sc())

controls_linecross = [countries, line_var, minyear_linecross,rolling_linecross,  legend_location_linecross, legend_location_ori_linecross,  note_linecross, note_linecross2,  note_linecross3,  note_linecross4,  maxyear_linecross, title_linecross, font_linecross]
for control in controls_linecross:
    control.on_change('value', lambda attr, old, new: update_plot_CROSS())

controls_nonstack = [ note_boxplot, countries, GroupSelect, boxplot_var, minyear_boxplot, maxyear_boxplot, title_boxplot, font_boxplot]
for control in controls_nonstack:
    control.on_change('value', lambda attr, old, new: update_plot_CROSS())

controls_bar_cross = [ countries, bar_height, group_years, bar_cross_var, minyear_bar_cross, maxyear_bar_cross, title_bar_cross, font_bar_cross, legend_location_ori_bar_cross, legend_location_bar_cross, round_bar_cross, bar_width, note_bar_cross]

bar_plot_options.on_change('active',  lambda attr, old, new: update_plot_CROSS())
linecross_scatter.on_change('active',  lambda attr, old, new: update_plot_CROSS())
bar_order.on_change('active',  lambda attr, old, new: update_plot_CROSS())
boxplot_options.on_change('active',  lambda attr, old, new: update_plot_CROSS())



for control in controls_bar_cross:
    control.on_change('value',  lambda attr, old, new: update_plot_CROSS())


######## heatmap widgets
controls_heatmap= [indicator_options, T_choice, minyear_heatmap, maxyear_heatmap, title_heatmap, font_heatmap, success]
# Generate Widgets for the Table
for control in controls_heatmap:
    control.on_change('value',  lambda attr, old, new: update_plot_CROSS())


#####################################################################
#####################################################################
#               Post to the Bokeh Server                            #
#####################################################################
#####################################################################




# Add to the Bokeh Server
curdoc().add_root(layout)



curdoc().title = "SET Interactive Data Portal"
