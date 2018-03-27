###########################################################################
##########         SET - Data Portal - Initial Post           ##############
############################################################################

# Author: Andrew Lightner
# Date: 23/01/2018
# Email: lightnera1@gmail.com


############################################################################
##############              Preamble                       #################
############################################################################


# Import the necessary python packages

from bokeh.io import curdoc
from bokeh.models import ColumnDataSource,MultiSelect, Span, Panel, Tabs, LabelSet, Select, Div, RangeSlider, Slider, TextInput, CategoricalColorMapper, HoverTool, CustomJS, FactorRange, TableColumn, DataTable, Label
from bokeh.plotting import figure
from bokeh.palettes import all_palettes
from bokeh.layouts import layout, widgetbox, row, column
from bokeh.models.widgets import Button
from bokeh.transform import factor_cmap
from bokeh.transform import dodge
from scipy.interpolate import spline
from bokeh.models.widgets import RadioButtonGroup
from bokeh.palettes import Category20
from bokeh.core.properties import value
from bokeh.models import Legend
import os
import pandas as pd
from os.path import dirname, join
import numpy as np
import sqlalchemy
from sqlalchemy import Table, MetaData, select, and_
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.types import Float, VARCHAR
from math import pi

import pandas as pd
from bokeh.palettes import Spectral4
from bokeh.plotting import figure, output_file, show
from bokeh.models import Legend
from bokeh.io import output_notebook
from bokeh.palettes import Category20b

############################################################################
##############              List of Vars                   #################
############################################################################

# This is first because we use the list for the inital column names upload.

########## Generate Axis Map for Variable Choice

# Generate the Axis Map for the choice of x and y vars
# !!! any changes need to be replicated in source = ColumnSourceData() !!!

#### Axis_map_notes
axis_map_notes = {
'Nominal FDI (Constant 2011 USD)': ['FDI_nom_UScontant', 'World Bank Development Indicators'],
'Domestic Credit to Private Sector (% of GDP)': ['dom_priv_cred_pct', 'World Bank Development Indicators'],
'Export Concentration Index': ['export_conc_index', 'World Bank Development Indicators'],
'Exports of Goods and Services - Constant 2011 USD': ['exports_GandS_constUS', 'World Bank Development Indicators'],
'FDI as Percenage of GDP': ['fdi_pct_GDP', 'World Bank Development Indicators'],
'Indicator of Economic Openness (1)' : ['open_indicator1', 'ODI Database'],
'World Bank Doing Business Score': ['DB', 'World Bank Development Indicators'],
'Access to Electricity (%)': ['access_elec_pct',
  'World Bank Development Indicators'],
 'CPIA Score - Financial Sector Development': ['fin_sec_CPIA',
  'World Bank Development Indicators'],
 'CPIA Score - Public Management': ['pub_man_CPIA',
  'World Bank Development Indicators'],
 'Change in Employment Share - Agriculture': ['CES_ag',
  'United Nations Statistical Division'],
 'Change in Employment Share - Construction': ['CES_constr',
  'United Nations Statistical Division'],
 'Change in Employment Share - Manufacturing': ['CES_manu',
  'United Nations Statistical Division'],
 'Change in Employment Share - Mining': ['CES_mining',
  'United Nations Statistical Division'],
 'Change in Employment Share - Other': ['CES_other',
  'United Nations Statistical Division'],
 'Change in Employment Share - Retail': ['CES_retail',
  'United Nations Statistical Division'],
 'Change in Employment Share - Transportation': ['CES_trans',
  'United Nations Statistical Division'],
 'Change in Labour Productivity - Agriculture': ['cngLP_ag',
  'United Nations Statistical Division'],
 'Change in Labour Productivity   - All': ['cngLP_all',
  'United Nations Statistical Division'],
 'Change in Labour Productivity   - Construction': ['cngLP_constr',
  'United Nations Statistical Division'],
 'Change in Labour Productivity   - Manufacturing': ['cngLP_manu',
  'United Nations Statistical Division'],
 'Change in Labour Productivity   - Mining': ['cngLP_mining',
  'United Nations Statistical Division'],
 'Change in Labour Productivity   - Other': ['cngLP_other',
  'United Nations Statistical Division'],
 'Change in Labour Productivity   - Retail': ['cngLP_retail',
  'United Nations Statistical Division'],
 'Change in Labour Productivity   - Transportation': ['cngLP_trans',
  'United Nations Statistical Division'],
 'Development Assistant (ODA) - Constant 2010 USD': ['ODI_constant2010',
  'World Bank Development Indicators'],
 'ODA (Proportion of Private Sector Investment)': ['ODA_priv_ratio',
  'World Bank Development Indicators'],
 'Employment Share - Agriculture': ['employ_share_ag',
  'United Nations Statistical Division'],
 'Employment Share - Construction': ['employ_share_constr',
  'United Nations Statistical Division'],
 'Employment Share - Manufacturing': ['employ_share_manu',
  'United Nations Statistical Division'],
 'Employment Share - Mining': ['employ_share_mining',
  'United Nations Statistical Division'],
 'Employment Share - Other': ['employ_share_other',
  'United Nations Statistical Division'],
 'Employment Share - Retail': ['employ_share_retail',
  'United Nations Statistical Division'],
 'Employment Share - Transportation': ['employ_share_trans',
  'United Nations Statistical Division'],
 'Export Value': ['export_value', 'World Bank Development Indicators'],
 'Export Value Index': ['exports_val_index',
  'World Bank Development Indicators'],
 'FDI Growth (%)': ['FDI_growth', 'World Bank Development Indicators'],
 'Fragile Index Score': ['FS_score_overtime',
  'World Bank Development Indicators'],
 'GDP Growth (%)': ['GDP_growth', 'World Bank Development Indicators'],
 'GDP Per Capita (Thousands; Constant 2010 USD)': ['GDPpc_2010',
  'World Bank Development Indicators'],
 'Goods and Service Exports (Constant 2010 USD)': ['exports_GandS_constUS',
  'World Bank Development Indicators'],
 'LP Growth (%) - Between Agriculture': ['LP_btw_ag',
  'United Nations Statistical Division'],
 'LP Growth (%) - Between Construction': ['LP_btw_constr',
  'United Nations Statistical Division'],
 'LP Growth (%) - Between Manufacturing': ['LP_btw_manu',
  'United Nations Statistical Division'],
 'LP Growth (%) - Between Mining': ['LP_btw_mining',
  'United Nations Statistical Division'],
 'LP Growth (%) - Between Other': ['LP_btw_other',
  'United Nations Statistical Division'],
 'LP Growth (%) - Between Retail': ['LP_btw_retail',
  'United Nations Statistical Division'],
 'LP Growth (%) - Between Transportation': ['LP_btw_trans',
  'United Nations Statistical Division'],
 'LP Growth (%) - Within Agriculture': ['LP_within_ag',
  'United Nations Statistical Division'],
 'LP Growth (%) - Within Construction': ['LP_within_constr',
  'United Nations Statistical Division'],
 'LP Growth (%) - Within Manufacturing': ['LP_within_manu',
  'United Nations Statistical Division'],
 'LP Growth (%) - Within Mining': ['LP_within_mining',
  'United Nations Statistical Division'],
 'LP Growth (%) - Within Other': ['LP_within_other',
  'United Nations Statistical Division'],
 'LP Growth (%) - Within Retail': ['LP_within_retail',
  'United Nations Statistical Division'],
 'LP Growth (%) - Within Transportation': ['LP_within_trans',
  'United Nations Statistical Division'],
 'Labour Productivity - Agriculture': ['LP_ag',
  'United Nations Statistical Division'],
 'Labour Productivity - All': ['LP_all',
  'United Nations Statistical Division'],
 'Labour Productivity - Construction': ['LP_constr',
  'United Nations Statistical Division'],
 'Labour Productivity - Manufacturing': ['LP_manu',
  'United Nations Statistical Division'],
 'Labour Productivity - Mining': ['LP_mining',
  'United Nations Statistical Division'],
 'Labour Productivity - Other': ['LP_other',
  'United Nations Statistical Division'],
 'Labour Productivity - Retail': ['LP_retail',
  'United Nations Statistical Division'],
 'Labour Productivity - Transportation': ['LP_trans',
  'United Nations Statistical Division'],
 'Liner Shipping Index (World Bank)': ['liner_ship_index',
  'World Bank Development Indicators'],
 'Mobile Phone Use (Per 100)': ['mobile_per100',
  'World Bank Development Indicators'],
 'Natural Resource Rent Dependence (% of GDP)': ['natural_rec_rent_pct',
  'World Bank Development Indicators'],
 'Number of Battle Deaths': ['BD', 'World Bank Development Indicators'],
 'Number of Displaced Persons': ['displaced_total',
  'World Bank Development Indicators'],
 'Number of Peace Keepers': ['peace_keep_pres',
  'World Bank Development Indicators'],
 'Relative Labour Productivity - Agriculture': ['relLP_ag',
  'United Nations Statistical Division'],
 'Relative Labour Productivity - Construction': ['relLP_constr',
  'United Nations Statistical Division'],
 'Relative Labour Productivity - Manufacturing': ['relLP_manu',
  'United Nations Statistical Division'],
 'Relative Labour Productivity - Mining': ['relLP_mining',
  'United Nations Statistical Division'],
 'Relative Labour Productivity - Other': ['relLP_other', 'United Nations Statistical Division'],
 'Relative Labour Productivity - Retail': ['relLP_retail',
  'United Nations Statistical Division'],
 'Relative Labour Productivity - Transportation': ['relLP_trans',
  'United Nations Statistical Division'],
 'Tax Revenue (% of GDP)': ['tax_rev_pct',
  'World Bank Development Indicators'],
 'Value Added Share - Agriculture': ['GVA_share_ag',
  'World Bank Development Indicators'],
 'Value Added Share - Construction': ['GVA_share_constr',
  'World Bank Development Indicators'],
 'Value Added Share - Manufacturing': ['GVA_share_manu',
  'World Bank Development Indicators'],
 'Value Added Share - Mining': ['GVA_shar_mining',
  'World Bank Development Indicators'],
 'Value Added Share - Other': ['GVA_share_other',
  'World Bank Development Indicators'],
 'Value Added Share - Retail': ['GVA_share_retail',
  'World Bank Development Indicators'],
 'Value Added Share - Transportation': ['GVA_share_trans',
  'World Bank Development Indicators'],
 'Year': ['year', '']}



############################################################################
##############              Load Data                      #################
############################################################################

user = os.environ.get('DATA_DB_USER')
passwd = os.environ.get('DATA_DB_PASS')
host = os.environ.get('DATA_DB_HOST')


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



#collect engine, con, meta
engine, connection, meta = connect(user, passwd, 'gonano', host=host)



# Create Data Table to load
try:
    data_table = Table('ODI4-march2018', MetaData(), autoload=True, autoload_with=engine)
except NoSuchTableError:
    print('error')


#####################################
##### Create Main selection
#####################################

stmt_main = select([data_table])

##########################################################3
###########################################################



#########################################################
########## Generate CountryChoice Widget Early
#########################################################


CountryList = [('ACT','Active Conflict'),('TFC', 'Transition From Conflict'), ('SNB', 'Subnational Conflict'),('ARC', 'At Risk of Conflict') , ('LTC', 'Limited Conflict'), ('ONI','Outside OECD Index'),('OFR', 'Within OECD Fragility Index'), ('WBN', 'Outside WB Index'), ('WBF', 'Within WB Index'), ('AFG', 'Afghanistan'), ('AGO', 'Angola'), ('ALB', 'Albania'), ('ARE', 'United Arab Emirates'), ('ARG', 'Argentina'), ('ARM', 'Armenia'), ('ATG', 'Antigua & Barbuda'), ('AZE', 'Azerbaijan'), ('BDI', 'Burundi'), ('BEN', 'Benin'), ('BFA', 'Burkina Faso'), ('BGD', 'Bangladesh'), ('BHR', 'Bahrain'), ('BHS', 'Bahamas'), ('BIH', 'Bosnia & Herzegovina'), ('BLR', 'Belarus'), ('BLZ', 'Belize'), ('BOL', 'Bolivia'), ('BRA', 'Brazil'), ('BRB', 'Barbados'), ('BRN', 'Brunei Darussalam'), ('BTN', 'Bhutan'), ('BWA', 'Botswana'), ('CAF', 'Central African Rep.'), ('CHL', 'Chile'), ('CHN', 'China'), ('CIV', "Côte d'Ivoire"), ('CMR', 'Cameroon'), ('COD', 'Congo DR'), ('COG', 'Congo'), ('COL', 'Colombia'), ('COM', 'Comoros'), ('CPV', 'Cape Verde'), ('CRI', 'Costa Rica'), ('CUB', 'Cuba'), ('DJI', 'Djibouti'), ('DMA', 'Dominica'), ('DOM', 'Dominican Rep.'), ('DZA', 'Algeria'), ('ECU', 'Ecuador'), ('EGY', 'Egypt'), ('ERI', 'Eritrea'), ('ETH', 'Ethiopia'), ('FJI', 'Fiji'), ('FSM', 'Micronesia (Fed. states of)'), ('GAB', 'Gabon'), ('GEO', 'Georgia'), ('GHA', 'Ghana'), ('GIN', 'Guinea'), ('GMB', 'Gambia'), ('GNB', 'Guinea Bissau'), ('GNQ', 'Equatorial Guinea'), ('GRD', 'Grenada'), ('GTM', 'Guatemala'), ('GUY', 'Guyana'), ('HKG', 'Hong Kong'), ('HND', 'Honduras'), ('HTI', 'Haiti'), ('IDN', 'Indonesia'), ('IND', 'India'), ('IRN', 'Iran'), ('IRQ', 'Iraq'), ('JAM', 'Jamaica'), ('JOR', 'Jordan'), ('KAZ', 'Kazakhstan'), ('KEN', 'Kenya'), ('KGZ', 'Kyrgyzstan'), ('KHM', 'Cambodia'), ('KIR', 'Kiribati'), ('KOR', 'Korea Rep.'), ('KWT', 'Kuwait'), ('LAO', 'Lao PDR'), ('LBN', 'Lebanon'), ('LBR', 'Liberia'), ('LBY', 'Libya'), ('LKA', 'Sri Lanka'), ('LSO', 'Lesotho'), ('MAC', 'Macau'), ('MAR', 'Morocco'), ('MDA', 'Moldova'), ('MDG', 'Madagascar'), ('MDV', 'Maldives'), ('MEX', 'Mexico'), ('MHL', 'Marshall Islands'), ('MKD', 'Macedonia'), ('MLI', 'Mali'), ('MMR', 'Myanmar'), ('MNE', 'Montenegro'), ('MNG', 'Mongolia'), ('MOZ', 'Mozambique'), ('MRT', 'Mauritania'), ('MUS', 'Mauritius'), ('MWI', 'Malawi'), ('MYS', 'Malaysia'), ('NAM', 'Namibia'), ('NER', 'Niger'), ('NGA', 'Nigeria'), ('NIC', 'Nicaragua'), ('NPL', 'Nepal'), ('NRU', 'Nauru'), ('OMN', 'Oman'), ('PAK', 'Pakistan'), ('PAN', 'Panama'), ('PER', 'Peru'), ('PHL', 'Philippines'), ('PLW', 'Palau'), ('PNG', 'Papua New Guinea'), ('PRK', 'Korea PDR'), ('PRY', 'Paraguay'), ('PSE', 'West Bank & Gaza (ILO)/State of Palestine (UN)'), ('QAT', 'Qatar'), ('RUS', 'Russian Fed.'), ('RWA', 'Rwanda'), ('SAU', 'Saudi Arabia'), ('SDN', 'Sudan (former)'), ('SEN', 'Senegal'), ('SLB', 'Solomon Islands'), ('SLE', 'Sierra Leone'), ('SLV', 'El Salvador'), ('SOM', 'Somalia'), ('SRB', 'Serbia'), ('SSD', 'South Sudan'), ('STP', 'Sao Tome & Principe'), ('SUR', 'Suriname'), ('SWZ', 'Swaziland'), ('SYC', 'Seychelles'), ('SYR', 'Syria'), ('TCD', 'Chad'), ('TGO', 'Togo'), ('THA', 'Thailand'), ('TJK', 'Tajikistan'), ('TKM', 'Turkmenistan'), ('TLS', 'Timor Leste'), ('TON', 'Tonga'), ('TTO', 'Trinidad & Tobago'), ('TUN', 'Tunisia'), ('TUR', 'Turkey'), ('TUV', 'Tuvalu'), ('TZA', 'Tanzania (= UN Mainland + Zanzibar, ILO U.R. Tanzania)'), ('UGA', 'Uganda'), ('UKR', 'Ukraine'), ('URY', 'Uruguay'), ('UZB', 'Uzbekistan'), ('VEN', 'Venezuela'), ('VNM', 'Vietnam'), ('VUT', 'Vanuatu'), ('WSM', 'Samoa'), ('XKX', 'Kosovo'), ('YEM', 'Yemen'), ('ZAF', 'South Africa'), ('ZMB', 'Zambia'), ('ZWE', 'Zimbabwe'), ('SDN', 'Sudan (current)')]

countrylistold = { 'Outside OECD Index': 'ONI' , 'Within OECD Fragility Index':'OFR' ,  'Subnational Conflict':'SNB' ,  'Limited Conflict':'LTC'  ,  'At Risk of Conflict':'ARC' ,  'Transition From Conflict':'TFC' ,  'Active Conflict':'ACT', 'Afghanistan': 'AFG', 'Angola': 'AGO', 'Albania': 'ALB', 'United Arab Emirates': 'ARE', 'Argentina': 'ARG', 'Armenia': 'ARM', 'Antigua & Barbuda': 'ATG', 'Azerbaijan': 'AZE', 'Burundi': 'BDI', 'Benin': 'BEN', 'Burkina Faso': 'BFA', 'Bangladesh': 'BGD', 'Bahrain': 'BHR', 'Bahamas': 'BHS', 'Bosnia & Herzegovina': 'BIH', 'Belarus': 'BLR', 'Belize': 'BLZ', 'Bolivia': 'BOL', 'Brazil': 'BRA', 'Barbados': 'BRB', 'Brunei Darussalam': 'BRN', 'Bhutan': 'BTN', 'Botswana': 'BWA', 'Central African Rep.': 'CAF', 'Chile': 'CHL', 'China': 'CHN', "Côte d'Ivoire": 'CIV', 'Cameroon': 'CMR', 'Congo DR': 'COD', 'Congo': 'COG', 'Colombia': 'COL', 'Comoros': 'COM', 'Cape Verde': 'CPV', 'Costa Rica': 'CRI', 'Cuba': 'CUB', 'Djibouti': 'DJI', 'Dominica': 'DMA', 'Dominican Rep.': 'DOM', 'Algeria': 'DZA', 'Ecuador': 'ECU', 'Egypt': 'EGY', 'Eritrea': 'ERI', 'Ethiopia': 'ETH', 'Fiji': 'FJI', 'Micronesia (Fed. states of)': 'FSM', 'Gabon': 'GAB', 'Georgia': 'GEO', 'Ghana': 'GHA', 'Guinea': 'GIN', 'Gambia': 'GMB', 'Guinea Bissau': 'GNB', 'Equatorial Guinea': 'GNQ', 'Grenada': 'GRD', 'Guatemala': 'GTM', 'Guyana': 'GUY', 'Hong Kong': 'HKG', 'Honduras': 'HND', 'Haiti': 'HTI', 'Indonesia': 'IDN', 'India': 'IND', 'Iran': 'IRN', 'Iraq': 'IRQ', 'Jamaica': 'JAM', 'Jordan': 'JOR', 'Kazakhstan': 'KAZ', 'Kenya': 'KEN', 'Kyrgyzstan': 'KGZ', 'Cambodia': 'KHM', 'Kiribati': 'KIR', 'Korea Rep.': 'KOR', 'Kuwait': 'KWT', 'Lao PDR': 'LAO', 'Lebanon': 'LBN', 'Liberia': 'LBR', 'Libya': 'LBY', 'Sri Lanka': 'LKA', 'Lesotho': 'LSO', 'Macau': 'MAC', 'Morocco': 'MAR', 'Moldova': 'MDA', 'Madagascar': 'MDG', 'Maldives': 'MDV', 'Mexico': 'MEX', 'Marshall Islands': 'MHL', 'Macedonia': 'MKD', 'Mali': 'MLI', 'Myanmar': 'MMR', 'Montenegro': 'MNE', 'Mongolia': 'MNG', 'Mozambique': 'MOZ', 'Mauritania': 'MRT', 'Mauritius': 'MUS', 'Malawi': 'MWI', 'Malaysia': 'MYS', 'Namibia': 'NAM', 'Niger': 'NER', 'Nigeria': 'NGA', 'Nicaragua': 'NIC', 'Nepal': 'NPL', 'Nauru': 'NRU', 'Oman': 'OMN', 'Pakistan': 'PAK', 'Panama': 'PAN', 'Peru': 'PER', 'Philippines': 'PHL', 'Palau': 'PLW', 'Papua New Guinea': 'PNG', 'Korea PDR': 'PRK', 'Paraguay': 'PRY', 'West Bank & Gaza (ILO)/State of Palestine (UN)': 'PSE', 'Qatar': 'QAT', 'Russian Fed.': 'RUS', 'Rwanda': 'RWA', 'Saudi Arabia': 'SAU', 'Sudan (former)': 'SDN', 'Senegal': 'SEN', 'Solomon Islands': 'SLB', 'Sierra Leone': 'SLE', 'El Salvador': 'SLV', 'Somalia': 'SOM', 'Serbia': 'SRB', 'South Sudan': 'SSD', 'Sao Tome & Principe': 'STP', 'Suriname': 'SUR', 'Swaziland': 'SWZ', 'Seychelles': 'SYC', 'Syria': 'SYR', 'Chad': 'TCD', 'Togo': 'TGO', 'Thailand': 'THA', 'Tajikistan': 'TJK', 'Turkmenistan': 'TKM', 'Timor Leste': 'TLS', 'Tonga': 'TON', 'Trinidad & Tobago': 'TTO', 'Tunisia': 'TUN', 'Turkey': 'TUR', 'Tuvalu': 'TUV', 'Tanzania (= UN Mainland + Zanzibar, ILO U.R. Tanzania)': 'TZA', 'Uganda': 'UGA', 'Ukraine': 'UKR', 'Uruguay': 'URY', 'Uzbekistan': 'UZB', 'Venezuela': 'VEN', 'Vietnam': 'VNM', 'Vanuatu': 'VUT', 'Samoa': 'WSM', 'Kosovo': 'XKX', 'Yemen': 'YEM', 'South Africa': 'ZAF', 'Zambia': 'ZMB', 'Zimbabwe': 'ZWE', 'Sudan (current)': 'SDN'
}

#### Individual Country Selections
CountrySTRUC = Select(title="Country or Group Category", value="MWI", options=CountryList)
CountryLP = Select(title="Country or Group Category", options=CountryList, value="MWI")
CountryFIRM = Select(title="Country or Group Category", options=CountryList, value="MWI")
CountryTRADE = Select(title="Country or Group Category", options=CountryList, value="MWI")
CountryCROSS = Select(title="Highlighted Country", options=CountryList, value="MWI")


# Multiple Country Selections - Must add when added to group

countries = MultiSelect(title="Country Selection", value=['ONI', 'OFR'],
                           options=CountryList)


LPcountries = MultiSelect(title="Country Selection", value=['ONI', 'OFR'],
                           options=CountryList)





axis_year = {}
for i in range(1991,2018):
    axis_year[str(i)] = (str(i))

############ Generate Axis Map for Font Size Choices

l = list(range(12, 32))
axis_font = {}
for i in range(0, len(l)):
    axis_font[str(l[i])] = (str(l[i]))

############ Generate Axis Map for Font Size Choices

l = list(range(0, 4))
axis_round = {}
for i in range(0, len(l)):
    axis_round[str(l[i])] = (str(l[i]))

############ Generate Axis Map for Font Size Choices

l = list(range(1, 10))
axis_barsize = {}
for i in range(0, len(l)):
    axis_barsize[str(l[i])] = (str(l[i]))

##### Label Options

label_opts = dict(
        x=0, y=0,
        x_units='screen', y_units='screen',
        text_font_size='10pt', text_font = 'arial'
    )

axis_legend = {'Bottom Right': 'bottom_right',
                'Bottom Left': 'bottom_left',
                'Top Right' : 'top_right',
                'Top Left': 'top_left'}

axis_legend_orientation = {'Horizontal': 'horizontal',
                'Vertical': 'vertical'}

############ Generate Axis Map for Group Year Choices

l = list(range(1, 10))
axis_groupyear = {}
for i in range(0, len(l)):
    axis_groupyear[str(l[i])] = (str(l[i]))



####################################################
############  Define Palettes Here  ################
####################################################

########## Generate a function which chooses the color pattern, should not be hard.
########## They choose the list of colors, then the rest is an index.

# this is so that if I need to change the color scheme, I can quickly

# Choose the palette for the visual
palette_figure_2 = ["#036564", "#D95B43"] #need to change this number as see fit
palette_figure_6 = all_palettes['Viridis'][6]
palette_figure_7 = ['#361c7f', '#b35900', '#990000', '#016450', '#02818a', '#3690c0', '#67a9cf', '#a6bddb', '#d0d1e6', '#fed976']
palette_figure_8 = all_palettes['Viridis'][8]
color_blender = ['#4d004b', '#810f7c', '#88419d', '#8c6bb1', '#8c96c6', '#9ebcda', '#bfd3e6', '#e0ecf4', '#f7fcfd']
large_color = Category20b[20]
tools = "pan,wheel_zoom,box_zoom,reset,save"

#####################################################################
#####################################################################
###    Bar Chart - Cross Sectional Comparison                    ####
#####################################################################
#####################################################################


title_bar_cross = TextInput(title="Title", value="Bar Chart - Cross Sectional")
bar_cross_var = Select(title="Variable of Interest", options=sorted(axis_map_notes.keys()), value="Employment Share - Agriculture")
minyear_bar_cross = Select(title="Start Year", options=sorted(axis_year.keys()), value="2008")
maxyear_bar_cross = Select(title="End Year", options=sorted(axis_year.keys()), value="2013")
font_bar_cross = Select(title="Title Font Size", options=sorted(axis_font.keys()), value="24")
note_bar_cross = TextInput(title="Additional Note Content - Line 1", value="")
note_bar_cross2 = TextInput(title="Additional Note Content - Line 2", value="")
note_bar_cross3 = TextInput(title="Additional Note Content - Line 3", value="")
note_bar_cross4 = TextInput(title="Additional Note Content - Line  4", value="")
legend_location_bar_cross = Select(title="Legend Location", options=sorted(axis_legend.keys()), value="Bottom Right")
legend_location_ori_bar_cross = Select(title="Legend Orientation", options=sorted(axis_legend_orientation.keys()), value="Vertical")
bar_plot_options = RadioButtonGroup(
        labels=["Averages", "By Country/Year", "By Year/Country"], active=0, width=400)
round_bar_cross = Select(title="Number of Decimal Points", options=sorted(axis_round.keys()), value="3")
bar_width = Slider(title='Width of Bars', start=0.05, end=.4, value=0.2, step=.04)
group_years = Select(title="Year Groupings", options=sorted(axis_groupyear.keys()), value="3")
bar_order = RadioButtonGroup(
        labels=["Ascending", "Descending"], active=0, width=250)




def select_obs_bar_cross():
    country_vals = countries.value
    Var_Interest = axis_map_notes[bar_cross_var.value][0]

    # Select the Correct Observations
    stmt = stmt_main.where(and_(
        data_table.columns.countrycode.in_(country_vals),
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

def create_figure_bar_cross():
    country_vals = countries.value
    Var_Interest = axis_map_notes[bar_cross_var.value][0]
    order = True
    if bar_order.active ==1:
        order=False
    data = select_obs_bar_cross()
    # Average Chart
    if bar_plot_options.active ==0:
        # Generate source for bar chart
        data = data.groupby('countryname').mean()
        data.reset_index(inplace=True)
        data[Var_Interest] = data[Var_Interest].round(int(round_bar_cross.value))
        data.sort_values(Var_Interest, ascending=order, inplace=True)
        C_list = list(data['countryname'])
        source = ColumnDataSource(data=dict(country=C_list, counts=list(data[Var_Interest])))
        p = figure(x_range=C_list, plot_height=700, plot_width=1000)
        colors = palette_figure_7[0:len(C_list)]
        if len(C_list) >7:
            colors = Category20b[len(C_list)]

        # Generate Plot
        p.vbar(x='country', top='counts', width=0.9, source=source, alpha=0.6,
               line_color='white', fill_color=factor_cmap('country', palette=colors, factors=C_list))

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


        colors = palette_figure_7[0:len(years)]
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



        p = figure(x_range=C_list, plot_height=700, plot_width=1000)

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
        colors = palette_figure_7[0:len(C_list)]
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


        p = figure(x_range=dictionary['years'], plot_height=700, plot_width=1000)

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
    if len(country_vals)>4:
        p.xaxis.major_label_orientation = pi/4


    #### SETTINGS TO MATCH Set
    p.xgrid.visible = False
    p.title.text_color = '#361c7f'
    p.title.text_font = "arial"
    p.title.text_font_style = "bold"
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
    p.axis.major_label_text_font_size = '16pt'
    p.axis.major_label_text_color='#999999'
    p.outline_line_color = None
    p.axis.axis_label_text_font_size = '16pt'
    p.legend.orientation = axis_legend_orientation[legend_location_ori_bar_cross.value]

    msg1 = 'Source: '+axis_map_notes[bar_cross_var.value][1]+'. '+note_bar_cross.value
    caption1 = Label(text=msg1, **label_opts, text_color='#999999')
    p.add_layout(caption1, 'below')

    msg2 = note_bar_cross2.value
    caption2 = Label(text=msg2, **label_opts, text_color='#999999')
    p.add_layout(caption2, 'below')

    msg3 = note_bar_cross3.value
    caption2 = Label(text=msg3, **label_opts, text_color='#999999')
    p.add_layout(caption2, 'below')

    msg4 = note_bar_cross4.value
    caption2 = Label(text=msg4, **label_opts, text_color='#999999')
    p.add_layout(caption2, 'below')

    p.legend.location = axis_legend[legend_location_bar_cross.value]


    return p


#####################################################################
#####################################################################
###    Area Chart Over Time - With Open for Color                ####
#####################################################################
#####################################################################

# Generate the Axis Map for the choice of x and y vars
# !!! any changes need to be replicated in source = ColumnSourceData() !!!

title_nonstack = TextInput(title="Title", value="Area Chart Over Time")
nonstack_var = Select(title="Variable of Interest", options=sorted(axis_map_notes.keys()), value="Mobile Phone Use (Per 100)")
minyear_nonstack = Select(title="Start Year", options=sorted(axis_year.keys()), value="1991")
maxyear_nonstack = Select(title="End Year", options=sorted(axis_year.keys()), value="2013")
font_nonstack = Select(title="Title Font Size", options=sorted(axis_font.keys()), value="14")
shade_nonstack = Slider(title="Shading Under the Curve", start=0.0, end=1, value=0.3, step=0.1)
note_nonstack = TextInput(title="Additional Note Content", value='')
note_nonstack2 = TextInput(title="Additional Note Content - Line 2", value="")
note_nonstack3 = TextInput(title="Additional Note Content - Line 3", value="")
note_nonstack4 = TextInput(title="Additional Note Content - Line  4", value="")
legend_location_nonstack = Select(title="Legend Location", options=sorted(axis_legend.keys()), value="Top Right")
legend_location_ori_nonstack = Select(title="Legend Orientation", options=sorted(axis_legend_orientation.keys()), value="Vertical")




def select_obs_nonstack():
    country_vals = countries.value
    Var_Interest = axis_map_notes[nonstack_var.value][0]

    minyear = int(minyear_nonstack.value)
    maxyear = int(maxyear_nonstack.value)

    # Select the Correct Observations
    stmt = stmt_main.where(and_(
        data_table.columns.countrycode.in_(country_vals),
        data_table.columns.year.between(minyear,maxyear)))

    dictionary = {'countrycode': [],
                  'year':[],
                  Var_Interest: []}
    for result in connection.execute(stmt):
        #dictionary['countryname'].append(result.countryname)
        dictionary['countrycode'].append(result.countrycode)
        dictionary['year'].append(result.year)
        dictionary[Var_Interest].append(result[Var_Interest])

    selected = pd.DataFrame(dictionary)


    # selected = data[['countrycode', 'year', Var_Interest]]
    # selected = selected[
    # (selected.year >= minyear) &
    # (selected.year <= maxyear)

    # Sources for line graph
    sources = {}
    for c in country_vals:
        df= selected[selected['countrycode']==c]
        sources[c] = ColumnDataSource(df)

    # min and max for the years
    m = min(selected['year'])
    M = max(selected['year'])

    ##########################################
    # Generate the areas for the charts

    def  stacked(df):
        zeros = pd.DataFrame(0, index=np.arange(len(df)), columns=list(df))
        df_stack = pd.concat([zeros, df], ignore_index=True)
        return df_stack

    areas = {}
    x2s = {}


    i = 0
    for c in country_vals:

        # Generate country specfic dataset - cleaner
        country_df = pd.DataFrame()
        country_df['year']= selected[selected['countrycode']==c]['year']
        country_df['yy0'] = selected[selected['countrycode']==c][Var_Interest]
        country_df = country_df.interpolate().dropna()

        # Generate the areas and x2s for each country
        df = pd.DataFrame()
        df['year'] = np.linspace(country_df['year'].min(), country_df['year'].max(), 180)
        df['yy0'] = spline(country_df['year'], country_df['yy0'], df['year'])
        df.set_index('year', drop=True, inplace=True)
        areas[c] = stacked(df)
        x2s[c] = np.hstack((df.index[::-1], df.index))
        i+=1


    return {'sources': sources, 'areas': areas, 'x2s':x2s, 'm':m, 'M':M}


def create_figure_nonstack():
    country_vals = countries.value
    Var_Interest = axis_map_notes[nonstack_var.value][0]

    data = select_obs_nonstack()
    areas = data['areas']
    x2s = data['x2s']
    m = data['m']
    M = data['M']
    sources = data['sources']

    fontvalue = font_nonstack.value+'pt'
    num = len(countries.value)
    colors = palette_figure_7[0:num]

    p = figure(x_range=(m, M), plot_height=800, plot_width=800)
    p.grid.minor_grid_line_color = '#eeeeee'

    l=0
    for n in country_vals:
        p.patches([x2s[n]] * areas[n].shape[1], [areas[n][c].values for c in areas[n]],
              color=colors[l], alpha=shade_nonstack.value, line_color=None, legend=n)
        l+=1


    l=0
    for n in country_vals:
        p.circle('year',Var_Interest, name=n, source=sources[n], color=colors[l])
        l+=1


    # Adjust the Labels of the Plot
    p.title.text = title_nonstack.value
    p.title.text_font_size = fontvalue
    p.xaxis.axis_label = 'Year'
    p.yaxis.axis_label = nonstack_var.value
    p.grid.grid_line_alpha=0.3


    var_hover = '@'+Var_Interest+'{0.00 a}'

    #Generate Hover
    hover = HoverTool(
        names=country_vals,
            tooltips=[
                ('Country', '@countrycode'),
                ( 'Year',   '@year'            ),
                ( nonstack_var.value,  var_hover ),
            ],
        # display a tooltip whenever the cursor is vertically in line with a glyph
        mode='vline'
    )
    p.add_tools(hover)
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
    p.axis.major_label_text_font_size = '16pt'
    p.axis.major_label_text_color='#999999'
    p.outline_line_color = None
    p.axis.axis_label_text_font_size = '16pt'
    p.axis.axis_label_text_font_style = 'normal'
    p.legend.orientation = axis_legend_orientation[legend_location_ori_nonstack.value]

    msg1 = 'Source: '+axis_map_notes[nonstack_var.value][1]+'. '+note_nonstack.value
    caption1 = Label(text=msg1, **label_opts, text_color='#999999')
    p.add_layout(caption1, 'below')

    msg2 = note_nonstack2.value
    caption2 = Label(text=msg2, **label_opts, text_color='#999999')
    p.add_layout(caption2, 'below')

    msg3 = note_nonstack3.value
    caption2 = Label(text=msg3, **label_opts, text_color='#999999')
    p.add_layout(caption2, 'below')

    msg4 = note_nonstack4.value
    caption2 = Label(text=msg4, **label_opts, text_color='#999999')
    p.add_layout(caption2, 'below')

    p.legend.location = axis_legend[legend_location_nonstack.value]
    p.legend.click_policy="mute"
    return p


def update_areacrossdata():
    source = ColumnDataSource(data=dict())
    country_vals = countries.value
    Var_Interest = axis_map_notes[nonstack_var.value][0]


    selected = data.loc[data['countrycode'].isin(country_vals)]
    # Select the Right Variables
    selected = selected[['countryname', 'countrycode', 'year', Var_Interest]]
    selected.columns = ['countryname', 'countrycode', 'year','Var_Interest']

    minyear = int(minyear_nonstack.value)
    maxyear = int(maxyear_nonstack.value)
    selected = selected[
        ( selected.year >= minyear) &
        ( selected.year <= maxyear)
    ]

    # Soruces for line graph
    source.data = {
		'Country'             : selected.countryname,
		'Year'           : selected.year,
		'Var_Interest' : selected.Var_Interest
        }

    return source



columns_areacross = [
    TableColumn(field="Country", title="Country"),
    TableColumn(field="Year", title="Year"),
    TableColumn(field="Var_Interest", title='Variable of Interest'),

]




#####################################################################
#####################################################################
###            Line Chart - Over Time Comparison                 ####
#####################################################################
#####################################################################


title_linecross = TextInput(title="Title", value="Line Chart Over Time")
line_var = Select(title="Variable of Interest", options=sorted(axis_map_notes.keys()), value="Employment Share - Agriculture")
minyear_linecross = Select(title="Start Year", options=sorted(axis_year.keys()), value="1991")
maxyear_linecross = Select(title="End Year", options=sorted(axis_year.keys()), value="2013")
font_linecross = Select(title="Title Font Size", options=sorted(axis_font.keys()), value="24")
note_linecross = TextInput(title="Additional Note Content - Line 1", value="")
note_linecross2 = TextInput(title="Additional Note Content - Line 2", value="")
note_linecross3 = TextInput(title="Additional Note Content - Line 3", value="")
note_linecross4 = TextInput(title="Additional Note Content - Line  4", value="")
legend_location_linecross = Select(title="Legend Location", options=sorted(axis_legend.keys()), value="Bottom Right")
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
        data_table.columns.countrycode.in_(country_vals),
        data_table.columns.year.between(minyear,maxyear)))

    dictionary = {'countryname': [],
             'countrycode': [],
             'year':[],
             Var_Interest: []}
    for result in connection.execute(stmt):
        dictionary['countryname'].append(result.countryname)
        dictionary['countrycode'].append(result.countrycode)
        dictionary['year'].append(result.year)
        dictionary[Var_Interest].append(result[Var_Interest])

    # Generate the moving average
    selected = pd.DataFrame(dictionary).sort_values('year')
    selected[Var_Interest] = selected.groupby(['countryname'])[Var_Interest].apply(pd.rolling_mean, rolling_linecross.value, min_periods=rolling_linecross.value)

    #selected[Var_Interest] = pd.rolling_mean(selected[Var_Interest], 5, min_periods=5, center=False)



    dict2 = {'a': [1]}

    # Generate dataset for plotting scatter with line charts
    if linecross_scatter.active ==1:
        # Select the Correct Observations
        stmt2 = stmt_main.where(
            data_table.columns.year.between(minyear,maxyear))

        dict2 = {'countryname': [],
                 'countrycode': [],
                 'year':[],
                 Var_Interest: []}

        for result in connection.execute(stmt2):
            dict2['countryname'].append(result.countryname)
            dict2['countrycode'].append(result.countrycode)
            dict2['year'].append(result.year)
            dict2[Var_Interest].append(result[Var_Interest])
        dict2 = pd.DataFrame(dict2)
        dict2[Var_Interest] = dict2.groupby(['countryname'])[Var_Interest].apply(pd.rolling_mean, rolling_linecross.value, min_periods=rolling_linecross.value)


    source_scatter = ColumnDataSource(dict2)

    legends = {}
    for c in country_vals:
        legends[c] = selected[selected['countrycode']==c]['countryname'].reset_index(drop=True)[0]
    selected.drop('countryname', axis=1, inplace=True)
    # Soruces for Cirlces in the line graph
    sources = {}
    for c in country_vals:
        df = selected[selected['countrycode']==c]
        sources[c] = ColumnDataSource(df)

    # min and max for the years
    m = min(selected['year'])
    M = max(selected['year'])

    # min and max for the years
    m = min(selected['year'])
    M = max(selected['year'])

    ##########################################
    # Generate the curned lines for the charts




    lines = {}

    i = 0
    for c in country_vals:

        # Generate country specfic dataset - cleaner
        country_df = pd.DataFrame()
        country_df['year']= selected[selected['countrycode']==c]['year']
        country_df[Var_Interest] = selected[selected['countrycode']==c][Var_Interest]
        country_df = country_df.interpolate().dropna()

        # Generate the areas and x2s for each country
        df = pd.DataFrame()
        df['year'] = np.linspace(country_df['year'].min(), country_df['year'].max(), 180)
        df[Var_Interest] = spline(country_df['year'], country_df[Var_Interest], df['year'])
        lines[c] = ColumnDataSource(df)

    selected.columns = ['countrycode', 'year', 'Var_Interest']


    return {'sources': sources, 'lines': lines, 'm':m, 'M':M, 'legends': legends, 'source_scatter':source_scatter}

def create_figure_linecross():
    country_vals = countries.value
    Var_Interest = axis_map_notes[line_var.value][0]

    data = select_obs_linecross()
    lines = data['lines']
    sources = data['sources']
    legends = data['legends']
    source_scatter = data['source_scatter']
    fontvalue = font_linecross.value+'pt'
    num = len(countries.value)
    colors = palette_figure_7[0:num]


    p = figure(plot_width=1000, plot_height=600)
    a = 1
    if linecross_scatter.active ==1:
        p = figure(plot_width=1000, plot_height=600)
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
            ('Country', '@countrycode'),
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
    p.axis.axis_label_text_font_size = '16pt'
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
    p.legend.click_policy="mute"

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
        data_table.columns.countrycode.in_(country_vals),
        data_table.columns.year.between(minyear,maxyear)))

    dictionary = {'countryname': [],
             'countrycode': [],
             'year':[],
             Var_Interest: []}
    for result in connection.execute(stmt):
        dictionary['countryname'].append(result.countryname)
        dictionary['countrycode'].append(result.countrycode)
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
max_yearTRADE = Select(title="End Year", options=sorted(axis_year.keys()), value="2013")
font_TRADE = Select(title="Title Font Size", options=sorted(axis_font.keys()), value="14")
note_TRADE = TextInput(title="Note Content", value="X-Axis: Proportion of Total Merchandise Exports. Y-axis: Year. Source: World Bank Development Indicators.")
note_TRADE = TextInput(title="Additional Note Content - Line 1", value="")
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
        data_table.columns.countrycode==country_val,
        data_table.columns.year.between(minyear,maxyear)))

    dictionary = {'countrycode': [],
             'year':[],
             'exports_food_pctsum':[],
             'exports_oresmet_pctsum':[],
             'exports_fuel_pctsum':[],
             'exports_manu_pctsum':[],
             'exports_agraw_pctsum':[],
             }
    for result in connection.execute(stmt):
        dictionary['countrycode'].append(result.countrycode)
        dictionary['exports_food_pctsum'].append(result.exports_food_pctsum)
        dictionary['exports_fuel_pctsum'].append(result.exports_fuel_pctsum)
        dictionary['exports_oresmet_pctsum'].append(result.exports_oresmet_pctsum)
        dictionary['exports_manu_pctsum'].append(result.exports_manu_pctsum)
        dictionary['exports_agraw_pctsum'].append(result.exports_agraw_pctsum)
        dictionary['year'].append(result.year)
        #dictionary[Var_Interest].append(result[Var_Interest])


    selected = pd.DataFrame(dictionary).dropna()

    return selected

def create_figure_trade():
    selected = select_obs_trade()
    m = min(selected['year'])
    M = max(selected['year'])
    selected.set_index('year', inplace=True, drop=True)
    selected.drop(['countrycode'], axis=1, inplace=True)

    selected.columns = ['yy0', 'yy1','yy2','yy3','yy4']
    source = ColumnDataSource(selected)

    def  stacked(selected):
        df_top = selected.cumsum(axis=1)
        df_bottom = df_top.shift(axis=1).fillna({'yy0': 0})[::-1]
        df_stack = pd.concat([df_bottom, df_top], ignore_index=True)
        return df_stack
    sectors = ['Food', 'Fuel', 'Ores', 'Manu', 'Agraw']
    areas = stacked(selected)
    colors = color_blender[0:5]
    x2 = np.hstack((selected.index[::-1], selected.index))
    p = figure(x_range=(m, M), y_range=(0, 100), plot_height=800, plot_width=800)
    p.grid.minor_grid_line_color = '#eeeeee'

    p.patches([x2] * areas.shape[1], [areas[c].values for c in areas],
              color=colors, alpha=0.8, line_color=None)
              #legend=["%s Sector" % c for c in sectors]

    p.line(x='year', y='yy0',source=source, color='#440154', line_width=.2)

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
    p.legend.orientation = "horizontal"

    return p


# Generate the HoverTool
hover_exports = HoverTool(

    tooltips=[
        ( 'Year',   '@year'            ),
        ( 'Raw Agricultural Goods', '@yy4{0.0 a}%' ),
        ( 'Manufactured Goods', '@yy3{0.0 a}%'     ),
	( 'Fuels', '@yy2{0.0 a}%'),
	( 'Ores and Metals', '@yy1{0.0 a}%'       ),
	( 'Food',  '@yy0{0.0 a}%'       )
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
		'Country'             : selected.countrycode,
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
max_year_taxrc = Select(title="End Year", options=sorted(axis_year.keys()), value="2013")
fontrc = Select(title="Title Font Size", options=sorted(axis_font.keys()), value="14")
note_taxrc = TextInput(title="Note Content", value="X-Axis: Proportion of Tax Revenue - Accounting for Resource Revenue. Y-axis: Year. Source: International Centre for Tax and Development.")

# Because I put this within the SET category, there is no need to create the year vars


tax_vars= ['tot_resource_revpct', 'direct_inc_sc_ex_resource_revpct', 'tax_g_spct', 'tax_int_trade_transpct', 'other_rc']



# Generate Call-backs
def select_obs_tax_rc():
    country_val = CountrySTRUC.value
    #selected = tax_data.loc[tax_data['countrycode']==country_val]
    minyear = int(min_year_taxrc.value)
    maxyear = int(max_year_taxrc.value)
    # Select the Correct Observations
    stmt = stmt_main.where(and_(
        data_table.columns.countrycode==country_val,
        data_table.columns.year.between(minyear,maxyear)))

    dictionary = {'countrycode': [],
             'year':[],
             'tot_resource_revpct':[],
             'direct_inc_sc_ex_resource_revpct':[],
             'tax_g_spct':[],
             'tax_int_trade_transpct':[],
             'other_rc':[],
             }
    for result in connection.execute(stmt):
        dictionary['countrycode'].append(result.countrycode)
        dictionary['tot_resource_revpct'].append(result.tot_resource_revpct)
        dictionary['direct_inc_sc_ex_resource_revpct'].append(result.direct_inc_sc_ex_resource_revpct)
        dictionary['tax_g_spct'].append(result.tax_g_spct)
        dictionary['tax_int_trade_transpct'].append(result.tax_int_trade_transpct)
        dictionary['other_rc'].append(result.other_rc)
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
    p = figure(x_range=(m, M), y_range=(0, 1), plot_height=800, plot_width=800)
    p.grid.minor_grid_line_color = '#eeeeee'

    p.patches([x2] * areas.shape[1], [areas[c].values for c in areas],
              color=colors, alpha=0.8, line_color=None)

    p.circle(x='year', y='tot_resource_revpct',source=source, color='#440154', size=10)

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
    p.legend.orientation = "horizontal"

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
    country_val = CountrySTRUC.value
    #selected = data_nonrc.loc[data_nonrc['countrycode']==country_val]
    minyear = int(min_year_taxrc.value)
    maxyear = int(max_year_taxrc.value)

    stmt = stmt_main.where(and_(
        data_table.columns.countrycode==country_val,
        data_table.columns.year.between(minyear,maxyear)))

    dictionary = {'countrycode': [],
             'year':[],
             'direct_inc_scpct':[],
             'tax_g_spct':[],
             'tax_int_trade_transpct':[],
             'other_nonrc':[],
             }
    for result in connection.execute(stmt):
        dictionary['countrycode'].append(result.countrycode)
        dictionary['direct_inc_scpct'].append(result.direct_inc_scpct)
        dictionary['tax_g_spct'].append(result.tax_g_spct)
        dictionary['tax_int_trade_transpct'].append(result.tax_int_trade_transpct)
        dictionary['other_nonrc'].append(result.other_nonrc)
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
    df.drop(['countrycode'], axis=1, inplace=True)

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
    p = figure(x_range=(m, M), y_range=(0, 1), plot_height=800, plot_width=800)
    p.grid.minor_grid_line_color = '#eeeeee'
    vars_list = ['r', 'a', 't', 't']
    p.patches([x2] * areas.shape[1], [areas[c].values for c in areas],
              color=colors, alpha=0.8, line_color=None)

    p.line(x='year', y='yy0',source=source, color='#440154', line_width=.2)

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
    #selected = data.loc[data['countrycode']==country_val]

    # Soruces for line graph
    source.data = {
		'Country'             : selected.countrycode,
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
		'Country'             : selected.countrycode,
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
max_yearSET = Select(title="End Year", options=sorted(axis_year.keys()), value="2013")
fontSET = Select(title="Title Font Size", options=sorted(axis_font.keys()), value="14")
title_name_emp = TextInput(title="Title - Employment Share", value="Employment Share Over Time 1990-2013")



def select_obs_emp():
    country_val = CountrySTRUC.value
    #selected = data.loc[data['countryname']==country_val]
    #filter_col = [col for col in data if col.startswith(('employ_share', 'year', 'countryname'))]
    #selected = selected[filter_col]
    minyear = int(min_yearSET.value)
    maxyear = int(max_yearSET.value)


    # Select the Correct Observations
    stmt = stmt_main.where(and_(
        data_table.columns.countrycode ==country_val,
        data_table.columns.year.between(minyear,maxyear)))

    dictionary = {'countrycode': [],
                'employ_share_ag': [],
                'employ_share_manu': [],
                'employ_share_trans': [],
                'employ_share_retail': [],
                'employ_share_constr': [],
                'employ_share_mining': [],
                'employ_share_other': [],
                'year':[]}
    for result in connection.execute(stmt):
        dictionary['countrycode'].append(result.countrycode)
        #dictionary['countrycode'].append(result.countrycode)
        dictionary['year'].append(result.year)
        dictionary['employ_share_ag'].append(result.employ_share_ag)
        dictionary['employ_share_manu'].append(result.employ_share_manu)
        dictionary['employ_share_trans'].append(result.employ_share_trans)
        dictionary['employ_share_retail'].append(result.employ_share_retail)
        dictionary['employ_share_constr'].append(result.employ_share_constr)
        dictionary['employ_share_mining'].append(result.employ_share_mining)
        dictionary['employ_share_other'].append(result.employ_share_other)

    selected = pd.DataFrame(dictionary).sort_values('year')

    m = min(selected['year'])
    M = max(selected['year'])

    selected.set_index('year', inplace=True, drop=True)
    selected.drop(['countrycode'], axis=1, inplace=True)
    selected = selected.dropna()

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
    lista = ['Agiculture', 'Manufacturing', 'Transportation', 'Retail', 'Contruction', 'Mining', 'Other']
    areas = stacked(df)
    tools = "pan,wheel_zoom,box_zoom,reset"
    x2 = np.hstack((df.index[::-1], df.index))

    p = figure(x_range=(m, M), y_range=(0, 100), plot_height=800, plot_width=800, tools=tools,)
    p.grid.minor_grid_line_color = '#eeeeee'

    p.patches([x2] * areas.shape[1], [areas[c].values for c in areas],
              color=color_blender[0:7], alpha=0.8, line_color=None)
    p.line(x='year', y='yy0',source=source, color='#016450', line_width=.2)

    # Adjust the Labels of the Plot
    fontvalue = fontSET.value+'pt'
    p.title.text = title_name_emp.value
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
        ( 'Other', '@yy6{0.0 a}%'      ),
	( 'Transportation', '@yy5{0.0 a}%' ),
	( 'Retail', '@yy4{0.0 a}%'       ),
	( 'Construction', '@yy3{0.0 a}%'       ),
        ( 'Manufacturing', '@yy2{0.0 a}%'        ),
        ( 'Mining', '@yy1{0.0 a}%'      ),
        ( 'Agriculture', '@yy0{0.0 a}%'       )
    ],

    # display a tooltip whenever the cursor is vertically in line with a glyph
    mode='vline'
)


################# GVA app
title_name_gva = TextInput(title="Title - GVA Share", value="GVA Percentage Over Time 1990-2013")


def select_obs_gva():
    country_val = CountrySTRUC.value
    #selected = data.loc[data['countryname']==country_val]
    #filter_col = [col for col in data if col.startswith(('GVA_shar', 'year', 'countryname'))]
    #selected = selected[filter_col]
    minyear = int(min_yearSET.value)
    maxyear = int(max_yearSET.value)

    # Select the Correct Observations
    stmt = stmt_main.where(and_(
        data_table.columns.countrycode ==country_val,
        data_table.columns.year.between(minyear,maxyear)))

    dictionary = {'countrycode': [],
                'GVA_share_ag': [],
                'GVA_share_manu': [],
                'GVA_share_trans': [],
                'GVA_share_retail': [],
                'GVA_share_constr': [],
                'GVA_shar_mining': [],
                'GVA_share_other': [],
                'year':[]}
    for result in connection.execute(stmt):
        dictionary['countrycode'].append(result.countrycode)
        #dictionary['countrycode'].append(result.countrycode)
        dictionary['year'].append(result.year)
        dictionary['GVA_share_ag'].append(result.GVA_share_ag)
        dictionary['GVA_share_manu'].append(result.GVA_share_manu)
        dictionary['GVA_share_trans'].append(result.GVA_share_trans)
        dictionary['GVA_share_retail'].append(result.GVA_share_retail)
        dictionary['GVA_share_constr'].append(result.GVA_share_constr)
        dictionary['GVA_shar_mining'].append(result.GVA_shar_mining)
        dictionary['GVA_share_other'].append(result.GVA_share_other)



    selected = pd.DataFrame(dictionary).sort_values('year')

    m = min(selected['year'])
    M = max(selected['year'])

    selected.set_index('year', inplace=True, drop=True)
    selected.drop(['countrycode'], axis=1, inplace=True)
    selected = selected.dropna()

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

    p = figure(x_range=(m, M), y_range=(0, 100), plot_height=800, plot_width=800)
    p.grid.minor_grid_line_color = '#eeeeee'

    p.patches([x2] * areas.shape[1], [areas[c].values for c in areas],
              color=color_blender[0:7], alpha=0.8, line_color=None)

    p.line(x='year', y='yy0',source=source, color='#016450', line_width=.2)

    fontvalue = fontSET.value+'pt'
    p.title.text = title_name_gva.value
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
        ( 'Other', '@yy6{0.0 a}%'      ),
	( 'Transportation', '@yy5{0.0 a}%' ),
	( 'Retail', '@yy4{0.0 a}%'       ),
	( 'Construction', '@yy3{0.0 a}%'       ),
        ( 'Manufacturing', '@yy2{0.0 a}%'        ),
        ( 'Mining', '@yy1{0.0 a}%'      ),
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
maxavyear = Select(title="Change in Employment Share Mean - End Year", options=sorted(axis_year.keys()), value="2013")
font_relemp = Select(title="Title Font Size", options=sorted(axis_font.keys()), value="14")
note_relemp = TextInput(title="Additional Note Content - Line 1", value="")
note_relemp2 = TextInput(title="Additional Note Content - Line 2", value="")
note_relemp3 = TextInput(title="Additional Note Content - Line 3", value="")
note_relemp4 = TextInput(title="Additional Note Content - Line  4", value="")

# Choose the title of the relemp visual
title_relemp = TextInput(title="Title - Employment Share", value="Changes in Employment Share - Over Time")
# Choose the Size of the Font
# Generate HoverTool
hover_relemp = HoverTool(name='scatter',
                         tooltips=
                         [('Country', '@countrycode'),
                          ('Industry', '@Industry'),
                          ('Change in Employment Share', '@empave{0.0 a}%'),
                          ('Relative Labour Productivity', '@relLP{0.0 a}'),
                          ('Share of Total Employment', '@ES{0.0 a}%')])


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
    stmt = stmt_main.where(and_(
        data_table.columns.countrycode ==country_val,
        data_table.columns.year.between(minyear,maxyear)))



    dictionary = {'countrycode': [],
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
        dictionary['countrycode'].append(result.countrycode)
        #dictionary['countrycode'].append(result.countrycode)
        dictionary['year'].append(result.year)
        dictionary['ES_Agriculture'].append(result.employ_share_ag)
        dictionary['ES_Manufacturing'].append(result.employ_share_manu)
        dictionary['ES_Transportation'].append(result.employ_share_trans)
        dictionary['ES_Retail'].append(result.employ_share_retail)
        dictionary['ES_Construction'].append(result.employ_share_constr)
        dictionary['ES_Mining'].append(result.employ_share_mining)
        dictionary['ES_Other'].append(result.employ_share_other)

        dictionary['relLP_Agriculture'].append(result.relLP_ag)
        dictionary['relLP_Manufacturing'].append(result.relLP_manu)
        dictionary['relLP_Transportation'].append(result.relLP_trans)
        dictionary['relLP_Retail'].append(result.relLP_retail)
        dictionary['relLP_Construction'].append(result.relLP_constr)
        dictionary['relLP_Mining'].append(result.relLP_mining)
        dictionary['relLP_Other'].append(result.relLP_other)

    selected = pd.DataFrame(dictionary).sort_values('year')

    # Set index to countrycode
    selected.set_index('countrycode', inplace=True, drop=True)

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
        df_empavea1 = df_empavea.groupby('countrycode').agg({'minyear' : 'sum'})
        df_empavea2 = df_empavea.groupby('countrycode').agg({'maxyear' : 'sum'})


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
    selected['countrycode'] = d[0]
    selected['Industry']= d[1]
    source = ColumnDataSource(selected)
    return source


#########################################
#### Make the Plot

def create_figure_relemp():
    source = select_obs_relemp()
    # Generate the Plot
    # Color Settings
    color_mapper = CategoricalColorMapper(factors=source.data['index'], palette=palette_figure_7)

    p = figure(plot_height=700, plot_width=850, title="")
    p.circle(x= 'empave', y= 'relLP', name='scatter', fill_alpha=0.6, source=source, size = 'ES_scale',
            color=dict(field='index', transform=color_mapper), legend=None)


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
    TableColumn(field="countrycode", title="Country"),
    TableColumn(field="Industry", title="Year"),
    TableColumn(field="empave", title='Change in Employment Over Time-Period Chosen'),
    TableColumn(field="relLP", title='Relative Labour Productivity at End Year')

]

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


axis_map_withinbtw = {
    "Agriculture": ["LP_within_ag", "LP_btw_ag"],
    "Mining": ["LP_within_mining", "LP_btw_mining"],
    "Manufacturing": ["LP_within_manu", "LP_btw_manu"],
    "Construction": ["LP_within_constr", "LP_btw_constr"],
    "Retail": ["LP_within_retail", "LP_btw_retail"],
    "Transportation": ["LP_within_trans", "LP_btw_trans"],
    "Other": ["LP_within_other", "LP_btw_other"]
    }

title_LP = TextInput(title="Title", value="Examine Labour Productivity Growth")
LP_var = Select(title="Variable of Interest", options=sorted(axis_map_withinbtw.keys()), value="Agriculture")
minyear_LP = Select(title="Start Year", options=sorted(axis_year.keys()), value="1996")
maxyear_LP = Select(title="End Year", options=sorted(axis_year.keys()), value="2013")
font_LP = Select(title="Title Font Size", options=sorted(axis_font.keys()), value="24")
note_LP = TextInput(title="Additional Note Content - Line 1", value="")
note_LP2 = TextInput(title="Additional Note Content - Line 2", value="")
note_LP3 = TextInput(title="Additional Note Content - Line 3", value="")
note_LP4 = TextInput(title="Additional Note Content - Line  4", value="")
order_barLP = RadioButtonGroup(
        labels=["Ascending", "Descending"], active=0, width=250)
bar_widthLP = Slider(title='Width of Bars', start=0.05, end=.5, value=0.2, step=.025)
group_yearsLP = Select(title="Year Groupings", options=sorted(axis_groupyear.keys()), value="6")
LP_variables = MultiSelect(title="Sector Selections", value=sector_list,
                           options=LPList)
order_barLP = RadioButtonGroup(
        labels=["Ascending", "Descending"], active=0, width=250)


def select_obs_withinLP():
    country_vals = LPcountries.value

    # Select the Correct Observations
    stmt = stmt_main.where(and_(
        data_table.columns.countrycode.in_(country_vals),
        data_table.columns.year.between(int(minyear_LP.value),int(maxyear_LP.value))))


    LP = axis_map_withinbtw[LP_var.value]

    dictionary = {'countrycode': [],
                'countryname': [],
                LP[0]: [],
                LP[1]: [],
                'year':[]}
    for result in connection.execute(stmt):
        dictionary['countrycode'].append(result.countrycode)
        dictionary['countryname'].append(result.countryname)
        dictionary['year'].append(result.year)
        dictionary[LP[0]].append(result[LP[0]])
        dictionary[LP[1]].append(result[LP[1]])

    data = pd.DataFrame(dictionary)
    return data



def create_figure_withinbtw():
    data = select_obs_withinLP()

    C_list = list(data['countryname'].unique())
    LP = axis_map_withinbtw[LP_var.value]

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

    palette_figure_7 = ['#361c7f','#9467bd', '#b35900', '#990000'] + Category20[20]
    sources  = {}
    for i, d in dictionary.items():
        sources[i] = ColumnDataSource(data=d)

    p = figure(x_range=dictionary[C_list[0]]['years'], title="Attempts at ", plot_height=850, plot_width=1100)

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
               color=palette_figure_7[n], name="Within: "+C_list[i])
        legend_it.append(("Within: "+C_list[i], [c]))
        names.append("Within: "+C_list[i])
        n +=1

        # Plot Between on top of the within plot
        c = p.vbar(x=dodge('years',  spacing[i],  range=p.x_range), top='Between', width=bar_widthLP.value, source=sources[C_list[i]], fill_alpha=0.65,
               color=palette_figure_7[n], name= "Between: "+C_list[i])
        n +=1
        legend_it.append(("Between: "+C_list[i], [c]))
        names.append("Within: "+C_list[i])


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
        data_table.columns.countrycode==country_vals,
        data_table.columns.year.between(int(minyear_LP.value),int(maxyear_LP.value))))


    dictionary = {'countrycode': [],
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
        dictionary['countrycode'].append(result.countrycode)
        dictionary['countryname'].append(result.countryname)
        dictionary['Agriculture'].append(result.cngLP_ag)
        dictionary['Manufacturing'].append(result.cngLP_manu)
        dictionary['Transportation'].append(result.cngLP_trans)
        dictionary['Retail'].append(result.cngLP_retail)
        dictionary['Construction'].append(result.cngLP_constr)
        dictionary['Mining'].append(result.cngLP_mining)
        dictionary['Other'].append(result.cngLP_other)
        dictionary['Total'].append(result.cngLP_all)
        dictionary['year'].append(result.year)

    data = pd.DataFrame(dictionary)
    data = data[LP_variables.value + ['countryname', 'year']]
    return data

def create_figure_annualLP():
    data = select_obs_annualLP()
    var_list = LP_variables.value
    # Chosen group year value
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



    palette_figure_7 = ['#361c7f','#9467bd', '#b35900', '#990000'] + Category20[20]


    sourceLP = ColumnDataSource(data)

    p = figure(x_range=list(data['year'].as_matrix()), plot_width=1050, plot_height=700)

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
               color=palette_figure_7[n], name = var_list[i])
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
Em = Slider(title="Minimum Exmployment Share - Agriculture (%)", value=0, start=0, end=100, step=2)
minyear = Select(title="Start Year", options=sorted(axis_year.keys()), value="1991")
maxyear = Select(title="End Year", options=sorted(axis_year.keys()), value="2013")
font_scatter = Select(title="Title Font Size", options=sorted(axis_font.keys()), value="14")
#Country_sc = TextInput(title="Country")
x_axis  = Select(title="X Axis", options=sorted(axis_map_notes.keys()), value="Employment Share - Agriculture")
y_axis = Select(title="Y Axis", options=sorted(axis_map_notes.keys()), value="GDP Per Capita (Thousands; Constant 2010 USD)")
title_name = TextInput(title="Title", value="Scatter Plot")
note_scatter = TextInput(title="Additional Note Content", value='')
### Adding notes to Scatter acts very weird.  Address at different point.


# Generate HoverTool
hover_scatter = HoverTool(tooltips=[('Country', '@countryname'),
                           ('Year', '@year'),
			   ('X Value', '@xx{0.00 a}'),
			   ('Y Value', '@yy{0.00 a}')])

# Generate the Blank Source
source = ColumnDataSource(data=dict(countrycode = [] , xx= [], yy= [], year= [], color=[], OECD_norm=[]))
source1 = ColumnDataSource(data=dict(rx = [], ry =[]))
#source2 = ColumnDataSource(data=dict(r2x = [], r2y =[]))

# Create Button which Downloads CSV file
button = Button(label="Download Data", button_type="success")
button.callback = CustomJS(args=dict(source=source),
                           code=open(join(dirname(__file__), 'models', 'download.js')).read())


# Generate the Plot

t = figure(plot_height=900, plot_width=900, output_backend="webgl")
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
         data_table.columns.GDPpc_2010 <= GDP.value,
         data_table.columns.year.between(minyeara,maxyeara)))

    dictionary = {'countrycode': [],
                  'countryname': [],
                  x_name : [],
                  y_name : [],
                  'year' : [],
                  'OECD_fragile': []
                  }

    for result in connection.execute(stmt):
         dictionary['countrycode'].append(result.countrycode)
         dictionary['countryname'].append(result.countryname)
         dictionary['year'].append(result.year)
         if x_name != 'year':
             dictionary[x_name].append(result[x_name])
         dictionary[y_name].append(result[y_name])
         dictionary['OECD_fragile'].append(result['OECD_fragile'])


    selected= pd.DataFrame(dictionary)

    selected["color"] = np.where(selected["OECD_fragile"] == 'Within OECD Index', '#b35900', '#361c7f' )


    t.title.text = title_name.value #+": %d observations " % len(df)
    #p.title.text = "%d Observations Selected" % len(df)
    fontvalue = font_scatter.value+'pt'
    t.title.text_font_size = fontvalue
    t.xaxis.axis_label = x_axis.value
    t.yaxis.axis_label = y_axis.value

    selected.dropna(inplace=True)
    source.data = dict(
        countrycode = selected['countrycode'],
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



# Add the HoverTool to the plot
t.add_tools(hover_scatter)

#Load the Initial Plot
update_sc()

#####################################################################
#####################################################################
#               Empty Plot                           #
#####################################################################
#####################################################################



empty = figure(plot_height=600, plot_width=700, title="")
label = Label(x=1.1, y=18, text='no data', text_font_size='70pt', text_color='#016450')
empty.add_layout(label)



#####################################################################
#####################################################################
#               Text Documents                             #
#####################################################################
#####################################################################


intro_portal = Div(text=open(join(dirname(__file__), 'tops', 'intro_portal.html')).read(), width=1200)


'''
intro_areaempgva = Div(text=open(join(dirname(__file__), 'tops', 'intro_areaempgva.html')).read(), width=800)
intro_relemp = Div(text=open(join(dirname(__file__), 'tops', 'intro_relemp.html')).read(), width=800)
intro_LP = Div(text=open(join(dirname(__file__), 'tops', 'intro_LP.html')).read(), width=800)
intro_TRADE = Div(text=open(join(dirname(__file__), 'tops', 'intro_TRADE.html')).read(), width=800)
intro_areataxrc = Div(text=open(join(dirname(__file__), 'tops', 'intro_areataxrc.html')).read(), width=800)
intro_scatter = Div(text=open(join(dirname(__file__), 'tops', 'intro_scatter.html')).read(), width=800)
'''
exit = Div(text=open(join(dirname(__file__), 'tops', 'exit.html')).read(), width=800)





##########################################################################
##########################################################################
#                    Generate plot choice section                        #
##########################################################################
##########################################################################

####### Choose the Section of plots:

axis_map_subject = {
        "Structural Transformation": "STR",
        "Upgrading Factors of Production": "UP",
	"Trade": 'TRADE',
	"Firm-Level Analysis": 'firms',
	"Cross-Sectional": 'cross'
	}
Subject_choice = Select(title='Choose Economic Transformation Subject', options=sorted(axis_map_subject.keys()), value='Cross-Sectional')


###########################################################
########### Create different first choices
###########################################################


#### Structual Transformation
axis_map_STRUC = { "Area Chart - Employment and GVA Composition": "empgva",
		     "Scatter - Changes in Employment Composition and Relative LP": "empave",
		'Area Chart - Tax Revenue Composition': 'tax_area'}
Plot_STRUC = Select(title='Type of Plot', options=sorted(axis_map_STRUC.keys()), value="Scatter - Changes in Employment Composition and Relative LP")


#### Labour Productivity Charts :
axis_map_LP = {"Between/Within Labour Productivity": "Bar - Between/Within Labour Productivity",
	       "Changes in Labour Productivity (Annualised)": "Bar - Between/Within Labour Productivity"
           }
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
                'Area Chart - Time Series': 'Area-Line',
                'Bar Chart - Comparison': 'Bar'}
Plot_CROSS = Select(title='Type of Plot', options=sorted(axis_map_cross.keys()), value='Line Chart - Time Series')




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


############################################
#Generate the callback to change the choices of plots
axis_map_subject = {
        "Structural Transformation": "STR",
        "Upgrading Factors of Production": "UP",
	"Trade": 'TRADE',
	"Firm-Level Analysis": 'firms',
	"Cross-Sectional": 'cross'
	}

UPDATE= Button(label="Update", button_type="success")
# Update the heading choices when the subject widgets changes
def update_start():
	if Subject_choice.value=="Structural Transformation":
		# generate the widgets within the update()
		# does not work otherwise
		layout.children[1] = widgetbox(Subject_choice, Plot_STRUC)
		update_plot_STRUC()
	elif Subject_choice.value=="Upgrading Factors of Production":
		layout.children[1] = widgetbox( Subject_choice, Plot_LP)
		update_plot_LP()
	elif Subject_choice.value=="Trade":
		layout.children[1] = widgetbox( Subject_choice, Plot_TRADE)
		update_plot_TRADE()
	elif Subject_choice.value=="Firm-Level Analysis":
		layout.children[1] = widgetbox( Subject_choice,Plot_FIRM)
		update_plot_FIRM()
	elif Subject_choice.value=="Cross-Sectional":
		layout.children[1] = widgetbox( Subject_choice, Plot_CROSS)
		update_plot_CROSS()
	else:
		print('there is something wrong')


###############################################################################
################## Generate the callbacks which change the plots
###############################################################################


#################################
### Structural Transformation

def update_plot_STRUC():
    # Generate the Area Chart - Employment/GVA
    if Plot_STRUC.value=='Area Chart - Employment and GVA Composition':
        print('why')
        # Text heading
        layout.children[2] = Div(text=open(join(dirname(__file__), 'tops', 'intro_areaempgva.html')).read(), width=800)
        # Widgets
        layout.children[3] = Tabs(tabs=[Panel(child=create_figure_emp(), title='Employment'), Panel(child=create_figure_gva(), title='Gross Value Added')])
        # Plot
        layout.children[4] = widgetbox(CountrySTRUC, title_name_emp, title_name_gva, fontSET, min_yearSET, max_yearSET)


    # Generate the Area Chart - Tax Revenue
    elif Plot_STRUC.value=='Area Chart - Tax Revenue Composition':
        print('why')
        # Text heading
        layout.children[2] = row(Div(text=open(join(dirname(__file__), 'tops', 'intro_areataxrc.html')).read(), width=800))
        # Widgets
        layout.children[3] = column(Div(text=open(join(dirname(__file__), 'tops', 'nodata.html')).read(), width=800), widgetbox(CountrySTRUC))
        layout.children[4] = empty
        layout.children[3] = Tabs(tabs=[Panel(child=create_figure_tax_nonrc(), title='Resource Revenue Within'), Panel(child=create_figure_tax_rc(), title='Resource Revenue Disaggregated')])
        # if data exist, replace the empty plot
        layout.children[4] = column(widgetbox(tax_nonrc_title, tax_rc_title, min_year_taxrc, max_year_taxrc, fontrc), widgetbox(Button(label="Download - Resource Dissagregated", button_type="success", callback = CustomJS(args=dict(source=update_taxdatarc()), code=open(join(dirname(__file__), 'models', "download_taxdatarc.js")).read())), DataTable(source=update_taxdatarc(), columns=columns_taxdatarc, width=800, fit_columns=False)), widgetbox(Button(label="Download - Resource Revenue Within", button_type="success", callback = CustomJS(args=dict(source=update_taxdata_nonrc()), code=open(join(dirname(__file__), 'models', "download_taxdata_nonrc.js")).read())), DataTable(source=update_taxdata_nonrc(), columns=columns_tax_nonrc, width=800, fit_columns=False)))


    # Else refers to the scatter - relemp plot
    else:
        # The heading stays the same
        layout.children[2] = row(Div(text=open(join(dirname(__file__), 'tops', 'intro_relemp.html')).read(), width=800))
        # Plot and Widgits (row)
        layout.children[3] = row(widgetbox(CountrySTRUC, minavyear, maxavyear,title_relemp, font_relemp, note_relemp, note_relemp2, note_relemp3, note_relemp4), create_figure_relemp())
        # Blank widgit box
        layout.children[4] = row(widgetbox(Button(label="Download Data", button_type="success", callback = CustomJS(args=dict(source=select_obs_relemp()), code=open(join(dirname(__file__), 'models', "download_relemp.js")).read())), DataTable(source=select_obs_relemp(), columns=columns_relemp, width=800, fit_columns=False)))



#############################
### Firms


def update_plot_FIRM():
	# no plots
	if Plot_FIRM.value=='Currently no visuals are available':
		print('why')
		# Appologies for not plot, else blanks
		layout.children[2] = Div(text=open(join(dirname(__file__), 'tops', 'intro_sorry.html')).read(), width=800)
		layout.children[3] = widgetbox()
		layout.children[4] = widgetbox()

############################
### Trade


def update_plot_TRADE():
	# Default is the no plots
	if Plot_TRADE.value=="Area Chart - Composition of Exports":
		# Appologies for not plot, else blanks
		layout.children[2] = Div(text=open(join(dirname(__file__), 'tops', 'intro_TRADE.html')).read(), width=800)
		layout.children[3] = row(widgetbox(CountryTRADE,min_yearTRADE, max_yearTRADE, exportarea_title, font_TRADE, note_TRADE, note_TRADE2, note_TRADE3, note_TRADE4), create_figure_trade())
		layout.children[4] = row(widgetbox(Button(label="Download Data", button_type="success", callback = CustomJS(args=dict(source=update_tradedata()), code=open(join(dirname(__file__), 'models', "download_tradedata.js")).read())), DataTable(source=update_tradedata(), columns=columns_tradedata, width=800, fit_columns=False)))

	else:
		print('why')


def update_plot_LP():
	print('Yes')
	# Generate the Area Chart when selected
	if Plot_LP.value=='Between/Within Labour Productivity':
		print('why')
		# Text heading
		layout.children[2] = Div(text=open(join(dirname(__file__), 'tops', 'intro_LP.html')).read(), width=800)	# Widgets
		layout.children[3] = row(widgetbox(LPcountries, LP_var, minyear_LP, maxyear_LP, group_yearsLP, bar_widthLP, title_LP, font_LP, note_LP, note_LP2), create_figure_withinbtw())
		# Plot
		layout.children[4] =  widgetbox() #row(widgetbox(LP_var, minyear_LP, maxyear_LP, group_years, bar_width, order_bar, title_LP, font_LP, note_LP, note_LP2, legend_location_LP, legend_location_ori_LP), widgetbox(button_withbtwLP, DataTable(source=update_table_withbtw(), columns=columns, width=800, fit_columns=False)))
		# The heading stays the same
	elif Plot_LP.value=='Changes in Labour Productivity (Annualised)':
		print('why3')
		layout.children[2] = Div(text=open(join(dirname(__file__), 'tops', 'intro_LP_annualised.html')).read(), width=800)		# Plot and Widgits (row)
		layout.children[3] = row(widgetbox(CountryLP, LP_variables, minyear_LP, maxyear_LP, group_yearsLP, order_barLP, title_LP, font_LP, bar_widthLP,  note_LP, note_LP2), create_figure_annualLP())
		# Blank widgit box
		layout.children[4] = widgetbox() #row(widgetbox(button_annualLP, DataTable(source=update_table_annual(), columns=columns, width=650, fit_columns=False)))
	print('I wonder why')


#############################
### Cross-sectional

def update_plot_CROSS():
	# Default is the Scatter
	if Plot_CROSS.value=='Scatter':
		# Scatter heading
		layout.children[2] = Div(text=open(join(dirname(__file__), 'tops', 'intro_scatter.html')).read(), width=800)
		# Widgit box and scatter plot
		layout.children[3] = row(widgetbox(button, *controls_scatter), t)
		# blank
		layout.children[4] = widgetbox()
	elif Plot_CROSS.value=='Line Chart - Time Series':
		# Scatter heading
		layout.children[2] = Div(text=open(join(dirname(__file__), 'tops', 'intro_line.html')).read(), width=800)
		# Widgit box and scatter plot
		layout.children[3] = row(widgetbox(linecross_scatter, countries, line_var, minyear_linecross, maxyear_linecross, rolling_linecross, title_linecross, font_linecross,legend_location_linecross, legend_location_ori_linecross, note_linecross, note_linecross2,  note_linecross3,  note_linecross4 ), create_figure_linecross())
		# blank
		layout.children[4] = widgetbox(Button(label="Download Data", button_type="success", callback = CustomJS(args=dict(source=update_linecrossdata()),
                                   code=open(join(dirname(__file__), 'models', "download_linecross.js")).read())), DataTable(source=update_linecrossdata(), columns=columns_linescross, width=800, fit_columns=False))

	elif Plot_CROSS.value=='Area Chart - Time Series':
		# Scatter heading
		layout.children[2] = Div(text=open(join(dirname(__file__), 'tops', 'intro_arealine.html')).read(), width=800)
		# Widgit box and scatter plot
		layout.children[3] = row(widgetbox(countries, nonstack_var, minyear_nonstack, maxyear_nonstack, title_nonstack, shade_nonstack, font_nonstack, legend_location_nonstack, legend_location_ori_nonstack, note_nonstack, note_nonstack2), create_figure_nonstack())
		layout.children[4] = widgetbox(Button(label="Download Data", button_type="success", callback = CustomJS(args=dict(source=update_areacrossdata()),
                                   code=open(join(dirname(__file__), 'models', "download_linecross.js")).read())), DataTable(source=update_areacrossdata(), columns=columns_areacross, width=800, fit_columns=False))
	elif Plot_CROSS.value=='Bar Chart - Comparison':
		# Scatter heading
		layout.children[2] = Div(text=open(join(dirname(__file__), 'tops', 'intro_bar_chart.html')).read(), width=800)
		# Widgit box and scatter plot
		if bar_plot_options.active==0:
			layout.children[3] = row(widgetbox(bar_plot_options, countries, bar_cross_var, bar_order, minyear_bar_cross, maxyear_bar_cross, title_bar_cross, font_bar_cross, round_bar_cross, note_bar_cross, note_bar_cross2,  note_bar_cross3,  note_bar_cross4), create_figure_bar_cross())
		elif bar_plot_options.active==1:
			layout.children[3] = row(widgetbox(bar_plot_options, countries, bar_cross_var, group_years, minyear_bar_cross, maxyear_bar_cross, title_bar_cross, font_bar_cross, bar_width, legend_location_ori_bar_cross, legend_location_bar_cross, note_bar_cross, note_bar_cross2), create_figure_bar_cross())
		elif bar_plot_options.active==2:
			layout.children[3] = row(widgetbox(bar_plot_options, countries, bar_cross_var, group_years, minyear_bar_cross, maxyear_bar_cross, title_bar_cross, font_bar_cross, bar_width, legend_location_ori_bar_cross, legend_location_bar_cross, note_bar_cross, note_bar_cross2), create_figure_bar_cross())

		layout.children[4] = widgetbox()
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



########### Generate the on.change updates for the plot widgets
########### These widgets recall the update_plot functions above.

CountrySTRUC.on_change('value', lambda attr, old, new: update_plot_STRUC())
CountryLP.on_change('value', lambda attr, old, new: update_plot_LP())
CountryFIRM.on_change('value', lambda attr, old, new: update_plot_FIRM())
CountryTRADE.on_change('value', lambda attr, old, new: update_plot_TRADE())


#### STRUC plots
# RELEMP
controls_relemp = [minavyear, maxavyear, title_relemp, font_relemp, note_relemp, note_relemp2, note_relemp3, note_relemp4]
for control in controls_relemp:
    control.on_change('value', lambda attr, old, new: update_plot_STRUC())

# Area chart
controls_SET = [title_name_emp, title_name_gva, min_yearSET, max_yearSET, fontSET]
for control in controls_SET:
    control.on_change('value', lambda attr, old, new: update_plot_STRUC())
# Generate the on.change updates()
controls_tax_rc = [tax_nonrc_title, tax_rc_title, min_year_taxrc, max_year_taxrc, fontrc]
for control in controls_tax_rc:
    control.on_change('value', lambda attr, old, new: update_plot_STRUC())


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
controls_scatter = [x_axis, y_axis, minyear, maxyear, GDP,Em, title_name, font_scatter ]
for control in controls_scatter:
    control.on_change('value', lambda attr, old, new: update_sc())

controls_linecross = [countries, line_var, minyear_linecross,rolling_linecross,  legend_location_linecross, legend_location_ori_linecross,  note_linecross, note_linecross2,  note_linecross3,  note_linecross4,  maxyear_linecross, title_linecross, font_linecross]
for control in controls_linecross:
    control.on_change('value', lambda attr, old, new: update_plot_CROSS())

controls_nonstack = [legend_location_nonstack, legend_location_ori_nonstack, note_nonstack, note_nonstack2,  note_nonstack3,  note_nonstack4, countries, nonstack_var, minyear_nonstack, maxyear_nonstack, title_nonstack, shade_nonstack, font_nonstack]
for control in controls_nonstack:
    control.on_change('value', lambda attr, old, new: update_plot_CROSS())

controls_bar_cross = [ countries, group_years, bar_cross_var, minyear_bar_cross, maxyear_bar_cross, title_bar_cross, font_bar_cross, legend_location_ori_bar_cross, legend_location_bar_cross, round_bar_cross, bar_width, note_bar_cross, note_bar_cross2,  note_bar_cross3,  note_bar_cross4]

bar_plot_options.on_change('active',  lambda attr, old, new: update_plot_CROSS())
linecross_scatter.on_change('active',  lambda attr, old, new: update_plot_CROSS())
bar_order.on_change('active',  lambda attr, old, new: update_plot_CROSS())

for control in controls_bar_cross:
    control.on_change('value',  lambda attr, old, new: update_plot_CROSS())



#####################################################################
#####################################################################
#               Post to the Bokeh Server                            #
#####################################################################
#####################################################################




# Add to the Bokeh Server
curdoc().add_root(layout)
curdoc().title = "SET Interactive Data Portal"
