# Import packackes
import pandas as pd
from os.path import dirname, join
import numpy as np
import sqlalchemy
import os
from sqlalchemy import Table, MetaData, select, and_
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.types import Float, VARCHAR
# Get connections

user = os.environ.get('DATA_DB_USER')
passwd = os.environ.get('DATA_DB_PASS')
host = os.environ.get('DATA_DB_HOST')

# Load Data Into Database

df = pd.read_csv('./Portal/data/ODI-Portal_March2018.csv', sep='|', low_memory=False)


# Names of Variables
names = ['BD', 'CES_ag', 'CES_constr', 'CES_manu', 'CES_mining', 'CES_other', 'CES_retail', 'CES_trans', 'DB', 'FDI_growth', 'FDI_nom_UScontant', 'FS_score_overtime', 'GDP_growth', 'GDPpc_2010', 'GVA_shar_mining', 'GVA_share_ag', 'GVA_share_constr', 'GVA_share_manu', 'GVA_share_other', 'GVA_share_retail', 'GVA_share_trans', 'LP_ag', 'LP_all', 'LP_btw_ag', 'LP_btw_all', 'LP_btw_constr', 'LP_btw_manu', 'LP_btw_mining', 'LP_btw_other', 'LP_btw_retail', 'LP_btw_trans', 'LP_constr', 'LP_manu', 'LP_mining', 'LP_other', 'LP_retail', 'LP_total_ag', 'LP_total_all', 'LP_total_constr', 'LP_total_manu', 'LP_total_mining', 'LP_total_other', 'LP_total_retail', 'LP_total_trans', 'LP_trans', 'LP_within_ag', 'LP_within_all', 'LP_within_constr', 'LP_within_manu', 'LP_within_mining', 'LP_within_other', 'LP_within_retail', 'LP_within_trans', 'ODA_priv_ratio', 'ODI_constant2010', 'OECD_binary', 'OECD_extreme', 'OECD_fragile', 'OECD_full', 'OECD_norm', 'SET Fragility Index', 'SET_string', 'WB_Fragile', 'access_elec_pct', 'av_Cpc_ex_ore_dol', 'ave_FDI_growth', 'ave_cng_pc_exports_food_dol', 'ave_cng_pc_exports_goods_dol', 'ave_cng_pc_exports_manu_dol', 'ave_cng_pc_exports_merch_dol', 'ave_cng_pc_exports_serv_dol', 'ave_growth_ymin5_y', 'cngLP_ag', 'cngLP_all', 'cngLP_constr', 'cngLP_manu', 'cngLP_mining', 'cngLP_other', 'cngLP_retail', 'cngLP_trans', 'cng_FDI_in_pct', 'cng_emp_share_all', 'countrycode', 'countryname', 'displaced_new', 'displaced_total', 'dom_priv_cred_pct', 'employ_share_ag', 'employ_share_constr', 'employ_share_manu', 'employ_share_mining', 'employ_share_other', 'employ_share_retail', 'employ_share_trans', 'export_conc_index', 'export_value', 'exports_GandS_constUS', 'exports_agraw_pctsum', 'exports_food_pctsum', 'exports_fuel_pctsum', 'exports_manu_pctsum', 'exports_merch_dol', 'exports_oresmet_pctsum', 'exports_val_index', 'fdi_pct_GDP', 'fin_sec_CPIA', 'import_conc_index', 'liner_ship_index', 'logistics_index', 'mobile_per100', 'natural_rec_rent_pct', 'nom_inv', 'open_indicator1', 'pc_exports_GandS_constUS', 'pc_exports_ICTgood_dol', 'pc_exports_ICTser_dol', 'pc_exports_agraw_dol', 'pc_exports_comm_dol', 'pc_exports_food_dol', 'pc_exports_fuel_dol', 'pc_exports_goods_dol', 'pc_exports_insfin_dol', 'pc_exports_manu_dol', 'pc_exports_merch_dol', 'pc_exports_oresmet_dol', 'pc_exports_serv_dol', 'pc_exports_tour_dol', 'pct_aid', 'peace_keep_pres', 'pub_man_CPIA', 'relLP_ag', 'relLP_all', 'relLP_constr', 'relLP_manu', 'relLP_mining', 'relLP_other', 'relLP_retail', 'relLP_trans', 'remit_pct', 'sumGVA', 'sumemploy', 'sumexport', 'tax_exp_pct', 'tax_goods_pct', 'tax_import_pct', 'tax_inc_pct', 'tax_other_pct', 'tax_rev_LCU', 'tax_rev_pct', 'var', 'year', 'tot_resource_revpct', 'direct_inc_sc_ex_resource_revpct', 'tax_g_spct', 'tax_int_trade_transpct', 'other_rc', 'other_nonrc', 'direct_inc_scpct']

print(set(list(df)) - set(names))

# Variable Types
variable_types = {'BD': Float(precision=3, asdecimal=True),
 'CES_ag': Float(precision=3, asdecimal=True),
 'CES_constr': Float(precision=3, asdecimal=True),
 'CES_manu': Float(precision=3, asdecimal=True),
 'CES_mining': Float(precision=3, asdecimal=True),
 'CES_other': Float(precision=3, asdecimal=True),
 'CES_retail': Float(precision=3, asdecimal=True),
 'CES_trans': Float(precision=3, asdecimal=True),
 'DB': Float(precision=3, asdecimal=True),
 'FDI_growth': Float(precision=3, asdecimal=True),
 'FDI_nom_UScontant': Float(precision=3, asdecimal=True),
 'FS_score_overtime': Float(precision=3, asdecimal=True),
 'GDP_growth': Float(precision=3, asdecimal=True),
 'GDPpc_2010': Float(precision=3, asdecimal=True),
 'GVA_shar_mining': Float(precision=3, asdecimal=True),
 'GVA_share_ag': Float(precision=3, asdecimal=True),
 'GVA_share_constr': Float(precision=3, asdecimal=True),
 'GVA_share_manu': Float(precision=3, asdecimal=True),
 'GVA_share_other': Float(precision=3, asdecimal=True),
 'GVA_share_retail': Float(precision=3, asdecimal=True),
 'GVA_share_trans': Float(precision=3, asdecimal=True),
 'LP_ag': Float(precision=3, asdecimal=True),
 'LP_all': Float(precision=3, asdecimal=True),
 'LP_btw_ag': Float(precision=3, asdecimal=True),
 'LP_btw_all': Float(precision=3, asdecimal=True),
 'LP_btw_constr': Float(precision=3, asdecimal=True),
 'LP_btw_manu': Float(precision=3, asdecimal=True),
 'LP_btw_mining': Float(precision=3, asdecimal=True),
 'LP_btw_other': Float(precision=3, asdecimal=True),
 'LP_btw_retail': Float(precision=3, asdecimal=True),
 'LP_btw_trans': Float(precision=3, asdecimal=True),
 'LP_constr': Float(precision=3, asdecimal=True),
 'LP_manu': Float(precision=3, asdecimal=True),
 'LP_mining': Float(precision=3, asdecimal=True),
 'LP_other': Float(precision=3, asdecimal=True),
 'LP_retail': Float(precision=3, asdecimal=True),
 'LP_total_ag': Float(precision=3, asdecimal=True),
 'LP_total_all': Float(precision=3, asdecimal=True),
 'LP_total_constr': Float(precision=3, asdecimal=True),
 'LP_total_manu': Float(precision=3, asdecimal=True),
 'LP_total_mining': Float(precision=3, asdecimal=True),
 'LP_total_other': Float(precision=3, asdecimal=True),
 'LP_total_retail': Float(precision=3, asdecimal=True),
 'LP_total_trans': Float(precision=3, asdecimal=True),
 'LP_trans': Float(precision=3, asdecimal=True),
 'LP_within_ag': Float(precision=3, asdecimal=True),
 'LP_within_all': Float(precision=3, asdecimal=True),
 'LP_within_constr': Float(precision=3, asdecimal=True),
 'LP_within_manu': Float(precision=3, asdecimal=True),
 'LP_within_mining': Float(precision=3, asdecimal=True),
 'LP_within_other': Float(precision=3, asdecimal=True),
 'LP_within_retail': Float(precision=3, asdecimal=True),
 'LP_within_trans': Float(precision=3, asdecimal=True),
 'ODA_priv_ratio': Float(precision=3, asdecimal=True),
 'ODI_constant2010': Float(precision=3, asdecimal=True),
 'OECD_binary': Float(precision=3, asdecimal=True),
 'OECD_extreme': Float(precision=3, asdecimal=True),
 'OECD_fragile': VARCHAR(length=255),
 'OECD_full': Float(precision=3, asdecimal=True),
 'OECD_norm': Float(precision=3, asdecimal=True),
 'SET Fragility Index': VARCHAR(length=255),
 'SET_string': VARCHAR(length=255),
 'WB_Fragile': VARCHAR(length=255),
 'access_elec_pct': Float(precision=3, asdecimal=True),
 'av_Cpc_ex_ore_dol': Float(precision=3, asdecimal=True),
 'ave_FDI_growth': Float(precision=3, asdecimal=True),
 'ave_cng_pc_exports_food_dol': Float(precision=3, asdecimal=True),
 'ave_cng_pc_exports_goods_dol': Float(precision=3, asdecimal=True),
 'ave_cng_pc_exports_manu_dol': Float(precision=3, asdecimal=True),
 'ave_cng_pc_exports_merch_dol': Float(precision=3, asdecimal=True),
 'ave_cng_pc_exports_serv_dol': Float(precision=3, asdecimal=True),
 'ave_growth_ymin5_y': Float(precision=3, asdecimal=True),
 'cngLP_ag': Float(precision=3, asdecimal=True),
 'cngLP_all': Float(precision=3, asdecimal=True),
 'cngLP_constr': Float(precision=3, asdecimal=True),
 'cngLP_manu': Float(precision=3, asdecimal=True),
 'cngLP_mining': Float(precision=3, asdecimal=True),
 'cngLP_other': Float(precision=3, asdecimal=True),
 'cngLP_retail': Float(precision=3, asdecimal=True),
 'cngLP_trans': Float(precision=3, asdecimal=True),
 'cng_FDI_in_pct': Float(precision=3, asdecimal=True),
 'cng_emp_share_all': Float(precision=3, asdecimal=True),
 'countrycode': VARCHAR(length=255),
 'countryname': VARCHAR(length=255),
 'direct_inc_sc_ex_resource_revpct': Float(precision=3, asdecimal=True),
 'direct_inc_scpct': Float(precision=3, asdecimal=True),
 'displaced_new': VARCHAR(length=255),
 'displaced_total': Float(precision=3, asdecimal=True),
 'dom_priv_cred_pct': Float(precision=3, asdecimal=True),
 'employ_share_ag': Float(precision=3, asdecimal=True),
 'employ_share_constr': Float(precision=3, asdecimal=True),
 'employ_share_manu': Float(precision=3, asdecimal=True),
 'employ_share_mining': Float(precision=3, asdecimal=True),
 'employ_share_other': Float(precision=3, asdecimal=True),
 'employ_share_retail': Float(precision=3, asdecimal=True),
 'employ_share_trans': Float(precision=3, asdecimal=True),
 'export_conc_index': Float(precision=3, asdecimal=True),
 'export_value': Float(precision=3, asdecimal=True),
 'exports_GandS_constUS': Float(precision=3, asdecimal=True),
 'exports_agraw_pctsum': Float(precision=3, asdecimal=True),
 'exports_food_pctsum': Float(precision=3, asdecimal=True),
 'exports_fuel_pctsum': Float(precision=3, asdecimal=True),
 'exports_manu_pctsum': Float(precision=3, asdecimal=True),
 'exports_merch_dol': Float(precision=3, asdecimal=True),
 'exports_oresmet_pctsum': Float(precision=3, asdecimal=True),
 'exports_val_index': Float(precision=3, asdecimal=True),
 'fdi_pct_GDP': Float(precision=3, asdecimal=True),
 'fin_sec_CPIA': Float(precision=3, asdecimal=True),
 'import_conc_index': Float(precision=3, asdecimal=True),
 'liner_ship_index': Float(precision=3, asdecimal=True),
 'logistics_index': Float(precision=3, asdecimal=True),
 'mobile_per100': Float(precision=3, asdecimal=True),
 'natural_rec_rent_pct': Float(precision=3, asdecimal=True),
 'nom_inv': Float(precision=3, asdecimal=True),
 'open_indicator1': Float(precision=3, asdecimal=True),
 'other_nonrc': Float(precision=3, asdecimal=True),
 'other_rc': Float(precision=3, asdecimal=True),
 'pc_exports_GandS_constUS': Float(precision=3, asdecimal=True),
 'pc_exports_ICTgood_dol': Float(precision=3, asdecimal=True),
 'pc_exports_ICTser_dol': Float(precision=3, asdecimal=True),
 'pc_exports_agraw_dol': Float(precision=3, asdecimal=True),
 'pc_exports_comm_dol': Float(precision=3, asdecimal=True),
 'pc_exports_food_dol': Float(precision=3, asdecimal=True),
 'pc_exports_fuel_dol': Float(precision=3, asdecimal=True),
 'pc_exports_goods_dol': Float(precision=3, asdecimal=True),
 'pc_exports_insfin_dol': Float(precision=3, asdecimal=True),
 'pc_exports_manu_dol': Float(precision=3, asdecimal=True),
 'pc_exports_merch_dol': Float(precision=3, asdecimal=True),
 'pc_exports_oresmet_dol': Float(precision=3, asdecimal=True),
 'pc_exports_serv_dol': Float(precision=3, asdecimal=True),
 'pc_exports_tour_dol': Float(precision=3, asdecimal=True),
 'pct_aid': Float(precision=3, asdecimal=True),
 'peace_keep_pres': Float(precision=3, asdecimal=True),
 'pub_man_CPIA': Float(precision=3, asdecimal=True),
 'relLP_ag': Float(precision=3, asdecimal=True),
 'relLP_all': Float(precision=3, asdecimal=True),
 'relLP_constr': Float(precision=3, asdecimal=True),
 'relLP_manu': Float(precision=3, asdecimal=True),
 'relLP_mining': Float(precision=3, asdecimal=True),
 'relLP_other': Float(precision=3, asdecimal=True),
 'relLP_retail': Float(precision=3, asdecimal=True),
 'relLP_trans': Float(precision=3, asdecimal=True),
 'remit_pct': Float(precision=3, asdecimal=True),
 'sumGVA': Float(precision=3, asdecimal=True),
 'sumemploy': Float(precision=3, asdecimal=True),
 'sumexport': Float(precision=3, asdecimal=True),
 'tax_exp_pct': Float(precision=3, asdecimal=True),
 'tax_g_spct': Float(precision=3, asdecimal=True),
 'tax_goods_pct': Float(precision=3, asdecimal=True),
 'tax_import_pct': Float(precision=3, asdecimal=True),
 'tax_inc_pct': Float(precision=3, asdecimal=True),
 'tax_int_trade_transpct': Float(precision=3, asdecimal=True),
 'tax_other_pct': Float(precision=3, asdecimal=True),
 'tax_rev_LCU': Float(precision=3, asdecimal=True),
 'tax_rev_pct': Float(precision=3, asdecimal=True),
 'tot_resource_revpct': Float(precision=3, asdecimal=True),
 'var': Float(precision=3, asdecimal=True),
 'year': Float(precision=3, asdecimal=True)}


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



#collect con, meta
engine, connection, meta = connect(user, passwd, 'gonano', host=host)



### load empty dataset for the first time, full dataset for the other uploads
try:
    data = pd.read_sql_table('ODI4-march2018', con=engine, schema ='public')
except ValueError:
    pd.DataFrame(columns=names).to_sql('ODI4-march2018', con=engine, schema='public', dtype=variable_types)


data = pd.read_sql_table('ODI4-march2018', con=engine, schema ='public')

# if dataset is empty, fill with CSV file
if len(data)==0:
    df[names].to_sql('ODI4-march2018', con=engine, schema='public', if_exists='append', dtype=variable_types)


# clear dataframes to clear some memory space.
data = pd.DataFrame()
df= pd.DataFrame()

# close connection
connection.close()
