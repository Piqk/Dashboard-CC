import pandas as pd
import streamlit as st
import plotly.express as px
import datetime
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

#ANALISIS DE PROMO DEL 2023 AL 2025
df = pd.read_excel("Mensual promo.xlsx", sheet_name=0)
df_historial = df
current_agent = ["ARCADIO","MARTHA","IVY","PAULINA","MONSERRAT","ENRIQUE","KARINA","EVA","KENIA","AIME","LOURDES","KATHIA O","KARLA","KATIA","EVELYN","NADINE","ROSARIO","FERNANDA"]
relevant_info_df = df[df["Agente"].isin(current_agent)]

promo_counts_todos = relevant_info_df.groupby("Agente").size().reset_index(name='Count')
promo_counts_hotel = relevant_info_df.groupby("Hotel").size().reset_index(name='Count')

fig_agente = px.pie(promo_counts_todos, values="Count", names="Agente", title="promos generadas por agente")
fig_hotel = px.bar(promo_counts_hotel, x=promo_counts_hotel["Hotel"], y=promo_counts_hotel["Count"], color="Count")

fig_promo_hist = df.groupby("Promoción").size().reset_index(name='Count')
fig_promo_hist = px.bar(fig_promo_hist, x="Promoción", y="Count", color="Promoción")


st.set_page_config(page_title="Dashboard", layout="wide")
st.title("Call Center Dashboard")
st.subheader("Dashboard created by Angel Nelson")
st.markdown("---")
st.title("Historical analysis 2023-2025")
df_final_historico = relevant_info_df.copy()

filtro_promo = st.sidebar.multiselect(
    "Filter by promotion (Historical)",
    options=relevant_info_df['Promoción'].unique(),
    default=[]
)

filtro_agente = st.sidebar.multiselect(
    "Filter by agent (Historical)",
    options=relevant_info_df['Agente'].unique(),
    default=[]
)

if filtro_promo:
    df_final_historico = df_final_historico[df_final_historico['Promoción'].isin(filtro_promo)]

if filtro_agente:
    df_final_historico = df_final_historico[df_final_historico['Agente'].isin(filtro_agente)]

promo_counts_todos = df_final_historico.groupby("Agente").size().reset_index(name='Count')
promo_counts_hotel = df_final_historico.groupby("Hotel").size().reset_index(name='Count')

fig_agente = px.pie(promo_counts_todos, values="Count", names="Agente", title="Promos by agent")
fig_hotel = px.bar(promo_counts_hotel, x=promo_counts_hotel["Hotel"], y=promo_counts_hotel["Count"], color="Count")

fig_promo_hist = df.groupby("Promoción").size().reset_index(name='Count')
fig_promo_hist = px.bar(fig_promo_hist, x="Promoción", y="Count", color="Promoción")

col1, col2 = st.columns(2)

with col1:
    st.metric(label="Total promos",
              value=df["Agente"].shape[0])
with col2:
    st.metric(label="Total promos by agent",
              value=df_final_historico["Agente"].shape[0])

g2col1, g2col2 = st.columns(2)

with g2col1:
    st.plotly_chart(fig_hotel, width="stretch")

with g2col2:
    st.plotly_chart(fig_agente, width="stretch")

#DASHBOARD DEL REPORTE SEMANAL ------------------------------------------------------------------------------------------------------------------------------
st.title("Weekly Progress")
st.markdown("---")

df_semanal = pd.read_excel("Mensual promo.xlsx", sheet_name=0)
df_semanal = df_semanal[df_semanal["Observacion"] != "REAGENDADA"]
df_para_filtro = df_semanal
df_semanal["Dia Ingreso"] = pd.to_datetime(df_semanal["Dia Ingreso"], format="%d/%m/%Y")

df_final_semanal = df_semanal.copy()

filtro_promo = st.sidebar.multiselect(
    "Filter by promotion (Weekly)",
    options=df_semanal['Promoción'].unique(),
    default=[]
)

filtro_agente = st.sidebar.multiselect(
    "Filter by agent (Weekly)",
    options=df_semanal['Agente'].unique(),
    default=[]
)

if filtro_promo:
    df_final_semanal = df_semanal[df_semanal['Promoción'].isin(filtro_promo)]

if filtro_agente:
    df_final_semanal = df_semanal[df_semanal['Agente'].isin(filtro_agente)]

df_filtered_semanal = df_final_semanal

semanas_mostrables = [{"15/11/2025":"21/11/2025"},
                      {"22/11/2025":"28/11/2025"},
                      {"29/11/2025":"05/12/2025"},
                      {"06/12/2025":"12/12/2025"},
                      {"13/12/2025":"19/12/2025"},
                      {"20/12/2025":"26/12/2025"},
                      {"27/12/2025":"02/01/2026"},
                      {"03/01/2026":"09/01/2026"},
                      {"10/01/2026":"16/01/2026"},
                      {"17/01/2025":"23/01/2026"},
                      {"24/01/2026":"30/01/2026"}]

today = datetime.datetime.strptime("15/11/2025", "%d/%m/%Y").date()

active_week = None

for week_dict in semanas_mostrables:
    start_week_date, end_week_date = list(week_dict.items())[0]
    
    start_date = datetime.datetime.strptime(start_week_date, "%d/%m/%Y").date()
    
    end_date = datetime.datetime.strptime(end_week_date, "%d/%m/%Y").date()
    extended_end_date = end_date + datetime.timedelta(days=2)
    df_semana_actual = df_filtered_semanal[
        (df_filtered_semanal["Dia Ingreso"].dt.date >= start_date) &
        (df_filtered_semanal["Dia Ingreso"].dt.date <= end_date)
    ].copy()
    key = f"{start_week_date}-{end_week_date}"


    if start_date <= today <= extended_end_date:
        active_week_dates = (start_date, end_date)
        shown_week_dates = (start_date, extended_end_date)
        break

if active_week_dates:
    df_filtered_semanal = df_filtered_semanal[
        (df_filtered_semanal["Dia Ingreso"].dt.date >= start_date) &
        (df_filtered_semanal["Dia Ingreso"].dt.date <= end_date)
    ]

if shown_week_dates:
    start_viz_date, end_viz_date = shown_week_dates



fig_promo_semal = df_filtered_semanal.groupby("Promoción").size().reset_index(name='Count')
fig_promo_semanal = px.bar(fig_promo_semal, x="Promoción", y="Count", color="Promoción")

fig_agente_semal = df_filtered_semanal.groupby("Agente").size().reset_index(name='Count')
fig_agente_semanal = px.pie(fig_agente_semal, values="Count", names="Agente", title="Promos by agent", hole=0.5)

if active_week_dates:
    start_date, end_date = active_week_dates
    formatted_start = start_date.strftime("%d/%m/%Y")
    formatted_end = end_date.strftime("%d/%m/%Y")
    display_value = f"{formatted_start} - {formatted_end}"


if active_week_dates:
    st.info(f"Active Week: {display_value}")

g3col1,= st.columns(1)

with g3col1:
    st.metric(label="Total promos weekly",
              value=df_filtered_semanal.shape[0])

g2col1, g2col2 = st.columns(2)

with g2col1:
    st.plotly_chart(fig_promo_semanal, width="stretch")

with g2col2:
    st.plotly_chart(fig_agente_semanal, width="stretch")


booking_activity = pd.read_excel("Mensual promo.xlsx", sheet_name=1)

promos = df_filtered_semanal
booking_activity.drop(columns="Room #", inplace=True)
booking_activity["Trx Date"] = pd.to_datetime(booking_activity["Trx Date"],format="%d/%m/%Y")
booking_activity["Trx Date"] = booking_activity["Trx Date"].dt.date

filtro = (booking_activity["Trx Date"] >= start_date) & (booking_activity["Trx Date"] <= end_date)
booking_activity = booking_activity[filtro]

inbound_team = ["llaurel","evhernandez","kgarcia","kangulo","majaime","mnieves","korozco","mosuna"]
inbound_team_name = ["LOURDES","EVELYN","KARLA","KATIA","FERNANDA","NADINE","KATHIA O","ROSARIO"]
outbound_team = ["aruiz","dovazquez","enarcio","istafford","kmacias","mairodriguez","khernandez","macontreras"]
outbound_team_name = ["ARCADIO","PAULINA","EVA","IVY","KARINA","AIME", "KENIA", "MARTHA"]
referral_team = ["SILVIA"]
supervisors_team = ["YUYENITH", "SARAHI","LUCY"]
inbound_team_name.sort()
outbound_team_name.sort()
full_team = inbound_team_name + outbound_team_name + referral_team
inbound_team_dict = {"llaurel":"LOURDES",
                     "evhernandez":"EVELYN",
                     "kgarcia":"KARLA",
                     "kangulo":"KATIA",
                     "majaime":"FERNANDA",
                     "mnieves":"NADINE",
                     "korozco":"KATHIA O",
                     "mosuna":"ROSARIO" }

inbound_team_dict_full_name = {"Lourdes Laurel": "LOURDES",
                               "Monserrat Vazquez":"MONSERRAT",
                               "Evelyn Hernandez": "EVELYN",
                               "Enrique Izquierdo": "ENRIQUE",
                               "Karla Garcia": "KARLA",
                               "Katia Angulo": "KATIA",
                               "Fernanda Jaime": "FERNANDA",
                               "Rayito Zatarain" :"RAYITO",
                               "Kathia Orozco": "KATHIA O",
                               "Rosario Osuna": "ROSARIO"}

outbound_team_dict_full_name = {"Arcadio Ruiz": "ARCADIO",
                                "Paulina Vazquez": "PAULINA",
                                "Eva Narcio": "EVA",
                                "Ivy Stafford": "IVY",
                                "Karina Macias": "KARINA",
                                "Aime Rodriguez": "AIME",
                                "Kenia Hernandez": "KENIA",
                                "Martha Contreras": "MARTHA"}

df4 = pd.DataFrame({
    "Agente": inbound_team_name
})

filtered_booking_activity = booking_activity[booking_activity["Modified By"].isin(inbound_team)]
filtered_wto_cxl = filtered_booking_activity[~filtered_booking_activity["Status"].str.contains("Cxl")].copy()
filtered_wto_cxl = filtered_wto_cxl[~filtered_wto_cxl["Status"].str.contains("Cancelled")].copy()
filtered_wto_cxl.reset_index(drop=True, inplace=True)
filtered_wto_cxl.replace(to_replace=inbound_team_dict, inplace=True)
booking_counts = filtered_wto_cxl["Modified By"].value_counts().reset_index()
booking_counts.columns = ["Agente", "Reservas Elegibles"]
df4 = pd.merge(df4, booking_counts, on="Agente", how="left")
df4["Reservas Elegibles"] = df4["Reservas Elegibles"].fillna(0).astype(int)

tlmk_counter = (promos["Campaña"] == "TLMK REFERIDO").sum()


promos_wto_tlmk = promos[~promos["Campaña"].str.contains("TLMK REFERIDO", na=False)].copy()
promos_wto_tlmk.reset_index(drop=True, inplace=True)


filtered_promos = promos_wto_tlmk[promos_wto_tlmk["Agente"].isin(inbound_team_name)]
agent_promo_counts = filtered_promos["Agente"].value_counts().reset_index()
agent_promo_counts.columns = ["Agente", "Promo Count"]
all_inbound_agents = pd.DataFrame({"Agente": inbound_team_name})
agent_promo_counts = pd.merge(all_inbound_agents, agent_promo_counts, on="Agente", how="left")
agent_promo_counts["Promo Count"] = agent_promo_counts["Promo Count"].fillna(0).astype(int)


df4 = pd.merge(df4, agent_promo_counts, on="Agente", how="left")
df4["Promo Count"] = df4["Promo Count"].fillna(0).astype(int)
df4["% HOOK"] = (df4["Promo Count"] / df4["Reservas Elegibles"])


condiciones = [

    (df4["% HOOK"] >= 0.10) & (df4["% HOOK"] <= 0.1999),
    (df4["% HOOK"] >= 0.20) & (df4["% HOOK"] <= 0.2999),
    (df4["% HOOK"] >= 0.30) & (df4["% HOOK"] <= 0.3999),
    (df4["% HOOK"] >= 0.40)

]

Bono_dias = [1,2,3,4]

df4["Dias libres"] = np.select(condiciones, Bono_dias, default=0)
df4.loc[len(df4)] = None
df4.loc[df4.index[-1], "Reservas Elegibles"] = df4["Reservas Elegibles"].sum()
df4.loc[df4.index[-1], "Promo Count"] = df4["Promo Count"].sum()
hook_percentage = (df4.iloc[8,2]/df4.iloc[8,1])
df4.loc[df4.index[-1], "% HOOK"] = hook_percentage


df4.loc[df4['Agente'] == 'EVELYN', 'Agente'] = 'Evelyn'
df4.loc[df4['Agente'] == 'LOURDES', 'Agente'] = 'Lourdes'
df4.loc[df4['Agente'] == 'FERNANDA', 'Agente'] = "Fernanda"
df4.loc[df4['Agente'] == 'KARLA', 'Agente'] = 'Karla'
df4.loc[df4['Agente'] == 'KATHIA O', 'Agente'] = 'Kathia'
df4.loc[df4['Agente'] == 'KATIA', 'Agente'] = 'Katia'
df4.loc[df4['Agente'] == 'NADINE', 'Agente'] = 'Nadine'
df4.loc[df4['Agente'] == 'ROSARIO', 'Agente'] = 'Rosario'

#Con filtros adicionales

df5 = pd.DataFrame({
    "Agente": inbound_team_name
})

col1, = st.columns(1)

with col1:
    st.subheader(f"Booking Activity from {start_date} to {end_date}")
    st.dataframe(filtered_wto_cxl)

df_wto_discount = filtered_wto_cxl[~filtered_wto_cxl["Disc Applied"].str.contains("35%", na=False)].copy()
df_wto_discount = df_wto_discount[~df_wto_discount["Disc Applied"].str.contains("50%", na=False)].copy()
df_wto_discount = df_wto_discount[~filtered_wto_cxl["LOS"].astype(str).str.contains("1", na=False)].copy()
df_wto_discount = df_wto_discount[~filtered_wto_cxl["LOS"].astype(str).str.contains("2", na=False)].copy()
df_wto_discount = df_wto_discount[~filtered_wto_cxl["LOS"].astype(str).str.contains("3", na=False)].copy()
df_wto_discount = df_wto_discount[~df_wto_discount["Subtype"].str.contains("Guest Certificate", na=False)].copy()
df_wto_discount = df_wto_discount[~df_wto_discount["Subtype"].str.contains("Consecutive Hooked", na=False)].copy()
df_wto_discount = df_wto_discount[~df_wto_discount["Market Code"].str.contains("VLO", na=False)].copy()
df_wto_discount = df_wto_discount[~df_wto_discount["Market Code"].str.contains("OwnerRelations", na=False)].copy()
booking_counts = df_wto_discount["Modified By"].value_counts().reset_index()
booking_counts.columns = ["Agente", "Reservas Elegibles"]

df5 = pd.merge(df5, booking_counts, on="Agente", how="left")
df5["Reservas Elegibles"] = df5["Reservas Elegibles"].fillna(0).astype(int)

filtered_promos = promos_wto_tlmk[promos_wto_tlmk["Agente"].isin(inbound_team_name)]
agent_promo_counts = filtered_promos["Agente"].value_counts().reset_index()
agent_promo_counts.columns = ["Agente", "Promo Count"]
all_inbound_agents = pd.DataFrame({"Agente": inbound_team_name})
agent_promo_counts = pd.merge(all_inbound_agents, agent_promo_counts, on="Agente", how="left")
agent_promo_counts["Promo Count"] = agent_promo_counts["Promo Count"].fillna(0).astype(int)

df5 = pd.merge(df5, agent_promo_counts, on="Agente", how="left")
df5["Promo Count"] = df5["Promo Count"].fillna(0).astype(int)
df5["% HOOK"] = (df5["Promo Count"] / df5["Reservas Elegibles"])
df5.loc[len(df5)] = None
df5.loc[df5.index[-1], "Reservas Elegibles"] = df5["Reservas Elegibles"].sum()
df5.loc[df5.index[-1], "Promo Count"] = df5["Promo Count"].sum()
hook_percentage = (df5.iloc[8,2]/df5.iloc[8,1])
df5.loc[df5.index[-1], "% HOOK"] = hook_percentage

condiciones = [
    (df5["% HOOK"] >= 0.10) & (df5["% HOOK"] <= 0.1999),
    (df5["% HOOK"] >= 0.20) & (df5["% HOOK"] <= 0.2999),
    (df5["% HOOK"] >= 0.30) & (df5["% HOOK"] <= 0.3999),
    (df5["% HOOK"] >= 0.40)

]

Bono_dias = [1,2,3,4]
df5["Dias libres"] = np.select(condiciones, Bono_dias, default=0)
df5.loc[df5.index[-1], "Dias libres"] = None

df5.loc[df5['Agente'] == 'EVELYN', 'Agente'] = 'Evelyn'
df5.loc[df5['Agente'] == 'LOURDES', 'Agente'] = 'Lourdes'
df5.loc[df5['Agente'] == 'FERNANDA', 'Agente'] = "Fernanda"
df5.loc[df5['Agente'] == 'KARLA', 'Agente'] = 'Karla'
df5.loc[df5['Agente'] == 'KATHIA O', 'Agente'] = 'Kathia'
df5.loc[df5['Agente'] == 'KATIA', 'Agente'] = 'Katia'
df5.loc[df5['Agente'] == 'NADINE', 'Agente'] = 'Nadine'
df5.loc[df5['Agente'] == 'ROSARIO', 'Agente'] = 'Rosario'

for col_index in range(0, 9):
    valor = df4.iloc[col_index, 3]
    if pd.isna(valor):
        df4.iloc[col_index, 3] = "-"
    else:
        try:
            df4.iloc[col_index, 3] = "{:.2%}".format(df4.iloc[col_index, 3])
        except (ValueError, TypeError):
            df4.iloc[col_index, 3] = "-"

for col_index in range(0, 9):
    valor = df5.iloc[col_index, 3]
    if pd.isna(valor):
        df5.iloc[col_index, 3] = "-"
    else:
        try:
            df5.iloc[col_index, 3] = "{:.2%}".format(df5.iloc[col_index, 3])
        except (ValueError, TypeError):
            df5.iloc[col_index, 3] = "-"


g4col1, g4col2 = st.columns(2)

with g4col1:
    st.subheader("Unfiltered IB results")
    st.dataframe(df4, width="stretch")


with g4col2:
    st.subheader("Filtered IB results")
    st.dataframe(df5, width="stretch")
#Dashboard mensual----------------------------------------------------------------------------------------------------------------------------------------------------------

st.title("Monthly Progress")
st.markdown("---")
mes_actual = "November"

st.info(f"Current month: {mes_actual}")

df_mensual  = pd.read_excel("Mensual promo.xlsx", sheet_name=0)
df_mensual = df_mensual[df_mensual["Observacion"] != "REAGENDADA"]

df_final = df_mensual.copy()

filtro_promo = st.sidebar.multiselect(
    "Filter by promotion (Monthly)",
    options=df_mensual['Promoción'].unique(),
    default=[]
)

filtro_agente = st.sidebar.multiselect(
    "Filter by agent (Monthly)",
    options=df_mensual['Agente'].unique(),
    default=[]
)

if filtro_promo:
    df_final = df_final[df_final['Promoción'].isin(filtro_promo)]

if filtro_agente:
    df_final = df_final[df_final['Agente'].isin(filtro_agente)]

fig_promo_mensual = df_final.groupby("Promoción").size().reset_index(name='Count')
fig_promo_mensual = px.bar(fig_promo_mensual, x="Promoción", y="Count", color="Promoción")
fig_agente_mensual = df_final.groupby("Agente").size().reset_index(name='Count')
fig_agente_mensual = px.pie(fig_agente_mensual, values="Count", names="Agente", title="Promos by agent", hole=0.5)

col1, = st.columns(1)
with col1:
    st.metric(label="Total monthly promos",
              value=df_final.shape[0])

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig_promo_mensual, width="stretch", key="promo_mensual")
with col2:
    st.plotly_chart(fig_agente_mensual, width="stretch", key="agente_mensual")

col1, = st.columns(1)

with col1:
    st.dataframe(df_final)

#Tiempo promedio en llamada semanal--- ------------------------------------------------------------------------------------------------------------------

st.title("Weekly Call Time Report")
st.markdown("---")

df_avaya = pd.read_excel("Avaya.xlsx")
df_avaya["Fecha"] = pd.to_datetime(df_avaya["Fecha"])

if active_week_dates:
    df_activo_semanal = (df_avaya[(df_avaya["Fecha"].dt.date >= start_date) & (df_avaya["Fecha"].dt.date <= end_date)].copy())

print(f"Filas antes: {len(df_avaya)}")
print(f"Filas después: {len(df_activo_semanal)}")


df_avaya = df_activo_semanal


df_avaya["Agente"] = df_avaya["Agente"].str.split("ID: ").str[1].str.split(" -").str[0]
df_avaya.insert(5, "Llamadas perdidas", df_avaya["Presented"]-df_avaya["Ans"])
df_avaya.drop(labels=df_avaya.columns[[6,8,9,11,14,15,16,17]], axis=1, inplace=True)

def format_timedelta(td):
    total_seconds = int(td)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"


df_avaya["Logged In Time"] = pd.to_timedelta(df_avaya["Logged In Time"])
df_avaya["Logged In Time"] = df_avaya["Logged In Time"].dt.total_seconds()

df_avaya["Active Time"] = pd.to_timedelta(df_avaya["Active Time"])
df_avaya["Active Time"] = df_avaya["Active Time"].dt.total_seconds()

df_avaya["Hold Time"] = pd.to_timedelta(df_avaya["Hold Time"])
df_avaya["Hold Time"] = df_avaya["Hold Time"].dt.total_seconds()

df_avaya["Not Ready"] = pd.to_timedelta(df_avaya["Not Ready"])
df_avaya["Not Ready"] = df_avaya["Not Ready"].dt.total_seconds()

df_avaya["Idle Time"] = pd.to_timedelta(df_avaya["Idle Time"])
df_avaya["Idle Time"] = df_avaya["Idle Time"].dt.total_seconds()

agg_funcs = {
    'Fecha': 'count', 
    'Logged In Time': 'sum',
    'Presented': 'sum',
    'Ans': 'sum',
    'Llamadas perdidas': 'sum',
    'Active Time': 'sum',
    'Hold Time': 'sum',
    'Not Ready': 'sum',
    'Idle Time': 'sum'
}

df_sumado = df_avaya.groupby("Agente").agg(agg_funcs).reset_index()
df_sumado = df_sumado.rename(columns={'Fecha': 'Dias Trabajados'})
df_avaya = df_sumado

df_avaya.insert(7, "Active Diario", df_avaya["Active Time"]/df_avaya["Dias Trabajados"])
df_avaya.insert(9, "Hold Diario", df_avaya["Hold Time"]/df_avaya["Dias Trabajados"])
df_avaya.insert(11, "Not Ready Diario", df_avaya["Not Ready"]/df_avaya["Dias Trabajados"])
df_avaya.insert(13, "Idle Diario", df_avaya["Idle Time"]/df_avaya["Dias Trabajados"])

df_avaya["Active Diario"] = df_avaya["Active Diario"].apply(format_timedelta)
df_avaya["Hold Diario"] = df_avaya["Hold Diario"].apply(format_timedelta)
df_avaya["Not Ready Diario"] = df_avaya["Not Ready Diario"].apply(format_timedelta)
df_avaya["Idle Diario"] = df_avaya["Idle Diario"].apply(format_timedelta)
df_avaya["Active Time"] = df_avaya["Active Time"].apply(format_timedelta)
df_avaya["Hold Time"] = df_avaya["Hold Time"].apply(format_timedelta)
df_avaya["Not Ready"] = df_avaya["Not Ready"].apply(format_timedelta)
df_avaya["Idle Time"] = df_avaya["Idle Time"].apply(format_timedelta)
df_avaya["Logged In Time"] = df_avaya["Logged In Time"].apply(format_timedelta)


Columnas_quitar_decmal = ["Active Diario","Hold Diario","Not Ready Diario","Idle Diario","Active Time","Hold Time","Not Ready","Idle Time","Logged In Time"]

for col in Columnas_quitar_decmal:
    df_avaya[col] = df_avaya[col].astype(str).str.split('.').str[0]

col1, = st.columns(1)

with col1:
    st.subheader(f"Time report from {display_value}")
    st.dataframe(df_avaya)

st.markdown("---")
st.markdown("Created by Angel Nelson")