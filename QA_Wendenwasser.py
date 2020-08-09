########################################################################################################################
# Quantitative Analyse - Wendenwasser
########################################################################################################################
########################################################################################################################
# Import
########################################################################################################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from matplotlib.dates import DateFormatter
import seaborn as sns
from sklearn.metrics import r2_score
import math
from collections import Counter
import matplotlib.dates as mdates
from datetime import datetime, timedelta as delta

# import data Wenden
dateparse = lambda x: pd.datetime.strptime(x,'%Y-%m-%d %H:%M:%S')
Wenden = pd.read_csv(r'" Pfad einfügen" Wenden.csv',
                    index_col=0, parse_dates=[0],date_parser=dateparse)

# import meteo data
dateparse = lambda x: pd.datetime.strptime(x,'%Y-%m-%d %H:%M:%S')
Meteo = pd.read_csv(r'"Pfad einfügen" Meteo.csv',
                    index_col=0, parse_dates=[0],date_parser=dateparse)
########################################################################################################################
# Saisonale Muster: Isotopen nach Saison
########################################################################################################################
###########
# Einteilen in Saisons
##########
# neue Zeile einfügen
Wenden['Saison'] = np.nan
# Einteilen nach Saison
Wenden.loc['2019-12-01':'2020-03-22','Saison'] = 'Winterlicher Basisabfluss'
Wenden.loc['2019-06-18':'2019-06-30','Saison'] = 'Schneeschmelze'
Wenden.loc['2019-07-01':'2019-08-31','Saison'] = 'Gletscherschmelze'
Wenden.loc['2019-09-01':'2019-11-30','Saison'] = 'Regen'
# Daten vom 9ten März sind 10' vorher vorhanden, diese einsetzen:
# 15' hat kein Wert, den von 10' verwenden:
Wenden.loc['2019-10-03-14:15','Abfluss'] = 1.37744
# Kontrolle verwendete Daten für Scatterplots:
Wenden_sel = Wenden[Wenden.d18O >= -100]

# Datengrundlage
Wenden_sel.describe()
# Insgesamt 26 Proben
# -> keine Schneeschmelze
Wenden_sel[Wenden_sel.Saison =='Gletscherschmelze'].describe()
# 4 Gletscherschmelze
Wenden_sel[Wenden_sel.Saison =='Regen'].describe()
# 18 Regen
Wenden_sel[Wenden_sel.Saison =='Winterlicher Basisabfluss'].describe()
# 4 Winterlicher Basisabfluss

###############################################################
# Scatterplots
###############################################################

#############
# d18O - d2H
#############
fig, ax1 = plt.subplots()
# LMWL
x = [-15,-7]
#8.07*-7+12.83
y = [-108.22, -43.66]
g =sns.scatterplot(x="d18O", y="d2H",data=Wenden, hue="Saison")
plt.plot(x,y, label= 'LMWL', color='black')
plt.ylabel('$δ^{2}H$ [‰]')
plt.xlabel('$δ^{18}O$ [‰]')
plt.legend()
# removing top and right borders
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
# adds major gridlines
ax1.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.8)

##############
# d18O - Abfluss
###############
fig, ax1 = plt.subplots()
g =sns.scatterplot(x="d18O", y="Abfluss",data=Wenden, hue="Saison")
plt.ylabel('Abfluss [m$^{3}$/s]')
plt.xlabel('$δ^{18}O$ [‰]')
ax1.legend(loc='lower left', bbox_to_anchor=(1.02, 0.5), ncol=1)
# removing top and right borders
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
# adds major gridlines
ax1.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.8)

############
# d18O - EC
############
fig, ax1 = plt.subplots()
g =sns.scatterplot(x="d18O", y="Cond",data=Wenden, hue="Saison")
plt.ylabel('Elektrische Leitfähigkeit [μS/cm]')
plt.xlabel('$δ^{18}O$ [‰]')
plt.legend()
ax1.legend(loc='lower left', bbox_to_anchor=(1.02, 0.5), ncol=1)
# removing top and right borders
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
# adds major gridlines
ax1.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.8)

#############
# EC - Abfluss
#############
# kalibierter Bereich wird bei weitem überstiegen, dadurch sind viel zu hohe Abflüsse vorhanden
# in den Tagesmittelwerten ist dieses Problem nicht vorhanden.
# realistische Obergrenze für Abfluss setzten:
Wenden.loc[Wenden.Abfluss >= 11, 'Abfluss'] = 11
# Achtung: das verändert die Abflussdaten, für stat.Zusammenhang mit original Daten rechnen

fig, ax1 = plt.subplots()
g =sns.scatterplot(x="Abfluss", y="Cond",data=Wenden, hue="Saison",alpha=0.5)
plt.ylabel('Elektrische Leitfähigkeit [μS/cm]')
plt.xlabel('Abfluss [m$^{3}$/s]')
plt.legend()
ax1.legend(loc='lower left', bbox_to_anchor=(1.02, 0.5), ncol=1)
# removing top and right borders
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
# adds major gridlines
ax1.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.8)

###############################################################
# statistischer Zusammenhang
###############################################################
############
# Rangkorrelation nach Spearman:
# aus Feiks M. (2019) Statistische Berechnungen mit Python. In: Empirische Sozialforschung mit Python. Springer VS, Wiesbaden
############
### Correlation def: Korrelations-Funktion
def correlation(x, y):
    n = len(x)
    # Mittelwerte berechnen
    x_mn = sum(x) / n
    y_mn = sum(y) / n
    # Varianzen berechnen
    var_x = (1 / (n-1)) * sum(map(lambda xi: (xi - x_mn) ** 2 , x))
    var_y = (1 / (n-1)) * sum(map(lambda yi: (yi - y_mn) ** 2 , y))
    # Standardabweichungen berechenen
    std_x, std_y = math.sqrt(var_x), math.sqrt(var_y)
    # Kovarianz berechnen
    xy_var = map(lambda xi, yi: (xi - x_mn) * (yi - y_mn), x, y)
    cov = (1 / (n-1)) * sum(xy_var)
    # Korrelationskoeffizient nach Pearson
    r = cov / (std_x * std_y)
    return float(f"{r:.3f}")

### ranking def: Ränge erstellen
def ranking(array):

    counts = Counter(array)
    array_sorted = sorted(set(array))

    rank = 1
    rankings = {}
    for num in array_sorted:
        count = counts.get(num)
        if count == 1:
            rankings[num] = rank
            rank += 1
        else:
            rankings[num] = sum(range(rank, rank+count)) / count
            rank += count
    return [float(rankings.get(num)) for num in array]

###############
# Zusammenhang EC-Abfluss
###############
######
#Spearman Rangkorrelation berechnen (Code Spearman Rangk nötig):
x = Wenden.Cond.dropna()
y = Wenden.Abfluss.dropna()
x_rank = ranking(x)
y_rank = ranking(y)
correlation(x_rank,y_rank)
# Faustregel: besteht ein echter Zusammnehang?
# Wurzel n-1 mal Sp grösser als 2
x.describe()
math.sqrt(33968)*0.095
# =17, ist grösser als 2
# d.h. mit 95% Sicherheit besteht ein echter Zusammenhang

######
# Gletscherabfluss:
#Spearman Rangkorrelation berechnen (Code Spearman Rangk nötig):
x = Wenden[Wenden.Saison =='Gletscherschmelze'].Cond.dropna()
y = Wenden[Wenden.Saison =='Gletscherschmelze'].Abfluss.dropna()
x_rank = ranking(x)
y_rank = ranking(y)
correlation(x_rank,y_rank)
# Faustregel: besteht ein echter Zusammnehang?
# Wurzel n-1 mal Sp grösser als 2
x.describe()
# Anzahl x = 8929
math.sqrt(6531)*0.498
# =40, ist grösser als 2
# d.h. mit 95% Sicherheit besteht ein echter Zusammenhang

######
# Regen:
#Spearman Rangkorrelation berechnen (Code Spearman Rangk nötig):
x = Wenden[Wenden.Saison =='Regen'].Cond.dropna()
y = Wenden[Wenden.Saison =='Regen'].Abfluss.dropna()
x_rank = ranking(x)
y_rank = ranking(y)
correlation(x_rank,y_rank)
# Faustregel: besteht ein echter Zusammnehang?
# Wurzel n-1 mal Sp grösser als 2
x.describe()
math.sqrt(13103)*0.342
# =40, ist grösser als 2
# d.h. mit 95% Sicherheit besteht ein echter Zusammenhang

######
# winterlicher Basisabfluss:
#Spearman Rangkorrelation berechnen (Code Spearman Rangk nötig):
x = Wenden[Wenden.Saison =='Winterlicher Basisabfluss'].Cond.dropna()
y = Wenden[Wenden.Saison =='Winterlicher Basisabfluss'].Abfluss.dropna()
x_rank = ranking(x)
y_rank = ranking(y)
correlation(x_rank,y_rank)
# Faustregel: besteht ein echter Zusammnehang?
# Wurzel n-1 mal Sp grösser als 2
x.describe()
math.sqrt(14332)*0.135

###############
# Zusammenhang Isotopen - EC
###############
# gesamte Periode:
# nur diese Zeitpunkte an denen Isotopendaten vorhanden sind
Wenden_iso = Wenden[Wenden.d18O >= -100]
# am 26.02 habe ich keinen EC Wert
x = Wenden_iso.d18O.dropna()
y = Wenden_iso.Cond.dropna()
x_rank = ranking(x)
y_rank = ranking(y)
correlation(x_rank,y_rank)
# Faustregel: besteht ein echter Zusammnehang?
# Wurzel n-1 mal Sp grösser als 2
x.describe()
math.sqrt(49)*0.444

######
# Pro Perioden:
x = Wenden_iso[Wenden_iso.Saison =='Gletscherschmelze'].d18O.dropna()
y = Wenden_iso[Wenden_iso.Saison =='Gletscherschmelze'].Cond.dropna()
x_rank = ranking(x)
y_rank = ranking(y)
correlation(x_rank,y_rank)
# Faustregel: besteht ein echter Zusammnehang?
# Wurzel n-1 mal Sp grösser als 2
x.describe()
math.sqrt(26)*0.521
# =x, ist grösser als 2
# d.h. mit 95% Sicherheit besteht ein echter Zusammenhang

###############
# Zusammenhang Isotopen - Abfluss
###############
# gesamte Periode:
x = Wenden_iso.d18O
y = Wenden_iso.Abfluss.dropna()
x_rank = ranking(x)
y_rank = ranking(y)
correlation(x_rank,y_rank)
# Faustregel: besteht ein echter Zusammnehang?
# Wurzel n-1 mal Sp grösser als 2
x.describe()
math.sqrt(25)*0.488

######
# Pro Perioden:
x = Wenden_iso[Wenden_iso.Saison =='Gletscherschmelze'].d18O.dropna()
y = Wenden_iso[Wenden_iso.Saison =='Gletscherschmelze'].Abfluss.dropna()
x_rank = ranking(x)
y_rank = ranking(y)
correlation(x_rank,y_rank)
# Faustregel: besteht ein echter Zusammnehang?
# Wurzel n-1 mal Sp grösser als 2
x.describe()
math.sqrt(3)*0.8
# =x, ist grösser als 2
# d.h. mit 95% Sicherheit besteht ein echter Zusammenhang

###############################################################
# Boxplots nach Saisons
###############################################################

##########
# EC
########
ax = plt.gca()
sns.set(style="ticks", rc={"lines.linewidth": 0.8})
flierprops = dict(marker="x",markersize=1)
# erste Farbe überspringen, da Schneeschmelze fehlt
my_pal = {"Gletscherschmelze": "C1", "Regen": "C2", "Winterlicher Basisabfluss":"C3"}
ax = sns.boxplot(x='Saison', y= 'Cond', data=Wenden,flierprops=flierprops, palette=my_pal)
for patch in ax.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, .5))
plt.title( '' )
plt.suptitle('')
plt.ylabel('Elektrische Leitfähigkeit [μS/cm]')
# removing top and right borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# adds major gridlines
ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.8)

##########
# Abfluss
########
Wenden.loc[Wenden.Abfluss >= 11, 'Abfluss'] = 11

ax = plt.gca()
sns.set(style="ticks", rc={"lines.linewidth": 0.8})
# erste Farbe überspringen, da Schneeschmelze fehlt
my_pal = {"Gletscherschmelze": "C1", "Regen": "C2", "Winterlicher Basisabfluss":"C3"}
flierprops = dict(marker="x",markersize=1)
ax = sns.boxplot(x='Saison', y= 'Abfluss', data=Wenden,flierprops=flierprops, palette=my_pal)
for patch in ax.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, .5))
plt.title( '' )
plt.suptitle('')
plt.ylabel('Abfluss [m$^{3}$/s]')
# removing top and right borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# adds major gridlines
ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.8)

##########
# Isotopen
########
ax = plt.gca()
sns.set(style="ticks", rc={"lines.linewidth": 0.8})
# erste Farbe überspringen, da Schneeschmelze fehlt
my_pal = {"Gletscherschmelze": "C1", "Regen": "C2", "Winterlicher Basisabfluss":"C3"}
flierprops = dict(marker="x",markersize=1)
ax = sns.boxplot(x='Saison', y= 'd18O', data=Wenden,flierprops=flierprops, palette=my_pal)
for patch in ax.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, .5))
plt.title( '' )
plt.suptitle('')
plt.ylabel('$δ^{18}O$ [‰]')
# removing top and right borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# adds major gridlines
ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.8)

########################################################################################################################
# prozentuale Abflussseparation
########################################################################################################################
###############################################################
# Daten Import: Isotopensignale
################
#import Schneeproben
data= r'"Pfad einfügen" Schneeproben.txt'
dateparse = lambda x: pd.datetime.strptime(x, '%d.%m.%Y')
headers =['Datum','EZG','Höhe','Beschriftung','d18O','d2H' ]

Schnee = pd.read_csv(data, parse_dates= ['Datum'],date_parser=dateparse,sep=';', skiprows=range(0,1),
                   names=headers, index_col=0, usecols=[0,1,2,3,4,5])

# Regendaten (eigene)
data= r'"Pfad einfügen" Regenproben.txt'
dateparse = lambda x: pd.datetime.strptime(x, '%d.%m.%Y')
headers = ['Datum_(Leerung)','Zeitraum','Anzahl_Tage','EZG','Sammler','Beschriftung','d18O','d2H']

Regen = pd.read_csv(data, parse_dates= ['Datum_(Leerung)'],date_parser=dateparse,sep=';', skiprows=range(0,1),
                   names=headers, index_col=0, usecols=[0,1,2,3,4,5,6,7])

###############
# Schnee
###############
# Darstellung Monatsmittel:
# fehlende Daten erschaffen
ndays = 265
start = datetime(2019, 6, 19)
dates = [start + delta(days=x) for x in range(0, ndays)]
# als Dataframe konvertieren
f = pd.DataFrame(dates)
# neue Spalte einfügen
f.insert(loc=1, column='d18O', value= 0)
# Datum als Index setzen
f.set_index(0, inplace= True)
# Monatsmittel einsezten:
#Schnee.d18O.loc['2020-03'].resample('M').mean()
# Im Juni hat es keine Proben, Mittelwert der ersten Juli Hälfte nehmen: -11.405000000000001
#(-12.06 + -10.75)/2
f.loc['2019-06-19':'2019-06-30'] = -11.405000000000001
f.loc['2019-07-01':'2019-07-31'] = -10.86
f.loc['2019-08-01':'2019-08-31'] = -10.51
f.loc['2019-09-01':'2019-09-30'] =  -9.95
f.loc['2019-10-01':'2019-10-31'] =   -9.905
f.loc['2019-11-01':'2019-11-30'] =  -19.253333
f.loc['2019-12-01':'2019-12-31'] =  -17.495
f.loc['2020-01-01':'2020-01-31'] =  -15.515
# Im Februar ist kein Wert vorhanden; Mitelwert von Januar und März genommen:
#(-15.515+ -14.7)/2
f.loc['2020-02-01':'2020-02-29'] =  -15.1075
f.loc['2020-03-01':'2020-03-31'] =  -14.7

###################
# Regen
###################
################
# Sammler: S
################
Sammler_S = Regen.loc[Regen.Sammler == 'S']
# fehlende Daten erschaffen
ndays = 123
start = datetime(2019, 6, 18)
dates = [start + delta(days=x) for x in range(0, ndays)]
# als Dataframe Sonvertieren
S = pd.DataFrame(dates)
# neue Spalte einfügen
S.insert(loc=1, column='d18O', value= 0)
# Datum als Index setzen
S.set_index(0, inplace= True)
# richtige Daten einsetzen
S.loc['2019-06-18 ':'2019-07-04', 'd18O'] = Sammler_S.loc['2019-07-04'].d18O
S.loc['2019-07-04 ':'2019-07-17', 'd18O'] = Sammler_S.loc['2019-07-17'].d18O
S.loc['2019-07-17 ':'2019-08-08', 'd18O'] = Sammler_S.loc['2019-08-08'].d18O
S.loc['2019-08-08 ':'2019-08-20', 'd18O'] = Sammler_S.loc['2019-08-20'].d18O
S.loc['2019-08-20 ':'2019-08-29', 'd18O'] = Sammler_S.loc['2019-08-29'].d18O
S.loc['2019-08-29 ':'2019-09-12', 'd18O'] = Sammler_S.loc['2019-09-12'].d18O
S.loc['2019-09-12 ':'2019-10-03', 'd18O'] = Sammler_S.loc['2019-10-03'].d18O
S.loc['2019-10-03 ':'2019-10-18', 'd18O'] = Sammler_S.loc['2019-10-18'].d18O

################
# Sammler: S mit d2H
################
# neue Spalte einfügen
S.insert(loc=1, column='d2H', value= 0)
# richtige Daten einsetzen
S.loc['2019-06-18 ':'2019-07-04', 'd2H'] = Sammler_S.loc['2019-07-04'].d2H
S.loc['2019-07-04 ':'2019-07-17', 'd2H'] = Sammler_S.loc['2019-07-17'].d2H
S.loc['2019-07-17 ':'2019-08-08', 'd2H'] = Sammler_S.loc['2019-08-08'].d2H
S.loc['2019-08-08 ':'2019-08-20', 'd2H'] = Sammler_S.loc['2019-08-20'].d2H
S.loc['2019-08-20 ':'2019-08-29', 'd2H'] = Sammler_S.loc['2019-08-29'].d2H
S.loc['2019-08-29 ':'2019-09-12', 'd2H'] = Sammler_S.loc['2019-09-12'].d2H
S.loc['2019-09-12 ':'2019-10-03', 'd2H'] = Sammler_S.loc['2019-10-03'].d2H
S.loc['2019-10-03 ':'2019-10-18', 'd2H'] = Sammler_S.loc['2019-10-18'].d2H

################
# Anwendung auf EZG
################
# Stein
S_HK = S.apply(lambda x : x -1.485507)
#Gigli
G_HK=S.apply(lambda x : x  -1.08452)
#Wenden
W_HK=S.apply(lambda x : x  -1.21295)

############
# Eis
#############
# Eis hinzufügen
f.insert(loc=1, column='Eis', value= -13.43)

###############################################################################
# Mischrechnung Wendenwasser
##############################################################################

#############
# Gletscherschmelze
#############
AG = Wenden_sel[Wenden_sel.Saison =='Gletscherschmelze']
# Minute und Sekunden aus Index entfernen:
AG.index = pd.to_datetime(AG.index, format = '%Y-%m-%d %H:%M').strftime('%Y-%m-%d')
# speichern der Berechnungen:
# Anteil Gletscher: G/ Anteil Schnee: S/ Anteil Regen:R
AG.G = np.nan
AG.R = np.nan
# Mischrechnung
# AG = ( δ18O Abfluss Tag x  - δ18ORegen ) / (δ18OGletscher - δ18ORegen)
# AG = Anteil Gletscherschmelze
for index, row in AG.iterrows():
    AG.loc[index,'G']=(((AG.d18O.loc[index]-W_HK.d18O.loc[index])/(f.Eis.loc[index]-W_HK.d18O.loc[index]))*100)
    AG.loc[index, 'R'] = (100 - (((AG.d18O.loc[index] - W_HK.d18O.loc[index]) / (f.Eis.loc[index] - W_HK.d18O.loc[index])) * 100))
#############
# Plot Gletscherschmelze
#############
AG = AG.apply(pd.to_numeric, errors='coerce')
AG.index = pd.to_datetime(AG.index)
# Plot
fig, ax1 = plt.subplots()
plt.plot(AG.G, marker='x', linestyle= 'dotted',color='black')
plt.plot(AG.G, marker='x',linestyle='',color='black', label='Probennahme')
plt.ylabel('[%]')
plt.legend()
ax1.xaxis.set_major_formatter(DateFormatter('%d.%m\n%Y'))
# removing top and right borders
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
# adds major gridlines
ax1.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.8)
# um die über 100% darzustellen:
ax1.fill_between(AG.index,100+AG.R-100, interpolate=True, color='green', alpha=0.2)
ax1.fill_between(AG.index, 100, interpolate=True, color='white')
ax1.fill_between(AG.index,100, interpolate=True, color='green', alpha=0.2, label='Anteil Regen')
ax1.fill_between(AG.index, AG.G, interpolate=True, color='white')
ax1.fill_between(AG.index, AG.G, interpolate=True, color='blue', alpha=0.2, label='Anteil Gletscherschmelze')
ax1.legend(loc='lower left')
x = AG.describe()

#############
# Regen Saison
#############
AR = Wenden_sel[Wenden_sel.Saison =='Regen']
# Achtung doppelte Werte an einem TAR löschen, jeweil manueller Wert behalten
AR.drop(AR.index[3], inplace = True)
# Minute und Sekunden aus Index entfernen:
AR.index = pd.to_datetime(AR.index, format = '%Y-%m-%d %H:%M').strftime('%Y-%m-%d')
# speichern der Berechnungen:
# Anteil Gletscher: G/ Anteil Schnee: S/ Anteil Regen:R
AR.S = np.nan
AR.R = np.nan
# Mischrechnung
# AG = ( δ18O Abfluss Tag x  - δ18ORegen ) / (δ18OSchnee - δ18ORegen)
# AG = Anteil Schneeschmelze
for index, row in AR.iterrows():
    AR.loc[index,'S']=(((AR.d18O.loc[index]-W_HK.d18O.loc[index])/(f.d18O.loc[index]-W_HK.d18O.loc[index]))*100)
    AR.loc[index, 'R'] = (100 - (((AR.d18O.loc[index] - W_HK.d18O.loc[index]) / (f.d18O.loc[index] - W_HK.d18O.loc[index])) * 100))
# problem mit Regen: ist nur bis 18.10 definiert
#############
# Plot Regen Saison
#############
AR = AR.apply(pd.to_numeric, errors='coerce')
AR.index = pd.to_datetime(AR.index)
# Nach 18.10 löschen, da hat es keine Berechnungen mehr da Isotopensignal Regen fehlt
# 09.03 nicht darstellen, extrem unrealistisch aufgrund anderem Isotopensignal Regen
AR = AR.loc['2019-09-16 ':'2019-10-18 ']

AZ = AR.loc['2019-10-03':]
AZ.describe()
# Plot
fig, ax1 = plt.subplots()
plt.plot(AR.R, marker='x', linestyle= 'dotted',color='black')
plt.plot(AR.R, marker='x',linestyle='',color='black', label='Probennahme')
plt.ylabel('[%]')
plt.legend()
ax1.xaxis.set_major_formatter(DateFormatter('%d.%m\n%Y'))
# removing top and right borders
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
# adds major gridlines
ax1.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.8)
# um die über 100% darzustellen:
ax1.fill_between(AR.index,100+AR.S-100, interpolate=True, color='red', alpha=0.2)
ax1.fill_between(AR.index, 100, interpolate=True, color='white')
ax1.fill_between(AR.index,100, interpolate=True, color='red', alpha=0.2, label='Anteil Schneeschmelze')
ax1.fill_between(AR.index, AR.R, interpolate=True, color='white')
ax1.fill_between(AR.index, AR.R, interpolate=True, color='green', alpha=0.2, label='Anteil Regen')
ax1.legend(loc='upper right')

y = AR.describe()
#############
# winterlicher Basisabfluss
#############
AW = Wenden_sel[Wenden_sel.Saison =='Winterlicher Basisabfluss']
# Minute und Sekunden aus Index entfernen:
AW.index = pd.to_datetime(AW.index, format = '%Y-%m-%d %H:%M').strftime('%Y-%m-%d')
# speichern der Berechnungen:
# Anteil Gletscher: G/ Anteil Schnee: S/ Anteil Regen:R
AW.S = np.nan
AW.G = np.nan
# Mischrechnung
# AG = ( δ18O Abfluss Tag x  - δ18OEis ) / (δ18OSchnee - δ18OEis)
# AG = Anteil Schneeschmelze
for index, row in AW.iterrows():
    AW.loc[index,'S']=(((AW.d18O.loc[index]-f.Eis.loc[index])/(f.d18O.loc[index]-f.Eis.loc[index]))*100)
    AW.loc[index, 'G'] = (100 - (((AW.d18O.loc[index] - f.Eis.loc[index]) / (f.d18O.loc[index] - f.Eis.loc[index])) * 100))

############################################################
# Plot Daten pro Saison
############################################################
############
# Gletscherschmelze Plot Daten
############
# Meteodaten auf länge Gletscherschmelze anpassen
Meteo_GS = Meteo.loc['2019-07-31':'2019-08-23']
fig, axs = plt.subplots(2, sharex=True, sharey=False)
fig.set_size_inches(60,30)
#axs[0].plot(Meteo_GS.Gadmen_Prec_d, linestyle='',marker='x')
axs[0].bar(Meteo_GS.index, Meteo_GS.Gadmen_Prec_d, 0.2,color='#32733c',label= 'Niederschlag')
axs[0].set_ylabel('[mm/Tag]')
axs[0].legend(loc='center left')
axs[0].grid(True)
# Regen von oben
axs[0].invert_yaxis()
axs[1].plot(Wenden.Abfluss.loc['2019-07-31':'2019-08-23'].resample('D').mean(), label='Abfluss',color= '#ff8000')
axs[1].set_ylabel('[$m^{3}$/s]')
axs[1].legend(loc='center left')
axs[1].grid(True)
#axs[2].twinx()
#axs[2].plot(Wenden.Cond.loc['2019-07-31':'2019-08-23'].resample('D').mean()
#            ,color= '#ff0000', label='Elektrische Leitfähigkeit')
#axs[2].set_ylabel('[μS/cm]')
#axs[2].legend(loc='center left')
#axs[2].grid(True)
axs[0].xaxis.set_major_formatter(DateFormatter('%d.%m\n%Y'))

#############
# Saison Regen Plot Daten
##############
# import Schneedecke
data= r'"Pfad einfügen" Schneedecke.txt'
dateparse = lambda x: pd.datetime.strptime(x, '%d.%m.%Y')
headers = ['Datum', 'Skala', 'Bemerkungen']
Sd = pd.read_csv(data, parse_dates= ['Datum'],date_parser=dateparse,sep=';', skiprows=range(0,1),
                   names=headers, index_col=0, usecols=[0,1,2])
# plot Regen Saison
Meteo_RS = Meteo.loc['2019-09-16':'2019-10-18']
fig, axs = plt.subplots(3, sharex=True, sharey=False)
fig.set_size_inches(60,30)
#axs[0].plot(Meteo_RS.Gadmen_Prec_d, linestyle='',marker='x')
axs[0].bar(Meteo_RS.index, Meteo_RS.Gadmen_Prec_d, 0.2,color='#32733c',label= 'Niederschlag')
axs[0].set_ylabel('[mm/Tag]')
axs[0].legend(loc='center left')
axs[0].grid(True)
# Regen von oben
axs[0].invert_yaxis()
axs[1].plot(Wenden.Abfluss.loc['2019-09-16':'2019-10-18'].resample('D').mean(), label='Abfluss',color= '#ff8000')
axs[1].set_ylabel('[$m^{3}$/s]')
axs[1].legend(loc='center left')
axs[1].grid(True)
#axs[2].plot(Wenden.Cond.loc['2019-09-16':'2019-10-18'].resample('D').mean()
#            ,color= '#ff0000', label='Elektrische Leitfähigkeit')

#axs[2].set_ylabel('[μS/cm]')
#axs[2].legend(loc='center left')
#axs[2].grid(True)

axs[2].plot(Sd.Skala.loc['2019-09-16':'2019-10-18'], label='Schneedecke', linestyle='',marker='x')
axs[2].set_ylabel('[1-5]')
axs[2].legend(loc='center left')
axs[2].grid(True)
axs[0].xaxis.set_major_formatter(DateFormatter('%d.%m\n%Y'))