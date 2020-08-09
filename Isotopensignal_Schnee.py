#########################################################################################################################
# Import
#########################################################################################################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import seaborn as sns
from sklearn.metrics import r2_score
from matplotlib.dates import DateFormatter
from datetime import datetime, timedelta as delta
################
# Daten Import
################
#import Schneeproben
data= r'"Pfad einfügen" Schneeproben.txt'
dateparse = lambda x: pd.datetime.strptime(x, '%d.%m.%Y')
headers =['Datum','EZG','Höhe','Beschriftung','d18O','d2H' ]
Schnee = pd.read_csv(data, parse_dates= ['Datum'],date_parser=dateparse,sep=';', skiprows=range(0,1),
                   names=headers, index_col=0, usecols=[0,1,2,3,4,5])

# import meteo data
dateparse = lambda x: pd.datetime.strptime(x,'%Y-%m-%d %H:%M:%S')
Meteo = pd.read_csv(r'"Pfad einfügen" Meteo.csv',
                    index_col=0, parse_dates=[0],date_parser=dateparse)

#########################################################################################################################
# Datengrundlage
#########################################################################################################################
Schnee.d18O.describe()
# Ablationsperiode
Schnee.d18O.loc['2019-07':'2019-10'].describe()
Schnee.loc['2019-07':'2019-10']
# Akkumulaitonsperiode
Schnee.d18O.loc['2019-11':'2020-03'].describe()
Schnee.loc['2019-11':'2020-03']

################
# Plot sämtliche Daten
################

# Schneedecke und d18o über Zeit
fig, ax1 = plt.subplots()
ax1.plot(Meteo.Gschlet_Schnee_d.resample('D').mean(), color='#326ba8', label='Schneehöhe')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('[cm]', color='#326ba8')
ax1.tick_params('y', colors='#326ba8')
#second x-axis
ax2 = ax1.twinx()
ax2.plot(Schnee.d18O, color= '#32733c', label='$δ^{18}O$', marker='.')
ax2.set_ylabel('[‰]', color='#32733c')
ax2.tick_params('y', colors='#32733c')
# ask matplotlib for the plotted objects and their labels
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)
# removing top and right borders
ax1.spines['top'].set_visible(False)
# adds major gridlines
ax1.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.8)
ax1.xaxis.set_major_formatter(DateFormatter('%d.%m\n%Y'))

# Höhe und d18o über Zeit
fig, ax1 = plt.subplots()
ax1.plot(Schnee.Höhe, color='#326ba8', label='Höhe', marker='.')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('[cm]', color='#326ba8')
ax1.tick_params('y', colors='#326ba8')
#Legende
plt.legend()
#second x-axis
ax2 = ax1.twinx()
ax2.plot(Schnee.d18O, color= '#32733c', label='$δ^{18}O$', marker='.')
ax2.set_ylabel('[‰]', color='#32733c')
ax2.tick_params('y', colors='#32733c')
plt.legend()
# removing top and right borders
ax1.spines['top'].set_visible(False)
# adds major gridlines
ax1.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.8)
ax1.xaxis.set_major_formatter(DateFormatter('%d.%m\n%Y'))

#########################################################################################################################
# Verdunstungsgerade Ablationsperiode eklusiv 03.10
#########################################################################################################################
# passen die Werte im Sommer auf die LMWL
# Wenn ja Verändert der Regen das Signal
# Wenn nein sind es Verdunstungs und Verlagerungseffekte -> das ist der Fall

# die Proben vom 10.03 weisen schwerere d2H Werte auf
# dieser Schnee war Neuschnee und es hatte nur sehr wenig
# er lag ev. schon etwas in der Sonne; war "angeschmolzen"
# ev. auch Schneeregen?..
# für Regressionsgerade ohne diese beiden Proben

# Regressionsgerade Juni-Sept
fig, ax1 = plt.subplots()
x = Schnee.d18O
y = Schnee.d2H
plt.plot(x.loc['2019-07':'2019-09'], y.loc['2019-07':'2019-09'], 'o', label='Schneeproben Ablationsperiode')
m, b = np.polyfit(x.loc['2019-07':'2019-09'], y.loc['2019-07':'2019-09'], 1)
print (m)
print (b)
plt.plot(x.loc['2019-07':'2019-09'], m*x.loc['2019-07':'2019-09'] + b
         , label= 'RG: $δ^{2}H$ = 7.48 · $δ^{18}O$ + 2.79')
plt.ylabel('$δ^{2}H$ [‰]')
plt.xlabel('$δ^{18}O$ [‰]')
# LMWL
x = [-12.2,-9.4]
8.07*-9.4+12.83
y = [-85.624, -63.028000000000006]
plt.plot(x,y, color='black',label = 'LMWL: $δ^{2}H$ = 8.07 · $δ^{18}O$ + 12.8')
plt.legend()
# removing top and right borders
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
# adds major gridlines
ax1.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.8)

# R^2 berechnen: 0.9751899301681923
y_pred = 7.481518755718188 * x + 2.789278031920078
print(r2_score(y, y_pred))


#########################################################################################################################
# Verdunstungsgerade Ablationsperiode inklusiv 03.10
#########################################################################################################################
# passen die Werte im Sommer auf die LMWL
# Wenn ja Verändert der Regen das Signal
# Wenn nein sind es Verdunstungs und Verlagerungseffekte -> das ist der Fall

# die Proben vom 10.03 weisen schwerere d2H Werte auf
# dieser Schnee war Neuschnee und es hatte nur sehr wenig
# er lag ev. schon etwas in der Sonne; war "angeschmolzen"
# ev. auch Schneeregen?..

# Regressionsgerade Juni-Sept
fig, ax1 = plt.subplots()
x = Schnee.d18O
y = Schnee.d2H
plt.plot(x.loc['2019-07':'2019-10'], y.loc['2019-07':'2019-10'], 'o', label='Schneeproben Ablationsperiode')
m, b = np.polyfit(x.loc['2019-07':'2019-10'], y.loc['2019-07':'2019-10'], 1)
print (m)
print (b)
plt.plot(x.loc['2019-07':'2019-10'], m*x.loc['2019-07':'2019-10'] + b
         , label= 'RG: $δ^{2}H$ = 7.48 · $δ^{18}O$ + 2.79')
plt.ylabel('$δ^{2}H$ [‰]')
plt.xlabel('$δ^{18}O$ [‰]')
# LMWL
x = [-12.2,-9.4]
8.07*-9.4+12.83
y = [-85.624, -63.028000000000006]
plt.plot(x,y, color='black',label = 'LMWL: $δ^{2}H$ = 8.07 · $δ^{18}O$ + 12.8')
plt.legend()
# removing top and right borders
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
# adds major gridlines
ax1.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.8)

# R^2 berechnen: 0.9751899301681923
y_pred = 7.481518755718188 * x + 2.789278031920078
print(r2_score(y, y_pred))

#########################################################################################################################
# Gerade  Akkumulationsperiode
#########################################################################################################################
# passen die Werte im Winter auf die LMWL
# Wenn ja Verändert der Regen das Signal
# Wenn nein sind es Verdunstungs und Verlagerungseffekte -> das ist der Fall

# Regressionsgerade Nov-März
fig, ax1 = plt.subplots()
x = Schnee.d18O
y = Schnee.d2H
plt.plot(x.loc['2019-11':'2020-03'], y.loc['2019-11':'2020-03'], 'o', label='Schneeproben Akkumulationsperiode')
m, b = np.polyfit(x.loc['2019-11':'2020-03'], y.loc['2019-11':'2020-03'], 1)
print (m)
print (b)
plt.plot(x.loc['2019-11':'2020-03'], m*x.loc['2019-11':'2020-03'] + b
         , label= 'RG: $δ^{2}H$ = 7.46 · $δ^{18}O$ + 1.77')
plt.ylabel('$δ^{2}H$ [‰]')
plt.xlabel('$δ^{18}O$ [‰]')
# LMWL
# LMWL
x = [-20,-14.3]
#8.07*-14.3+12.83
y = [-148.57, -102.57100000000001]
plt.plot(x,y, color='black',label = 'LMWL: $δ^{2}H$ = 8.07 · $δ^{18}O$ + 12.8')
plt.legend()
# removing top and right borders
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
# adds major gridlines
ax1.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.8)

#########################################################################################################################
# Deuterium Excess
#########################################################################################################################
DE = Schnee.d2H - 8*Schnee.d18O
DE.plot()
DE.describe()
print(DE)
# ist bei Probe von 3.10 deutlich höher

#########################################################################################################################
# Monatsmittel -> Isotopensignal Schnee
#########################################################################################################################
Schnee.d18O.loc['2019-06':'2020-03'].describe()
# Darstellung Monatsmittel:
ndays = 264
start = datetime(2019, 6, 19)
dates = [start + delta(days=x) for x in range(0, ndays)]
# als Dataframe konvertieren
S = pd.DataFrame(dates)
# neue Spalte einfügen
S.insert(loc=1, column='d18O', value= 0)
# Datum als Index setzen
S.set_index(0, inplace= True)
# Monatsmittel einsezten:
#Schnee.d18O.loc['2020-03'].resample('M').mean()
# Im Juni hat es keine Proben, Mittelwert der ersten Juli Hälfte nehmen: -11.405000000000001
#(-12.06 + -10.75)/2
S.loc['2019-06-19':'2019-06-30'] = -11.405000000000001
S.loc['2019-07-01':'2019-07-31'] = -10.86
S.loc['2019-08-01':'2019-08-31'] = -10.51
S.loc['2019-09-01':'2019-09-30'] =  -9.95
S.loc['2019-10-01':'2019-10-31'] =   -9.905
S.loc['2019-11-01':'2019-11-30'] =  -19.253333
S.loc['2019-12-01':'2019-12-31'] =  -17.495
S.loc['2020-01-01':'2020-01-31'] =  -15.515
# Im Februar ist kein Wert vorhanden; Mitelwert von Januar und März genommen:
#(-15.515+ -14.7)/2
S.loc['2020-02-01':'2020-02-29'] =  -15.1075
S.loc['2020-03-01':'2020-03-31'] =  -14.7

################
# Plot Monatsmittel
###############
fig, ax1 = plt.subplots()
plt.plot(S, label='Isotopensignal Schnee', marker='_',linestyle='')
plt.ylabel('$δ^{18}O$ [‰]')
plt.legend(numpoints=15)
ax1.xaxis.set_major_formatter(DateFormatter('%d.%m.%Y'))
# removing top and right borders
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
# adds major gridlines
ax1.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.8)

################
# Zahlen Monatsmittel
###############
j = Schnee.loc['2019-07']
print(j)
j.describe()
a = Schnee.loc['2019-08']
print(a)
a.describe()
s = Schnee.loc['2019-09']
print(s)
s.describe()
o = Schnee.loc['2019-10']
print(o)
o.describe()
n = Schnee.loc['2019-11']
print(n)
n.describe()
d = Schnee.loc['2019-12']
print(d)
d.describe()
ja = Schnee.loc['2020-01']
print(ja)
ja.describe()
m = Schnee.loc['2020-03']
print(m)

# sd Juni gerechnet in excel 0.93
print(j)

# sd feb 0.61
print(ja)
print(m)