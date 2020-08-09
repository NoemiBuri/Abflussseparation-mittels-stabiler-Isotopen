################
# Import
################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import seaborn as sns
#specify the file location and name
data= r'"Pfad einfügen" Eisproben.txt'
dateparse = lambda x: pd.datetime.strptime(x, '%d.%m.%Y')
headers = ['Datum','Höhe','Beschriftung','d18O','d2H','Probenart']
Eis = pd.read_csv(data, parse_dates= ['Datum'],date_parser=dateparse,sep=';', skiprows=range(0,1),
                   names=headers, index_col=0, usecols=[0,1,2,3,4,5])

################
# Datengrundlage
################
# Mittelwert alle Proben: -12.13
Eis.d18O.describe()
# Mittelwert Eisproben:  -13.43
Eis.loc[Eis.Probenart=='Festeisproben'].describe()
# Mittelwert Schmelzwasserproben: -11.15
Eis.loc[Eis.Probenart=='Schmelzeisprobe'].describe()

################
# Plot sämtliche Daten
################
fig, ax1 = plt.subplots()
# LMWL
x = [-15,-7]
#8.07*-7+12.83
y = [-108.22, -43.66]
g =sns.scatterplot(x="d18O", y="d2H",data=Eis, hue="Probenart",  palette=['#f07a13','#271fc2'])
plt.plot(x,y, label= 'LMWL', color='black')
plt.ylabel('$δ^{2}H$ [‰]')
plt.xlabel('$δ^{18}O$ [‰]')
# Add labels to the plot
style = dict(size=10, color='gray')
#ax1.text(-12.83, -93.59, "21.08.2019",ha='left')
ax1.text(-12.73, -93.59, "21.08.2019",ha='left')
ax1.text(-13.43, -95.44, "21.08.2019",ha='right')
ax1.text(-13.92, -101.69, "21.08.2019",ha='left')
ax1.text( -8., -54.55, "21.08.2019",ha='left')
ax1.text( -11.68, -83.17, "21.08.2019",ha='left')
ax1.text( -12.12, -87.91, "29.08.2019",ha='left')
ax1.text( -12.36 , -84.91, "16.09.2019",ha='right')
plt.legend()
# removing top and right borders
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
# adds major gridlines
ax1.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.8)

################
# Plot Mittelwert und SD von Festeiproben
################
#Eis.loc[Eis.Probenart=='Festeisproben'].describe()
# X = Mitelwert/ xerr= Standardabweichung
x = -13.426667
y = -96.906667
xerr = 0.650410
yerr = 4.244506

fig, ax1 = plt.subplots()

ax1.errorbar(x, y,
            xerr=xerr,
            yerr=yerr,
            fmt='-o', label= 'Mittelwert und Standardabweichung Festeisproben')
# LMWL
x = [-14.1,-12.73]
8.07*-12.73+12.83
y = [-100.95700000000001, -89.90110000000001]
plt.plot(x,y, label= 'LMWL', color='black')

plt.ylabel('$δ^{2}H$ [‰]')
plt.xlabel('$δ^{18}O$ [‰]')
plt.legend()
# removing top and right borders
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
# adds major gridlines
ax1.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.8)