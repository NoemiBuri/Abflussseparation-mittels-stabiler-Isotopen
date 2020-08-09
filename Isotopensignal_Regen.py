################
# Import
################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import seaborn as sns
from sklearn.metrics import r2_score
from datetime import datetime, timedelta as delta
from matplotlib.dates import DateFormatter
# Regendaten (eigen erhoben)
data= r' "Pfad einfügen" Regenproben.txt'
dateparse = lambda x: pd.datetime.strptime(x, '%d.%m.%Y')
headers = ['Datum_(Leerung)','Zeitraum','Anzahl_Tage','EZG','Sammler','Beschriftung','d18O','d2H']
Regen = pd.read_csv(data, parse_dates= ['Datum_(Leerung)'],date_parser=dateparse,sep=';', skiprows=range(0,1),
                   names=headers, index_col=0, usecols=[0,1,2,3,4,5,6,7])

# Regendaten Bafu 2019
data= r'"Pfad einfügen" ISOT_2019.txt'
dateparse = lambda x: pd.datetime.strptime(x, '%d.%m.%Y')
headers = ['Datum','d18O','d2H','Sammler']
ISOT19 = pd.read_csv(data, parse_dates= ['Datum'],date_parser=dateparse,sep=';', skiprows=range(0,1),
                   names=headers, index_col=0, usecols=[0,1,2,3])

# Regendaten Bafu 1992-2014
data= r'"Pfad einfügen" ISOT_9214.txt'
headers = [ 'Guttannen_d2H','Guttannen_d18O','Grimsel_d2H','Grimsel_d18O']
ISOT92 = pd.read_csv(data,sep=';', skiprows=range(0,1),
                   names=headers, usecols=[0,1,2,3])
ISOT92.dropna(inplace=True)

################
# Höhe hinzufügen
################
Regen.insert(loc=4, column='Hoehe', value=Regen.Sammler)
Regen.Hoehe= Regen.Hoehe.replace('S', 1430)
Regen.Hoehe= Regen['Hoehe'].replace('Su1', 1842)
Regen.Hoehe= Regen['Hoehe'].replace('Su2', 2210)
Regen.Hoehe= Regen['Hoehe'].replace('W', 1542)

#########################################################################################################################
# Zeitliche Abfolge richtig darstellen
#########################################################################################################################

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
# Sammler: Su1
################
Sammler_Su1 = Regen.loc[Regen.Sammler == 'Su1']
# fehlende Daten erschaffen
ndays = 88
start = datetime(2019, 7, 8)
dates = [start + delta(days=x) for x in range(0, ndays)]
# als Dataframe Sonvertieren
Su1 = pd.DataFrame(dates)
# neue Spalte einfügen
Su1.insert(loc=1, column='d18O', value= 0)
# Datum als Index setzen
Su1.set_index(0, inplace= True)
# richtige Daten einsetzen
Su1.loc['2019-07-08 ':'2019-07-27', 'd18O'] = Sammler_Su1.loc['2019-07-27'].d18O
Su1.loc['2019-07-27 ':'2019-08-14', 'd18O'] = Sammler_Su1.loc['2019-08-14'].d18O
Su1.loc['2019-08-14 ':'2019-08-29', 'd18O'] = Sammler_Su1.loc['2019-08-29'].d18O
Su1.loc['2019-08-29 ':'2019-09-16', 'd18O'] = Sammler_Su1.loc['2019-09-16'].d18O
Su1.loc['2019-09-16 ':'2019-10-03', 'd18O'] = Sammler_Su1.loc['2019-10-03'].d18O

################
# Sammler: Su2
################
Sammler_Su2 = Regen.loc[Regen.Sammler == 'Su2']
# fehlende Daten erschaffen
ndays = 88
start = datetime(2019, 7, 8)
dates = [start + delta(days=x) for x in range(0, ndays)]
# als Dataframe Sonvertieren
Su2 = pd.DataFrame(dates)
# neue Spalte einfügen
Su2.insert(loc=1, column='d18O', value= 0)
# Datum als Index setzen
Su2.set_index(0, inplace= True)
# richtige Daten einsetzen
Su2.loc['2019-07-08 ':'2019-07-27', 'd18O'] = Sammler_Su2.loc['2019-07-27'].d18O
Su2.loc['2019-07-27 ':'2019-08-14', 'd18O'] = Sammler_Su2.loc['2019-08-14'].d18O
Su2.loc['2019-08-14 ':'2019-08-29', 'd18O'] = Sammler_Su2.loc['2019-08-29'].d18O
Su2.loc['2019-08-29 ':'2019-09-16', 'd18O'] = Sammler_Su2.loc['2019-09-16'].d18O
Su2.loc['2019-09-16 ':'2019-10-03', 'd18O'] = Sammler_Su2.loc['2019-10-03'].d18O

################
# Sammler: W
################
Sammler_W = Regen.loc[Regen.Sammler == 'W']
# fehlende Daten erWchaffen
ndays = 70
start = datetime(2019, 7, 31)
dates = [start + delta(days=x) for x in range(0, ndays)]
# alW Dataframe Wonvertieren
W = pd.DataFrame(dates)
# neue Wpalte einfügen
W.insert(loc=1, column='d18O', value= 0)
# Datum alW Index Wetzen
W.set_index(0, inplace= True)
# richtige Daten einWetzen
W.loc['2019-07-31 ':'2019-08-13', 'd18O'] = Sammler_W.loc['2019-08-13'].d18O
W.loc['2019-08-13 ':'2019-08-23', 'd18O'] = Sammler_W.loc['2019-08-23'].d18O
W.loc['2019-08-23 ':'2019-09-03', 'd18O'] = Sammler_W.loc['2019-09-03'].d18O
W.loc['2019-09-03 ':'2019-09-25', 'd18O'] = Sammler_W.loc['2019-09-25'].d18O
W.loc['2019-09-25 ':'2019-10-08', 'd18O'] = Sammler_W.loc['2019-10-08'].d18O

#########################################################################################################################
# Datenverteilung: Kapitel Regen, Resultate
#########################################################################################################################

# Boxplot pro Sammler
Regen.boxplot(column= ['d18O'], by='Sammler', meanline=True)
plt.title( '' )
plt.suptitle('')
plt.ylabel('[$δ^{18}O$]')
plt.show()

# Achtung das sind wieder die ungewichteten Werte
# Statistische Werte pro Sammler:
Regen.groupby('Sammler').d18O.describe()
# Statistische Werte über alle:
Regen.describe()

# gewichtete Mittelwerte pro Sammler
S.describe()
Su1.describe()
Su2.describe()
W.describe()

# diese Werte mit Bafu Werten vergleichen (langzeit)
# Mittelwert von den Monaten Juni bis Oktober berechnen

# Plot: Sammler über Zeit
fig, ax1 = plt.subplots()
plt.plot(S, label='Stein', marker='_',linestyle='')
plt.plot(Su1, label='Susten 1', marker='_',linestyle='')
plt.plot(Su2, label='Susten 2', marker='_',linestyle='')
plt.plot(W, label='Wenden', marker='_',linestyle='')
plt.ylabel('$δ^{18}O$ [‰]')
plt.legend(numpoints=15)
ax1.xaxis.set_major_formatter(DateFormatter('%d.%m.%Y'))
# removing top and right borders
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
# adds major gridlines
ax1.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.8)

#########################################################################################################################
# Höheneffekt
#########################################################################################################################
################
# Literatur:
################
# Höheneffekt aus der Literatur:
# Abnahme um etwa 0,2‰ im δ18O pro 100 m Höhenzunahme ist über mehrere Jahre gemittelt sehr stabil.
# Schotterer et. al, 2010 aus Abbildung 3
# Meiringen: 595müm -10.53 d18O
# Guttannen: 1055   -11.93
# Grimsel:  1980    -13.25

# Differenz M-Gut: 460  -1.4
#                  100  -0.3043
#-1.4/ 460 *100

# Differenz Gut-Gri: 925  -1.32
#                    100  -0.1427
#-1.32 / 925 * 100

# Durchschnitt der beiden: -0.2235
#-0.14270270270270272 + -0.30434782608695654
#-0.44705052878965923/2

# Für mich ist der Höheneffekt von Gut-Grim representativer als wenn Meiringen miteinbezogen wird
# daher besser mit -0.1427 rechnen

################
# eigene Daten
################
# Mittelwert pro Sammler
# gewichtetes Mittel ist genauer; besser (ist hier gewichtet, weil an jedem Tag ein Wert steht)
#Su1.mean()
#Su2.mean()
#W.mean()
# Problem: die Zeitreihen sind nicht gleich lang; nur dort wo sie sich überschneiden ist ein Vergleich möglich
# damit ein längerer Vergleich möglich ist; Wenden separat behandeln
# S kürzen für Su1 und Su2 Beginn 2019-07-08 Ende: 2019-10-03
S_kurz = S.loc['2019-07-08': '2019-10-03']
#S_kurz.mean()
# S kürzen für W '2019-07-31' : '2019-10-08'
S_kurz2 = S.loc['2019-07-31' : '2019-10-08']
#S_kurz2.mean()
#     müm   d18O
# S   = 1430 -8.197317
# S_kurz =    -8.395227
# S_kurz2=    -9.353857
# Su1 = 1842 -9.206477
# Su2 = 2210 -9.291818
# W   = 1542 -8.834429

# Differenz S-Su1
# 412  -0.8112499999999994
# 100  -0.19690533980582509
#-0.8112499999999994/ 412 * 100

# Differenz Su1-Su2
# 368    -0.08534099999999967
# 100     -0.023190489130434692
#-0.08534099999999967/ 368 *100

# Differenz S - W
# 112   0.5194279999999996
# 100   0.4637749999999996
#0.5194279999999996/112 *100

# Höheneffekt Wenden macht keinen Sinn:
# Wenn ich mir die Zeitserien unten anschaue, sieht es so aus, dass vor allem die beiden Perioden
# vom ca. 20.8-1.9 und 20.9-10.10 deutliche Unterschiede zu den anderen Sammler zeigen.
# Es könnte also auch sein, dass vor allem diese beiden Proben durch Fraktionierungsprozesse verfälscht wurden.
# Die anderen Proben zeigen einen ähnlichen saisonalen Trend wie bei den anderen Sammler.
# Um dies zu überprüfen würde ich den Höheneffekt S-W nochmals ohne diese beiden Proben bestimmen.

# Mittelwert ohne die gennanten Perioden berechnen:
W_P = W.loc[(W.index < '2019-08-23 ') | (W.index > '2019-09-03')]
W_P2 = W_P.loc[(W_P.index < '2019-09-25 ') | (W_P.index > '2019-10-08')]
W_P2.mean()

# Differenz S - W exkl. gennante Perioden
# 112   0.023175000000000168
# 100   0.020691964285714435
#0.023175000000000168/112 *100
#-9.330682 - -9.353857
# schwächt den Effekt ab, kaum noch vorhanden. aber in die falsche Richtung

# Mittlerer Höheneffekt S-Su1-Su2: -0.11004791446812989
#(-0.19690533980582509 + -0.023190489130434692)/2

# Differenz S- Su2
# 780     -1.0945009999999993
# 100    -0.14032064102564093
#-1.0945009999999993/780*100

################
# Schlussfolgerung
################
# Unterschied von S nach Su1 ist viel grösser als von Su1 nach Su2
# meine Datenmenge ist zu gering und die Zeitreihe ist zu kurz um anhand von meinen Daten einen
# Höheneffekt zu berechnen.
# siehe meine Einleitung: Es bedarf jedoch Daten einiger Jahre um einen Höheneffekt berechnen zu können (Leibundgut et al., 2009).

# Es ist jedoch ein Trend in die richtige Richtung ersichtlich.
# Ausserdem Unterstützt es die Wahl auf den Höheneffekt Gut-Grim zurückzugreiffen
# Gut-Grim: -0.1427/ S-Su1-Su2: -0.11004791446812989

# Räumliche und Zeitliche Variabilität:
# Im Mittelwert können die Unterschiede zwische S/ Su1/ Su2 durch den Höheneffekt erklärt werden
# damit kann durch die oben berechnete Höhenkorrektur der räumlichen Variabilität sorge getragen werden
# Die Zeitliche Variabilität wird beachtet, in dem jede Probe eine gewisse Zeitspanne abdeckt
# Wenden: ist beinahe gleich hoch wie S (geringer Höheneffekt ist vorhanden) aber in einem anderen Tal
# daher bei Wenden Sammler W verwenden, räumliche Variabilität miteinbeziehen
# Paper Penna 2019; räumliche Variabilität nicht unterschätzen; ich sehe bei mir aber Höheneffekt

################
# Anwendung auf EZG
################
# Regensammler S als Ausgangspunkt für Steinwasser und Giglibach nehmen;
# ist der qualitativere Regensammler und habe längere Datenreihe dazu
# bei Wenden Regensammler Wenden verwenden um der räumlichen Variabilität gerecht zu werden; doch nid

###Stein###
# Mittlere Höhe EZG Stein: 2 471
# Höhe Regensammler S: 1 430
# Höhendifferenz: 1041

# 100m  -0.1427
# 1041m  -1.485507
#-0.1427/100*1041

S_HK = S.apply(lambda x : x -1.485507)

###Gigli
# Mittlere Höhe EZG Gigli: 2190
# Höhe Regensammler S: 1 430
# Höhendifferenz:760m

# 100m  -0.1427
# 760m  -1.08452
#0.1427/100*760

G_HK=S.apply(lambda x : x  -1.08452)

###Wenden
# Mittlere Höhe EZG Wenden: 2280
# Höhe Regensammler S: 1 430
# Höhendifferenz:850m

# 100m  -0.1427
# 850  -1.21295
#-0.1427/100*850

W_HK=S.apply(lambda x : x  -1.21295)

#### Plot
fig, ax1 = plt.subplots()
plt.plot(S_HK, label='Steinwasser', marker='_',linestyle='',color= '#F61919')
#plt.plot(G_HK, label='Giglibach', marker='_',linestyle='', color= '#1010F3')
#plt.plot(W_HK, label='Wendenwasser', marker='_',linestyle='', color= '#9F16DA')
plt.ylabel('$δ^{18}O$ [‰]')
plt.legend(numpoints=15)
ax1.xaxis.set_major_formatter(DateFormatter('%d.%m.%Y'))
# removing top and right borders
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
# adds major gridlines
ax1.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.8)