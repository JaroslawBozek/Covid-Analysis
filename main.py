import pandas as pd

import numpy as np

import json
from netCDF4 import Dataset
import math
from scipy.stats import chisquare, chi2_contingency, normaltest, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def main():

    ##Load data
    df_confirmed = pd.read_csv('data/time_series_covid19_confirmed_global.csv').fillna(value="None")
    df_deaths = pd.read_csv('data/time_series_covid19_deaths_global.csv').fillna(value="None")
    df_recovered = pd.read_csv('data/time_series_covid19_recovered_global.csv').fillna(value="None")

    ##Divide Canada into regions and add them to df_recovered
    df_canada = df_confirmed.loc[df_confirmed['Country/Region'] == 'Canada']
    df_canada.iloc[:][:] = 0
    df_recovered = df_recovered[~df_recovered['Country/Region'].isin(['Canada'])].reset_index(drop=True)
    df_recovered = pd.concat([df_recovered, df_canada], axis=0)
    df_confirmed = df_confirmed.sort_values(by=['Country/Region','Province/State']).reset_index(drop=True)
    df_deaths = df_deaths.sort_values(by=['Country/Region','Province/State']).reset_index(drop=True)
    df_recovered = df_recovered.sort_values(by=['Country/Region','Province/State']).reset_index(drop=True)

    ##Remove zero death and confirmed countries
    #Remove zero deaths countries
    zero_deaths_indexes = list(df_deaths.loc[df_deaths.iloc[:, -1] == 0].index)
    df_confirmed = df_confirmed.drop(zero_deaths_indexes).reset_index(drop=True)
    df_deaths = df_deaths.drop(zero_deaths_indexes).reset_index(drop=True)
    df_recovered = df_recovered.drop(zero_deaths_indexes).reset_index(drop=True)

    #Remove zero confirmed countries
    zero_confirmed_indexes = list(df_confirmed.loc[df_confirmed.iloc[:, -1] == 0].index)
    df_confirmed = df_confirmed.drop(zero_confirmed_indexes).reset_index(drop=True)
    df_deaths = df_deaths.drop(zero_confirmed_indexes).reset_index(drop=True)
    df_recovered = df_recovered.drop(zero_confirmed_indexes).reset_index(drop=True)

    ##Active cases
    #Active cases
    df_deaths_no_cols = df_deaths.drop(df_deaths.columns[0:4], axis=1)
    df_recovered_no_cols = df_recovered.drop(df_recovered.columns[0:4], axis=1)
    df_active_cases = df_confirmed.drop(df_confirmed.columns[0:4], axis=1)

    df_active_cases = df_active_cases.subtract(df_deaths_no_cols, fill_value=0)
    df_active_cases = df_active_cases.subtract(df_recovered_no_cols, fill_value=0)

    four_columns = df_deaths.iloc[:, 0:4]

    #Active Cases for no recovery info (countries with 0 recoveries)
    df_active_cases_fill = df_confirmed
    df_active_cases_fill = df_active_cases_fill.drop(df_active_cases_fill.columns[0:4], axis=1)
    df_active_cases_fill = df_active_cases_fill[df_recovered.iloc[:, -1] == 0]

    df_active_cases_fill_shifted = df_active_cases_fill.shift(periods=14, axis="columns").fillna(0)
    df_active_cases_fill = df_active_cases_fill.subtract(df_active_cases_fill_shifted, fill_value=0).dropna()
    df_active_cases = df_active_cases.add(df_active_cases_fill, fill_value=0)

    indexes_to_drop = df_active_cases_fill.index.values.tolist()
    df_active_cases = df_active_cases.drop(indexes_to_drop)
    df_active_cases = pd.concat([df_active_cases, df_active_cases_fill], axis=0).sort_index()

    df_active_cases = pd.concat([four_columns, df_active_cases], axis=1)

    ##Create Monthly mortality
    df_monthly_deaths = df_deaths
    df_monthly_recovered = df_recovered
    four_columns = df_monthly_deaths.iloc[:, 0:4]

    #Monthly_deaths
    df_monthly_deaths = df_monthly_deaths.drop(df_monthly_deaths.columns[0:4], axis=1)
    df_monthly_deaths = df_monthly_deaths.T.groupby(
        [i.split("/")[0] + i.split("/")[2] for i in df_monthly_deaths.T.index.values]).max().T
    df_monthly_deaths = df_monthly_deaths.rename(
        columns=lambda i: i[-2:] + "/" + i[:-2] if len(i) == 4 else i[-2:] + "/0" + i[:-2])
    df_monthly_deaths = df_monthly_deaths.sort_index(axis=1)
    df_monthly_deaths_shifted = df_monthly_deaths.shift(periods=1, axis="columns").fillna(0)
    df_monthly_deaths = df_monthly_deaths.subtract(df_monthly_deaths_shifted, fill_value=0)

    #Monthly_recovered
    df_monthly_recovered = df_monthly_recovered.drop(df_monthly_recovered.columns[0:4], axis=1)
    df_monthly_recovered = df_monthly_recovered.T.groupby(
        [i.split("/")[0] + i.split("/")[2] for i in df_monthly_recovered.T.index.values]).max().T
    df_monthly_recovered = df_monthly_recovered.rename(
        columns=lambda i: i[-2:] + "/" + i[:-2] if len(i) == 4 else i[-2:] + "/0" + i[:-2])
    df_monthly_recovered = df_monthly_recovered.sort_index(axis=1)
    df_monthly_recovered_shifted = df_monthly_recovered.shift(periods=1, axis="columns").fillna(0)
    df_monthly_recovered = df_monthly_recovered.subtract(df_monthly_recovered_shifted, fill_value=0)

    #Monthly mortality (deaths/recovered)
    df_monthly_mortality = df_monthly_deaths.div(df_monthly_recovered).fillna(0)
    df_monthly_mortality = pd.concat([four_columns, df_monthly_mortality], axis=1)

    ##Active cases from 1 week
    df_active_cases_1week = df_active_cases.drop(df_active_cases.columns[0:4], axis=1)
    df_active_cases_copy = df_active_cases.drop(df_active_cases.columns[0:4], axis=1)
    for i in range(1,7):
        df_active_cases_copy = df_active_cases_copy.shift(periods=1, axis="columns").fillna(0)
        df_active_cases_1week = df_active_cases_1week.add(df_active_cases_copy, fill_value=0)

    ##Reproduction coefficient
    df_reproduction_coefficient = df_active_cases_1week
    df_active_cases_copy = df_active_cases_1week.shift(periods=5, axis="columns").fillna(0)

    # Set active cases with < 100 cases to NaN
    for column in df_active_cases_copy:
        df_active_cases_copy.loc[df_active_cases_copy[column] < 100, column] = float('nan')

    df_reproduction_coefficient = df_reproduction_coefficient.div(df_active_cases_copy)


    ###Weather data
    weather_max = Dataset('./data/TerraClimate_tmax_2018.nc')
    weather_min = Dataset('./data/TerraClimate_tmin_2018.nc')

    Country_name = df_monthly_mortality['Country/Region'].tolist()
    Long = df_monthly_mortality['Long'].tolist()
    Lat = df_monthly_mortality['Lat'].tolist()

    df_avg_temps = pd.DataFrame(np.zeros((len(Long), 12)))

    # df_avg_temps = pd.read_json('avg_temps.json')

    #Generate temperatures
    print("Converting weather data into arrays")
    print("It may take a moment...")
    matrix_max = np.asarray(weather_max['tmax'])
    print("Halfway there...")
    matrix_min = np.asarray(weather_min['tmin'])

    for i in range(12):
        for j in range(len(Long)):
            Long_temp = Long[j]
            Lat_temp = Lat[j]

            Long_temp += 180.0
            Lat_temp -= 90.0
            Lat_temp = -Lat_temp

            Long_temp = int(Long_temp*(8640.0/360.0))
            Lat_temp = int(Lat_temp*(4320.0/180.0))


            w_min = matrix_min[i][Lat_temp][Long_temp]
            w_max = matrix_max[i][Lat_temp][Long_temp]
            w_avg = (w_min+w_max)/2.0
            df_avg_temps[i][j] = w_avg

    # df_avg_temps.to_json('avg_temps.json')

    #Add january to average temperatures
    col = df_avg_temps[0]
    df_avg_temps = pd.concat([df_avg_temps, col], axis=1)
    df_avg_temps = pd.concat([four_columns, df_avg_temps], axis=1)
    indexes_to_drop = list(df_avg_temps.loc[df_avg_temps.iloc[:, -1] > 30000].index)
    df_avg_temps = df_avg_temps.drop(indexes_to_drop).reset_index(drop=True)
    df_avg_temps = df_avg_temps.drop(df_avg_temps.columns[0:4], axis=1)

    #drop columns from df_deaths and df_confirmed for future purposes
    df_deaths = df_deaths.drop(indexes_to_drop).reset_index(drop=True)
    df_confirmed = df_confirmed.drop(indexes_to_drop).reset_index(drop=True)

    ###TASK 1
    #Squeeze reproduction coefficient to months
    df_reproduction_coefficient = df_reproduction_coefficient.T.groupby(
        [i.split("/")[0] + i.split("/")[2] for i in df_reproduction_coefficient.T.index.values]).mean().T
    df_reproduction_coefficient = df_reproduction_coefficient.rename(
        columns=lambda i: i[-2:] + "/" + i[:-2] if len(i) == 4 else i[-2:] + "/0" + i[:-2])
    df_reproduction_coefficient = df_reproduction_coefficient.sort_index(axis=1)
    df_reproduction_coefficient = df_reproduction_coefficient.drop(indexes_to_drop).reset_index(drop=True)

    df_avg_temps = df_avg_temps.T.reset_index(drop=True).T
    df_reproduction_coefficient = df_reproduction_coefficient.apply(lambda x: x / x.max(), axis=1) #Normalization


    #Removing an outlier (Iran)
    df_reproduction_coefficient = df_reproduction_coefficient.drop(125).reset_index(drop=True)
    df_avg_temps = df_avg_temps.drop(125).reset_index(drop=True)

    coeffs = [[],[],[],[],[]]
    x, y = df_avg_temps.shape
    for i in range(x):
        for j in range(y):
            temp = df_avg_temps.iloc[i][j]

            if math.isnan(df_reproduction_coefficient.iloc[i][j]):
                continue
            else:
                coeff = df_reproduction_coefficient.iloc[i][j]

            if temp < 0:
                coeffs[0].append(coeff)
            if temp >= 0 and temp < 10:
                coeffs[1].append(coeff)
            if temp >= 10 and temp < 20:
                coeffs[2].append(coeff)
            if temp >= 20 and temp < 30:
                coeffs[3].append(coeff)
            if temp >= 30:
                coeffs[4].append(coeff)

    ##Compare temperatures to reproduction coefficients
    #Normal test
    print("\nZADANIE 1")
    print("\nNormaltests:")
    for i in range(len(coeffs)):
        print(normaltest(coeffs[i]))
    print("Dla temperatur >30C najlepszym z rozwiązań do zminimalizowania p_value okazało się usunięcię outliera jakim"
          " były dane z Iranu.")

    #ANOVA test
    print("\nf_oneway: ")
    f_value, p_value = f_oneway(coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4])
    print("f_value:", f_value, "p_value:", p_value)
    print("p_value wynoszące ~3e-13 oznacza, iż hipoteza zerowa jest odrzucona z czego wynika, że współczynnik"
          " reprodukcji jest zależny od temperatury.")

    print("\npairwise_tukeyhsd: ")
    print(pairwise_tukeyhsd(np.concatenate([coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4]]),
                            np.concatenate([['<0'] * len(coeffs[0]), ['0-10'] * len(coeffs[1]), ['10-20'] * len(coeffs[2]), ['20-30'] * len(coeffs[3]), ['>30'] * len(coeffs[4])])))
    print("Przy niektórych porównaniach można zauważyć, że hipoteza zerowa została odrzucona, aczkolwiek ze względu"
          " na wątpliwą jakość spreparowanych danych i różnicy między ilością danych dla poszczególnych \"kubełków\","
          " trudno jest stwierdzić jak wiarygodne są to wyniki")


    ###TASK 2
    #Remove non-european countries
    array_confirmed = df_confirmed.to_numpy()
    indexes_to_drop = []
    european_countries_list = ['Albania','Andorra','Austria','Belarus','Belgium','Bosnia and Herzegovina','Bulgaria',
                               'Croatia','Cyprus','Czechia','Denmark','Estonia','Finland','France','Georgia','Germany',
                               'Greece','Hungary','Iceland','Ireland','Italy','Kosovo','Latvia','Liechtenstein',
                               'Lithuania','Luxembourg','Monaco','Montenegro','Netherlands','Norway','Poland',
                               'Portugal','Russia','San Marino','Slovakia','Slovenia','Spain','Sweden','Switzerland',
                               'Turkey','Ukraine','United Kingdom']
    for i in range(len(array_confirmed)):
        if array_confirmed[i][1] not in european_countries_list:
            indexes_to_drop.append(i)

    df_confirmed = df_confirmed.drop(indexes_to_drop).reset_index(drop=True)
    df_deaths = df_deaths.drop(indexes_to_drop).reset_index(drop=True)


    ##Chi2_contingency deaths/confirmed
    all_confirmed = df_confirmed.iloc[:,-1:].to_numpy()
    all_deaths = df_deaths.iloc[:,-1:].to_numpy()

    data = [all_confirmed,all_deaths]
    g, p, df, _ = chi2_contingency(data)
    print("\nZADANIE 2")
    print("\nChi2_contingency dla zależności śmierci/zarażenia")
    print("g_value:", g, "p_value:", p)
    print("p_value wynosi praktycznie zero z czego wynika, iż hipoteza zerowa została odrzucona i ilość śmierci zależy"
            " od ilości zakażeń.")


    ##ANOVA monthly mortality
    #Remove non-european countries from df_monthly_mortality
    array_monthly_mortality = df_monthly_mortality.to_numpy()
    indexes_to_drop = []

    for i in range(len(array_monthly_mortality)):
        if array_monthly_mortality[i][1] not in european_countries_list:
            indexes_to_drop.append(i)

    df_monthly_mortality = df_monthly_mortality.drop(indexes_to_drop).reset_index(drop=True)

    #Convert 0s and infs to nans
    for column in df_monthly_mortality.columns[4:]:
        df_monthly_mortality.loc[df_monthly_mortality[column] <= 0, column] = float('nan')
        df_monthly_mortality.loc[df_monthly_mortality[column] == np.inf, column] = float('nan')

    array_monthly_mortality = df_monthly_mortality.to_numpy()

    #Remove countries with nans only
    rows_to_delete = []
    for i, x in enumerate(array_monthly_mortality):
        x = x[4:]
        x = x.tolist()
        if np.count_nonzero(~np.isnan(x)) <= 0:
            rows_to_delete.append(i)

    array_monthly_mortality = np.delete(array_monthly_mortality, rows_to_delete, 0)

    #Remove countries with variance > 1
    rows_to_delete = []
    var_list = np.nanvar(array_monthly_mortality[:,4:], axis=1)

    for i, var in enumerate(var_list):
        if var > 1:
            rows_to_delete.append(i)

    array_monthly_mortality = np.delete(array_monthly_mortality, rows_to_delete, 0)


    #ANOVA test
    arrays = []
    names = []
    for i, x in enumerate(array_monthly_mortality):
        country_name = [x[0] + "_" + x[1]]
        x = x[4:].astype('float64')
        x = x[~np.isnan(x)]
        arrays.append(x)
        name = country_name * len(x)
        names.append(name)


    f_value, p_value = f_oneway(*arrays)
    print("\nf_oneway dla skumulowanej śmiertelności w różnych krajach: ")
    print("f_value:", f_value, "p_value:", p_value)
    print("Dla poziomu istotności p<0.05 hipoteza zerowa została potwierdzona ze względu na to, iż p_value>0.05"
        " i nie istnieje zależność między śmiertelnościami w różnych krajach europejskich.")

    print("\npairwise_tukeyhsd:")
    print(pairwise_tukeyhsd(np.concatenate([*arrays]),
                            np.concatenate([*names])))
    print("Użycie pairwise_tukeyhsd jest zbędne po potwierdzeniu hipotezy zerowej, aczkolwiek można dokładniej zauważyć,"
          " iż nie występują żadne szczególne zależności między śmiertelnościami w różnych krajach z czego wynika"
          " dominująca ilość wartości p_adj = 0.9 przy porównaniach.")

if __name__ == '__main__':
    main()