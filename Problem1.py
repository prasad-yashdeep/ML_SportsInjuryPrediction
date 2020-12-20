from sklearn.metrics import plot_roc_curve
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost as xgb
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss
import itertools

play_df = pd.read_csv('Dataset/PlayList.csv')
# lets start cleaning up and extracting information.
unique_players = play_df.PlayerKey.nunique()
unique_plays = play_df.PlayKey.nunique()
unique_games = play_df.GameID.nunique()
print('There are {} players in the dataset.'.format(unique_players))
print('There are {} games in the dataset.'.format(unique_games))
print('There are {} plays in the dataset.'.format(unique_plays))
# play_df.head()

game_df = play_df[['GameID', 'StadiumType', 'FieldType', 'Weather',
                   'Temperature']].drop_duplicates().reset_index().drop(columns=['index'])


def add_value_labels(ax, spacing=5, decimals=0):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'
        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        format_str = "{:." + str(decimals) + "f}"
        label = format_str.format(y_value)

        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points",  # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va)                      # Vertically align label differently for
        # positive and negative values


def visualize_game_features(game_df, rotation=90, add_labels=False, figsize=(10, 10)):
    plt.style.use('ggplot')
    fig = plt.figure(figsize=figsize)
    grid = plt.GridSpec(4, 3, hspace=0.2, wspace=0.2)
    stadium_ax = fig.add_subplot(grid[0, :2])
    fieldtype_ax = fig.add_subplot(grid[0, 2])
    weather_ax = fig.add_subplot(grid[1, 0:])
    temperature_ax = fig.add_subplot(grid[2, 0:])

    stadium_ax.bar(game_df.StadiumType.value_counts().keys(),
                   game_df.StadiumType.value_counts().values, color='#00c2c7')
    stadium_ax.set_title('StadiumType')
    stadium_ax.set_xticklabels(
        game_df.StadiumType.value_counts().keys(), rotation=rotation)

    if add_labels:
        add_value_labels(stadium_ax, spacing=5)

    fieldtype_ax.bar(game_df.FieldType.value_counts().keys(
    ), game_df.FieldType.value_counts().values, color=['#00c2c7', '#ff9e15'])
    fieldtype_ax.set_title('FieldType')
    fieldtype_ax.set_xticklabels(
        game_df.FieldType.value_counts().keys(), rotation=0)

    if add_labels:
        add_value_labels(fieldtype_ax, spacing=5)

    weather_ax.bar(game_df.Weather.value_counts().keys(),
                   game_df.Weather.value_counts().values, color='#00c2c7')
    weather_ax.set_title('Weather')
    weather_ax.set_xticklabels(
        game_df.Weather.value_counts().keys(), rotation=rotation)

    if add_labels:
        add_value_labels(weather_ax, spacing=5)

    temperature_ax.hist(game_df.Temperature.astype(
        int).values, bins=30, range=(0, 90))
    temperature_ax.set_xlim(0, 110)
    temperature_ax.set_xticks(range(0, 110, 10))
    temperature_ax.set_xticklabels(range(0, 110, 10))
    temperature_ax.set_title('Temperature')

    plt.suptitle('Game-Level Exploration', fontsize=16)
    plt.show()


def clean_weather(row):
    cloudy = ['Cloudy 50% change of rain', 'Hazy', 'Cloudy.', 'Overcast', 'Mostly Cloudy',
              'Cloudy, fog started developing in 2nd quarter', 'Partly Cloudy',
              'Mostly cloudy', 'Rain Chance 40%', ' Partly cloudy', 'Party Cloudy',
              'Rain likely, temps in low 40s', 'Partly Clouidy', 'Cloudy, 50% change of rain', 'Mostly Coudy', '10% Chance of Rain',
              'Cloudy, chance of rain', '30% Chance of Rain', 'Cloudy, light snow accumulating 1-3"',
              'cloudy', 'Coudy', 'Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.',
              'Cloudy fog started developing in 2nd quarter', 'Cloudy light snow accumulating 1-3"',
              'Cloudywith periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.',
              'Cloudy 50% change of rain', 'Cloudy and cold',
              'Cloudy and Cool', 'Partly cloudy']

    clear = ['Clear, Windy', ' Clear to Cloudy', 'Clear, highs to upper 80s',
             'Clear and clear', 'Partly sunny',
             'Clear, Windy', 'Clear skies', 'Sunny', 'Partly Sunny', 'Mostly Sunny', 'Clear Skies',
             'Sunny Skies', 'Partly clear', 'Fair', 'Sunny, highs to upper 80s', 'Sun & clouds', 'Mostly sunny', 'Sunny, Windy',
             'Mostly Sunny Skies', 'Clear and Sunny', 'Clear and sunny', 'Clear to Partly Cloudy', 'Clear Skies',
             'Clear and cold', 'Clear and warm', 'Clear and Cool', 'Sunny and cold', 'Sunny and warm', 'Sunny and clear']

    rainy = ['Rainy', 'Scattered Showers', 'Showers', 'Cloudy Rain', 'Light Rain',
             'Rain shower', 'Rain likely, temps in low 40s.', 'Cloudy, Rain']

    snow = ['Heavy lake effect snow']

    indoor = ['Controlled Climate', 'Indoors', 'N/A Indoor', 'N/A (Indoors)']

    if row.Weather in cloudy:
        return 'Cloudy'

    if row.Weather in indoor:
        return 'Indoor'

    if row.Weather in clear:
        return 'Clear'

    if row.Weather in rainy:
        return 'Rain'

    if row.Weather in snow:
        return 'Snow'

    if row.Weather in ['Cloudy.', 'Heat Index 95', 'Cold']:
        return np.nan

    return row.Weather


def clean_stadiumtype(row):
    if row.StadiumType in ['Bowl', 'Heinz Field', 'Cloudy']:
        return np.nan
    else:
        return row.StadiumType


def clean_play_df(play_df):
    play_df_cleaned = play_df.copy()

    # clean StadiumType
    play_df_cleaned['StadiumType'] = play_df_cleaned['StadiumType'].str.replace(
        r'Oudoor|Outdoors|Ourdoor|Outddors|Outdor|Outside', 'Outdoor')
    play_df_cleaned['StadiumType'] = play_df_cleaned['StadiumType'].str.replace(
        r'Indoors|Indoor, Roof Closed|Indoor, Open Roof', 'Indoor')
    play_df_cleaned['StadiumType'] = play_df_cleaned['StadiumType'].str.replace(
        r'Closed Dome|Domed, closed|Domed, Open|Domed, open|Dome, closed|Domed', 'Dome')
    play_df_cleaned['StadiumType'] = play_df_cleaned['StadiumType'].str.replace(
        r'Retr. Roof-Closed|Outdoor Retr Roof-Open|Retr. Roof - Closed|Retr. Roof-Open|Retr. Roof - Open|Retr. Roof Closed', 'Retractable Roof')
    play_df_cleaned['StadiumType'] = play_df_cleaned['StadiumType'].str.replace(
        'Open', 'Outdoor')
    play_df_cleaned['StadiumType'] = play_df_cleaned.apply(
        lambda row: clean_stadiumtype(row), axis=1)

    # clean Weather
    play_df_cleaned['Weather'] = play_df_cleaned.apply(
        lambda row: clean_weather(row), axis=1)

    return play_df_cleaned


play_df_cleaned = clean_play_df(play_df)
game_df_cleaned = play_df_cleaned[['GameID', 'StadiumType', 'FieldType', 'Weather',
                                   'Temperature']].drop_duplicates().reset_index().drop(columns=['index'])
visualize_game_features(game_df_cleaned, rotation=0,
                        add_labels=True, figsize=(12, 16))
# game_df_cleaned.head()

player_data_df = play_df_cleaned[[
    'PlayerKey', 'RosterPosition', 'PlayerGamePlay', 'Position', 'PositionGroup']]


def visualize_player_features(player_df, figsize=(25, 20), add_labels=False):
    plt.style.use('ggplot')
    fig = plt.figure(figsize=figsize)

    grid = plt.GridSpec(3, 4, hspace=0.2, wspace=0.2)

    plays_ax = fig.add_subplot(grid[0, 0:2])
    max_rolling_plays_ax = fig.add_subplot(grid[1, 0:2])

    rosterposition_ax = fig.add_subplot(grid[0, 2:])
    positiongroup_ax = fig.add_subplot(grid[1, 2:])
    position_ax = fig.add_subplot(grid[2, 0:])

    plays_ax.hist(player_df.groupby(by=['PlayerKey']).count()[
                  'RosterPosition'].values, color='#00c2c7', bins=50)
    plays_ax.set_title('Number of plays per player')
    # play_ax.set

    max_rolling_plays_ax.hist(player_df.groupby(
        by=['PlayerKey']).PlayerGamePlay.max().values, bins=30, color='#00c2c7')
    max_rolling_plays_ax.set_title(
        'Maximum number of rolling plays per player')

    rosterposition_ax.bar(player_df.RosterPosition.value_counts(
    ).keys().values, player_df.RosterPosition.value_counts().values)
    rosterposition_ax.set_xticklabels(
        player_df.RosterPosition.value_counts().keys().values, rotation=20)
    rosterposition_ax.set_title('Roster Position')
    if add_labels:
        add_value_labels(rosterposition_ax, spacing=5)

    position_ax.bar(player_df.Position.value_counts().keys().values,
                    player_df.Position.value_counts().values, color='#ff9e15')
    position_ax.set_title('Position')
    if add_labels:
        add_value_labels(position_ax, spacing=5)

    positiongroup_ax.bar(player_df.PositionGroup.value_counts(
    ).keys().values, player_df.PositionGroup.value_counts().values)
    positiongroup_ax.set_title('Position Group')
    if add_labels:
        add_value_labels(positiongroup_ax, spacing=5)

    plt.suptitle('Player-Level Exploration', fontsize=16)
    plt.show()


visualize_player_features(player_data_df, add_labels=True)


def visualize_play(play_df_cleaned):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))

    plt.bar(play_df_cleaned.PlayType.value_counts().keys().values,
            play_df_cleaned.PlayType.value_counts().values)
    plt.xticks(range(len(play_df_cleaned.PlayType.value_counts().keys().values)),
               play_df_cleaned.PlayType.value_counts().keys().values, rotation=20)
    add_value_labels(ax, spacing=5)
    plt.title('Play-Level Exploration: PlayType', fontsize=16)

    plt.show()


visualize_play(play_df_cleaned)

injury_df = pd.read_csv('Dataset/InjuryRecord.csv')
df1 = play_df_cleaned.drop_duplicates('GameID')

game_injury_df = injury_df.set_index('GameID').join(
    df1.set_index('GameID'), how='outer', lsuffix='_left', rsuffix='_right')
copy_game_injury = game_injury_df.copy()
game_injury_df['DM_M1'] = game_injury_df['DM_M1'].fillna(0).astype(int)
game_injury_df['DM_M7'] = game_injury_df['DM_M7'].fillna(0).astype(int)
game_injury_df['DM_M28'] = game_injury_df['DM_M28'].fillna(0).astype(int)
game_injury_df['DM_M42'] = game_injury_df['DM_M42'].fillna(0).astype(int)
game_injury_df['Injury'] = game_injury_df['DM_M1'] + \
    game_injury_df['DM_M7'] + game_injury_df['DM_M28'] + \
    game_injury_df['DM_M42']
lis = []
for i in range(len(game_injury_df['Injury'])):
    if(game_injury_df['Injury'][i] > 0):
        lis.append(1)
    else:
        lis.append(0)
game_injury_df['Injury'] = lis

game_injury_df.drop(columns=['PlayerKey_left', 'BodyPart', 'PlayKey_right', 'PlayKey_left', 'PlayerKey_right',
                             'DM_M1', 'DM_M7', 'DM_M28', 'DM_M42', 'Surface', 'PlayerGamePlay'], axis=1, inplace=True)
condition = game_injury_df['Injury'] == 0
condition1 = game_injury_df['Injury'] == 1
game_injury_df1 = game_injury_df[condition][:int(len(game_injury_df)*1)]
game_injury_df2 = game_injury_df[condition1]
frames = [game_injury_df1, game_injury_df2]
features_df = pd.concat(frames)

features_df = pd.get_dummies(features_df, dummy_na=False)
# features_df

df = pd.read_csv('Dataset/file1.csv')
df.loc[(df.Temperature == -999), 'Temperature'] = 55

# changing the playdate values
df["PlayerDay"] = df["PlayerDay"].abs()
features_df = df.set_index('GameID')
# features_df


y = features_df['Injury']
X = features_df.drop(columns=['Injury'])
y = np.array(y)
X = np.array(X)
# print(len(X[0]))

res = RandomOverSampler(random_state=0, sampling_strategy=0.5)
X_resampled, y_resampled = res.fit_resample(X, y)
dt_yresam = pd.DataFrame(y_resampled)
dt_yresam.columns = ['T']
# print(dt_yresam['T'].value_counts())
#print(X_resampled, y_resampled)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=21, shuffle=True)
#y_train, y_test = y[train_index], y[test_index]

print("GaussianNB")
new1 = GaussianNB()
new1.fit(X_train, y_train)
y_pred = new1.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print('Accuracy: {}'.format(accuracy))
print('Confusion Matrix: \n {}'.format(conf_matrix))
#plot_confusion_matrix(new1 , X_test, y_test)
print('Precision')
print(precision_score(y_test, y_pred))
print('Racall')
print(recall_score(y_test, y_pred))
plot_roc_curve(new1, X_test, y_test)

y = features_df['Injury']
X = features_df.drop(columns=['Injury'])
y = np.array(y)
X = np.array(X)
# print(len(X[0]))


res = RandomOverSampler(random_state=0, sampling_strategy=0.5)
X_resampled, y_resampled = res.fit_resample(X, y)
dt_yresam = pd.DataFrame(y_resampled)
dt_yresam.columns = ['T']
# print(dt_yresam['T'].value_counts())
#print(X_resampled, y_resampled)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=21, shuffle=True)
#y_train, y_test = y[train_index], y[test_index]

print("Logistic Regression")
new1 = LogisticRegression(max_iter=5000)
new1.fit(X_train, y_train)
y_pred = new1.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print('Accuracy: {}'.format(accuracy))
print('Confusion Matrix: \n {}'.format(conf_matrix))
#plot_confusion_matrix(new1 , X_test, y_test)
print('Precision')
print(precision_score(y_test, y_pred))
print('Racall')
print(recall_score(y_test, y_pred))
plot_roc_curve(new1, X_test, y_test)


y = features_df['Injury']
X = features_df.drop(columns=['Injury'])
y = np.array(y)
X = np.array(X)
X = StandardScaler().fit_transform(X)
# print(X.shape)
skf = StratifiedKFold(n_splits=2)
res = RandomOverSampler(random_state=0)
X_resampled, y_resampled = res.fit_resample(X, y)
for train_index, test_index in skf.split(X_resampled, y_resampled):

    X_train, X_test = X_resampled[train_index], X_resampled[test_index]
    y_train, y_test = y_resampled[train_index], y_resampled[test_index]

    new = DecisionTreeClassifier(max_depth=8)
    new.fit(X_train, y_train)

    y_pred = new.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("DecisionTree")
    print('Accuracy: {}'.format(accuracy))
    print('Confusion Matrix: \n {}'.format(conf_matrix))
    plot_confusion_matrix(new, X_test, y_test)
    print('Precision')
    print(precision_score(y_test, y_pred))
    print('Racall')
    print(recall_score(y_test, y_pred))
    plot_roc_curve(new, X_test, y_test)


model = xgb.XGBClassifier(max_depth=3,
                          learning_rate=0.1,
                          n_estimators=150,
                          objective='binary:logistic',
                          booster='gbtree',
                          tree_method='auto',
                          n_jobs=50,
                          gamma=0,
                          min_child_weight=1,
                          max_delta_step=0,
                          subsample=1,
                          colsample_bytree=1,
                          colsample_bylevel=1,
                          colsample_bynode=1,
                          reg_alpha=0,
                          reg_lambda=1,
                          scale_pos_weight=1,
                          base_score=0.5,
                          random_state=0)

y = features_df['Injury']
X = features_df.drop(columns=['Injury'])
res = RandomOverSampler(random_state=0, sampling_strategy=0.5)
X_resampled, y_resampled = res.fit_resample(X, y)
dt_yresam = pd.DataFrame(y_resampled)
dt_yresam.columns = ['T']
# print(dt_yresam['T'].value_counts())
# print(X_resampled.shape,y_resampled.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, shuffle=True)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print('Accuracy: {}'.format(accuracy))
print('Confusion Matrix: \n {}'.format(conf_matrix))
plot_confusion_matrix(model, X_test, y_test)
print('Precision')
print(precision_score(y_test, y_pred))
print('Racall')
print(recall_score(y_test, y_pred))
plot_roc_curve(model, X_test, y_test)