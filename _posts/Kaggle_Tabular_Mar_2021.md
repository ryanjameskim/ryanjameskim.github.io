# Importing and Data upload



```
# Import OS for navigation and environment set up
import os

# Import Numpy
import numpy as np

# Import Pandas
import pandas as pd

# Needed for cleaner code
from functools import partial

# Import graphing libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.graph_objects as go

# Import encoding libraries
from sklearn.preprocessing import LabelEncoder
from copy import deepcopy
!pip install category_encoders
from category_encoders import CatBoostEncoder, LeaveOneOutEncoder

# Import baseline models and other tools
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from lightgbm import LGBMRegressor
from lightgbm import LGBMClassifier
!pip install catboost
from catboost import CatBoostClassifier
from sklearn.linear_model import RidgeClassifier
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# Import for Ensemble Second Level Classification
from scipy.special import expit
from sklearn.calibration import CalibratedClassifierCV

# Import cross validation tools
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# Import measurement tools
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score, mean_squared_error

# Import Optuna for tuning
!pip install optuna
import optuna

#mount google drive
from google.colab import drive
drive.mount('/gdrive')
```

    Requirement already satisfied: category_encoders in /usr/local/lib/python3.7/dist-packages (2.2.2)
    Requirement already satisfied: numpy>=1.14.0 in /usr/local/lib/python3.7/dist-packages (from category_encoders) (1.19.5)
    Requirement already satisfied: statsmodels>=0.9.0 in /usr/local/lib/python3.7/dist-packages (from category_encoders) (0.10.2)
    Requirement already satisfied: pandas>=0.21.1 in /usr/local/lib/python3.7/dist-packages (from category_encoders) (1.1.5)
    Requirement already satisfied: scikit-learn>=0.20.0 in /usr/local/lib/python3.7/dist-packages (from category_encoders) (0.22.2.post1)
    Requirement already satisfied: patsy>=0.5.1 in /usr/local/lib/python3.7/dist-packages (from category_encoders) (0.5.1)
    Requirement already satisfied: scipy>=1.0.0 in /usr/local/lib/python3.7/dist-packages (from category_encoders) (1.4.1)
    Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.7/dist-packages (from pandas>=0.21.1->category_encoders) (2018.9)
    Requirement already satisfied: python-dateutil>=2.7.3 in /usr/local/lib/python3.7/dist-packages (from pandas>=0.21.1->category_encoders) (2.8.1)
    Requirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.7/dist-packages (from scikit-learn>=0.20.0->category_encoders) (1.0.1)
    Requirement already satisfied: six in /usr/local/lib/python3.7/dist-packages (from patsy>=0.5.1->category_encoders) (1.15.0)
    Requirement already satisfied: catboost in /usr/local/lib/python3.7/dist-packages (0.24.4)
    Requirement already satisfied: pandas>=0.24.0 in /usr/local/lib/python3.7/dist-packages (from catboost) (1.1.5)
    Requirement already satisfied: plotly in /usr/local/lib/python3.7/dist-packages (from catboost) (4.4.1)
    Requirement already satisfied: six in /usr/local/lib/python3.7/dist-packages (from catboost) (1.15.0)
    Requirement already satisfied: graphviz in /usr/local/lib/python3.7/dist-packages (from catboost) (0.10.1)
    Requirement already satisfied: scipy in /usr/local/lib/python3.7/dist-packages (from catboost) (1.4.1)
    Requirement already satisfied: numpy>=1.16.0 in /usr/local/lib/python3.7/dist-packages (from catboost) (1.19.5)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.7/dist-packages (from catboost) (3.2.2)
    Requirement already satisfied: python-dateutil>=2.7.3 in /usr/local/lib/python3.7/dist-packages (from pandas>=0.24.0->catboost) (2.8.1)
    Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.7/dist-packages (from pandas>=0.24.0->catboost) (2018.9)
    Requirement already satisfied: retrying>=1.3.3 in /usr/local/lib/python3.7/dist-packages (from plotly->catboost) (1.3.3)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.7/dist-packages (from matplotlib->catboost) (0.10.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->catboost) (1.3.1)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->catboost) (2.4.7)
    Collecting optuna
    [?25l  Downloading https://files.pythonhosted.org/packages/50/67/ed0af7c66bcfb9a9a56fbafcca7d848452d78433208b59b003741879cc69/optuna-2.6.0-py3-none-any.whl (293kB)
    [K     |████████████████████████████████| 296kB 7.3MB/s 
    [?25hRequirement already satisfied: numpy<1.20.0 in /usr/local/lib/python3.7/dist-packages (from optuna) (1.19.5)
    Collecting cmaes>=0.8.2
      Downloading https://files.pythonhosted.org/packages/01/1f/43b01223a0366171f474320c6e966c39a11587287f098a5f09809b45e05f/cmaes-0.8.2-py3-none-any.whl
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.7/dist-packages (from optuna) (20.9)
    Requirement already satisfied: sqlalchemy>=1.1.0 in /usr/local/lib/python3.7/dist-packages (from optuna) (1.3.23)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.7/dist-packages (from optuna) (4.41.1)
    Collecting alembic
    [?25l  Downloading https://files.pythonhosted.org/packages/f7/29/ed5c134aba874053859ba3e8d4705b4a5c1b66156deabc26cbe643e83f2e/alembic-1.5.7-py2.py3-none-any.whl (159kB)
    [K     |████████████████████████████████| 163kB 12.8MB/s 
    [?25hCollecting cliff
    [?25l  Downloading https://files.pythonhosted.org/packages/a2/d6/7d9acb68a77acd140be7fececb7f2701b2a29d2da9c54184cb8f93509590/cliff-3.7.0-py3-none-any.whl (80kB)
    [K     |████████████████████████████████| 81kB 7.8MB/s 
    [?25hCollecting colorlog
      Downloading https://files.pythonhosted.org/packages/51/62/61449c6bb74c2a3953c415b2cdb488e4f0518ac67b35e2b03a6d543035ca/colorlog-4.8.0-py2.py3-none-any.whl
    Requirement already satisfied: scipy!=1.4.0 in /usr/local/lib/python3.7/dist-packages (from optuna) (1.4.1)
    Requirement already satisfied: pyparsing>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from packaging>=20.0->optuna) (2.4.7)
    Requirement already satisfied: python-dateutil in /usr/local/lib/python3.7/dist-packages (from alembic->optuna) (2.8.1)
    Collecting python-editor>=0.3
      Downloading https://files.pythonhosted.org/packages/c6/d3/201fc3abe391bbae6606e6f1d598c15d367033332bd54352b12f35513717/python_editor-1.0.4-py3-none-any.whl
    Collecting Mako
    [?25l  Downloading https://files.pythonhosted.org/packages/f3/54/dbc07fbb20865d3b78fdb7cf7fa713e2cba4f87f71100074ef2dc9f9d1f7/Mako-1.1.4-py2.py3-none-any.whl (75kB)
    [K     |████████████████████████████████| 81kB 8.0MB/s 
    [?25hCollecting pbr!=2.1.0,>=2.0.0
    [?25l  Downloading https://files.pythonhosted.org/packages/fb/48/69046506f6ac61c1eaa9a0d42d22d54673b69e176d30ca98e3f61513e980/pbr-5.5.1-py2.py3-none-any.whl (106kB)
    [K     |████████████████████████████████| 112kB 9.0MB/s 
    [?25hCollecting stevedore>=2.0.1
    [?25l  Downloading https://files.pythonhosted.org/packages/d4/49/b602307aeac3df3384ff1fcd05da9c0376c622a6c48bb5325f28ab165b57/stevedore-3.3.0-py3-none-any.whl (49kB)
    [K     |████████████████████████████████| 51kB 5.8MB/s 
    [?25hRequirement already satisfied: PyYAML>=3.12 in /usr/local/lib/python3.7/dist-packages (from cliff->optuna) (3.13)
    Requirement already satisfied: PrettyTable>=0.7.2 in /usr/local/lib/python3.7/dist-packages (from cliff->optuna) (2.1.0)
    Collecting cmd2>=1.0.0
    [?25l  Downloading https://files.pythonhosted.org/packages/15/8b/15061b32332bb35ea2a2f6263d0f616779d576e82739ec8e7fcf3c94abf5/cmd2-1.5.0-py3-none-any.whl (133kB)
    [K     |████████████████████████████████| 143kB 12.8MB/s 
    [?25hRequirement already satisfied: six>=1.5 in /usr/local/lib/python3.7/dist-packages (from python-dateutil->alembic->optuna) (1.15.0)
    Requirement already satisfied: MarkupSafe>=0.9.2 in /usr/local/lib/python3.7/dist-packages (from Mako->alembic->optuna) (1.1.1)
    Requirement already satisfied: importlib-metadata>=1.7.0; python_version < "3.8" in /usr/local/lib/python3.7/dist-packages (from stevedore>=2.0.1->cliff->optuna) (3.7.2)
    Requirement already satisfied: wcwidth in /usr/local/lib/python3.7/dist-packages (from PrettyTable>=0.7.2->cliff->optuna) (0.2.5)
    Collecting colorama>=0.3.7
      Downloading https://files.pythonhosted.org/packages/44/98/5b86278fbbf250d239ae0ecb724f8572af1c91f4a11edf4d36a206189440/colorama-0.4.4-py2.py3-none-any.whl
    Requirement already satisfied: attrs>=16.3.0 in /usr/local/lib/python3.7/dist-packages (from cmd2>=1.0.0->cliff->optuna) (20.3.0)
    Collecting pyperclip>=1.6
      Downloading https://files.pythonhosted.org/packages/a7/2c/4c64579f847bd5d539803c8b909e54ba087a79d01bb3aba433a95879a6c5/pyperclip-1.8.2.tar.gz
    Requirement already satisfied: typing-extensions>=3.6.4; python_version < "3.8" in /usr/local/lib/python3.7/dist-packages (from importlib-metadata>=1.7.0; python_version < "3.8"->stevedore>=2.0.1->cliff->optuna) (3.7.4.3)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata>=1.7.0; python_version < "3.8"->stevedore>=2.0.1->cliff->optuna) (3.4.1)
    Building wheels for collected packages: pyperclip
      Building wheel for pyperclip (setup.py) ... [?25l[?25hdone
      Created wheel for pyperclip: filename=pyperclip-1.8.2-cp37-none-any.whl size=11107 sha256=9e50df6e45efa660c86562056da6dfd12dca5501da1bcd765dc37724584929cb
      Stored in directory: /root/.cache/pip/wheels/25/af/b8/3407109267803f4015e1ee2ff23be0c8c19ce4008665931ee1
    Successfully built pyperclip
    Installing collected packages: cmaes, python-editor, Mako, alembic, pbr, stevedore, colorama, pyperclip, cmd2, cliff, colorlog, optuna
    Successfully installed Mako-1.1.4 alembic-1.5.7 cliff-3.7.0 cmaes-0.8.2 cmd2-1.5.0 colorama-0.4.4 colorlog-4.8.0 optuna-2.6.0 pbr-5.5.1 pyperclip-1.8.2 python-editor-1.0.4 stevedore-3.3.0
    Drive already mounted at /gdrive; to attempt to forcibly remount, call drive.mount("/gdrive", force_remount=True).


```
# Enable the Kaggle environment, use the path to the directory your Kaggle API JSON is stored in
os.environ['KAGGLE_CONFIG_DIR'] = '/gdrive/MyDrive/Kaggle'

# install Kaggle library for kaggle terminal commands
!pip install kaggle

```

    Requirement already satisfied: kaggle in /usr/local/lib/python3.7/dist-packages (1.5.10)
    Requirement already satisfied: requests in /usr/local/lib/python3.7/dist-packages (from kaggle) (2.23.0)
    Requirement already satisfied: certifi in /usr/local/lib/python3.7/dist-packages (from kaggle) (2020.12.5)
    Requirement already satisfied: python-dateutil in /usr/local/lib/python3.7/dist-packages (from kaggle) (2.8.1)
    Requirement already satisfied: urllib3 in /usr/local/lib/python3.7/dist-packages (from kaggle) (1.24.3)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.7/dist-packages (from kaggle) (4.41.1)
    Requirement already satisfied: python-slugify in /usr/local/lib/python3.7/dist-packages (from kaggle) (4.0.1)
    Requirement already satisfied: six>=1.10 in /usr/local/lib/python3.7/dist-packages (from kaggle) (1.15.0)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests->kaggle) (2.10)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests->kaggle) (3.0.4)
    Requirement already satisfied: text-unidecode>=1.3 in /usr/local/lib/python3.7/dist-packages (from python-slugify->kaggle) (1.3)


```
#zip save directory
zip_dir = '/gdrive/MyDrive/Kaggle/Tabular Playground Series - Mar 2021'

#change directory to save zips
os.chdir(zip_dir)

!kaggle competitions download -c tabular-playground-series-mar-2021
```

    Warning: Looks like you're using an outdated API Version, please consider updating (server 1.5.12 / client 1.5.4)
    train.csv.zip: Skipping, found more recently modified local copy (use --force to force download)
    test.csv.zip: Skipping, found more recently modified local copy (use --force to force download)
    sample_submission.csv.zip: Skipping, found more recently modified local copy (use --force to force download)


```
# Complete path to storage location of the .zip file of data
zip_files = []

#collect all zip files
for root, dirs, files in os.walk(zip_dir):
  for name in files:
    if 'zip' in name:
      zip_files.append(os.path.join(root, name))

# Change to colabs VM directory
os.chdir('/content')

#unzip
for zip_file in zip_files:
  !cp '{zip_file}' .
  root, name = os.path.split(zip_file)
  !unzip -q '{name}'

#show directory
os.listdir()
```




    ['.config',
     'test.csv.zip',
     'sample_submission.csv.zip',
     'test.csv',
     'sample_submission.csv',
     'train.csv',
     'train.csv.zip',
     'sample_data']



```
sample_sub = pd.read_csv('sample_submission.csv')
print(sample_sub.head())
print(len(sample_sub))

```

       id  target
    0   5     0.5
    1   6     0.5
    2   8     0.5
    3   9     0.5
    4  11     0.5
    200000


```
train = pd.read_csv('train.csv')

print(train.head())
print(len(train))
```

       id cat0 cat1 cat2 cat3  ...     cont7     cont8     cont9    cont10 target
    0   0    A    I    A    B  ...  0.791921  0.815254  0.965006  0.665915      0
    1   1    A    I    A    A  ...  0.408701  0.399353  0.927406  0.493729      0
    2   2    A    K    A    A  ...  0.388835  0.412303  0.292696  0.549452      0
    3   3    A    K    A    C  ...  0.897617  0.633669  0.760318  0.934242      0
    4   4    A    I    G    B  ...  0.279167  0.351103  0.357084  0.328960      1
    
    [5 rows x 32 columns]
    300000


```
test = pd.read_csv('test.csv')
```

```
train.columns
```




    Index(['id', 'cat0', 'cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7',
           'cat8', 'cat9', 'cat10', 'cat11', 'cat12', 'cat13', 'cat14', 'cat15',
           'cat16', 'cat17', 'cat18', 'cont0', 'cont1', 'cont2', 'cont3', 'cont4',
           'cont5', 'cont6', 'cont7', 'cont8', 'cont9', 'cont10', 'target'],
          dtype='object')



```
train.target.value_counts()

train.target.value_counts().div(len(train))
```




    0    0.73513
    1    0.26487
    Name: target, dtype: float64



```
cat_columns = [col_name for col_name in train.columns if 'cat' in col_name]
cont_columns = [col_name for col_name in train.columns if 'cont' in col_name]
```

```
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 300000 entries, 0 to 299999
    Data columns (total 32 columns):
     #   Column  Non-Null Count   Dtype  
    ---  ------  --------------   -----  
     0   id      300000 non-null  int64  
     1   cat0    300000 non-null  object 
     2   cat1    300000 non-null  object 
     3   cat2    300000 non-null  object 
     4   cat3    300000 non-null  object 
     5   cat4    300000 non-null  object 
     6   cat5    300000 non-null  object 
     7   cat6    300000 non-null  object 
     8   cat7    300000 non-null  object 
     9   cat8    300000 non-null  object 
     10  cat9    300000 non-null  object 
     11  cat10   300000 non-null  object 
     12  cat11   300000 non-null  object 
     13  cat12   300000 non-null  object 
     14  cat13   300000 non-null  object 
     15  cat14   300000 non-null  object 
     16  cat15   300000 non-null  object 
     17  cat16   300000 non-null  object 
     18  cat17   300000 non-null  object 
     19  cat18   300000 non-null  object 
     20  cont0   300000 non-null  float64
     21  cont1   300000 non-null  float64
     22  cont2   300000 non-null  float64
     23  cont3   300000 non-null  float64
     24  cont4   300000 non-null  float64
     25  cont5   300000 non-null  float64
     26  cont6   300000 non-null  float64
     27  cont7   300000 non-null  float64
     28  cont8   300000 non-null  float64
     29  cont9   300000 non-null  float64
     30  cont10  300000 non-null  float64
     31  target  300000 non-null  int64  
    dtypes: float64(11), int64(2), object(19)
    memory usage: 73.2+ MB


#Data exploration

## Continuous Features

```
#adapted from @AndresHG 

graph_rows, graph_cols = 4, 3
fig, axes = plt.subplots(nrows=graph_rows, ncols=graph_cols, figsize=(12,12))
fig.suptitle('Distribution of Features', fontsize=16)

for index, column in enumerate(train[cont_columns].columns):
  i, j = (index // graph_cols, index % graph_cols)
  sns.kdeplot(train.loc[train.target == 0, column], color='m', shade=True, ax=axes[i,j])
  sns.kdeplot(train.loc[train.target == 1, column], color='b', shade=True, ax=axes[i,j])

fig.delaxes(axes[3,2])
plt.tight_layout(pad=1.10, rect=(0,0,1,.95))
plt.show()
```


![png](Kaggle_Tabular_Mar_2021_files/output_13_0.png)


```
#adapted from @AndresHG 

corr = train[cont_columns].corr().abs()
mask = np.triu(np.ones_like(corr, dtype=np.bool))

fig, ax = plt.subplots(figsize=(10, 10))

fig.suptitle('Feature Correlation Heatmap', fontsize=16)

# plot heatmap
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
            cbar_kws={"shrink": .8}, vmin=0, vmax=1)
# yticks
plt.yticks(rotation=0)

plt.tight_layout(pad=1.10, rect=(0,0,1,.99))

plt.show()


```


![png](Kaggle_Tabular_Mar_2021_files/output_14_0.png)


```
#adapted from @dwin183287 via @AndresHG 

background_color = "#f6f5f5"

fig = plt.figure(figsize=(12, 8), facecolor=background_color)
gs = fig.add_gridspec(1, 1)
ax0 = fig.add_subplot(gs[0, 0])

ax0.set_facecolor(background_color)
ax0.text(-1.1, 0.26, 'Correlation of Continuous Features with Target',
         fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-1.1, 0.24, 'No large feature - target correlation',
         fontsize=13, fontweight='light', fontfamily='serif')

chart_df = pd.DataFrame(train[cont_columns].corrwith(train['target']))
chart_df.columns = ['corr']
sns.barplot(x=chart_df.index, y=chart_df['corr'], ax=ax0, color='blue',
            zorder=3, edgecolor='black', linewidth=1.5)
ax0.grid(which='major', axis='y', zorder=0, color='gray', linestyle=':', dashes=(5,10))
ax0.set_ylabel('')

for s in ["top","right", 'left']:
    ax0.spines[s].set_visible(False)

plt.show()
```


![png](Kaggle_Tabular_Mar_2021_files/output_15_0.png)


```
chart_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>corr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>cont0</th>
      <td>-0.015172</td>
    </tr>
    <tr>
      <th>cont1</th>
      <td>0.164655</td>
    </tr>
    <tr>
      <th>cont2</th>
      <td>0.140459</td>
    </tr>
    <tr>
      <th>cont3</th>
      <td>-0.148316</td>
    </tr>
    <tr>
      <th>cont4</th>
      <td>-0.075585</td>
    </tr>
    <tr>
      <th>cont5</th>
      <td>0.215184</td>
    </tr>
    <tr>
      <th>cont6</th>
      <td>0.189832</td>
    </tr>
    <tr>
      <th>cont7</th>
      <td>-0.040646</td>
    </tr>
    <tr>
      <th>cont8</th>
      <td>0.183726</td>
    </tr>
    <tr>
      <th>cont9</th>
      <td>0.059242</td>
    </tr>
    <tr>
      <th>cont10</th>
      <td>-0.047077</td>
    </tr>
  </tbody>
</table>
</div>



## Categorical Features

```
for cat_col in cat_columns:
  print(train[cat_col].value_counts())
```

    A    223525
    B     76475
    Name: cat0, dtype: int64
    I    90809
    F    43818
    K    41870
    L    31891
    H    17257
    N    13231
    M    11354
    G    11248
    A    10547
    J    10036
    O     8740
    B     6847
    C     1703
    D      414
    E      235
    Name: cat1, dtype: int64
    A    168694
    C     38875
    D     22720
    G     18225
    Q     10901
    F      9877
    J      9102
    M      8068
    I      5287
    L      3997
    O      2749
    N       340
    H       219
    B       218
    S       197
    U       166
    R       129
    K       126
    E       110
    Name: cat2, dtype: int64
    A    187251
    B     79951
    C     15957
    D      8676
    E      3318
    F      2489
    K       846
    G       372
    L       292
    J       286
    H       274
    I       177
    N       111
    Name: cat3, dtype: int64
    E    129385
    F     76678
    G     30754
    D     27919
    H     23388
    J      4307
    I      3241
    K      1481
    M       547
    C       506
    O       330
    B       301
    S       285
    T       215
    L       214
    Q       117
    P       100
    A        92
    N        81
    R        59
    Name: cat4, dtype: int64
    BI    238563
    AB     41639
    BU      6740
    K       2713
    G        683
           ...  
    ZZ        25
    B         24
    BP        19
    AG        19
    CB        18
    Name: cat5, Length: 84, dtype: int64
    A    187896
    C     71427
    E     16581
    G     11198
    I      6648
    M      2182
    K      1552
    O       673
    S       583
    F       312
    D       214
    Y       212
    B       172
    U       155
    Q       124
    W        71
    Name: cat6, dtype: int64
    AH    45818
    E     39601
    AS    25326
    J     16135
    AN    16097
    U     15674
    N     14983
    AF    11455
    AK     9697
    AV     7958
    S      7921
    AI     7668
    A      6432
    K      6264
    Y      5896
    G      5656
    F      5550
    AW     5322
    C      4324
    AA     3692
    R      2991
    AX     2635
    O      2538
    AP     2311
    AD     1826
    V      1798
    AY     1749
    AO     1734
    AG     1706
    H      1692
    AL     1572
    W      1525
    B      1408
    Q      1306
    AM     1183
    AR     1121
    L       941
    AT      918
    M       907
    D       835
    BA      781
    AU      671
    X       662
    AC      660
    I       640
    P       611
    AB      539
    AE      458
    T       379
    AJ      229
    AQ      205
    Name: cat7, dtype: int64
    BM    42380
    AE    24442
    AX    22129
    Y     20864
    H     15561
          ...  
    AQ       69
    T        67
    B        57
    AC       57
    AR       33
    Name: cat8, Length: 61, dtype: int64
    A    201945
    E     33046
    C     23360
    F     14371
    J      8982
    I      7931
    N      4785
    L      2957
    R       862
    V       360
    B       280
    G       214
    Q       211
    D       189
    W       125
    O       122
    U       101
    X        99
    S        60
    Name: cat9, dtype: int64
    DJ    31584
    HK    30998
    DP    23679
    GS    16619
    CR    14382
          ...  
    LK        1
    ML        1
    GH        1
    AW        1
    MW        1
    Name: cat10, Length: 299, dtype: int64
    A    258932
    B     41068
    Name: cat11, dtype: int64
    A    257139
    B     42861
    Name: cat12, dtype: int64
    A    292712
    B      7288
    Name: cat13, dtype: int64
    A    160166
    B    139834
    Name: cat14, dtype: int64
    B    203574
    D     83188
    A     11072
    C      2166
    Name: cat15, dtype: int64
    D    206906
    B     84541
    C      5369
    A      3184
    Name: cat16, dtype: int64
    D    247125
    B     26136
    C     25325
    A      1414
    Name: cat17, dtype: int64
    B    255482
    D     22394
    C     21414
    A       710
    Name: cat18, dtype: int64


```
#adapted from @AndresHG 

train_0 = train.loc[train['target']==0]
train_1 = train.loc[train['target']==1]

graph_rows, graph_cols = 10,2
fig = plotly.subplots.make_subplots(rows=graph_rows, cols=graph_cols)

for index, feature in enumerate(cat_columns):
  i,j = ((index // graph_cols)+1, (index % graph_cols)+1)
  data = train_0.groupby(feature)[feature].count().sort_values(ascending=False)
  data = data if len(data) < 10 else data[:10]
  fig.add_trace(  go.Bar(x = data.index,
                          y = data.values,
                          name='Label: 0',),
                row=i, col=j)

  data = train_1.groupby(feature)[feature].count().sort_values(ascending=False)
  data = data if len(data) < 10 else data[:10]
  fig.add_trace(  go.Bar(x = data.index,
                          y = data.values,
                          name='Label: 1'),
                row=i, col=j)
  
  fig.update_xaxes(title=feature, row=i, col=j)
  fig.update_layout(barmode='stack')

fig.update_layout(
    autosize=False,
    width=1000,
    height=1600,
    showlegend=False,
)

fig.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>
                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>    
            <div id="e888b39c-3164-4dd9-ae05-eeeb85d0e335" class="plotly-graph-div" style="height:1600px; width:1000px;"></div>
            <script type="text/javascript">

                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("e888b39c-3164-4dd9-ae05-eeeb85d0e335")) {
                    Plotly.newPlot(
                        'e888b39c-3164-4dd9-ae05-eeeb85d0e335',
                        [{"name": "Label: 0", "type": "bar", "x": ["A", "B"], "xaxis": "x", "y": [148852, 71687], "yaxis": "y"}, {"name": "Label: 1", "type": "bar", "x": ["A", "B"], "xaxis": "x", "y": [74673, 4788], "yaxis": "y"}, {"name": "Label: 0", "type": "bar", "x": ["I", "F", "K", "L", "A", "N", "H", "J", "M", "O"], "xaxis": "x2", "y": [78807, 35110, 31846, 15654, 10081, 8614, 8573, 8478, 7599, 5203], "yaxis": "y2"}, {"name": "Label: 1", "type": "bar", "x": ["L", "I", "K", "F", "H", "G", "N", "M", "O", "B"], "xaxis": "x2", "y": [16237, 12002, 10024, 8708, 8684, 6451, 4617, 3755, 3537, 2699], "yaxis": "y2"}, {"name": "Label: 0", "type": "bar", "x": ["A", "C", "D", "G", "F", "J", "I", "M", "Q", "L"], "xaxis": "x3", "y": [133307, 32743, 18020, 12564, 6016, 5407, 3029, 2811, 2733, 1978], "yaxis": "y3"}, {"name": "Label: 1", "type": "bar", "x": ["A", "Q", "C", "G", "M", "D", "F", "J", "I", "O"], "xaxis": "x3", "y": [35387, 8168, 6132, 5661, 5257, 4700, 3861, 3695, 2258, 2024], "yaxis": "y3"}, {"name": "Label: 0", "type": "bar", "x": ["A", "B", "C", "D", "E", "F", "H", "J", "G", "K"], "xaxis": "x4", "y": [139026, 58620, 10841, 7074, 2438, 1526, 197, 196, 177, 158], "yaxis": "y4"}, {"name": "Label: 1", "type": "bar", "x": ["A", "B", "C", "D", "F", "E", "K", "G", "L", "N"], "xaxis": "x4", "y": [48225, 21331, 5116, 1602, 963, 880, 688, 195, 144, 93], "yaxis": "y4"}, {"name": "Label: 0", "type": "bar", "x": ["E", "F", "D", "G", "H", "J", "K", "I", "C", "M"], "xaxis": "x5", "y": [105953, 55373, 21762, 20310, 11742, 2101, 785, 700, 429, 326], "yaxis": "y5"}, {"name": "Label: 1", "type": "bar", "x": ["E", "F", "H", "G", "D", "I", "J", "K", "M", "O"], "xaxis": "x5", "y": [23432, 21305, 11646, 10444, 6157, 2541, 2206, 696, 221, 219], "yaxis": "y5"}, {"name": "Label: 0", "type": "bar", "x": ["BI", "AB", "BU", "K", "G", "BQ", "N", "CL", "BO", "AY"], "xaxis": "x6", "y": [175558, 28523, 5008, 2109, 653, 455, 431, 313, 223, 221], "yaxis": "y6"}, {"name": "Label: 1", "type": "bar", "x": ["BI", "AB", "BU", "K", "AL", "I", "BC", "R", "G", "AT"], "xaxis": "x6", "y": [63005, 13116, 1732, 604, 57, 51, 34, 32, 30, 29], "yaxis": "y6"}, {"name": "Label: 0", "type": "bar", "x": ["A", "C", "E", "G", "I", "K", "M", "S", "O", "F"], "xaxis": "x7", "y": [147717, 48720, 11410, 5894, 3196, 1318, 671, 367, 286, 225], "yaxis": "y7"}, {"name": "Label: 1", "type": "bar", "x": ["A", "C", "G", "E", "I", "M", "O", "K", "S", "F"], "xaxis": "x7", "y": [40179, 22707, 5304, 5171, 3452, 1511, 387, 234, 216, 87], "yaxis": "y7"}, {"name": "Label: 0", "type": "bar", "x": ["E", "AH", "AS", "U", "N", "AN", "J", "AK", "AV", "AI"], "xaxis": "x8", "y": [32474, 29056, 16452, 13077, 12452, 12434, 11082, 8302, 6640, 6440], "yaxis": "y8"}, {"name": "Label: 1", "type": "bar", "x": ["AH", "AS", "E", "AF", "J", "AN", "C", "U", "N", "A"], "xaxis": "x8", "y": [16762, 8874, 7127, 7112, 5053, 3663, 2602, 2597, 2531, 2221], "yaxis": "y8"}, {"name": "Label: 0", "type": "bar", "x": ["BM", "AE", "Y", "AX", "S", "L", "AD", "X", "H", "AT"], "xaxis": "x9", "y": [33136, 19655, 16885, 16865, 12272, 11603, 10962, 10062, 9345, 8697], "yaxis": "y9"}, {"name": "Label: 1", "type": "bar", "x": ["BM", "K", "H", "AX", "AE", "AT", "X", "Y", "AD", "S"], "xaxis": "x9", "y": [9244, 8734, 6216, 5264, 4787, 4519, 4157, 3979, 3701, 2719], "yaxis": "y9"}, {"name": "Label: 0", "type": "bar", "x": ["A", "E", "C", "F", "J", "I", "N", "L", "R", "V"], "xaxis": "x10", "y": [138767, 29518, 18693, 12164, 7439, 5906, 3412, 2594, 668, 219], "yaxis": "y10"}, {"name": "Label: 1", "type": "bar", "x": ["A", "C", "E", "F", "I", "J", "N", "L", "R", "V"], "xaxis": "x10", "y": [63178, 4667, 3528, 2207, 2025, 1543, 1373, 363, 194, 141], "yaxis": "y10"}, {"name": "Label: 0", "type": "bar", "x": ["DJ", "HK", "DP", "GS", "CR", "HX", "CK", "DC", "LN", "HQ"], "xaxis": "x11", "y": [24072, 20432, 17774, 13726, 13415, 10364, 7680, 6396, 6144, 6095], "yaxis": "y11"}, {"name": "Label: 1", "type": "bar", "x": ["HK", "DJ", "DP", "DC", "HQ", "CK", "GS", "HX", "MD", "LF"], "xaxis": "x11", "y": [10566, 7512, 5905, 3887, 3485, 2907, 2893, 2807, 2621, 2095], "yaxis": "y11"}, {"name": "Label: 0", "type": "bar", "x": ["A", "B"], "xaxis": "x12", "y": [203340, 17199], "yaxis": "y12"}, {"name": "Label: 1", "type": "bar", "x": ["A", "B"], "xaxis": "x12", "y": [55592, 23869], "yaxis": "y12"}, {"name": "Label: 0", "type": "bar", "x": ["A", "B"], "xaxis": "x13", "y": [185179, 35360], "yaxis": "y13"}, {"name": "Label: 1", "type": "bar", "x": ["A", "B"], "xaxis": "x13", "y": [71960, 7501], "yaxis": "y13"}, {"name": "Label: 0", "type": "bar", "x": ["A", "B"], "xaxis": "x14", "y": [219374, 1165], "yaxis": "y14"}, {"name": "Label: 1", "type": "bar", "x": ["A", "B"], "xaxis": "x14", "y": [73338, 6123], "yaxis": "y14"}, {"name": "Label: 0", "type": "bar", "x": ["A", "B"], "xaxis": "x15", "y": [137706, 82833], "yaxis": "y15"}, {"name": "Label: 1", "type": "bar", "x": ["B", "A"], "xaxis": "x15", "y": [57001, 22460], "yaxis": "y15"}, {"name": "Label: 0", "type": "bar", "x": ["B", "D", "A", "C"], "xaxis": "x16", "y": [176567, 33439, 8954, 1579], "yaxis": "y16"}, {"name": "Label: 1", "type": "bar", "x": ["D", "B", "A", "C"], "xaxis": "x16", "y": [49749, 27007, 2118, 587], "yaxis": "y16"}, {"name": "Label: 0", "type": "bar", "x": ["D", "B", "C", "A"], "xaxis": "x17", "y": [183031, 31016, 4003, 2489], "yaxis": "y17"}, {"name": "Label: 1", "type": "bar", "x": ["B", "D", "C", "A"], "xaxis": "x17", "y": [53525, 23875, 1366, 695], "yaxis": "y17"}, {"name": "Label: 0", "type": "bar", "x": ["D", "B", "C", "A"], "xaxis": "x18", "y": [195154, 14356, 9875, 1154], "yaxis": "y18"}, {"name": "Label: 1", "type": "bar", "x": ["D", "C", "B", "A"], "xaxis": "x18", "y": [51971, 15450, 11780, 260], "yaxis": "y18"}, {"name": "Label: 0", "type": "bar", "x": ["B", "C", "D", "A"], "xaxis": "x19", "y": [207067, 6880, 6037, 555], "yaxis": "y19"}, {"name": "Label: 1", "type": "bar", "x": ["B", "D", "C", "A"], "xaxis": "x19", "y": [48415, 16357, 14534, 155], "yaxis": "y19"}],
                        {"autosize": false, "barmode": "stack", "height": 1600, "showlegend": false, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "width": 1000, "xaxis": {"anchor": "y", "domain": [0.0, 0.45], "title": {"text": "cat0"}}, "xaxis10": {"anchor": "y10", "domain": [0.55, 1.0], "title": {"text": "cat9"}}, "xaxis11": {"anchor": "y11", "domain": [0.0, 0.45], "title": {"text": "cat10"}}, "xaxis12": {"anchor": "y12", "domain": [0.55, 1.0], "title": {"text": "cat11"}}, "xaxis13": {"anchor": "y13", "domain": [0.0, 0.45], "title": {"text": "cat12"}}, "xaxis14": {"anchor": "y14", "domain": [0.55, 1.0], "title": {"text": "cat13"}}, "xaxis15": {"anchor": "y15", "domain": [0.0, 0.45], "title": {"text": "cat14"}}, "xaxis16": {"anchor": "y16", "domain": [0.55, 1.0], "title": {"text": "cat15"}}, "xaxis17": {"anchor": "y17", "domain": [0.0, 0.45], "title": {"text": "cat16"}}, "xaxis18": {"anchor": "y18", "domain": [0.55, 1.0], "title": {"text": "cat17"}}, "xaxis19": {"anchor": "y19", "domain": [0.0, 0.45], "title": {"text": "cat18"}}, "xaxis2": {"anchor": "y2", "domain": [0.55, 1.0], "title": {"text": "cat1"}}, "xaxis20": {"anchor": "y20", "domain": [0.55, 1.0]}, "xaxis3": {"anchor": "y3", "domain": [0.0, 0.45], "title": {"text": "cat2"}}, "xaxis4": {"anchor": "y4", "domain": [0.55, 1.0], "title": {"text": "cat3"}}, "xaxis5": {"anchor": "y5", "domain": [0.0, 0.45], "title": {"text": "cat4"}}, "xaxis6": {"anchor": "y6", "domain": [0.55, 1.0], "title": {"text": "cat5"}}, "xaxis7": {"anchor": "y7", "domain": [0.0, 0.45], "title": {"text": "cat6"}}, "xaxis8": {"anchor": "y8", "domain": [0.55, 1.0], "title": {"text": "cat7"}}, "xaxis9": {"anchor": "y9", "domain": [0.0, 0.45], "title": {"text": "cat8"}}, "yaxis": {"anchor": "x", "domain": [0.9269999999999999, 0.9999999999999999]}, "yaxis10": {"anchor": "x10", "domain": [0.515, 0.588]}, "yaxis11": {"anchor": "x11", "domain": [0.412, 0.485]}, "yaxis12": {"anchor": "x12", "domain": [0.412, 0.485]}, "yaxis13": {"anchor": "x13", "domain": [0.30899999999999994, 0.38199999999999995]}, "yaxis14": {"anchor": "x14", "domain": [0.30899999999999994, 0.38199999999999995]}, "yaxis15": {"anchor": "x15", "domain": [0.206, 0.27899999999999997]}, "yaxis16": {"anchor": "x16", "domain": [0.206, 0.27899999999999997]}, "yaxis17": {"anchor": "x17", "domain": [0.103, 0.176]}, "yaxis18": {"anchor": "x18", "domain": [0.103, 0.176]}, "yaxis19": {"anchor": "x19", "domain": [0.0, 0.073]}, "yaxis2": {"anchor": "x2", "domain": [0.9269999999999999, 0.9999999999999999]}, "yaxis20": {"anchor": "x20", "domain": [0.0, 0.073]}, "yaxis3": {"anchor": "x3", "domain": [0.824, 0.8969999999999999]}, "yaxis4": {"anchor": "x4", "domain": [0.824, 0.8969999999999999]}, "yaxis5": {"anchor": "x5", "domain": [0.721, 0.7939999999999999]}, "yaxis6": {"anchor": "x6", "domain": [0.721, 0.7939999999999999]}, "yaxis7": {"anchor": "x7", "domain": [0.618, 0.691]}, "yaxis8": {"anchor": "x8", "domain": [0.618, 0.691]}, "yaxis9": {"anchor": "x9", "domain": [0.515, 0.588]}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('e888b39c-3164-4dd9-ae05-eeeb85d0e335');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };

            </script>
        </div>
</body>
</html>


```
#adapted from @AndresHG 

num_rows, num_cols = 10,1
fig = plotly.subplots.make_subplots(rows=num_rows, cols=num_cols)
cont = 1

for index, feature in enumerate(cat_columns):
  data = train_0.groupby(feature)[feature].count().sort_values(ascending=False) #find values of feature for target 0
  if len(data) < 10:    #if less than 10, skip
      continue
  data = data if len(data) < 25 else data[:25]   #if greater than 25, choose largest 25
  i,j = (cont, 1)   #set place in graphing
  cont+=1
  
  fig.add_trace(  go.Bar(x = data.index,
                          y = data.values,
                          name='Label: 0',),
                row=i, col=j)
  
  target_0_values = set(data.index)  #make a set of all the categories which have target 0
  
  data = train_1.groupby(feature)[feature].count().sort_values(ascending=False)
  data = data if len(data) < 25 else data[:25]
  
  fig.add_trace(  go.Bar(x = data.index,
                          y = data.values,
                          name='Label: 1'),
                row=i, col=j)
  
  target_1_values = set(data.index)   #make a set of all the categories which have target 1
  
  print('----------------------{}----------------------'.format(feature))
  print('Unique values for class 0: {}'.format(target_0_values - target_1_values))
  print('Unique values for class 1: {}'.format(target_1_values - target_0_values))
  
  fig.update_xaxes(title=feature, row=i, col=j)
  fig.update_layout(barmode='stack')
    
fig.update_layout(
    autosize=False,
    width=900,
    height=2000,
    showlegend=False,
)
fig.show()
```

    ----------------------cat1----------------------
    Unique values for class 0: set()
    Unique values for class 1: set()
    ----------------------cat2----------------------
    Unique values for class 0: set()
    Unique values for class 1: set()
    ----------------------cat3----------------------
    Unique values for class 0: set()
    Unique values for class 1: set()
    ----------------------cat4----------------------
    Unique values for class 0: set()
    Unique values for class 1: set()
    ----------------------cat5----------------------
    Unique values for class 0: {'BO', 'CI', 'BK', 'BV', 'N', 'AY', 'T', 'BG', 'CA', 'BS'}
    Unique values for class 1: {'BL', 'AH', 'AM', 'AQ', 'C', 'X', 'CK', 'F', 'BJ', 'I'}
    ----------------------cat6----------------------
    Unique values for class 0: set()
    Unique values for class 1: set()
    ----------------------cat7----------------------
    Unique values for class 0: {'Y', 'AA', 'R', 'O', 'AY', 'AP'}
    Unique values for class 1: {'V', 'L', 'AD', 'AL', 'BA', 'AC'}
    ----------------------cat8----------------------
    Unique values for class 0: {'I', 'AS', 'BJ', 'AW', 'BC'}
    Unique values for class 1: {'BD', 'AO', 'AG', 'F', 'A'}
    ----------------------cat9----------------------
    Unique values for class 0: set()
    Unique values for class 1: set()
    ----------------------cat10----------------------
    Unique values for class 0: {'EK', 'LN', 'HJ', 'DF', 'GE', 'IG', 'HB'}
    Unique values for class 1: {'GK', 'FR', 'HC', 'GI', 'CD', 'JR', 'MC'}



<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>
                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>    
            <div id="adf46dbe-617c-40c4-82a4-50bd6c14fe39" class="plotly-graph-div" style="height:2000px; width:900px;"></div>
            <script type="text/javascript">

                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("adf46dbe-617c-40c4-82a4-50bd6c14fe39")) {
                    Plotly.newPlot(
                        'adf46dbe-617c-40c4-82a4-50bd6c14fe39',
                        [{"name": "Label: 0", "type": "bar", "x": ["I", "F", "K", "L", "A", "N", "H", "J", "M", "O", "G", "B", "C", "D", "E"], "xaxis": "x", "y": [78807, 35110, 31846, 15654, 10081, 8614, 8573, 8478, 7599, 5203, 4797, 4148, 1265, 226, 138], "yaxis": "y"}, {"name": "Label: 1", "type": "bar", "x": ["L", "I", "K", "F", "H", "G", "N", "M", "O", "B", "J", "A", "C", "D", "E"], "xaxis": "x", "y": [16237, 12002, 10024, 8708, 8684, 6451, 4617, 3755, 3537, 2699, 1558, 466, 438, 188, 97], "yaxis": "y"}, {"name": "Label: 0", "type": "bar", "x": ["A", "C", "D", "G", "F", "J", "I", "M", "Q", "L", "O", "N", "B", "H", "U", "S", "R", "E", "K"], "xaxis": "x2", "y": [133307, 32743, 18020, 12564, 6016, 5407, 3029, 2811, 2733, 1978, 725, 276, 200, 160, 154, 134, 95, 94, 93], "yaxis": "y2"}, {"name": "Label: 1", "type": "bar", "x": ["A", "Q", "C", "G", "M", "D", "F", "J", "I", "O", "L", "N", "S", "H", "R", "K", "B", "E", "U"], "xaxis": "x2", "y": [35387, 8168, 6132, 5661, 5257, 4700, 3861, 3695, 2258, 2024, 2019, 64, 63, 59, 34, 33, 18, 16, 12], "yaxis": "y2"}, {"name": "Label: 0", "type": "bar", "x": ["A", "B", "C", "D", "E", "F", "H", "J", "G", "K", "L", "I", "N"], "xaxis": "x3", "y": [139026, 58620, 10841, 7074, 2438, 1526, 197, 196, 177, 158, 148, 120, 18], "yaxis": "y3"}, {"name": "Label: 1", "type": "bar", "x": ["A", "B", "C", "D", "F", "E", "K", "G", "L", "N", "J", "H", "I"], "xaxis": "x3", "y": [48225, 21331, 5116, 1602, 963, 880, 688, 195, 144, 93, 90, 77, 57], "yaxis": "y3"}, {"name": "Label: 0", "type": "bar", "x": ["E", "F", "D", "G", "H", "J", "K", "I", "C", "M", "S", "T", "B", "O", "L", "P", "Q", "N", "A", "R"], "xaxis": "x4", "y": [105953, 55373, 21762, 20310, 11742, 2101, 785, 700, 429, 326, 211, 180, 159, 111, 100, 80, 74, 62, 45, 36], "yaxis": "y4"}, {"name": "Label: 1", "type": "bar", "x": ["E", "F", "H", "G", "D", "I", "J", "K", "M", "O", "B", "L", "C", "S", "A", "Q", "T", "R", "P", "N"], "xaxis": "x4", "y": [23432, 21305, 11646, 10444, 6157, 2541, 2206, 696, 221, 219, 142, 114, 77, 74, 47, 43, 35, 23, 20, 19], "yaxis": "y4"}, {"name": "Label: 0", "type": "bar", "x": ["BI", "AB", "BU", "K", "G", "BQ", "N", "CL", "BO", "AY", "AL", "CI", "BA", "AW", "M", "T", "BG", "AT", "R", "BS", "BC", "BV", "CA", "BK", "L"], "xaxis": "x5", "y": [175558, 28523, 5008, 2109, 653, 455, 431, 313, 223, 221, 215, 202, 189, 184, 177, 176, 166, 165, 157, 150, 149, 148, 141, 138, 133], "yaxis": "y5"}, {"name": "Label: 1", "type": "bar", "x": ["BI", "AB", "BU", "K", "AL", "I", "BC", "R", "G", "AT", "BQ", "BJ", "M", "L", "CL", "CK", "AH", "AW", "F", "BL", "X", "BA", "AQ", "AM", "C"], "xaxis": "x5", "y": [63005, 13116, 1732, 604, 57, 51, 34, 32, 30, 29, 28, 27, 26, 25, 23, 22, 22, 21, 21, 21, 20, 20, 19, 19, 17], "yaxis": "y5"}, {"name": "Label: 0", "type": "bar", "x": ["A", "C", "E", "G", "I", "K", "M", "S", "O", "F", "Y", "D", "B", "U", "Q", "W"], "xaxis": "x6", "y": [147717, 48720, 11410, 5894, 3196, 1318, 671, 367, 286, 225, 178, 169, 153, 120, 77, 38], "yaxis": "y6"}, {"name": "Label: 1", "type": "bar", "x": ["A", "C", "G", "E", "I", "M", "O", "K", "S", "F", "Q", "D", "U", "Y", "W", "B"], "xaxis": "x6", "y": [40179, 22707, 5304, 5171, 3452, 1511, 387, 234, 216, 87, 47, 45, 35, 34, 33, 19], "yaxis": "y6"}, {"name": "Label: 0", "type": "bar", "x": ["E", "AH", "AS", "U", "N", "AN", "J", "AK", "AV", "AI", "S", "Y", "F", "AW", "AF", "A", "K", "G", "AA", "R", "O", "AP", "AX", "C", "AY"], "xaxis": "x7", "y": [32474, 29056, 16452, 13077, 12452, 12434, 11082, 8302, 6640, 6440, 6104, 5569, 4863, 4499, 4343, 4211, 4100, 3709, 3380, 2704, 2171, 2016, 1819, 1722, 1585], "yaxis": "y7"}, {"name": "Label: 1", "type": "bar", "x": ["AH", "AS", "E", "AF", "J", "AN", "C", "U", "N", "A", "K", "G", "S", "AK", "AV", "AI", "AL", "AW", "AX", "V", "F", "BA", "AC", "AD", "L"], "xaxis": "x7", "y": [16762, 8874, 7127, 7112, 5053, 3663, 2602, 2597, 2531, 2221, 2164, 1947, 1817, 1395, 1318, 1228, 854, 823, 816, 786, 687, 594, 453, 417, 390], "yaxis": "y7"}, {"name": "Label: 0", "type": "bar", "x": ["BM", "AE", "Y", "AX", "S", "L", "AD", "X", "H", "AT", "N", "I", "BC", "AS", "K", "AN", "BJ", "AF", "Q", "AK", "BN", "AJ", "M", "AW", "J"], "xaxis": "x8", "y": [33136, 19655, 16885, 16865, 12272, 11603, 10962, 10062, 9345, 8697, 6687, 6382, 5078, 4568, 4376, 4162, 3646, 3401, 3345, 3111, 3046, 2350, 2113, 1955, 1879], "yaxis": "y8"}, {"name": "Label: 1", "type": "bar", "x": ["BM", "K", "H", "AX", "AE", "AT", "X", "Y", "AD", "S", "AF", "L", "A", "BD", "BN", "AG", "AN", "N", "F", "M", "AK", "AO", "Q", "AJ", "J"], "xaxis": "x8", "y": [9244, 8734, 6216, 5264, 4787, 4519, 4157, 3979, 3701, 2719, 2481, 1970, 1925, 1891, 1859, 1845, 1546, 1244, 1173, 996, 932, 844, 784, 711, 700], "yaxis": "y8"}, {"name": "Label: 0", "type": "bar", "x": ["A", "E", "C", "F", "J", "I", "N", "L", "R", "V", "G", "D", "B", "Q", "W", "O", "X", "U", "S"], "xaxis": "x9", "y": [138767, 29518, 18693, 12164, 7439, 5906, 3412, 2594, 668, 219, 193, 184, 172, 163, 118, 110, 90, 81, 48], "yaxis": "y9"}, {"name": "Label: 1", "type": "bar", "x": ["A", "C", "E", "F", "I", "J", "N", "L", "R", "V", "B", "Q", "G", "U", "O", "S", "X", "W", "D"], "xaxis": "x9", "y": [63178, 4667, 3528, 2207, 2025, 1543, 1373, 363, 194, 141, 108, 48, 21, 20, 12, 12, 9, 7, 5], "yaxis": "y9"}, {"name": "Label: 0", "type": "bar", "x": ["DJ", "HK", "DP", "GS", "CR", "HX", "CK", "DC", "LN", "HQ", "LM", "IE", "MD", "LF", "IG", "HB", "DF", "HG", "KW", "LB", "HJ", "EK", "HV", "LO", "GE"], "xaxis": "x10", "y": [24072, 20432, 17774, 13726, 13415, 10364, 7680, 6396, 6144, 6095, 5080, 4624, 4196, 4119, 3968, 3840, 3736, 3648, 3548, 3145, 2553, 2468, 2234, 2072, 1816], "yaxis": "y10"}, {"name": "Label: 1", "type": "bar", "x": ["HK", "DJ", "DP", "DC", "HQ", "CK", "GS", "HX", "MD", "LF", "KW", "IE", "HC", "CD", "HV", "LO", "HG", "GK", "CR", "LM", "JR", "FR", "GI", "MC", "LB"], "xaxis": "x10", "y": [10566, 7512, 5905, 3887, 3485, 2907, 2893, 2807, 2621, 2095, 1612, 1566, 1501, 1235, 1212, 1082, 1068, 987, 967, 837, 721, 714, 695, 653, 573], "yaxis": "y10"}],
                        {"autosize": false, "barmode": "stack", "height": 2000, "showlegend": false, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "width": 900, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": "cat1"}}, "xaxis10": {"anchor": "y10", "domain": [0.0, 1.0], "title": {"text": "cat10"}}, "xaxis2": {"anchor": "y2", "domain": [0.0, 1.0], "title": {"text": "cat2"}}, "xaxis3": {"anchor": "y3", "domain": [0.0, 1.0], "title": {"text": "cat3"}}, "xaxis4": {"anchor": "y4", "domain": [0.0, 1.0], "title": {"text": "cat4"}}, "xaxis5": {"anchor": "y5", "domain": [0.0, 1.0], "title": {"text": "cat5"}}, "xaxis6": {"anchor": "y6", "domain": [0.0, 1.0], "title": {"text": "cat6"}}, "xaxis7": {"anchor": "y7", "domain": [0.0, 1.0], "title": {"text": "cat7"}}, "xaxis8": {"anchor": "y8", "domain": [0.0, 1.0], "title": {"text": "cat8"}}, "xaxis9": {"anchor": "y9", "domain": [0.0, 1.0], "title": {"text": "cat9"}}, "yaxis": {"anchor": "x", "domain": [0.9269999999999999, 0.9999999999999999]}, "yaxis10": {"anchor": "x10", "domain": [0.0, 0.073]}, "yaxis2": {"anchor": "x2", "domain": [0.824, 0.8969999999999999]}, "yaxis3": {"anchor": "x3", "domain": [0.721, 0.7939999999999999]}, "yaxis4": {"anchor": "x4", "domain": [0.618, 0.691]}, "yaxis5": {"anchor": "x5", "domain": [0.515, 0.588]}, "yaxis6": {"anchor": "x6", "domain": [0.412, 0.485]}, "yaxis7": {"anchor": "x7", "domain": [0.30899999999999994, 0.38199999999999995]}, "yaxis8": {"anchor": "x8", "domain": [0.206, 0.27899999999999997]}, "yaxis9": {"anchor": "x9", "domain": [0.103, 0.176]}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('adf46dbe-617c-40c4-82a4-50bd6c14fe39');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };

            </script>
        </div>
</body>
</html>


```
def print_value_counts(cat):
  label0 = train_0[cat].value_counts()
  label1 = train_1[cat].value_counts()
  output = pd.DataFrame(data={'label 0': label0,
                              'label 1': label1,
                              'total' : train[cat].value_counts()})
  output['label1share'] = output['label 1'].div(output['total'], axis=0).round(2)
  print(output.sort_values('total', ascending=False))
  return output

```

```
#----------------------cat0----------------------

output = print_value_counts('cat0')
#Almost no label 1's in category B
```

       label 0  label 1   total  label1share
    A   148852    74673  223525         0.33
    B    71687     4788   76475         0.06


```
#----------------------cat1----------------------

output = print_value_counts('cat1')
#D and E are small but split evenly
```

       label 0  label 1  total  label1share
    I    78807    12002  90809         0.13
    F    35110     8708  43818         0.20
    K    31846    10024  41870         0.24
    L    15654    16237  31891         0.51
    H     8573     8684  17257         0.50
    N     8614     4617  13231         0.35
    M     7599     3755  11354         0.33
    G     4797     6451  11248         0.57
    A    10081      466  10547         0.04
    J     8478     1558  10036         0.16
    O     5203     3537   8740         0.40
    B     4148     2699   6847         0.39
    C     1265      438   1703         0.26
    D      226      188    414         0.45
    E      138       97    235         0.41


```
#----------------------cat2----------------------

output = print_value_counts('cat2')
#'N', 'H', 'B', 'S', 'U', 'R', 'K', 'E' all have less than 400 representation, average total label share
# Could possibly combine, but portions aren't all that similar
```

       label 0  label 1   total  label1share
    A   133307    35387  168694         0.21
    C    32743     6132   38875         0.16
    D    18020     4700   22720         0.21
    G    12564     5661   18225         0.31
    Q     2733     8168   10901         0.75
    F     6016     3861    9877         0.39
    J     5407     3695    9102         0.41
    M     2811     5257    8068         0.65
    I     3029     2258    5287         0.43
    L     1978     2019    3997         0.51
    O      725     2024    2749         0.74
    N      276       64     340         0.19
    H      160       59     219         0.27
    B      200       18     218         0.08
    S      134       63     197         0.32
    U      154       12     166         0.07
    R       95       34     129         0.26
    K       93       33     126         0.26
    E       94       16     110         0.15


```
output.loc['subtotal'] = output.loc[['N', 'H', 'B', 'S', 'U', 'R', 'K', 'E']].sum(axis=0)[['label 0', 'label 1', 'total']]
output.loc['subtotal', 'label1share'] = output.loc['subtotal', 'label 1'] / output.loc['subtotal', 'total']
output.loc[['N', 'H', 'B', 'S', 'U', 'R', 'K', 'E', 'subtotal']]

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label 0</th>
      <th>label 1</th>
      <th>total</th>
      <th>label1share</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>N</th>
      <td>276.0</td>
      <td>64.0</td>
      <td>340.0</td>
      <td>0.190000</td>
    </tr>
    <tr>
      <th>H</th>
      <td>160.0</td>
      <td>59.0</td>
      <td>219.0</td>
      <td>0.270000</td>
    </tr>
    <tr>
      <th>B</th>
      <td>200.0</td>
      <td>18.0</td>
      <td>218.0</td>
      <td>0.080000</td>
    </tr>
    <tr>
      <th>S</th>
      <td>134.0</td>
      <td>63.0</td>
      <td>197.0</td>
      <td>0.320000</td>
    </tr>
    <tr>
      <th>U</th>
      <td>154.0</td>
      <td>12.0</td>
      <td>166.0</td>
      <td>0.070000</td>
    </tr>
    <tr>
      <th>R</th>
      <td>95.0</td>
      <td>34.0</td>
      <td>129.0</td>
      <td>0.260000</td>
    </tr>
    <tr>
      <th>K</th>
      <td>93.0</td>
      <td>33.0</td>
      <td>126.0</td>
      <td>0.260000</td>
    </tr>
    <tr>
      <th>E</th>
      <td>94.0</td>
      <td>16.0</td>
      <td>110.0</td>
      <td>0.150000</td>
    </tr>
    <tr>
      <th>subtotal</th>
      <td>1206.0</td>
      <td>299.0</td>
      <td>1505.0</td>
      <td>0.198671</td>
    </tr>
  </tbody>
</table>
</div>



```
#----------------------cat3----------------------

output = print_value_counts('cat3')
#'G', 'L', 'J', 'H', 'I', 'N' all have less than 400 representation, fairly split between classes
```

       label 0  label 1   total  label1share
    A   139026    48225  187251         0.26
    B    58620    21331   79951         0.27
    C    10841     5116   15957         0.32
    D     7074     1602    8676         0.18
    E     2438      880    3318         0.27
    F     1526      963    2489         0.39
    K      158      688     846         0.81
    G      177      195     372         0.52
    L      148      144     292         0.49
    J      196       90     286         0.31
    H      197       77     274         0.28
    I      120       57     177         0.32
    N       18       93     111         0.84


```
output.loc['subtotal'] = output.loc[['G', 'L', 'J', 'H', 'I', 'N']].sum(axis=0)[['label 0', 'label 1', 'total']]
output.loc['subtotal', 'label1share'] = output.loc['subtotal', 'label 1'] / output.loc['subtotal', 'total']
print(output.loc[['G', 'L', 'J', 'H', 'I', 'N', 'subtotal']])

```

              label 0  label 1   total  label1share
    G           177.0    195.0   372.0     0.520000
    L           148.0    144.0   292.0     0.490000
    J           196.0     90.0   286.0     0.310000
    H           197.0     77.0   274.0     0.280000
    I           120.0     57.0   177.0     0.320000
    N            18.0     93.0   111.0     0.840000
    subtotal    856.0    656.0  1512.0     0.433862


```
#----------------------cat4----------------------

output = print_value_counts('cat4')
#'O', 'B', 'S', 'T', 'L', 'Q', 'P', 'A', 'N', 'R' are below 400
```

       label 0  label 1   total  label1share
    E   105953    23432  129385         0.18
    F    55373    21305   76678         0.28
    G    20310    10444   30754         0.34
    D    21762     6157   27919         0.22
    H    11742    11646   23388         0.50
    J     2101     2206    4307         0.51
    I      700     2541    3241         0.78
    K      785      696    1481         0.47
    M      326      221     547         0.40
    C      429       77     506         0.15
    O      111      219     330         0.66
    B      159      142     301         0.47
    S      211       74     285         0.26
    T      180       35     215         0.16
    L      100      114     214         0.53
    Q       74       43     117         0.37
    P       80       20     100         0.20
    A       45       47      92         0.51
    N       62       19      81         0.23
    R       36       23      59         0.39


```
output.loc[output['total'] < 400].sort_values('total', ascending=False).index
```




    Index(['O', 'B', 'S', 'T', 'L', 'Q', 'P', 'A', 'N', 'R'], dtype='object')



```
#----------------------cat5----------------------
#Unique values for class 0: {'BG', 'BK', 'BO', 'BV', 'T', 'CI', 'CA', 'BS', 'AY', 'N'}
#Unique values for class 1: {'CK', 'I', 'AH', 'F', 'BJ', 'C', 'AQ', 'AM', 'BL', 'X'}

output = print_value_counts('cat5')
#'BI', 'AB', 'BU', 'K', 'G', 'BQ', 'N' are the only ones above 400
#Interesting that G, BQ, N, CL and BO have almost no positive representation
```

        label 0  label 1   total  label1share
    BI   175558  63005.0  238563         0.26
    AB    28523  13116.0   41639         0.31
    BU     5008   1732.0    6740         0.26
    K      2109    604.0    2713         0.22
    G       653     30.0     683         0.04
    ..      ...      ...     ...          ...
    ZZ       25      NaN      25          NaN
    B        24      NaN      24          NaN
    BP       19      NaN      19          NaN
    AG       19      NaN      19          NaN
    CB       18      NaN      18          NaN
    
    [84 rows x 4 columns]


```
output.sort_values('total', ascending=False).iloc[:10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label 0</th>
      <th>label 1</th>
      <th>total</th>
      <th>label1share</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BI</th>
      <td>175558</td>
      <td>63005.0</td>
      <td>238563</td>
      <td>0.26</td>
    </tr>
    <tr>
      <th>AB</th>
      <td>28523</td>
      <td>13116.0</td>
      <td>41639</td>
      <td>0.31</td>
    </tr>
    <tr>
      <th>BU</th>
      <td>5008</td>
      <td>1732.0</td>
      <td>6740</td>
      <td>0.26</td>
    </tr>
    <tr>
      <th>K</th>
      <td>2109</td>
      <td>604.0</td>
      <td>2713</td>
      <td>0.22</td>
    </tr>
    <tr>
      <th>G</th>
      <td>653</td>
      <td>30.0</td>
      <td>683</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>BQ</th>
      <td>455</td>
      <td>28.0</td>
      <td>483</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>N</th>
      <td>431</td>
      <td>16.0</td>
      <td>447</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>CL</th>
      <td>313</td>
      <td>23.0</td>
      <td>336</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>AL</th>
      <td>215</td>
      <td>57.0</td>
      <td>272</td>
      <td>0.21</td>
    </tr>
    <tr>
      <th>BO</th>
      <td>223</td>
      <td>16.0</td>
      <td>239</td>
      <td>0.07</td>
    </tr>
  </tbody>
</table>
</div>



```
output.loc[output['total'] > 400].sort_values('total', ascending=False).index
```




    Index(['BI', 'AB', 'BU', 'K', 'G', 'BQ', 'N'], dtype='object')



```
#----------------------cat6----------------------

output = print_value_counts('cat6')
#'F', 'D', 'Y', 'B', 'U', 'Q', 'W' are less than 400
```

       label 0  label 1   total  label1share
    A   147717    40179  187896         0.21
    C    48720    22707   71427         0.32
    E    11410     5171   16581         0.31
    G     5894     5304   11198         0.47
    I     3196     3452    6648         0.52
    M      671     1511    2182         0.69
    K     1318      234    1552         0.15
    O      286      387     673         0.58
    S      367      216     583         0.37
    F      225       87     312         0.28
    D      169       45     214         0.21
    Y      178       34     212         0.16
    B      153       19     172         0.11
    U      120       35     155         0.23
    Q       77       47     124         0.38
    W       38       33      71         0.46


```
output.loc[output['total'] < 400].sort_values('total', ascending=False).index
```




    Index(['F', 'D', 'Y', 'B', 'U', 'Q', 'W'], dtype='object')



```
#----------------------cat7----------------------

#Unique values for class 0: {'O', 'R', 'AP', 'AA', 'Y', 'AY'}
#Unique values for class 1: {'BA', 'AL', 'AD', 'AC', 'L', 'V'}

output = print_value_counts('cat7')
```

        label 0  label 1  total  label1share
    AH    29056    16762  45818         0.37
    E     32474     7127  39601         0.18
    AS    16452     8874  25326         0.35
    J     11082     5053  16135         0.31
    AN    12434     3663  16097         0.23
    U     13077     2597  15674         0.17
    N     12452     2531  14983         0.17
    AF     4343     7112  11455         0.62
    AK     8302     1395   9697         0.14
    AV     6640     1318   7958         0.17
    S      6104     1817   7921         0.23
    AI     6440     1228   7668         0.16
    A      4211     2221   6432         0.35
    K      4100     2164   6264         0.35
    Y      5569      327   5896         0.06
    G      3709     1947   5656         0.34
    F      4863      687   5550         0.12
    AW     4499      823   5322         0.15
    C      1722     2602   4324         0.60
    AA     3380      312   3692         0.08
    R      2704      287   2991         0.10
    AX     1819      816   2635         0.31
    O      2171      367   2538         0.14
    AP     2016      295   2311         0.13
    AD     1409      417   1826         0.23
    V      1012      786   1798         0.44
    AY     1585      164   1749         0.09
    AO     1398      336   1734         0.19
    AG     1495      211   1706         0.12
    H      1490      202   1692         0.12
    AL      718      854   1572         0.54
    W      1223      302   1525         0.20
    B      1195      213   1408         0.15
    Q      1074      232   1306         0.18
    AM      961      222   1183         0.19
    AR      828      293   1121         0.26
    L       551      390    941         0.41
    AT      845       73    918         0.08
    M       529      378    907         0.42
    D       793       42    835         0.05
    BA      187      594    781         0.76
    AU      604       67    671         0.10
    X       588       74    662         0.11
    AC      207      453    660         0.69
    I       302      338    640         0.53
    P       497      114    611         0.19
    AB      461       78    539         0.14
    AE      319      139    458         0.30
    T       321       58    379         0.15
    AJ      182       47    229         0.21
    AQ      146       59    205         0.29


```
#Low number (<1000) criteria too restrictive (not enough to merge)
output.loc[(output['label1share'] < .15) &
           (output['total'] < 1000)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label 0</th>
      <th>label 1</th>
      <th>total</th>
      <th>label1share</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AB</th>
      <td>461</td>
      <td>78</td>
      <td>539</td>
      <td>0.14</td>
    </tr>
    <tr>
      <th>AT</th>
      <td>845</td>
      <td>73</td>
      <td>918</td>
      <td>0.08</td>
    </tr>
    <tr>
      <th>AU</th>
      <td>604</td>
      <td>67</td>
      <td>671</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>D</th>
      <td>793</td>
      <td>42</td>
      <td>835</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>X</th>
      <td>588</td>
      <td>74</td>
      <td>662</td>
      <td>0.11</td>
    </tr>
  </tbody>
</table>
</div>



```
#Low number (<1000) criteria too restrictive (not enough to merge)
output.loc[(output['label1share'] > .40) &
           (output['total'] < 1000)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label 0</th>
      <th>label 1</th>
      <th>total</th>
      <th>label1share</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AC</th>
      <td>207</td>
      <td>453</td>
      <td>660</td>
      <td>0.69</td>
    </tr>
    <tr>
      <th>BA</th>
      <td>187</td>
      <td>594</td>
      <td>781</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th>I</th>
      <td>302</td>
      <td>338</td>
      <td>640</td>
      <td>0.53</td>
    </tr>
    <tr>
      <th>L</th>
      <td>551</td>
      <td>390</td>
      <td>941</td>
      <td>0.41</td>
    </tr>
    <tr>
      <th>M</th>
      <td>529</td>
      <td>378</td>
      <td>907</td>
      <td>0.42</td>
    </tr>
  </tbody>
</table>
</div>



```
#@AndresHG suggestion 
subtotal_name = 'subtotal_Andres'
criteria = ['Y', 'AA', 'R', 'O', 'AP', 'AY']

output.loc[subtotal_name] = output.loc[criteria].loc[:,['label 0', 'label 1', 'total']].sum(axis=0)

output.loc[subtotal_name, 'label1share'] = output.loc[subtotal_name, 'label 1'] \
                                                    / output.loc[subtotal_name, 'total']

mask = output.loc[criteria].sort_values('total', ascending=False).index
mask = mask.append(pd.Index([subtotal_name]))

print(output.loc[mask])
#all low label 1 share
```

                     label 0  label 1    total  label1share
    Y                 5569.0    327.0   5896.0     0.060000
    AA                3380.0    312.0   3692.0     0.080000
    R                 2704.0    287.0   2991.0     0.100000
    O                 2171.0    367.0   2538.0     0.140000
    AP                2016.0    295.0   2311.0     0.130000
    AY                1585.0    164.0   1749.0     0.090000
    subtotal_Andres  17425.0   1752.0  19177.0     0.091359


```
#sorting by low positive and below 6000 (max of @AndresHG)
subtotal_name = 'subtotal_low_poz_6000'
criteria = (output['label1share'] < .15) & (output['total'] < 6000)

output.loc[subtotal_name] = output.loc[criteria].loc[:,['label 0', 'label 1', 'total']].sum(axis=0)

output.loc[subtotal_name, 'label1share'] = output.loc[subtotal_name, 'label 1'] \
                                                    / output.loc[subtotal_name, 'total']

mask = output.loc[criteria].sort_values('total', ascending=False).index
mask = mask.append(pd.Index([subtotal_name]))

print(output.loc[mask])

#Y and F seem significant enough to have on their own, think the cut shoudld be 4000
```

                           label 0  label 1    total  label1share
    Y                       5569.0    327.0   5896.0     0.060000
    F                       4863.0    687.0   5550.0     0.120000
    AA                      3380.0    312.0   3692.0     0.080000
    R                       2704.0    287.0   2991.0     0.100000
    O                       2171.0    367.0   2538.0     0.140000
    AP                      2016.0    295.0   2311.0     0.130000
    AY                      1585.0    164.0   1749.0     0.090000
    AG                      1495.0    211.0   1706.0     0.120000
    H                       1490.0    202.0   1692.0     0.120000
    AT                       845.0     73.0    918.0     0.080000
    D                        793.0     42.0    835.0     0.050000
    AU                       604.0     67.0    671.0     0.100000
    X                        588.0     74.0    662.0     0.110000
    AB                       461.0     78.0    539.0     0.140000
    subtotal_low_poz_6000  28564.0   3186.0  31750.0     0.100346


```
#sorting by low positive and below 4000
subtotal_name = 'subtotal_low_poz_4000'
criteria = (output['label1share'] < .15) & (output['total'] < 4000)

output.loc[subtotal_name] = output.loc[criteria].loc[:,['label 0', 'label 1', 'total']].sum(axis=0)

output.loc[subtotal_name, 'label1share'] = output.loc[subtotal_name, 'label 1'] \
                                                    / output.loc[subtotal_name, 'total']

mask = output.loc[criteria].sort_values('total', ascending=False).index

print(f'Final: {mask}')

mask = mask.append(pd.Index([subtotal_name]))

print(output.loc[mask])

#Final criteria
```

    Final: Index(['AA', 'R', 'O', 'AP', 'AY', 'AG', 'H', 'AT', 'D', 'AU', 'X', 'AB'], dtype='object')
                           label 0  label 1    total  label1share
    AA                      3380.0    312.0   3692.0     0.080000
    R                       2704.0    287.0   2991.0     0.100000
    O                       2171.0    367.0   2538.0     0.140000
    AP                      2016.0    295.0   2311.0     0.130000
    AY                      1585.0    164.0   1749.0     0.090000
    AG                      1495.0    211.0   1706.0     0.120000
    H                       1490.0    202.0   1692.0     0.120000
    AT                       845.0     73.0    918.0     0.080000
    D                        793.0     42.0    835.0     0.050000
    AU                       604.0     67.0    671.0     0.100000
    X                        588.0     74.0    662.0     0.110000
    AB                       461.0     78.0    539.0     0.140000
    subtotal_low_poz_4000  18132.0   2172.0  20304.0     0.106974


```
#Second @AndresHG suggestion 
subtotal_name = 'subtotal_Andres2'
criteria = ['AL', 'V', 'BA', 'AC', 'AD', 'L']

output.loc[subtotal_name] = output.loc[criteria].loc[:,['label 0', 'label 1', 'total']].sum(axis=0)

output.loc[subtotal_name, 'label1share'] = output.loc[subtotal_name, 'label 1'] \
                                                    / output.loc[subtotal_name, 'total']

mask = output.loc[criteria].sort_values('total', ascending=False).index
mask = mask.append(pd.Index([subtotal_name]))

print(output.loc[mask])

#all high positivity, low representation < 2000
```

                      label 0  label 1   total  label1share
    AD                 1409.0    417.0  1826.0     0.230000
    V                  1012.0    786.0  1798.0     0.440000
    AL                  718.0    854.0  1572.0     0.540000
    L                   551.0    390.0   941.0     0.410000
    BA                  187.0    594.0   781.0     0.760000
    AC                  207.0    453.0   660.0     0.690000
    subtotal_Andres2   4084.0   3494.0  7578.0     0.461072


```
#sorting by high positive and below 2000 (max of @AndresHG)
subtotal_name = 'subtotal_hi_poz_2000'
criteria = (output['label1share'] > .40) & (output['total'] < 2000)

output.loc[subtotal_name] = output.loc[criteria].loc[:,['label 0', 'label 1', 'total']].sum(axis=0)

output.loc[subtotal_name, 'label1share'] = output.loc[subtotal_name, 'label 1'] \
                                                    / output.loc[subtotal_name, 'total']

mask = output.loc[criteria].sort_values('total', ascending=False).index

print(f'Final: {mask}')

mask = mask.append(pd.Index([subtotal_name]))

print(output.loc[mask])
#includes additional categories to merge, M and I
```

    Final: Index(['V', 'AL', 'L', 'M', 'BA', 'AC', 'I'], dtype='object')
                          label 0  label 1   total  label1share
    V                      1012.0    786.0  1798.0      0.44000
    AL                      718.0    854.0  1572.0      0.54000
    L                       551.0    390.0   941.0      0.41000
    M                       529.0    378.0   907.0      0.42000
    BA                      187.0    594.0   781.0      0.76000
    AC                      207.0    453.0   660.0      0.69000
    I                       302.0    338.0   640.0      0.53000
    subtotal_hi_poz_2000   3506.0   3793.0  7299.0      0.51966


```
#----------------------cat8----------------------
#Unique values for class 0: {'AS', 'I', 'BJ', 'BC', 'AW'}
#Unique values for class 1: {'F', 'AG', 'BD', 'AO', 'A'}

output = print_value_counts('cat8')
```

        label 0  label 1  total  label1share
    BM    33136   9244.0  42380         0.22
    AE    19655   4787.0  24442         0.20
    AX    16865   5264.0  22129         0.24
    Y     16885   3979.0  20864         0.19
    H      9345   6216.0  15561         0.40
    ..      ...      ...    ...          ...
    AQ       58     11.0     69         0.16
    T        66      1.0     67         0.01
    B        55      2.0     57         0.04
    AC       57      NaN     57          NaN
    AR       32      1.0     33         0.03
    
    [61 rows x 4 columns]


```
#low positivity, low total
subtotal_name = 'subtotal_low_poz_2000'
criteria = (output['label1share'] < .15) & (output['total'] < 2000)

output.loc[subtotal_name] = output.loc[criteria].loc[:,['label 0', 'label 1', 'total']].sum(axis=0)

output.loc[subtotal_name, 'label1share'] = output.loc[subtotal_name, 'label 1'] \
                                                    / output.loc[subtotal_name, 'total']

mask = output.loc[criteria].sort_values('total', ascending=False).index

print(f'Final: {mask}')

mask = mask.append(pd.Index([subtotal_name]))

print(output.loc[mask])
```

    Final: Index(['BK', 'AM', 'AY', 'AI', 'BE', 'E', 'V', 'BB', 'AP', 'AL', 'C', 'T', 'B',
           'AR'],
          dtype='object')
                           label 0  label 1   total  label1share
    BK                       864.0    101.0   965.0     0.100000
    AM                       752.0     68.0   820.0     0.080000
    AY                       622.0     61.0   683.0     0.090000
    AI                       520.0     49.0   569.0     0.090000
    BE                       278.0      8.0   286.0     0.030000
    E                        219.0      2.0   221.0     0.010000
    V                        210.0      9.0   219.0     0.040000
    BB                       196.0     12.0   208.0     0.060000
    AP                       170.0     19.0   189.0     0.100000
    AL                       160.0      8.0   168.0     0.050000
    C                        129.0     13.0   142.0     0.090000
    T                         66.0      1.0    67.0     0.010000
    B                         55.0      2.0    57.0     0.040000
    AR                        32.0      1.0    33.0     0.030000
    subtotal_low_poz_2000   4273.0    354.0  4627.0     0.076507


```
#high positivity, low total
subtotal_name = 'subtotal_hi_poz_2000'
criteria = (output['label1share'] > .40) & (output['total'] < 2000)

output.loc[subtotal_name] = output.loc[criteria].loc[:,['label 0', 'label 1', 'total']].sum(axis=0)

output.loc[subtotal_name, 'label1share'] = output.loc[subtotal_name, 'label 1'] \
                                                    / output.loc[subtotal_name, 'total']

mask = output.loc[criteria].sort_values('total', ascending=False).index

print(f'Final: {mask}')

mask = mask.append(pd.Index([subtotal_name]))

print(output.loc[mask])
```

    Final: Index(['AO', 'F', 'AV'], dtype='object')
                          label 0  label 1   total  label1share
    AO                      973.0    844.0  1817.0     0.460000
    F                       533.0   1173.0  1706.0     0.690000
    AV                      305.0    269.0   574.0     0.470000
    subtotal_hi_poz_2000   1811.0   2286.0  4097.0     0.557969


```
#----------------------cat9----------------------

output = print_value_counts('cat9')
```

       label 0  label 1   total  label1share
    A   138767    63178  201945         0.31
    E    29518     3528   33046         0.11
    C    18693     4667   23360         0.20
    F    12164     2207   14371         0.15
    J     7439     1543    8982         0.17
    I     5906     2025    7931         0.26
    N     3412     1373    4785         0.29
    L     2594      363    2957         0.12
    R      668      194     862         0.23
    V      219      141     360         0.39
    B      172      108     280         0.39
    G      193       21     214         0.10
    Q      163       48     211         0.23
    D      184        5     189         0.03
    W      118        7     125         0.06
    O      110       12     122         0.10
    U       81       20     101         0.20
    X       90        9      99         0.09
    S       48       12      60         0.20


```
#below 400
subtotal_name = 'subtotal_below_400'
criteria = (output['total'] < 400)

output.loc[subtotal_name] = output.loc[criteria].loc[:,['label 0', 'label 1', 'total']].sum(axis=0)

output.loc[subtotal_name, 'label1share'] = output.loc[subtotal_name, 'label 1'] \
                                                    / output.loc[subtotal_name, 'total']

mask = output.loc[criteria].sort_values('total', ascending=False).index

print(f'Final: {mask}')

mask = mask.append(pd.Index([subtotal_name]))

print(output.loc[mask])

```

    Final: Index(['V', 'B', 'G', 'Q', 'D', 'W', 'O', 'U', 'X', 'S'], dtype='object')
                        label 0  label 1   total  label1share
    V                     219.0    141.0   360.0      0.39000
    B                     172.0    108.0   280.0      0.39000
    G                     193.0     21.0   214.0      0.10000
    Q                     163.0     48.0   211.0      0.23000
    D                     184.0      5.0   189.0      0.03000
    W                     118.0      7.0   125.0      0.06000
    O                     110.0     12.0   122.0      0.10000
    U                      81.0     20.0   101.0      0.20000
    X                      90.0      9.0    99.0      0.09000
    S                      48.0     12.0    60.0      0.20000
    subtotal_below_400   1378.0    383.0  1761.0      0.21749


```
#----------------------cat10----------------------
#Unique values for class 0: {'LN', 'EK', 'HJ', 'DF', 'GE', 'IG', 'HB'}
#Unique values for class 1: {'GI', 'HC', 'GK', 'CD', 'MC', 'FR', 'JR'}

output = print_value_counts('cat10')
```

        label 0  label 1  total  label1share
    DJ    24072   7512.0  31584         0.24
    HK    20432  10566.0  30998         0.34
    DP    17774   5905.0  23679         0.25
    GS    13726   2893.0  16619         0.17
    CR    13415    967.0  14382         0.07
    ..      ...      ...    ...          ...
    AW        1      NaN      1          NaN
    LK        1      NaN      1          NaN
    MK        1      NaN      1          NaN
    ML        1      NaN      1          NaN
    MR        1      NaN      1          NaN
    
    [299 rows x 4 columns]


```
#low total
subtotal_name = 'subtotal_below_2000'
criteria = (output['total'] < 2000) 

output.loc[subtotal_name] = output.loc[criteria].loc[:,['label 0', 'label 1', 'total']].sum(axis=0)

output.loc[subtotal_name, 'label1share'] = output.loc[subtotal_name, 'label 1'] / output.loc[subtotal_name, 'total']

mask = output.loc[criteria].sort_values('total', ascending=False).index

print(f'Final: {len(mask)} items \n {mask}')
test_mask_tot = mask.copy()

mask = mask.append(pd.Index([subtotal_name]))

print(output.loc[mask])
```

    Final: 273 items 
     Index(['LY', 'GE', 'GK', 'CS', 'MJ', 'HH', 'CD', 'MC', 'HA', 'GQ',
           ...
           'CX', 'MW', 'GH', 'MR', 'MO', 'ML', 'MK', 'FW', 'LK', 'GG'],
          dtype='object', length=273)
                         label 0  label 1    total  label1share
    LY                    1556.0    420.0   1976.0     0.210000
    GE                    1816.0    159.0   1975.0     0.080000
    GK                     957.0    987.0   1944.0     0.510000
    CS                    1165.0    570.0   1735.0     0.330000
    MJ                    1236.0    278.0   1514.0     0.180000
    ...                      ...      ...      ...          ...
    MK                       1.0      NaN      1.0          NaN
    FW                       1.0      NaN      1.0          NaN
    LK                       1.0      NaN      1.0          NaN
    GG                       1.0      NaN      1.0          NaN
    subtotal_below_2000  41970.0  22249.0  64219.0     0.346455
    
    [274 rows x 4 columns]


```
#low total, low poz
subtotal_name = 'subtotal_low_poz_2000'
criteria = (output['total'] < 2000) & (pd.isna(output['label1share']) | (output['label1share'] < .15))

#74

#110

output.loc[subtotal_name] = output.loc[criteria].loc[:,['label 0', 'label 1', 'total']].sum(axis=0)

output.loc[subtotal_name, 'label1share'] = output.loc[subtotal_name, 'label 1'] / output.loc[subtotal_name, 'total']

mask = output.loc[criteria].sort_values('total', ascending=False).index

print(f'Final: {len(mask)} items', end='')
for i, category in enumerate(mask):
  if i % 11 == 0:
    print('\n', end='')
  print(f'\'{category}\'', end=',')
print('\n')

test_mask_low = mask.copy()

mask = mask.append(pd.Index([subtotal_name]))

print(output.loc[mask])
```

    Final: 110 items
    'GE','HH','KB','AV','KV','AD','BM','BI','JP','O','JN',
    'LQ','KS','BG','IT','CJ','CI','KR','CW','K','CO','CG',
    'KA','EC','EA','IC','DK','GX','EP','CP','BX','ME','CN',
    'LL','KN','DD','DN','DU','BD','ED','KU','GR','FG','KJ',
    'KQ','KI','KD','HP','GD','EG','GJ','MA','IQ','CT','IU',
    'KT','CF','HF','MQ','MP','HY','IM','IP','BO','LR','EH',
    'FA','MU','EB','IY','CM','DM','JU','EF','DL','MI','HI',
    'FF','AF','GV','DX','IN','LT','AJ','GF','DA','DT','EN',
    'JF','LH','CQ','GY','BS','KK','CH','JC','JE','ML','MR',
    'CX','MW','MO','BA','MK','FW','GG','GH','IL','AW','LK',
    
                           label 0  label 1    total  label1share
    GE                      1816.0    159.0   1975.0     0.080000
    HH                      1315.0     64.0   1379.0     0.050000
    KB                       852.0    141.0    993.0     0.140000
    AV                       673.0     87.0    760.0     0.110000
    KV                       509.0     64.0    573.0     0.110000
    ...                        ...      ...      ...          ...
    GH                         1.0      NaN      1.0          NaN
    IL                         1.0      NaN      1.0          NaN
    AW                         1.0      NaN      1.0          NaN
    LK                         1.0      NaN      1.0          NaN
    subtotal_low_poz_2000   9374.0    753.0  10127.0     0.074356
    
    [111 rows x 4 columns]


```
#low total, mid poz
subtotal_name = 'subtotal_mid_poz_2000'
criteria = (output['total'] < 2000) & (output['label1share'] >= .15) & (output['label1share'] < .40) 

output.loc[subtotal_name] = output.loc[criteria].loc[:,['label 0', 'label 1', 'total']].sum(axis=0)

output.loc[subtotal_name, 'label1share'] = output.loc[subtotal_name, 'label 1'] / output.loc[subtotal_name, 'total']

mask = output.loc[criteria].sort_values('total', ascending=False).index

print(f'Final: {len(mask)} items', end='')
for i, category in enumerate(mask):
  if i % 11 == 0:
    print('\n', end='')
  print(f'\'{category}\'', end=',')
print('\n')

test_mask_mid = mask.copy()

mask = mask.append(pd.Index([subtotal_name]))

print(output.loc[mask])
```

    Final: 87 items
    'LY','CS','MJ','HA','GQ','LI','JW','CB','R','JG','GU',
    'CU','FS','HL','BY','JD','FN','IJ','JK','MB','BV','IX',
    'IK','KC','BL','I','DO','HO','MT','AU','P','MH','LU',
    'DY','AE','LW','FP','T','AA','AN','LX','FL','JI','IB',
    'MF','W','C','AS','DV','JA','AR','FM','HE','FK','EL',
    'IH','KG','LC','AK','MS','HM','JB','CY','DW','FX','EU',
    'EO','S','DH','IF','GN','BQ','X','LJ','FE','Q','GT',
    'ES','LG','LD','HR','GL','FD','CL','JL','EE','JJ',
    
                           label 0  label 1    total  label1share
    LY                      1556.0    420.0   1976.0     0.210000
    CS                      1165.0    570.0   1735.0     0.330000
    MJ                      1236.0    278.0   1514.0     0.180000
    HA                       939.0    280.0   1219.0     0.230000
    GQ                       869.0    229.0   1098.0     0.210000
    ...                        ...      ...      ...          ...
    CL                        10.0      4.0     14.0     0.290000
    JL                         8.0      5.0     13.0     0.380000
    EE                        10.0      2.0     12.0     0.170000
    JJ                         6.0      3.0      9.0     0.330000
    subtotal_mid_poz_2000  22542.0   7888.0  30430.0     0.259218
    
    [88 rows x 4 columns]


```
#low total, high poz
subtotal_name = 'subtotal_hi_poz_2000'
criteria = (output['total'] < 2000) & (output['label1share'] >= .40)

output.loc[subtotal_name] = output.loc[criteria].loc[:,['label 0', 'label 1', 'total']].sum(axis=0)

output.loc[subtotal_name, 'label1share'] = output.loc[subtotal_name, 'label 1'] / output.loc[subtotal_name, 'total']

mask = output.loc[criteria].sort_values('total', ascending=False).index

print(f'Final: {len(mask)} items', end='')
for i, category in enumerate(mask):
  if i % 11 == 0:
    print('\n', end='')
  print(f'\'{category}\'', end=',')
print('\n')

test_mask_hi = mask.copy()

mask = mask.append(pd.Index([subtotal_name]))

print(output.loc[mask])
```

    Final: 76 items
    'GK','CD','MC','GI','JR','FR','HN','MG','BF','LV','EQ',
    'EV','V','DI','JX','IA','JT','FI','FC','CC','JH','J',
    'JO','BC','AP','EY','IV','FO','BP','KX','IO','LE','AB',
    'GA','IR','AT','E','KF','AH','DQ','FT','D','GB','FJ',
    'BB','KH','JY','G','KL','EI','AY','AM','MV','BT','FQ',
    'AC','HU','KP','JV','F','FH','FB','DS','ID','M','AG',
    'L','GW','HW','AL','GM','EW','DE','DR','FV','Y',
    
                          label 0  label 1    total  label1share
    GK                      957.0    987.0   1944.0     0.510000
    CD                       25.0   1235.0   1260.0     0.980000
    MC                      567.0    653.0   1220.0     0.540000
    GI                      279.0    695.0    974.0     0.710000
    JR                      119.0    721.0    840.0     0.860000
    ...                       ...      ...      ...          ...
    DE                        5.0      6.0     11.0     0.550000
    DR                        4.0      5.0      9.0     0.560000
    FV                        3.0      2.0      5.0     0.400000
    Y                         2.0      2.0      4.0     0.500000
    subtotal_hi_poz_2000  10054.0  13608.0  23662.0     0.575099
    
    [77 rows x 4 columns]


```
test_mask_tot = set(test_mask_tot)
test_mask_low = set(test_mask_low)
test_mask_mid = set(test_mask_mid)
test_mask_hi = set(test_mask_hi)

leftovers = test_mask_tot - test_mask_low - test_mask_mid - test_mask_hi
```

```
print(leftovers)
print(len(leftovers))
```

    {'CX', 'DK', 'FW', 'BA', 'MR', 'DA', 'EG', 'HI', 'EB', 'CN', 'IQ', 'FA', 'IP', 'KD', 'HF', 'FF', 'MU', 'KK', 'CT', 'KU', 'IY', 'IM', 'EH', 'JF', 'BD', 'IN', 'GF', 'IU', 'ME', 'LH', 'JE', 'EN', 'KN', 'JC', 'GD', 'BS', 'CQ', 'MP', 'DM', 'BO', 'CH', 'GG', 'MI', 'JU', 'DX', 'GV', 'GR', 'DN', 'MO', 'MA', 'BX', 'AF', 'AW', 'HY', 'ML', 'MW', 'CF', 'DL', 'LR', 'GH', 'MQ', 'GY', 'ED', 'EF', 'MK', 'DU', 'IL', 'KI', 'LT', 'LK', 'CM', 'AJ', 'GJ', 'DT'}
    74


```
print(len(test_mask_tot), len(test_mask_low), len(test_mask_mid), len(test_mask_hi))
273-36-87-76

```

    273 36 87 76





    74



```
leftover_index = pd.Index(leftovers)

leftover_index

output.loc[leftover_index].loc[pd.notna(output['label 1'])]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label 0</th>
      <th>label 1</th>
      <th>total</th>
      <th>label1share</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



Comments
* cat0: almost no positives in the B category
* cat1: D and E are not large and fairly evenly split
* cat2: ['N', 'H', 'B', 'S', 'U', 'R', 'K', 'E'] all have less than 400 representation
* cat3: ['G', 'L', 'J', 'H', 'I', 'N'] all have less than 400 representation, fairly split between classes
* cat4: ['O', 'B', 'S', 'T', 'L', 'Q', 'P', 'A', 'N', 'R'] are below 400
* cat5: steep drop off in relevant categories, ['BI', 'AB', 'BU', 'K', 'G', 'BQ', 'N'] are the only ones above 400
* cat6: ['F', 'D', 'Y', 'B', 'U', 'Q', 'W'] are less than 400
* cat7: ['AA', 'R', 'O', 'AP', 'AY', 'AG', 'H', 'AT', 'D', 'AU', 'X', 'AB'] can be merged as low positive, low rep; ['V', 'AL', 'L', 'M', 'BA', 'AC', 'I'] can be merged as high positive, low rep
* cat8: ['BK', 'AM', 'AY', 'AI', 'BE', 'E', 'V', 'BB', 'AP', 'AL', 'C', 'T', 'B', 'AR'] can be merged as low poz, low rep; ['AO', 'F', 'AV'] can be merged as hi poz, low rep
* cat9: ['V', 'B', 'G', 'Q', 'D', 'W', 'O', 'U', 'X', 'S'] all low rep
* cat10: three categories
1.   Low poz, low total: ['GE','HH','KB','AV','KV','AD','BM','BI','JP','O','JN', 'LQ','KS','BG','IT','CJ','CI','KR','CW','K','CO','CG','KA','EC','EA','IC','DK','GX','EP','CP','BX','ME','CN','LL','KN','DD','DN','DU','BD','ED','KU','GR','FG','KJ','KQ','KI','KD','HP','GD','EG','GJ','MA','IQ','CT','IU','KT','CF','HF','MQ','MP','HY','IM','IP','BO','LR','EH','FA','MU','EB','IY','CM','DM','JU','EF','DL','MI','HI','FF','AF','GV','DX','IN','LT','AJ','GF','DA','DT','EN','JF','LH','CQ','GY','BS','KK','CH','JC','JE','ML','MR','CX','MW','MO','BA','MK','FW','GG','GH','IL','AW','LK'] 110 items
2.   Mid poz, low total: ['LY','CS','MJ','HA','GQ','LI','JW','CB','R','JG','GU','CU','FS','HL','BY','JD','FN','IJ','JK','MB','BV','IX', 'IK','KC','BL','I','DO','HO','MT','AU','P','MH','LU','DY','AE','LW','FP','T','AA','AN','LX','FL','JI','IB','MF','W','C','AS','DV','JA','AR','FM','HE','FK','EL', 'IH','KG','LC','AK','MS','HM','JB','CY','DW','FX','EU', 'EO','S','DH','IF','GN','BQ','X','LJ','FE','Q','GT','ES','LG','LD','HR','GL','FD','CL','JL','EE','JJ'] 87 items
3.   Hi poz, low total: ['GK','CD','MC','GI','JR','FR','HN','MG','BF','LV','EQ','EV','V','DI','JX','IA','JT','FI','FC','CC','JH','J','JO','BC','AP','EY','IV','FO','BP','KX','IO','LE','AB','GA','IR','AT','E','KF','AH','DQ','FT','D','GB','FJ','BB','KH','JY','G','KL','EI','AY','AM','MV','BT','FQ','AC','HU','KP','JV','F','FH','FB','DS','ID','M','AG', 'L','GW','HW','AL','GM','EW','DE','DR','FV','Y'] 76 items
* cat11-18: not a large number of categories




# Feature Engineering


```
def Clean_Categories(df):
  #generic separating function (to be partialed)
  def sep_fn(low_crit, mid_crit, hi_crit, name, x):
    if x in low_crit:
      return ''.join([name, 'LOPOZ'])
    elif x in mid_crit:
      return ''.join([name, 'MIDPOZ'])
    elif x in hi_crit:
      return ''.join([name, 'HIPOZ'])
    else:
      return x

  #cat 1
  df['cat1'] = df['cat1'].apply(lambda x:
                                x if x not in ['D' 'E']
                                else 'CAT1SMALL')

  #cat 2
  df['cat2'] = df['cat2'].apply(lambda x:
                                x if x not in ['N', 'H', 'B', 'S', 'U', 'R', 'K', 'E']
                                else 'CAT2SMALL')

  #cat3
  df['cat3'] = df['cat3'].apply(lambda x:
                                x if x not in ['G', 'L', 'J', 'H', 'I', 'N']
                                else 'CAT3SMALL')
  
  #cat4
  df['cat4'] = df['cat4'].apply(lambda x:
                                x if x not in ['O', 'B', 'S', 'T', 'L', 'Q', 'P', 'A', 'N', 'R']
                                else 'CAT4SMALL')
  
  #cat5
  df['cat5'] = df['cat5'].apply(lambda x:
                                x if x in ['BI', 'AB', 'BU', 'K', 'G', 'BQ', 'N']
                                else 'CAT5SMALL')
  
  #cat6
  df['cat6'] = df['cat6'].apply(lambda x:
                                x if x not in ['F', 'D', 'Y', 'B', 'U', 'Q', 'W']
                                else 'CAT6SMALL')
  
  #cat7
  low_crit = ['AA', 'R', 'O', 'AP', 'AY', 'AG', 'H', 'AT', 'D', 'AU', 'X', 'AB']
  mid_crit = []
  hi_crit = ['V', 'AL', 'L', 'M', 'BA', 'AC', 'I']
  cat7_mini_fn = partial(sep_fn, low_crit, mid_crit, hi_crit, 'CAT7')
  df['cat7'] = df['cat7'].apply(cat7_mini_fn)

  #cat8
  low_crit = ['BK', 'AM', 'AY', 'AI', 'BE', 'E', 'V', 'BB', 'AP', 'AL', 'C', 'T', 'B', 'AR']
  mid_crit = []
  hi_crit = ['AO', 'F', 'AV']
  cat8_mini_fn = partial(sep_fn, low_crit, mid_crit, hi_crit, 'CAT8')
  df['cat8'] = df['cat8'].apply(cat8_mini_fn)

  #cat9
  df['cat9'] = df['cat9'].apply(lambda x:
                                x if x not in ['V', 'B', 'G', 'Q', 'D', 'W', 'O', 'U', 'X', 'S']
                                else 'CAT9SMALL')
  
  #cat10
  low_crit = ['GE','HH','KB','AV','KV','AD','BM','BI','JP','O','JN',
              'LQ','KS','BG','IT','CJ','CI','KR','CW','K','CO','CG',
              'KA','EC','EA','IC','DK','GX','EP','CP','BX','ME','CN',
              'LL','KN','DD','DN','DU','BD','ED','KU','GR','FG','KJ',
              'KQ','KI','KD','HP','GD','EG','GJ','MA','IQ','CT','IU',
              'KT','CF','HF','MQ','MP','HY','IM','IP','BO','LR','EH',
              'FA','MU','EB','IY','CM','DM','JU','EF','DL','MI','HI',
              'FF','AF','GV','DX','IN','LT','AJ','GF','DA','DT','EN',
              'JF','LH','CQ','GY','BS','KK','CH','JC','JE','ML','MR',
              'CX','MW','MO','BA','MK','FW','GG','GH','IL','AW','LK',]
  mid_crit = ['LY','CS','MJ','HA','GQ','LI','JW','CB','R','JG','GU',
              'CU','FS','HL','BY','JD','FN','IJ','JK','MB','BV','IX',
              'IK','KC','BL','I','DO','HO','MT','AU','P','MH','LU',
              'DY','AE','LW','FP','T','AA','AN','LX','FL','JI','IB',
              'MF','W','C','AS','DV','JA','AR','FM','HE','FK','EL',
              'IH','KG','LC','AK','MS','HM','JB','CY','DW','FX','EU',
              'EO','S','DH','IF','GN','BQ','X','LJ','FE','Q','GT',
              'ES','LG','LD','HR','GL','FD','CL','JL','EE','JJ']
  hi_crit = ['GK','CD','MC','GI','JR','FR','HN','MG','BF','LV','EQ',
             'EV','V','DI','JX','IA','JT','FI','FC','CC','JH','J',
             'JO','BC','AP','EY','IV','FO','BP','KX','IO','LE','AB',
             'GA','IR','AT','E','KF','AH','DQ','FT','D','GB','FJ',
             'BB','KH','JY','G','KL','EI','AY','AM','MV','BT','FQ',
             'AC','HU','KP','JV','F','FH','FB','DS','ID','M','AG',
             'L','GW','HW','AL','GM','EW','DE','DR','FV','Y',]
  cat10_mini_fn = partial(sep_fn, low_crit, mid_crit, hi_crit, 'CAT10')
  df['cat10'] = df['cat10'].apply(cat10_mini_fn)
  
  return df  

```

```
train_clean = Clean_Categories(train)
test_clean = Clean_Categories(test)
```

```
print('For Train')
for cat_col in cat_columns:
  print(f'Category: {cat_col}, Number: {len(train_clean[cat_col].value_counts())}')

```

    For Train
    Category: cat0, Number: 2
    Category: cat1, Number: 15
    Category: cat2, Number: 12
    Category: cat3, Number: 8
    Category: cat4, Number: 11
    Category: cat5, Number: 8
    Category: cat6, Number: 10
    Category: cat7, Number: 34
    Category: cat8, Number: 46
    Category: cat9, Number: 10
    Category: cat10, Number: 29
    Category: cat11, Number: 2
    Category: cat12, Number: 2
    Category: cat13, Number: 2
    Category: cat14, Number: 2
    Category: cat15, Number: 4
    Category: cat16, Number: 4
    Category: cat17, Number: 4
    Category: cat18, Number: 4


## Testing

```
  #generic separating function (to be partialed)
  def sep_fn(low_crit, mid_crit, hi_crit, name, x):
    if x in low_crit:
      return ''.join([name, 'LOPOZ'])
    elif x in mid_crit:
      return ''.join([name, 'MIDPOZ'])
    elif x in hi_crit:
      return ''.join([name, 'HIPOZ'])
    else:
      return x
```

```
low_crit = ['AA', 'R', 'O', 'AP', 'AY', 'AG', 'H', 'AT', 'D', 'AU', 'X', 'AB']
mid_crit = ['XXXXX']
hi_crit = ['V', 'AL', 'L', 'M', 'BA', 'AC', 'I']

cat7_mini_fn = partial(sep_fn, low_crit, mid_crit, hi_crit, 'CAT7')
  
```

```
cat7_mini_fn('XXXXX')
```




    'CAT7MIDPOZ'



```
train_cop = train.copy()

```

```
len(train_cop['cat7'].value_counts()) - len(train_cop['cat7'].apply(cat7_mini_fn).value_counts())

```




    17



```
train_cop['cat7'].apply(cat7_mini_fn).value_counts()
```




    AH           45818
    E            39601
    AS           25326
    CAT7LOPOZ    20304
    J            16135
    AN           16097
    U            15674
    N            14983
    AF           11455
    AK            9697
    AV            7958
    S             7921
    AI            7668
    CAT7HIPOZ     7299
    A             6432
    K             6264
    Y             5896
    G             5656
    F             5550
    AW            5322
    C             4324
    AX            2635
    AD            1826
    AO            1734
    W             1525
    B             1408
    Q             1306
    AM            1183
    AR            1121
    P              611
    AE             458
    T              379
    AJ             229
    AQ             205
    Name: cat7, dtype: int64



```
  low_crit = ['GE','HH','KB','AV','KV','AD','BM','BI','JP','O','JN',
              'LQ','KS','BG','IT','CJ','CI','KR','CW','K','CO','CG',
              'KA','EC','EA','IC','DK','GX','EP','CP','BX','ME','CN',
              'LL','KN','DD','DN','DU','BD','ED','KU','GR','FG','KJ',
              'KQ','KI','KD','HP','GD','EG','GJ','MA','IQ','CT','IU',
              'KT','CF','HF','MQ','MP','HY','IM','IP','BO','LR','EH',
              'FA','MU','EB','IY','CM','DM','JU','EF','DL','MI','HI',
              'FF','AF','GV','DX','IN','LT','AJ','GF','DA','DT','EN',
              'JF','LH','CQ','GY','BS','KK','CH','JC','JE','ML','MR',
              'CX','MW','MO','BA','MK','FW','GG','GH','IL','AW','LK',]
  mid_crit = ['LY','CS','MJ','HA','GQ','LI','JW','CB','R','JG','GU',
              'CU','FS','HL','BY','JD','FN','IJ','JK','MB','BV','IX',
              'IK','KC','BL','I','DO','HO','MT','AU','P','MH','LU',
              'DY','AE','LW','FP','T','AA','AN','LX','FL','JI','IB',
              'MF','W','C','AS','DV','JA','AR','FM','HE','FK','EL',
              'IH','KG','LC','AK','MS','HM','JB','CY','DW','FX','EU',
              'EO','S','DH','IF','GN','BQ','X','LJ','FE','Q','GT',
              'ES','LG','LD','HR','GL','FD','CL','JL','EE','JJ']
  hi_crit = ['GK','CD','MC','GI','JR','FR','HN','MG','BF','LV','EQ',
             'EV','V','DI','JX','IA','JT','FI','FC','CC','JH','J',
             'JO','BC','AP','EY','IV','FO','BP','KX','IO','LE','AB',
             'GA','IR','AT','E','KF','AH','DQ','FT','D','GB','FJ',
             'BB','KH','JY','G','KL','EI','AY','AM','MV','BT','FQ',
             'AC','HU','KP','JV','F','FH','FB','DS','ID','M','AG',
             'L','GW','HW','AL','GM','EW','DE','DR','FV','Y',]
  cat10_mini_fn = partial(sep_fn, low_crit, mid_crit, hi_crit, 'CAT10')
```

```
len(train_cop['cat10'].value_counts()) - len(train_cop['cat10'].apply(cat10_mini_fn).value_counts())
```




    270



```
train_cop['cat10'].apply(cat10_mini_fn).value_counts()
```




    DJ             31584
    HK             30998
    CAT10MIDPOZ    30430
    DP             23679
    CAT10HIPOZ     23662
    GS             16619
    CR             14382
    HX             13171
    CK             10587
    DC             10283
    CAT10LOPOZ     10127
    HQ              9580
    MD              6817
    LN              6709
    LF              6214
    IE              6190
    LM              5917
    KW              5160
    HG              4716
    IG              4442
    HB              4212
    DF              3800
    LB              3718
    HV              3446
    LO              3154
    HC              3011
    HJ              2834
    EK              2508
    GC              2050
    Name: cat10, dtype: int64



#Encoding


```
#adapted from @AndresHG 

#Label Encoding
def label_encoder(train_df, test_df, column):
  le = LabelEncoder()
  new_feature = f'{column}_le'
  le.fit(train_df[column].unique().tolist() + test_df[column].unique().tolist())
  train_df[new_feature] = le.transform(train_df[column])
  test_df[new_feature] = le.transform(test_df[column])
  return new_feature

#Leave One Out Encoding
def loo_encoder(train_df, test_df, column):
  loo = LeaveOneOutEncoder()
  new_feature = f'{column}_loo'
  loo.fit(train_df[column], train_df['target'])
  train_df[new_feature] = loo.transform(train_df[column])
  test_df[new_feature] = loo.transform(test_df[column])
  return new_feature

#for collecting encoded column names
label_encoded_features  = []
loo_encoded_features = []

for feature in cat_columns:
  label_encoded_features.append(label_encoder(train_clean, test_clean, feature))
  loo_encoded_features.append(loo_encoder(train_clean, test_clean, feature))

xgb_cat_features = deepcopy(loo_encoded_features)
lgb_cat_features = deepcopy(label_encoded_features)
cb_cat_features = deepcopy(list(cat_columns))
ridge_cat_features = deepcopy(loo_encoded_features)


```

```
train_clean.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>cat0</th>
      <th>cat1</th>
      <th>cat2</th>
      <th>cat3</th>
      <th>cat4</th>
      <th>cat5</th>
      <th>cat6</th>
      <th>cat7</th>
      <th>cat8</th>
      <th>cat9</th>
      <th>cat10</th>
      <th>cat11</th>
      <th>cat12</th>
      <th>cat13</th>
      <th>cat14</th>
      <th>cat15</th>
      <th>cat16</th>
      <th>cat17</th>
      <th>cat18</th>
      <th>cont0</th>
      <th>cont1</th>
      <th>cont2</th>
      <th>cont3</th>
      <th>cont4</th>
      <th>cont5</th>
      <th>cont6</th>
      <th>cont7</th>
      <th>cont8</th>
      <th>cont9</th>
      <th>cont10</th>
      <th>target</th>
      <th>cat0_le</th>
      <th>cat0_loo</th>
      <th>cat1_le</th>
      <th>cat1_loo</th>
      <th>cat2_le</th>
      <th>cat2_loo</th>
      <th>cat3_le</th>
      <th>cat3_loo</th>
      <th>cat4_le</th>
      <th>cat4_loo</th>
      <th>cat5_le</th>
      <th>cat5_loo</th>
      <th>cat6_le</th>
      <th>cat6_loo</th>
      <th>cat7_le</th>
      <th>cat7_loo</th>
      <th>cat8_le</th>
      <th>cat8_loo</th>
      <th>cat9_le</th>
      <th>cat9_loo</th>
      <th>cat10_le</th>
      <th>cat10_loo</th>
      <th>cat11_le</th>
      <th>cat11_loo</th>
      <th>cat12_le</th>
      <th>cat12_loo</th>
      <th>cat13_le</th>
      <th>cat13_loo</th>
      <th>cat14_le</th>
      <th>cat14_loo</th>
      <th>cat15_le</th>
      <th>cat15_loo</th>
      <th>cat16_le</th>
      <th>cat16_loo</th>
      <th>cat17_le</th>
      <th>cat17_loo</th>
      <th>cat18_le</th>
      <th>cat18_loo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>A</td>
      <td>I</td>
      <td>A</td>
      <td>B</td>
      <td>CAT4SMALL</td>
      <td>BI</td>
      <td>A</td>
      <td>S</td>
      <td>Q</td>
      <td>A</td>
      <td>LO</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>D</td>
      <td>D</td>
      <td>B</td>
      <td>0.629858</td>
      <td>0.855349</td>
      <td>0.759439</td>
      <td>0.795549</td>
      <td>0.681917</td>
      <td>0.621672</td>
      <td>0.592184</td>
      <td>0.791921</td>
      <td>0.815254</td>
      <td>0.965006</td>
      <td>0.665915</td>
      <td>0</td>
      <td>0</td>
      <td>0.33407</td>
      <td>8</td>
      <td>0.132168</td>
      <td>0</td>
      <td>0.209770</td>
      <td>1</td>
      <td>0.266801</td>
      <td>1</td>
      <td>0.410256</td>
      <td>1</td>
      <td>0.264102</td>
      <td>0</td>
      <td>0.213836</td>
      <td>29</td>
      <td>0.229390</td>
      <td>41</td>
      <td>0.189876</td>
      <td>0</td>
      <td>0.312848</td>
      <td>35</td>
      <td>0.343056</td>
      <td>0</td>
      <td>0.214697</td>
      <td>0</td>
      <td>0.279849</td>
      <td>0</td>
      <td>0.250547</td>
      <td>0</td>
      <td>0.140230</td>
      <td>1</td>
      <td>0.132664</td>
      <td>3</td>
      <td>0.115391</td>
      <td>3</td>
      <td>0.210302</td>
      <td>1</td>
      <td>0.189505</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>A</td>
      <td>I</td>
      <td>A</td>
      <td>A</td>
      <td>E</td>
      <td>BI</td>
      <td>K</td>
      <td>W</td>
      <td>AD</td>
      <td>F</td>
      <td>HJ</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>B</td>
      <td>D</td>
      <td>B</td>
      <td>D</td>
      <td>B</td>
      <td>0.370727</td>
      <td>0.328929</td>
      <td>0.386385</td>
      <td>0.541366</td>
      <td>0.388982</td>
      <td>0.357778</td>
      <td>0.600044</td>
      <td>0.408701</td>
      <td>0.399353</td>
      <td>0.927406</td>
      <td>0.493729</td>
      <td>0</td>
      <td>0</td>
      <td>0.33407</td>
      <td>8</td>
      <td>0.132168</td>
      <td>0</td>
      <td>0.209770</td>
      <td>0</td>
      <td>0.257542</td>
      <td>3</td>
      <td>0.181103</td>
      <td>1</td>
      <td>0.264102</td>
      <td>6</td>
      <td>0.150773</td>
      <td>32</td>
      <td>0.198033</td>
      <td>3</td>
      <td>0.252404</td>
      <td>4</td>
      <td>0.153573</td>
      <td>20</td>
      <td>0.099153</td>
      <td>0</td>
      <td>0.214697</td>
      <td>1</td>
      <td>0.175008</td>
      <td>0</td>
      <td>0.250547</td>
      <td>1</td>
      <td>0.407633</td>
      <td>3</td>
      <td>0.598031</td>
      <td>1</td>
      <td>0.633125</td>
      <td>3</td>
      <td>0.210302</td>
      <td>1</td>
      <td>0.189505</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>A</td>
      <td>K</td>
      <td>A</td>
      <td>A</td>
      <td>E</td>
      <td>BI</td>
      <td>A</td>
      <td>E</td>
      <td>BM</td>
      <td>L</td>
      <td>DJ</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>D</td>
      <td>D</td>
      <td>B</td>
      <td>0.502272</td>
      <td>0.322749</td>
      <td>0.343255</td>
      <td>0.616352</td>
      <td>0.793687</td>
      <td>0.552877</td>
      <td>0.352113</td>
      <td>0.388835</td>
      <td>0.412303</td>
      <td>0.292696</td>
      <td>0.549452</td>
      <td>0</td>
      <td>0</td>
      <td>0.33407</td>
      <td>10</td>
      <td>0.239408</td>
      <td>0</td>
      <td>0.209770</td>
      <td>0</td>
      <td>0.257542</td>
      <td>3</td>
      <td>0.181103</td>
      <td>1</td>
      <td>0.264102</td>
      <td>0</td>
      <td>0.213836</td>
      <td>21</td>
      <td>0.179970</td>
      <td>26</td>
      <td>0.218122</td>
      <td>7</td>
      <td>0.122760</td>
      <td>11</td>
      <td>0.237842</td>
      <td>0</td>
      <td>0.214697</td>
      <td>1</td>
      <td>0.175008</td>
      <td>0</td>
      <td>0.250547</td>
      <td>0</td>
      <td>0.140230</td>
      <td>1</td>
      <td>0.132664</td>
      <td>3</td>
      <td>0.115391</td>
      <td>3</td>
      <td>0.210302</td>
      <td>1</td>
      <td>0.189505</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>A</td>
      <td>K</td>
      <td>A</td>
      <td>C</td>
      <td>E</td>
      <td>BI</td>
      <td>A</td>
      <td>Y</td>
      <td>AD</td>
      <td>F</td>
      <td>CAT10LOPOZ</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>D</td>
      <td>D</td>
      <td>B</td>
      <td>0.934242</td>
      <td>0.707663</td>
      <td>0.831147</td>
      <td>0.807807</td>
      <td>0.800032</td>
      <td>0.619147</td>
      <td>0.221789</td>
      <td>0.897617</td>
      <td>0.633669</td>
      <td>0.760318</td>
      <td>0.934242</td>
      <td>0</td>
      <td>0</td>
      <td>0.33407</td>
      <td>10</td>
      <td>0.239408</td>
      <td>0</td>
      <td>0.209770</td>
      <td>2</td>
      <td>0.320612</td>
      <td>3</td>
      <td>0.181103</td>
      <td>1</td>
      <td>0.264102</td>
      <td>0</td>
      <td>0.213836</td>
      <td>33</td>
      <td>0.055461</td>
      <td>3</td>
      <td>0.252404</td>
      <td>4</td>
      <td>0.153573</td>
      <td>4</td>
      <td>0.074356</td>
      <td>0</td>
      <td>0.214697</td>
      <td>0</td>
      <td>0.279849</td>
      <td>0</td>
      <td>0.250547</td>
      <td>0</td>
      <td>0.140230</td>
      <td>1</td>
      <td>0.132664</td>
      <td>3</td>
      <td>0.115391</td>
      <td>3</td>
      <td>0.210302</td>
      <td>1</td>
      <td>0.189505</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>A</td>
      <td>I</td>
      <td>G</td>
      <td>B</td>
      <td>E</td>
      <td>BI</td>
      <td>C</td>
      <td>G</td>
      <td>Q</td>
      <td>A</td>
      <td>DP</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>B</td>
      <td>B</td>
      <td>D</td>
      <td>B</td>
      <td>0.254427</td>
      <td>0.274514</td>
      <td>0.338818</td>
      <td>0.277308</td>
      <td>0.610578</td>
      <td>0.128291</td>
      <td>0.578764</td>
      <td>0.279167</td>
      <td>0.351103</td>
      <td>0.357084</td>
      <td>0.328960</td>
      <td>1</td>
      <td>0</td>
      <td>0.33407</td>
      <td>8</td>
      <td>0.132168</td>
      <td>5</td>
      <td>0.310617</td>
      <td>1</td>
      <td>0.266801</td>
      <td>3</td>
      <td>0.181103</td>
      <td>1</td>
      <td>0.264102</td>
      <td>1</td>
      <td>0.317905</td>
      <td>23</td>
      <td>0.344236</td>
      <td>41</td>
      <td>0.189876</td>
      <td>0</td>
      <td>0.312848</td>
      <td>12</td>
      <td>0.249377</td>
      <td>0</td>
      <td>0.214697</td>
      <td>0</td>
      <td>0.279849</td>
      <td>0</td>
      <td>0.250547</td>
      <td>1</td>
      <td>0.407633</td>
      <td>1</td>
      <td>0.132664</td>
      <td>1</td>
      <td>0.633125</td>
      <td>3</td>
      <td>0.210302</td>
      <td>1</td>
      <td>0.189505</td>
    </tr>
  </tbody>
</table>
</div>



# Baseline models

## Function for plotting

```
def Plot_AUC_Curve(predictions, y_valid): 
  # Plot AUC curve
  plotting_df = pd.DataFrame({'preds': predictions, 'actuals': y_valid})
  plotting_df = plotting_df.sort_values('preds', ascending=False)

  class_count = plotting_df['actuals'].value_counts()
  pos_count = class_count[1]
  neg_count = class_count[0]

  coords = [(0,0)]
  fp = 0 #horizontal
  tp = 0 #vertical

  for actual in plotting_df['actuals']:
    if actual == 1:
      tp += 1
    else:
      fp += 1
    coords.append((fp, tp))

  fp, tp = map(list, zip(*coords))  #unpacks the coordinates and then produces separate lists

  tpr = tp/pos_count
  fpr = fp/neg_count

  plt.scatter(fpr, tpr)
```

## XGBoost: 88.36

```
# Set x and y
y = train_clean["target"]
xgb_cls = XGBClassifier()

xgb_features = xgb_cat_features + list(cont_columns)
x = train_clean[xgb_features]

x_train, x_valid, y_train, y_valid=train_test_split(x, y, test_size=0.2, random_state=42)
```

```
# Fit XGB model
xgb_cls.fit(x_train, y_train, verbose=False)
predictions = xgb_cls.predict_proba(x_valid)[:,1]

auc = roc_auc_score(y_valid, predictions)

print(f'Baseline Score: {auc}')
```

    Baseline Score: 0.8836326133856207


```
Plot_AUC_Curve(predictions, y_valid)
```


![png](Kaggle_Tabular_Mar_2021_files/output_81_0.png)


## LGBM: 88.93

```
# Set x and y
y = train_clean["target"]
lgbm_features = lgb_cat_features + list(cont_columns)
x = train_clean[lgbm_features]

x_train, x_valid, y_train, y_valid=train_test_split(x, y, test_size=0.2, random_state=42)
```

```
# Use LGBM Classifier as first base model
lgbm = LGBMClassifier()

lgbm.fit(x_train, y_train,
         eval_set=(x_valid,y_valid),
         early_stopping_rounds=150, verbose=False)
predictions = lgbm.predict_proba(x_valid)[:,1]

auc = roc_auc_score(y_valid, predictions)

print(f'Baseline Score: {auc}')
```

    Baseline Score: 0.8893417667451045


```
Plot_AUC_Curve(predictions, y_valid)

```


![png](Kaggle_Tabular_Mar_2021_files/output_85_0.png)


## CatBoost: 88.83

```
# Set x and y
y = train_clean["target"]
cb_features = cb_cat_features + list(cont_columns)
x = train_clean[cb_features]

x_train, x_valid, y_train, y_valid=train_test_split(x, y, test_size=0.2, random_state=42)
```

```
# Fit Model
cat_cls = CatBoostClassifier(
    verbose = 1,
    eval_metric = 'AUC',
    loss_function = 'Logloss',
    task_type = 'GPU',
    cat_features = [x for x in range(len(cb_cat_features))]    
)

cat_cls.fit(x_train, y_train, verbose=False)
predictions = cat_cls.predict_proba(x_valid)[:,1]

auc = roc_auc_score(y_valid, predictions)

print(f'Baseline Score: {auc}')
```

    Baseline Score: 0.8882935937104544


```
Plot_AUC_Curve(predictions, y_valid)
```


![png](Kaggle_Tabular_Mar_2021_files/output_89_0.png)


## Ridge Model: 87.50

```
# Set x and y
y = train_clean["target"]
ridge_features = ridge_cat_features + list(cont_columns)
x = train_clean[ridge_features]

x_train, x_valid, y_train, y_valid=train_test_split(x, y, test_size=0.2, random_state=42)
```

```
# Fit Model
ridge_cls = RidgeClassifier(random_state=42)

ridge_cls.fit(x_train, y_train)
predictions = ridge_cls.decision_function(x_valid)

auc = roc_auc_score(y_valid, predictions)

print(f'Baseline Score: {auc}')
```

    Baseline Score: 0.8749915125973005


```
Plot_AUC_Curve(predictions, y_valid)
```


![png](Kaggle_Tabular_Mar_2021_files/output_93_0.png)


# Essemble models


```
# Prepare four different models for k-fold cross validation
random_state = 42
n_folds = 10
k_fold = StratifiedKFold(n_splits=n_folds,
                         random_state=random_state,
                         shuffle=True)

y = train_clean["target"]

# XGB model uses leave one out encoded cat features
xgb_train_preds = np.zeros(len(train_clean.index), )
xgb_test_preds = np.zeros(len(test_clean.index), )
xgb_features = xgb_cat_features + list(cont_columns)

# LGB model uses label encoded cat features
lgbm_train_preds = np.zeros(len(train_clean.index), )
lgbm_test_preds = np.zeros(len(test_clean.index), )
lgbm_features = lgb_cat_features + list(cont_columns)

# Cat Boost uses cat features as is
cb_train_preds = np.zeros(len(train_clean.index), )
cb_test_preds = np.zeros(len(test_clean.index), )
cb_features = cb_cat_features + list(cont_columns)

# Ridge uses leave one out encoded cat features
ridge_train_preds = np.zeros(len(train_clean.index), )
ridge_test_preds = np.zeros(len(test_clean.index), )
ridge_features = ridge_cat_features + list(cont_columns)
```

## K Fold Cross Validation, First Level

* XGB - ROC AUC Score = 0.8833
* LGB - ROC AUC Score = 0.8920
* CB - ROC AUC Score = 0.8895
* Ridge - ROC AUC Score = 0.875

```
for fold, (train_index, valid_index) in enumerate(k_fold.split(train_clean, y)):
    print("--> Fold {}".format(fold + 1))
    y_train = y.iloc[train_index]
    y_valid = y.iloc[valid_index]

    ########## Generate train and valid sets ##########
    xgb_x_train = pd.DataFrame(train_clean[xgb_features].iloc[train_index])
    xgb_x_valid = pd.DataFrame(train_clean[xgb_features].iloc[valid_index])

    lgbm_x_train = pd.DataFrame(train_clean[lgbm_features].iloc[train_index])
    lgbm_x_valid = pd.DataFrame(train_clean[lgbm_features].iloc[valid_index])
    
    cb_x_train = pd.DataFrame(train_clean[cb_features].iloc[train_index])
    cb_x_valid = pd.DataFrame(train_clean[cb_features].iloc[valid_index])

    ridge_x_train = pd.DataFrame(train_clean[ridge_features].iloc[train_index])
    ridge_x_valid = pd.DataFrame(train_clean[ridge_features].iloc[valid_index])

    ########## XGBoost model ##########
    xgb_model = XGBClassifier(
        seed=random_state,
        verbosity=1,
        eval_metric="auc",
        tree_method="gpu_hist",
        gpu_id=0,
        n_jobs = 12,
    )
    xgb_model.fit(
        xgb_x_train,
        y_train,
        eval_set=[(xgb_x_valid, y_valid)], 
        verbose=0,
        early_stopping_rounds=200
    )

    train_oof_preds = xgb_model.predict_proba(xgb_x_valid)[:,1]
    test_oof_preds = xgb_model.predict_proba(test_clean[xgb_features])[:,1]
    xgb_train_preds[valid_index] = train_oof_preds
    xgb_test_preds += test_oof_preds / n_folds
    
    print(": XGB - ROC AUC Score = {}".format(roc_auc_score(y_valid, train_oof_preds, average="micro")))
    
    ########## LGBM model ##########
    lgbm_model = LGBMClassifier(
        cat_feature=[x for x in range(len(lgb_cat_features))],
        random_state=random_state,
        metric="auc",
        n_jobs=12,
    )
    lgbm_model.fit(
        lgbm_x_train,
        y_train,
        eval_set=[(lgbm_x_valid, y_valid)], 
        verbose=0,
    )

    train_oof_preds = lgbm_model.predict_proba(lgbm_x_valid)[:,1]
    test_oof_preds = lgbm_model.predict_proba(test_clean[lgbm_features])[:,1]
    lgbm_train_preds[valid_index] = train_oof_preds
    lgbm_test_preds += test_oof_preds / n_folds
    
    print(": LGB - ROC AUC Score = {}".format(roc_auc_score(y_valid, train_oof_preds, average="micro")))

    ########## CatBoost model ##########
    cb_model = CatBoostClassifier(
        verbose=0,
        eval_metric="AUC",
        loss_function="Logloss",
        random_state=random_state,
        task_type="GPU",
        devices="0",
        cat_features=[x for x in range(len(cb_cat_features))],
    )
    cb_model.fit(
        cb_x_train,
        y_train,
        eval_set=[(cb_x_valid, y_valid)], 
        verbose=0,
    )

    train_oof_preds = cb_model.predict_proba(cb_x_valid)[:,1]
    test_oof_preds = cb_model.predict_proba(test_clean[cb_features])[:,1]
    cb_train_preds[valid_index] = train_oof_preds
    cb_test_preds += test_oof_preds / n_folds
    
    print(": CB - ROC AUC Score = {}".format(roc_auc_score(y_valid, train_oof_preds, average="micro")))
    
    ########## Ridge model ##########
    ridge_model = RidgeClassifier(
        random_state=random_state,
    )
    ridge_model.fit(
        ridge_x_train,
        y_train,
    )

    train_oof_preds = ridge_model.decision_function(ridge_x_valid)
    test_oof_preds = ridge_model.decision_function(test_clean[ridge_features])
    ridge_train_preds[valid_index] = train_oof_preds
    ridge_test_preds += test_oof_preds / n_folds
    
    print(": Ridge - ROC AUC Score = {}".format(roc_auc_score(y_valid, train_oof_preds, average="micro")))
    print("")


print("--> Overall metrics")
print(": XGB - ROC AUC Score = {}".format(
    roc_auc_score(y, xgb_train_preds, average="micro")
))
print(": LGB - ROC AUC Score = {}".format(
    roc_auc_score(y, lgbm_train_preds, average="micro")
))
print(": CB - ROC AUC Score = {}".format(
    roc_auc_score(y, cb_train_preds, average="micro")
))
print(": Ridge - ROC AUC Score = {}".format(
    roc_auc_score(y, ridge_train_preds, average="micro")
))
```

    --> Fold 1
    : XGB - ROC AUC Score = 0.8817187098660038
    : LGB - ROC AUC Score = 0.8905359287779799
    : CB - ROC AUC Score = 0.8870812166398149
    : Ridge - ROC AUC Score = 0.8738232810748878
    
    --> Fold 2
    : XGB - ROC AUC Score = 0.8834380641014523
    : LGB - ROC AUC Score = 0.8920187460150613
    : CB - ROC AUC Score = 0.8888588363217382
    : Ridge - ROC AUC Score = 0.8748566232333965
    
    --> Fold 3
    : XGB - ROC AUC Score = 0.8812229328597396
    : LGB - ROC AUC Score = 0.8907625052125333
    : CB - ROC AUC Score = 0.8878478690533551
    : Ridge - ROC AUC Score = 0.8732208995009413
    
    --> Fold 4
    : XGB - ROC AUC Score = 0.8830248362307551
    : LGB - ROC AUC Score = 0.8916760124583569
    : CB - ROC AUC Score = 0.8896774856745351
    : Ridge - ROC AUC Score = 0.8741859186399463
    
    --> Fold 5
    : XGB - ROC AUC Score = 0.885059427616871
    : LGB - ROC AUC Score = 0.8942221619674527
    : CB - ROC AUC Score = 0.891589982403898
    : Ridge - ROC AUC Score = 0.877637135593158
    
    --> Fold 6
    : XGB - ROC AUC Score = 0.8837433606607911
    : LGB - ROC AUC Score = 0.8915335972242673
    : CB - ROC AUC Score = 0.8886989651353675
    : Ridge - ROC AUC Score = 0.8761425659749972
    
    --> Fold 7
    : XGB - ROC AUC Score = 0.8834450402052979
    : LGB - ROC AUC Score = 0.8925625682616756
    : CB - ROC AUC Score = 0.8903226825508568
    : Ridge - ROC AUC Score = 0.8743578360882543
    
    --> Fold 8
    : XGB - ROC AUC Score = 0.8821838433731669
    : LGB - ROC AUC Score = 0.8898527585003981
    : CB - ROC AUC Score = 0.8882713199833893
    : Ridge - ROC AUC Score = 0.8751758177893947
    
    --> Fold 9
    : XGB - ROC AUC Score = 0.8864643835460411
    : LGB - ROC AUC Score = 0.894987102453669
    : CB - ROC AUC Score = 0.8924243814880762
    : Ridge - ROC AUC Score = 0.8783978704445814
    
    --> Fold 10
    : XGB - ROC AUC Score = 0.8828966954821897
    : LGB - ROC AUC Score = 0.8921953587098029
    : CB - ROC AUC Score = 0.8904678891936502
    : Ridge - ROC AUC Score = 0.874318433169834
    
    --> Overall metrics
    : XGB - ROC AUC Score = 0.8833117954380614
    : LGB - ROC AUC Score = 0.8920333076878812
    : CB - ROC AUC Score = 0.8895156261430668
    : Ridge - ROC AUC Score = 0.8752132102079165


```
#save results
os.getcwd()

path = '/gdrive/MyDrive/Kaggle/Tabular Playground Series - Mar 2021/Results and Submissions'

np.save('/'.join([path, 'xgb_train']),xgb_train_preds)
np.save('/'.join([path, 'lgbm_train']),lgbm_train_preds)
np.save('/'.join([path, 'cb_train']),cb_train_preds)
np.save('/'.join([path, 'ridge_train']),ridge_train_preds)

np.save('/'.join([path, 'xgb_test']),xgb_train_preds)
np.save('/'.join([path, 'lgbm_test']),lgbm_test_preds)
np.save('/'.join([path, 'cb_test']),cb_test_preds)
np.save('/'.join([path, 'ridge_test']),ridge_test_preds)

```

## Manual Blended classification


```
#Manually choose weightings, drop Ridge
y_train_preds = (
    0.3 * xgb_train_preds +
    0.4 * lgbm_train_preds +
    0.3 * cb_train_preds
)

print(": Essemble train test - ROC AUC Score = {}".format(
    roc_auc_score(y, y_train_preds, average="micro")
))

y_test_preds = (
    0.3 * xgb_test_preds +
    0.4 * lgbm_test_preds +
    0.3 * cb_test_preds
)

sample_sub['target'] = y_test_preds
sample_sub.to_csv('/'.join([path, 'submission_base_essemble.csv']),index=False)

```

    : Essemble train test - ROC AUC Score = 0.8909404471050102


# Second Level Classification

```
random_state = 42
n_folds = 10
k_fold = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=True)

y = train_clean['target']

l1_train = pd.DataFrame(data={
    "xgb": xgb_train_preds.tolist(),
    "lgbm": lgbm_train_preds.tolist(),
    "cb": cb_train_preds.tolist(),
    "ridge": ridge_train_preds.tolist(),
    "target": y.tolist()
})

l1_test = pd.DataFrame(data={
    "xgb": xgb_test_preds.tolist(),
    "lgbm": lgbm_test_preds.tolist(),
    "cb": cb_test_preds.tolist(),
    "ridge": ridge_test_preds.tolist(),    
})

train_preds = np.zeros(len(l1_train.index), )
test_preds = np.zeros(len(l1_test.index), )
features = ["xgb", "lgbm", "cb", "ridge"]

for fold, (train_index, valid_index) in enumerate(k_fold.split(l1_train, y)):
    print("--> Fold {}".format(fold + 1))
    y_train = y.iloc[train_index]
    y_valid = y.iloc[valid_index]

    x_train = pd.DataFrame(l1_train[features].iloc[train_index])
    x_valid = pd.DataFrame(l1_train[features].iloc[valid_index])
    
    model = CalibratedClassifierCV(
        RidgeClassifier(random_state=random_state), 
        cv=3
    )
    model.fit(
        x_train,
        y_train,
    )

    train_oof_preds = model.predict_proba(x_valid)[:,-1]
    test_oof_preds = model.predict_proba(l1_test[features])[:,-1]
    train_preds[valid_index] = train_oof_preds
    test_preds += test_oof_preds / n_folds
    
    print(": ROC AUC Score = {}".format(roc_auc_score(y_valid, train_oof_preds, average="micro")))
    print("")
    
print("--> Overall metrics")
print(": ROC AUC Score = {}".format(roc_auc_score(y, train_preds, average="micro")))
```

    --> Fold 1
    : ROC AUC Score = 0.8907889259575682
    
    --> Fold 2
    : ROC AUC Score = 0.8922990627015296
    
    --> Fold 3
    : ROC AUC Score = 0.8915431440723113
    
    --> Fold 4
    : ROC AUC Score = 0.8924845272013953
    
    --> Fold 5
    : ROC AUC Score = 0.8949637917099393
    
    --> Fold 6
    : ROC AUC Score = 0.8918673602817933
    
    --> Fold 7
    : ROC AUC Score = 0.8933580552377776
    
    --> Fold 8
    : ROC AUC Score = 0.8907186456344905
    
    --> Fold 9
    : ROC AUC Score = 0.8956073508424544
    
    --> Fold 10
    : ROC AUC Score = 0.89312838100185
    
    --> Overall metrics
    : ROC AUC Score = 0.8926600181506125


# Hyper parameter tuning with Optuna

```
#Tune LGB model since consistently highest AUC

def objective(trial, X=train_clean[lgbm_features], y=y, meta_random_seed = 42):

  X_train,X_valid,y_train,y_valid=train_test_split(X,y,test_size=0.2,random_state=meta_random_seed)

  lgb_params={
      'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2),
      'max_depth': trial.suggest_int('max_depth', 6, 200),
      'num_leaves': trial.suggest_int('num_leaves', 31, 120),
      'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0),
      'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0),
      'random_state': meta_random_seed,
      'metric': 'auc',
      'n_estimators': trial.suggest_int('n_estimators', 6, 300000),
      'n_jobs': 12,
      'cat_feature': [x for x in range(len(cat_columns))],
      'bagging_seed': 2021,
      'feature_fraction_seed': 2021,
      'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 0.9),
      'min_child_samples': trial.suggest_int('min_child_samples', 1, 500),
      'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),
      'subsample': trial.suggest_float('subsample', 0.3, 0.9),
      'max_bin': trial.suggest_int('max_bin', 128, 1024),
      'min_data_per_group': trial.suggest_int('min_data_per_group', 50, 350),
      'cat_smooth': trial.suggest_int('cat_smooth', 10, 250),
      'cat_l2': trial.suggest_int('cat_l2', 1, 20)
  }

  lgb = LGBMClassifier(
      **lgb_params
  )
  
  lgb.fit(
      X_train,
      y_train,
      eval_set=(X_valid,y_valid),
      eval_metric='auc',
      early_stopping_rounds=100,
      verbose=False
  )

  predictions=lgb.predict_proba(X_valid)[:,1]

  return roc_auc_score(y_valid,predictions)
```

```
study = optuna.create_study(direction='maximize')
study.optimize(objective, timeout=3600*7, n_trials=15)
```

    [32m[I 2021-03-23 15:28:15,613][0m A new study created in memory with name: no-name-fad9d58e-e7f0-461e-b366-ff611a08c682[0m
    [32m[I 2021-03-23 15:33:48,426][0m Trial 0 finished with value: 0.8971520973319906 and parameters: {'learning_rate': 0.006786406114306524, 'max_depth': 10, 'num_leaves': 49, 'reg_alpha': 7.18520826715226, 'reg_lambda': 3.8109203540198067, 'n_estimators': 197006, 'colsample_bytree': 0.26964063096272617, 'min_child_samples': 140, 'subsample_freq': 3, 'subsample': 0.4972731841894771, 'max_bin': 166, 'min_data_per_group': 160, 'cat_smooth': 12, 'cat_l2': 6}. Best is trial 0 with value: 0.8971520973319906.[0m
    [32m[I 2021-03-23 15:40:20,204][0m Trial 1 finished with value: 0.8966528747869886 and parameters: {'learning_rate': 0.005759410861659484, 'max_depth': 22, 'num_leaves': 98, 'reg_alpha': 7.776144748979597, 'reg_lambda': 3.2485130414362215, 'n_estimators': 297264, 'colsample_bytree': 0.657468335072982, 'min_child_samples': 36, 'subsample_freq': 5, 'subsample': 0.33432501249944235, 'max_bin': 392, 'min_data_per_group': 339, 'cat_smooth': 236, 'cat_l2': 18}. Best is trial 0 with value: 0.8971520973319906.[0m
    [32m[I 2021-03-23 15:54:44,180][0m Trial 2 finished with value: 0.897037044915039 and parameters: {'learning_rate': 0.0030106970831838623, 'max_depth': 83, 'num_leaves': 78, 'reg_alpha': 9.888107390731124, 'reg_lambda': 5.043506409769575, 'n_estimators': 39811, 'colsample_bytree': 0.5602134653916706, 'min_child_samples': 35, 'subsample_freq': 1, 'subsample': 0.5743902081697952, 'max_bin': 499, 'min_data_per_group': 255, 'cat_smooth': 117, 'cat_l2': 13}. Best is trial 0 with value: 0.8971520973319906.[0m
    [32m[I 2021-03-23 16:07:14,992][0m Trial 3 finished with value: 0.8967108501214222 and parameters: {'learning_rate': 0.003399895250036552, 'max_depth': 54, 'num_leaves': 117, 'reg_alpha': 7.729594111287509, 'reg_lambda': 5.135016973180108, 'n_estimators': 26625, 'colsample_bytree': 0.7726382854195353, 'min_child_samples': 160, 'subsample_freq': 4, 'subsample': 0.649458591608344, 'max_bin': 958, 'min_data_per_group': 178, 'cat_smooth': 234, 'cat_l2': 5}. Best is trial 0 with value: 0.8971520973319906.[0m
    [32m[I 2021-03-23 16:17:34,690][0m Trial 4 finished with value: 0.8973855695326747 and parameters: {'learning_rate': 0.004277549026567215, 'max_depth': 167, 'num_leaves': 81, 'reg_alpha': 7.942324971676128, 'reg_lambda': 0.511460245836029, 'n_estimators': 261317, 'colsample_bytree': 0.3315180208179222, 'min_child_samples': 162, 'subsample_freq': 10, 'subsample': 0.6512684717168147, 'max_bin': 606, 'min_data_per_group': 267, 'cat_smooth': 214, 'cat_l2': 20}. Best is trial 4 with value: 0.8973855695326747.[0m
    [32m[I 2021-03-23 16:21:38,401][0m Trial 5 finished with value: 0.8956368325826582 and parameters: {'learning_rate': 0.007980157556144908, 'max_depth': 198, 'num_leaves': 57, 'reg_alpha': 6.084239580524572, 'reg_lambda': 6.048731856504431, 'n_estimators': 109347, 'colsample_bytree': 0.8832669311049091, 'min_child_samples': 115, 'subsample_freq': 10, 'subsample': 0.38178147567955234, 'max_bin': 214, 'min_data_per_group': 323, 'cat_smooth': 142, 'cat_l2': 12}. Best is trial 4 with value: 0.8973855695326747.[0m
    [32m[I 2021-03-23 16:25:36,746][0m Trial 6 finished with value: 0.8972856321803562 and parameters: {'learning_rate': 0.008870412868274256, 'max_depth': 24, 'num_leaves': 71, 'reg_alpha': 0.7391761011917338, 'reg_lambda': 5.3909815733137645, 'n_estimators': 231228, 'colsample_bytree': 0.3675707137760426, 'min_child_samples': 210, 'subsample_freq': 7, 'subsample': 0.45961630835510503, 'max_bin': 990, 'min_data_per_group': 272, 'cat_smooth': 100, 'cat_l2': 9}. Best is trial 4 with value: 0.8973855695326747.[0m
    [32m[I 2021-03-23 17:07:06,453][0m Trial 7 finished with value: 0.8972560861597488 and parameters: {'learning_rate': 0.0008263750464975777, 'max_depth': 195, 'num_leaves': 75, 'reg_alpha': 1.833490376588891, 'reg_lambda': 8.659784992415881, 'n_estimators': 211981, 'colsample_bytree': 0.5814090864441459, 'min_child_samples': 424, 'subsample_freq': 7, 'subsample': 0.8279444598441861, 'max_bin': 309, 'min_data_per_group': 248, 'cat_smooth': 124, 'cat_l2': 9}. Best is trial 4 with value: 0.8973855695326747.[0m
    [32m[I 2021-03-23 17:19:05,388][0m Trial 8 finished with value: 0.8970326427620513 and parameters: {'learning_rate': 0.002696237401930733, 'max_depth': 81, 'num_leaves': 82, 'reg_alpha': 6.640836551939658, 'reg_lambda': 1.6720451088893609, 'n_estimators': 254766, 'colsample_bytree': 0.430458050965674, 'min_child_samples': 112, 'subsample_freq': 6, 'subsample': 0.3394002978154948, 'max_bin': 901, 'min_data_per_group': 314, 'cat_smooth': 28, 'cat_l2': 8}. Best is trial 4 with value: 0.8973855695326747.[0m
    [32m[I 2021-03-23 17:22:44,254][0m Trial 9 finished with value: 0.8965125807726088 and parameters: {'learning_rate': 0.008355511131656909, 'max_depth': 191, 'num_leaves': 86, 'reg_alpha': 6.934590551886994, 'reg_lambda': 0.8873477534196866, 'n_estimators': 223557, 'colsample_bytree': 0.6777667926078743, 'min_child_samples': 283, 'subsample_freq': 7, 'subsample': 0.5589425840664104, 'max_bin': 173, 'min_data_per_group': 113, 'cat_smooth': 172, 'cat_l2': 3}. Best is trial 4 with value: 0.8973855695326747.[0m
    [32m[I 2021-03-23 17:23:04,154][0m Trial 10 finished with value: 0.8822877570850877 and parameters: {'learning_rate': 0.00012495563103414892, 'max_depth': 133, 'num_leaves': 117, 'reg_alpha': 3.368879789513586, 'reg_lambda': 0.0022318372977182532, 'n_estimators': 137141, 'colsample_bytree': 0.20417467985641397, 'min_child_samples': 323, 'subsample_freq': 10, 'subsample': 0.7438438906238584, 'max_bin': 712, 'min_data_per_group': 72, 'cat_smooth': 174, 'cat_l2': 20}. Best is trial 4 with value: 0.8973855695326747.[0m
    [32m[I 2021-03-23 17:27:08,970][0m Trial 11 finished with value: 0.8973490645748272 and parameters: {'learning_rate': 0.009755268888745273, 'max_depth': 145, 'num_leaves': 63, 'reg_alpha': 3.854759870966789, 'reg_lambda': 7.468087976605721, 'n_estimators': 290228, 'colsample_bytree': 0.34002195630753823, 'min_child_samples': 234, 'subsample_freq': 9, 'subsample': 0.4673513130203878, 'max_bin': 737, 'min_data_per_group': 261, 'cat_smooth': 66, 'cat_l2': 16}. Best is trial 4 with value: 0.8973855695326747.[0m
    [32m[I 2021-03-23 17:30:49,452][0m Trial 12 finished with value: 0.8970046704787998 and parameters: {'learning_rate': 0.009988170284524318, 'max_depth': 148, 'num_leaves': 31, 'reg_alpha': 4.106863745378388, 'reg_lambda': 8.478701995598927, 'n_estimators': 294179, 'colsample_bytree': 0.3906594542338139, 'min_child_samples': 344, 'subsample_freq': 9, 'subsample': 0.6965728370927864, 'max_bin': 694, 'min_data_per_group': 224, 'cat_smooth': 59, 'cat_l2': 17}. Best is trial 4 with value: 0.8973855695326747.[0m
    [32m[I 2021-03-23 17:40:22,554][0m Trial 13 finished with value: 0.8971817640776554 and parameters: {'learning_rate': 0.004495156770393727, 'max_depth': 154, 'num_leaves': 56, 'reg_alpha': 9.643756728991963, 'reg_lambda': 7.229198081500561, 'n_estimators': 288019, 'colsample_bytree': 0.2849619745421524, 'min_child_samples': 227, 'subsample_freq': 9, 'subsample': 0.4766379376134908, 'max_bin': 739, 'min_data_per_group': 279, 'cat_smooth': 81, 'cat_l2': 15}. Best is trial 4 with value: 0.8973855695326747.[0m
    [32m[I 2021-03-23 17:49:52,133][0m Trial 14 finished with value: 0.8973488116270878 and parameters: {'learning_rate': 0.0051019922988232, 'max_depth': 166, 'num_leaves': 38, 'reg_alpha': 4.620142563695225, 'reg_lambda': 9.586813033655627, 'n_estimators': 172119, 'colsample_bytree': 0.45649863383914424, 'min_child_samples': 451, 'subsample_freq': 9, 'subsample': 0.8989990901130349, 'max_bin': 569, 'min_data_per_group': 201, 'cat_smooth': 197, 'cat_l2': 20}. Best is trial 4 with value: 0.8973855695326747.[0m


```
study.best_params

```




    {'cat_l2': 20,
     'cat_smooth': 214,
     'colsample_bytree': 0.3315180208179222,
     'learning_rate': 0.004277549026567215,
     'max_bin': 606,
     'max_depth': 167,
     'min_child_samples': 162,
     'min_data_per_group': 267,
     'n_estimators': 261317,
     'num_leaves': 81,
     'reg_alpha': 7.942324971676128,
     'reg_lambda': 0.511460245836029,
     'subsample': 0.6512684717168147,
     'subsample_freq': 10}



```
!pip install joblib
import joblib
import pickle

path = '/gdrive/MyDrive/Kaggle/Tabular Playground Series - Mar 2021/Results and Submissions'


joblib.dump(study, '/'.join([path, 'lgb_study.pkl']))


```

    Requirement already satisfied: joblib in /usr/local/lib/python3.7/dist-packages (1.0.1)





    ['/gdrive/MyDrive/Kaggle/Tabular Playground Series - Mar 2021/Results and Submissions/lgb_study.pkl']



## LGM Model with optimized hyperparameters

```
path = '/gdrive/MyDrive/Kaggle/Tabular Playground Series - Mar 2021/Results and Submissions'

study = joblib.load('/'.join([path, 'lgb_study.pkl']))

lgb_opt_params = {**study.best_params, **{
    'random_state': 32,
    'metric': 'auc',
    'n_jobs': 12,
    'cat_feature': [x for x in range(len(cat_columns))],
    'bagging_seed': 2021,
    'feature_fraction_seed': 2021}}

print(lgb_opt_params)
```

    {'learning_rate': 0.004277549026567215, 'max_depth': 167, 'num_leaves': 81, 'reg_alpha': 7.942324971676128, 'reg_lambda': 0.511460245836029, 'n_estimators': 261317, 'colsample_bytree': 0.3315180208179222, 'min_child_samples': 162, 'subsample_freq': 10, 'subsample': 0.6512684717168147, 'max_bin': 606, 'min_data_per_group': 267, 'cat_smooth': 214, 'cat_l2': 20, 'random_state': 32, 'metric': 'auc', 'n_jobs': 12, 'cat_feature': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], 'bagging_seed': 2021, 'feature_fraction_seed': 2021}


```
########## Set up cross val ##########
random_state = 42
n_folds = 10
k_fold = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=True)
y = train_clean["target"]

########## 
lgbm_opt_train_preds = np.zeros(len(train_clean.index), )
lgbm_opt_test_preds = np.zeros(len(test_clean.index), )

for fold, (train_index, valid_index) in enumerate(k_fold.split(train_clean, y)):
  print("--> Fold {}".format(fold + 1))
  y_train = y.iloc[train_index]
  y_valid = y.iloc[valid_index]

  ########## LGBM Optimized X and Y ##############
  lgbm_x_train = pd.DataFrame(train_clean[lgbm_features].iloc[train_index])
  lgbm_x_valid = pd.DataFrame(train_clean[lgbm_features].iloc[valid_index])

  ########## LGBM Optimized model ##########
  lgbm_opt_model = LGBMClassifier(
      **lgb_opt_params
  )

  ########## LGBM Optimized Fit ###############
  lgbm_opt_model.fit(
      lgbm_x_train,
      y_train,
      eval_set=[(lgbm_x_valid, y_valid)], 
      eval_metric='auc',
      early_stopping_rounds=200,
      verbose=0,
  )

  ########## Collect Out of Fold predictions ####
  train_oof_preds = lgbm_model.predict_proba(lgbm_x_valid)[:,1]
  test_oof_preds = lgbm_model.predict_proba(test_clean[lgbm_features])[:,1]
  lgbm_opt_train_preds[valid_index] = train_oof_preds
  lgbm_opt_test_preds += test_oof_preds / n_folds

  sample_sub['target'] = lgbm_opt_test_preds
  save_file_name = f'lgbm_opt_{fold}.csv'
  sample_sub.to_csv('/'.join([path, save_file_name]),index=False)

  print(": LGB - ROC AUC Score = {}".format(roc_auc_score(y_valid, train_oof_preds, average="micro")))


sample_sub['target'] = lgbm_opt_test_preds
sample_sub.to_csv('/'.join([path, 'lgbm_opt.csv']),index=False)

```

    --> Fold 1
    : LGB - ROC AUC Score = 0.8996460670147416
    --> Fold 2
    : LGB - ROC AUC Score = 0.9004835875130743
    --> Fold 3
    : LGB - ROC AUC Score = 0.8989699099327644
    --> Fold 4
    : LGB - ROC AUC Score = 0.9003322417247772
    --> Fold 5
    : LGB - ROC AUC Score = 0.902690595659634
    --> Fold 6
    : LGB - ROC AUC Score = 0.8999689079759401
    --> Fold 7
    : LGB - ROC AUC Score = 0.9011589628149071
    --> Fold 8
    : LGB - ROC AUC Score = 0.8988345278667643
    --> Fold 9
    : LGB - ROC AUC Score = 0.9031854339590824
    --> Fold 10
    : LGB - ROC AUC Score = 0.8921953587098029

