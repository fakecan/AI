import pandas as pd
import numpy as np

df = pd.read_csv('./data/train_ver2.csv')

# print(df.shape)   # (13647309, 48)
# print(df.head())
# fecha_dato  ncodpers ind_empleado pais_residencia sexo  age  fecha_alta  ind_nuevo  ... ind_pres_fin_ult1  ind_reca_fin_ult1 ind_tjcr_fin_ult1 ind_valo_fin_ult1 ind_viv_fin_ult1 ind_nomina_ult1 ind_nom_pens_ult1 ind_recibo_ult1
# 0  2015-01-28   1375586            N              ES    H   35  2015-01-12        0.0  ...                 0                  0                 0                 0                0             0.0               0.0               0
# 1  2015-01-28   1050611            N              ES    V   23  2012-08-10        0.0  ...                 0                  0                 0                 0                0             0.0               0.0               0
# 2  2015-01-28   1050612            N              ES    V   23  2012-08-10        0.0  ...                 0                  0                 0                 0                0             0.0               0.0               0
# 3  2015-01-28   1050613            N              ES    H   22  2012-08-10        0.0  ...                 0                  0                 0                 0                0             0.0               0.0               0
# 4  2015-01-28   1050614            N              ES    V   23  2012-08-10        0.0  ...                 0                  0                 0                 0                0             0.0               0.0               0

# for col in df.columns:  # 모든 column 변수 미리보기
#     print('{}\n'.format(df[col].head()))

df.info()   # 정보 요약

# num_cols = [col for col in df.columns[:24] if df[col].dtype in ['int64', 'float64']]
# df[num_cols].describe()

# cat_cols = [col for col in df.columns[:24] if df[col].dtype in ['0']]
# df[cat_cols].describe()

# for col in cat_cols = 