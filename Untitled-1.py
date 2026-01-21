# %%
import pandas as pd
data = pd.read_csv(r'D:\python\hydhouse\Hyderbad_House_price.csv')
data


# %%
data.info()

# %%
data=data.drop('id'
               ,axis=1)


# %%
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
category = ['title','location','building_status']

for col in category:
    data[col] = le.fit_transform(data[col])

    mapping_df = pd.DataFrame({
        'category': le.classes_,
        'code': le.transform(le.classes_)
    })
    mapping_df.to_csv(f"{col}_mapping.csv", index=False)


# %%
data

# %%
x = data.drop(['price(L)','rate_persqft'],axis = 1)
x

# %%
y = data[['price(L)','rate_persqft']]
y

# %%
from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test = tts(x,y,test_size=0.3)


# %%
x_train

# %%
x_test

# %%
y_train

# %%
y_test

# %%
from sklearn.ensemble import RandomForestRegressor
rfc = RandomForestRegressor()
rfc_model = rfc.fit(x_train,y_train)
rfc_model

# %%
rfc_predict = rfc_model.predict(x_test)
rfc_predict

# %%
from sklearn.metrics import r2_score
rfc_r2 = r2_score(y_test, rfc_predict)
rfc_r2

# %%
from lightgbm import LGBMRegressor
lgbm = LGBMRegressor()
lgbm_model = lgbm.fit(x_train, y_train['price(L)'])
lgbm_model

# %%
lgbm_predict = lgbm_model.predict(x_test)
lgbm_predict

# %%
lgbm_r2 = r2_score(y_test['price(L)'], lgbm_predict)
lgbm_r2

# %%
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

xgb = XGBRegressor()
multi_xgb = MultiOutputRegressor(xgb)
multi_xgb_model = multi_xgb.fit(x_train, y_train)
multi_xgb_model

# %%
multi_xgb_predict = multi_xgb_model.predict(x_test)
multi_xgb_predict

# %%
multi_xgb_r2 = r2_score(y_test, multi_xgb_predict)
multi_xgb_r2

# %%
print("rfc" if rfc_r2 > lgbm_r2 else "lgbm")

# %%
import pickle
with open('multi_xgb_model', 'wb') as file:
    pickle.dump(multi_xgb_model, file)
    print(type(multi_xgb_model),"Model saved successfully.")

# %%


# %%


# %%



