import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
    
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize 
from sklearn.model_selection import cross_validate, KFold, GridSearchCV

from scipy import interp

from imblearn.over_sampling import SMOTE
import shap
import joblib

from tableone import TableOne

# 数据处理
df = pd.read_excel("F:\\VSCode_Python\\AMI第二次处理后数据.xlsx", sheet_name="Sheet3")
df1 = df.drop(["killIP"], axis=1)
df2 = df.drop(["killip分级"], axis=1)

label_4 = df["killip分级"] - 1 
label_2 = df["killIP"]
features = df.drop(["killip分级", "killIP"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(features, label_4, test_size=0.3,
                                                    random_state=44,
                                                    stratify=label_4)

sampling_strategy = {2: 100, 3: 80}
smote = SMOTE(random_state=42, sampling_strategy = sampling_strategy)
x_resample, y_resample = smote.fit_resample(X_train, y_train)

# 模型
rf1 = RandomForestClassifier(random_state=42)
param_grid_simple = {"criterion": ["squared_error","poisson"],
                     'n_estimators': [*range(20,100,5)],
                     'max_depth': [*range(10,25,2)],
                     "max_features": ["log2","sqrt",16,32,64,"auto"],
                     "min_impurity_decrease": [*np.arange(0,5,10)]
                    }
cv = KFold(n_splits=5,shuffle=True,random_state=83)
search = GridSearchCV(estimator=rf1
                     param_grid=param_grid_simple,
                     scoring = "neg_mean_squared_error",
                     verbose = True,
                     cv = cv,
                     n_jobs=-1)
search.fit(x_resample, y_resample)
search.best_estimator_
# rf.fit(X_train, y_train)
rf = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42)
rf.fit(x_resample, y_resample)
rf_preds = rf.predict(X_test)

xgb = XGBClassifier(random_state=42)
# rf.fit(X_train, y_train)
xgb.fit(x_resample, y_resample)
xgb_preds = rf.predict(X_test)

tabnet = TabNetClassifier()
tabnet.fit(x_resample.values, y_resample.values,
           max_epochs=50,
           batch_size=32)
y_pred = tabnet.predict(X_test.values)

mlp = RandomForestClassifier(n_estimators=70, max_depth=5, random_state=42)  # 设置随机种子
# mlp.fit(X_train, y_train)
mlp.fit(x_resample, y_resample)
mlp_pred = mlp.predict(X_test)

rf_score = rf.predict_proba(X_test)
fpr = {}
tpr = {}
roc_auc = {}
for i in range(4):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], rf_score[:, i])
    roc_auc[i] =  auc(fpr[i], tpr[i])
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(4)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(4):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
mean_tpr /= 4
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# 计算各模型的ROC曲线和AUC值
roc_values = {}

# 计算RF的ROC曲线
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf.predict_proba(X_test.values))
rf_auc = auc(fpr_rf, tpr_rf)
roc_values['RF'] = (fpr_rf, tpr_rf, roc_auc_score(y_test, rf.predict_proba(X_test), multi_class="ovo", average="macro"))

# 计算XgBoost的ROC曲线
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb.predict_proba(X_test.values))
xgb_auc = auc(fpr_xgb, tpr_xgb)
roc_values['XGBoost'] = (fpr_xgb, tpr_xgb, roc_auc_score(y_test, xgb.predict_proba(X_test), multi_class="ovo", average="macro"))

# 计算MLP的ROC曲线
fpr_mlp, tpr_mlp, _ = roc_curve(y_test, mlp.predict_proba(X_test))
mlp_auc = auc(fpr_mlp, tpr_mlp)
roc_values['MLP'] = (fpr_mlp, tpr_mlp, roc_auc_score(y_test, mlp.predict_proba(X_test), multi_class="ovo", average="macro"))

# 计算Tabnet的ROC曲线
fpr_tabnet, tpr_tabnet, _ = roc_curve(y_test, tabnet.predict_proba(X_test.values))
tabnet_auc = auc(fpr_tabnet, tpr_tabnet)
roc_values['TabNet'] = (fpr_tabnet, tpr_tabnet, roc_auc_score(y_test, tabnet.predict_proba(X_test), multi_class="ovo", average="macro"))

# 按照AUC值对模型进行排序
sorted_roc_values = sorted(roc_values.items(), key=lambda x: x[1][2], reverse=True)

# 绘制ROC曲线
plt.figure(figsize=(10, 8))

for model_name, (fpr, tpr, roc_auc) in sorted_roc_values:
    plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')

# 绘制对角线
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')

# 添加图例和标签
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# shap可解释性分析
explanier = shap.TreeExplainer(rf)
shap_values = explanier.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type='bar', class_names=["killip 1", "killip 2", "killip 3", "killip 4"])
shap.summary_plot(shap_values[0], X_test)
shap.dependence_plot("GRACE risk score", shap_values[0], X_test, interaction_index=None)
shap.dependence_plot("NT-pro BNP", shap_values[0], X_test, interaction_index=None)
shap.dependence_plot("TIMI risk score", shap_values[0], X_test, interaction_index=None)
shap.dependence_plot("LVEF", shap_values[0], X_test, interaction_index=None)
shap.dependence_plot("CCR", shap_values[0], X_test, interaction_index=None)
shap.dependence_plot("Creatinine", shap_values[0], X_test, interaction_index=None)
shap.dependence_plot("Age", shap_values[0], X_test, interaction_index=None)
shap.dependence_plot("Length of Hospitalization", shap_values[0], X_test, interaction_index=None)
shap.dependence_plot("hs-CRP", shap_values[0], X_test, interaction_index=None)
exp = explanier(X_train)
shap_values_train = explanier.shap_values(X_train)
shap.initjs()
shap.force_plot(explanier.expected_value[0], shap_values[0], X_test)
