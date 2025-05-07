import pandas as pd
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_curve, auc
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

# شروع زمان‌سنجی
start_time = time.time()

# خواندن داده‌ها
arenges_file = pd.read_csv('./permutations_with_repetition.csv')
df = pd.read_excel('./pro_ELCR_data2.xlsx')

with open('my_intervals.json', 'r') as f:
    intervals = json.load(f)

# تعریف توابع مربوط به بازه‌ها (همانند کد شما)
# ...

# محاسبه شیب‌های میانگین (همانند کد شما)
# ...

# محاسبه معیارهای ارزیابی
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

# رسم نمودار ROC
def plot_roc_curve(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

# تقسیم داده‌ها به داده‌های آموزش و آزمون
X = df[['Age', 'weight', 'rice use', 'pasta use', 'ED', 'EF', 'As_RBA', 'Cr_RBA', 'Hg_RBA', 'Cd_RBA', 'Pb_RBA']]
y = df['cr']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# آموزش مدل رگرسیون خطی
model = LinearRegression()
model.fit(X_train, y_train)

# پیش‌بینی روی داده‌های آزمون
y_pred = model.predict(X_test)

# محاسبه معیارهای ارزیابی
mae, rmse = evaluate_model(y_test, y_pred)
print(f'MAE: {mae}')
print(f'RMSE: {rmse}')

# رسم نمودار ROC (برای مدل‌های طبقه‌بندی)
# در اینجا فرض می‌کنیم که y یک متغیر طبقه‌بندی است
# اگر y یک متغیر پیوسته است، باید آن را به طبقه‌بندی تبدیل کنید
# برای مثال، می‌توانید از label_binarize استفاده کنید
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
n_classes = y_test_bin.shape[1]

# آموزش مدل طبقه‌بندی
classifier = OneVsRestClassifier(SVC(kernel='linear', probability=True, random_state=42))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# رسم نمودار ROC برای هر کلاس
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# رسم نمودار ROC برای همه کلاس‌ها
plt.figure()
colors = ['blue', 'red', 'green']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for multi-class')
plt.legend(loc="lower right")
plt.show()

# پایان زمان‌سنجی
end_time = time.time()
print("زمان اجرای برنامه:", end_time - start_time, "ثانیه")