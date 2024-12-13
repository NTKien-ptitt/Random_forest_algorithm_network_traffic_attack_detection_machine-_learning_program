import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
sns.set(style='darkgrid')
import matplotlib.pyplot as plt
# Load dataset
dataBenign = pd.read_csv('Begin.pcap_Flow.csv')
dataAtt1 = pd.read_csv('data_kien_test.pcap_Flow.csv')
dataAtt2 = pd.read_csv('data_kiett.pcap_Flow.csv')

data_list = [dataBenign, dataAtt1, dataAtt2]
data_names = ['Data Benign', 'Data Att1', 'Data Att2']

print('Kích thước dữ liệu: ')
for name, data in zip(data_names, data_list):
    rows, cols = data.shape
    print(f'{name} -> {rows} hàng, {cols} cột')

data = pd.concat(data_list)
rows, cols = data.shape

print('Kích thước mới:')
print(f'Số hàng: {rows}')
print(f'Số cột: {cols}')
print(f'Tổng số ô: {rows * cols}')
columns_to_keep = [
    'Src Port',
    'Dst Port',
    'Flow Duration',
    'TotLen Fwd Pkts',
    'Fwd Pkt Len Max',
    'Flow Byts/s',
    'Flow Pkts/s',
    'Flow IAT Mean',
    'Flow IAT Max',
    'Bwd Pkt Len Mean',
    'FIN Flag Cnt',
    'SYN Flag Cnt',
    'PSH Flag Cnt',
    'ACK Flag Cnt',
    'Down/Up Ratio',
    'Subflow Fwd Byts',
    'Subflow Fwd Pkts',
    'Label'
]

# Lọc các cột cần giữ lại
data = data[columns_to_keep]
data.info()

# Làm sạch dữ liệu
# Xác định các giá trị trùng lặp

dups = data[data.duplicated()]
print(f'Số lượng giá trị trùng lặp: {len(dups)}')
data.drop_duplicates(inplace = True)
data.shape
#### Xác định các giá trị bị thiếu
missing_val = data.isna().sum()
print(missing_val.loc[missing_val > 0])
# Kiểm tra các giá trị vô hạn
numeric_cols = data.select_dtypes(include = np.number).columns
inf_count = np.isinf(data[numeric_cols]).sum()
print(inf_count[inf_count > 0])
# Thay thế bất kỳ giá trị vô hạn nào (dương hoặc âm) bằng NaN (không phải là số)
print(f'Giá trị ban đầu bị thiếu: {data.isna().sum().sum()}')

data.replace([np.inf, -np.inf], np.nan, inplace = True)

print(f'Các giá trị bị thiếu sau khi xử lý các giá trị vô hạn: {data.isna().sum().sum()}')
missing = data.isna().sum()
print(missing.loc[missing > 0])
# Tính toán phần trăm giá trị bị thiếu trong tập dữ liệu
mis_per = (missing / len(data)) * 100
mis_table = pd.concat([missing, mis_per.round(2)], axis = 1)
mis_table = mis_table.rename(columns = {0 : 'Giá trị bị thiếu', 1 : 'Tỷ lệ phần trăm trên tổng giá trị'})

print(mis_table.loc[mis_per > 0])
# Xử lý các giá trị bị thiếu (Các cột có dữ liệu bị thiếu)
med_flow_bytes = data['Flow Byts/s'].median()
med_flow_packets = data['Flow Pkts/s'].median()

print('Trung vị của Flow Byts/s: ', med_flow_bytes)
print('Trung vị của Flow Pkts/s/s: ', med_flow_packets)
# Điền giá trị còn thiếu bằng trung vị
data['Flow Byts/s'] = data['Flow Byts/s'].fillna(med_flow_bytes)
data['Flow Pkts/s'] = data['Flow Pkts/s'].fillna(med_flow_packets)

# 1. Tách đặc trưng (features) và nhãn (labels)
X = data.drop(columns=['Label'])
y = data['Label']

# 2. Chia dữ liệu train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 3. Huấn luyện mô hình Random Forest
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# 4. Đánh giá mô hình
from sklearn.metrics import classification_report, accuracy_score
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Đọc tập dữ liệu mới cần dự đoán
new_data = pd.read_csv('begin.pcap_Flow.csv')
new_data = new_data[columns_to_keep].drop('Label', axis=1)
# Dự đoán lớp
predictions = model.predict(new_data)
print("Predictions:", predictions)

# Lưu kết quả vào file CSV
output = pd.DataFrame({'Prediction': predictions})
output.to_csv('predictions.csv', index=False)

from joblib import dump

# Lưu mô hình sau khi huấn luyện
dump(model, 'model.pkl')  # Hoặc .joblib
