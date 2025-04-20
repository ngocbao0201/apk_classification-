import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer


def getModel():
    with open("RF/random_forest_model.pkl", 'rb') as file:
        model = pickle.load(file)
    return model
def getTFIDS():
    with open("RF/tfidf.pkl", 'rb') as file:
        model = pickle.load(file)
    return model

def predict(permisions, vector):
    tfidf = getTFIDS()
    pandas_permisions = pd.Series(str(permisions).replace('.', '_'))
    permission_features = tfidf.transform(pandas_permisions)
    X = np.hstack((vector, permission_features.toarray()))
    model = getModel()
    print("Malware") if model.predict(X)[0] == 0 else print("Begin")
    
def count_samples_in_tree(tree):
    """
    Đếm số lượng mẫu trong mỗi cây quyết định trong mô hình Random Forest.
    
    Parameters:
    tree (DecisionTreeClassifier or DecisionTreeRegressor): Cây quyết định trong mô hình Random Forest.
    
    Returns:
    int: Số lượng mẫu trong cây quyết định.
    """
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    n_samples = tree.tree_.n_node_samples
    
    def count_samples(node_id):
        if children_left[node_id] == children_right[node_id]:
            return n_samples[node_id]
        else:
            left_samples = count_samples(children_left[node_id])
            right_samples = count_samples(children_right[node_id])
            return left_samples + right_samples
    
    return count_samples(0)  # Bắt đầu đếm từ nút gốc

if __name__ == '__main__':

    # Đọc dữ liệu từ file CSV
    data = pd.read_csv("RF/csv/train_final.csv")

    # Xử lý cột "permission" để chuyển đổi thành đặc trưng số học
    tfidf = TfidfVectorizer()
    permission_features = tfidf.fit_transform(data['Permissions'])
    # Chọn các đặc trưng từ 1 đến 40
    X_other = data.iloc[:, 1:41]
    # Hợp nhất các đặc trưng
    X = np.hstack((X_other.values, permission_features.toarray()))
    #print(X.shape[1])
    # Chia dữ liệu thành features (X) và nhãn (y)
    y = data['label']  # Chọn cột label là nhãn

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Huấn luyện mô hình Random Forest Classifier
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    # Dự đoán trên tập kiểm tra
    y_pred = clf.predict(X_test)

    # Đánh giá mô hình
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    
        # Lấy một cây quyết định từ mô hình Random Forest
    tree = clf.estimators_[99]

    # Kiểm tra số lượng mẫu trong cây quyết định
    samples_in_tree = count_samples_in_tree(tree)
    print(samples_in_tree)

    # Lưu mô hình
    # with open('RF/random_forest_model.pkl', 'wb') as file:
    #     pickle.dump(clf, file)
    # with open('RF/tfidf.pkl', 'wb') as f:
    #     pickle.dump(tfidf, f)

    # In ra kết quả
    print(f"Độ chính xác: {accuracy:.3f}")
    print("\nBáo cáo phân loại:\n", classification_rep)