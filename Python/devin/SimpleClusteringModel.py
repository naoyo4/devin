"""
K-meansクラスタリングを使用した簡単なクラスタ分析の実装。
アヤメデータセットを使用して、教師なし学習によるクラスタリングを行い、散布図で可視化します。
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.datasets import load_iris, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import matplotlib.font_manager as fm
import os

plt.rcParams['font.family'] = 'sans-serif'
font_candidates = ['IPAGothic', 'IPAPGothic', 'VL Gothic', 'Noto Sans CJK JP', 
                  'Meiryo', 'MS Gothic', 'Hiragino Sans GB', 'TakaoGothic']

font_found = False
for font in font_candidates:
    if any(f.name == font for f in fm.fontManager.ttflist):
        plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
        logging.info(f"日本語フォント '{font}' を使用します")
        font_found = True
        break

if not font_found:
    logging.warning("日本語フォントが見つかりませんでした。代替設定を使用します。")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def load_data(use_synthetic=False):
    """データセットを読み込む"""
    if use_synthetic:
        logging.info("合成データセットを生成しています...")
        X, y_true = make_blobs(
            n_samples=300, 
            centers=4, 
            cluster_std=0.60, 
            random_state=42
        )
        feature_names = [f"特徴量{i+1}" for i in range(X.shape[1])]
        return X, y_true, feature_names
    else:
        logging.info("アヤメデータセットを読み込んでいます...")
        iris = load_iris()
        X = iris.data
        y_true = iris.target  # 実際のクラスラベル（評価用）
        feature_names = iris.feature_names
        
        logging.info(f"データセットの形状: {X.shape}")
        logging.info(f"特徴量: {feature_names}")
        
        return X, y_true, feature_names

def preprocess_data(X):
    """データの前処理: スケーリング"""
    logging.info("データを前処理しています...")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    logging.info(f"スケーリング後のデータ形状: {X_scaled.shape}")
    
    return X_scaled, scaler

def find_optimal_clusters(X, max_clusters=10):
    """最適なクラスタ数を見つける"""
    logging.info("最適なクラスタ数を探索しています...")
    
    silhouette_scores = []
    calinski_scores = []
    inertia_values = []
    
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        
        calinski_avg = calinski_harabasz_score(X, cluster_labels)
        calinski_scores.append(calinski_avg)
        
        inertia_values.append(kmeans.inertia_)
        
        logging.info(f"クラスタ数 {n_clusters}: シルエットスコア = {silhouette_avg:.4f}, "
                    f"Calinski-Harabasz指標 = {calinski_avg:.4f}, "
                    f"イナーシャ = {kmeans.inertia_:.4f}")
    
    optimal_n_clusters = np.argmax(silhouette_scores) + 2  # +2 because we started from 2
    logging.info(f"最適なクラスタ数: {optimal_n_clusters}")
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(range(2, max_clusters + 1), silhouette_scores, 'o-')
    plt.axvline(x=optimal_n_clusters, color='r', linestyle='--')
    plt.xlabel('クラスタ数')
    plt.ylabel('シルエットスコア')
    plt.title('シルエット分析')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(range(2, max_clusters + 1), calinski_scores, 'o-')
    plt.xlabel('クラスタ数')
    plt.ylabel('Calinski-Harabasz指標')
    plt.title('Calinski-Harabasz分析')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(range(2, max_clusters + 1), inertia_values, 'o-')
    plt.xlabel('クラスタ数')
    plt.ylabel('イナーシャ')
    plt.title('エルボー法')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('cluster_evaluation.png')
    logging.info("クラスタ評価指標を 'cluster_evaluation.png' に保存しました")
    
    return optimal_n_clusters

def train_kmeans_model(X, n_clusters):
    """K-meansモデルの訓練"""
    logging.info(f"K-meansモデルを訓練しています（クラスタ数: {n_clusters}）...")
    
    kmeans = KMeans(
        n_clusters=n_clusters,  # クラスタ数
        init='k-means++',       # 初期化方法
        max_iter=300,           # 最大イテレーション数
        n_init=10,              # 異なる初期値で実行する回数
        random_state=42         # 乱数シード
    )
    
    kmeans.fit(X)
    
    logging.info(f"訓練が完了しました。イテレーション数: {kmeans.n_iter_}")
    logging.info(f"最終イナーシャ: {kmeans.inertia_:.6f}")
    
    return kmeans

def visualize_clusters(X, kmeans, feature_names, y_true=None, use_pca=True):
    """クラスタリング結果の可視化"""
    logging.info("クラスタリング結果を可視化しています...")
    
    labels = kmeans.labels_
    
    if X.shape[1] > 2 and use_pca:
        logging.info("PCAで2次元に削減しています...")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        explained_variance = pca.explained_variance_ratio_
        logging.info(f"PCAの説明分散比: {explained_variance}")
        
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 2, 1)
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50, alpha=0.8)
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                   c='red', marker='X', s=200, label='クラスタ中心')
        plt.colorbar(scatter, label='クラスタラベル')
        plt.title('K-meansクラスタリング結果 (PCA)')
        plt.xlabel(f'主成分1 ({explained_variance[0]:.2%})')
        plt.ylabel(f'主成分2 ({explained_variance[1]:.2%})')
        plt.legend()
        plt.grid(True)
        
        if y_true is not None:
            plt.subplot(2, 2, 2)
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='plasma', s=50, alpha=0.8)
            plt.colorbar(scatter, label='実際のクラス')
            plt.title('実際のクラス (PCA)')
            plt.xlabel(f'主成分1 ({explained_variance[0]:.2%})')
            plt.ylabel(f'主成分2 ({explained_variance[1]:.2%})')
            plt.grid(True)
    
    if X.shape[1] >= 2:
        if use_pca:
            plt.subplot(2, 2, 3)
        else:
            plt.figure(figsize=(12, 10))
            
        scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.8)
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                   c='red', marker='X', s=200, label='クラスタ中心')
        plt.colorbar(scatter, label='クラスタラベル')
        plt.title('K-meansクラスタリング結果 (元の特徴量)')
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.legend()
        plt.grid(True)
        
        if y_true is not None and use_pca:
            plt.subplot(2, 2, 4)
            scatter = plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='plasma', s=50, alpha=0.8)
            plt.colorbar(scatter, label='実際のクラス')
            plt.title('実際のクラス (元の特徴量)')
            plt.xlabel(feature_names[0])
            plt.ylabel(feature_names[1])
            plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('clustering_results.png')
    logging.info("クラスタリング結果を 'clustering_results.png' に保存しました")
    
    if X.shape[1] >= 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis', s=50, alpha=0.8)
        ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                  kmeans.cluster_centers_[:, 2], c='red', marker='X', s=200, label='クラスタ中心')
        
        plt.colorbar(scatter, label='クラスタラベル')
        ax.set_title('K-meansクラスタリング結果 (3D)')
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.set_zlabel(feature_names[2])
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('clustering_results_3d.png')
        logging.info("3Dクラスタリング結果を 'clustering_results_3d.png' に保存しました")
    
    plt.show()

def main():
    """メイン関数"""
    logging.info("K-meansクラスタリングの実行を開始します...")
    
    X, y_true, feature_names = load_data(use_synthetic=False)
    
    X_scaled, scaler = preprocess_data(X)
    
    optimal_n_clusters = find_optimal_clusters(X_scaled, max_clusters=10)
    
    kmeans = train_kmeans_model(X_scaled, n_clusters=optimal_n_clusters)
    
    visualize_clusters(X_scaled, kmeans, feature_names, y_true)
    
    logging.info("処理が完了しました。")


if __name__ == "__main__":
    main()
