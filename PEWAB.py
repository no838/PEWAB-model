#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 10:57:18 2024

@author: raism
"""


import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from tigramite.data_processing import DataFrame
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split

# 设置全局中文字体支持
plt.rcParams['font.sans-serif'] = ['Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
output_dir = "output"
figures_dir = os.path.join(output_dir, "figures")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)



# 第一步：数据准备
data_file = 'data.csv'
data = pd.read_csv(data_file, parse_dates=['日期'], low_memory=False)

# 初始化用于存储每个断面结果的DataFrame
results_df = pd.DataFrame(columns=[
    '断面名称', 'MSE_LSTM', 'MSE_GNN', 'MSE_CNN', 'MSE_BNN', 'MSE_Ensemble', 
    'RMSE', 'MAE', 'R²', 'MAPE', 'Composite_Score',
    'LSTM Best Lead Time', 'GNN Best Lead Time', 'CNN Best Lead Time', 
    'BNN Best Lead Time', 'Ensemble Best Lead Time', 'Best_Model'
])

# 定义综合评分计算函数
def calculate_composite_score(mse, rmse, mae, r2, mape):
    scaler = MinMaxScaler()
    standardized_scores = scaler.fit_transform(np.array([mse, rmse, mae, 1-r2, mape]).reshape(-1, 1)).ravel()
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # 可以根据实际需求调整权重
    composite_score = np.dot(standardized_scores, weights)
    return composite_score

# 遍历每个断面生成预测数据
for section_name in data['断面名称'].unique():
    if section_name in skip_sections:
        print(f"Skipping section {section_name} as per the skip list.")
        continue

    try:
        print(f"Processing section: {section_name}")

        # 筛选当前断面的数据
        section_data = data[data['断面名称'] == section_name].copy()

        # 检查数据量是否足够，少于10行数据则跳过该断面
        if section_data.shape[0] < 10:
            print(f"Skipping section {section_name} due to insufficient data.")
            continue

        # 考虑的特征列
        feature_columns = [
            '水温(℃)', 'pH(无量纲)', '溶解氧(mg/L)', '电导率(μS/cm)', '浊度(NTU)',
            '高锰酸盐指数(mg/L)', '氨氮(mg/L)', '总磷(mg/L)', '总氮(mg/L)',
            'TEMP', 'DEWP', 'SLP', 'STP', 'VISIB', 'WDSP',
            'MXSPD', 'MAX', 'MIN', 'PRCP',  '叶绿素α(mg/L)', '藻密度(cells/L)'
        ]

        # 计算累计特征
        section_data['10_day_cumulative_temp'] = section_data['水温(℃)'].rolling(window=10).sum()
        section_data['10_day_cumulative_tp'] = section_data['总磷(mg/L)'].rolling(window=10).sum()
        section_data['10_day_cumulative_tn'] = section_data['总氮(mg/L)'].rolling(window=10).sum()
        section_data['TN_TP_ratio'] = section_data['总氮(mg/L)'] / section_data['总磷(mg/L)']

        # 更新特征列表
        feature_columns += ['10_day_cumulative_temp', '10_day_cumulative_tp', '10_day_cumulative_tn', 'TN_TP_ratio']

        # 删除NaN值
        section_data = section_data.dropna(subset=feature_columns + ['叶绿素α(mg/L)'])

        # 再次检查数据量，如果数据量不足则跳过
        if section_data.shape[0] < 10:
            print(f"Skipping section {section_name} due to insufficient data after dropping NaNs.")
            continue

        # 第二步：使用随机森林计算特征重要性
        X = section_data[feature_columns].values
        y = section_data['叶绿素α(mg/L)'].values

        # 数据标准化
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X = scaler_X.fit_transform(X)
        y = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()  # 标准化后保持y为一维

        # 使用随机森林模型
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)

        # 获取特征重要性
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]

        # 获取排名前15的重要特征
        top_n = 15
        top_features = [feature_columns[i] for i in indices[:top_n]]
        top_importances = importances[indices[:top_n]]

        # 绘制特征重要性图并保存
        plt.figure(figsize=(12, 8))
        sns.barplot(x=top_importances, y=top_features)
        plt.title(f"Top 15 Feature Importances for {section_name}")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.savefig(os.path.join(figures_dir, f"{section_name}_feature_importances.png"))
        plt.close()

        # 更新特征列为重要特征
        feature_columns = top_features

        # 第三步：使用Tigramite进行因果发现并引入滞后特征
        df = DataFrame(section_data[feature_columns].values, var_names=feature_columns)
        parcorr = ParCorr()
        pcmci = PCMCI(dataframe=df, cond_ind_test=parcorr)
        results = pcmci.run_pcmci(tau_max=5, pc_alpha=0.05)

        # 手动提取显著的因果链及其滞后效应
        significant_links = {}
        lag_features = []
        alpha_level = 0.05
        for i, var in enumerate(feature_columns):
            parents = np.where(results['p_matrix'][:, i, 0] < alpha_level)[0]
            if len(parents) > 0:
                significant_links[var] = parents.tolist()
                for parent in parents:
                    # 修改：选择绝对值最大的滞后时间
                    lag = np.argmax(np.abs(results['val_matrix'][parent, i]))
                    lag_feature_name = f"{feature_columns[parent]}_lag{lag}"
                    lag_features.append(lag_feature_name)
                    section_data[lag_feature_name] = section_data[feature_columns[parent]].shift(int(lag))

        # 删除包含NaN值的滞后特征行
        section_data = section_data.dropna(subset=lag_features)

        # 将滞后特征加入特征列
        feature_columns += lag_features

        # 将数据分为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # LSTM 模型定义
        class LSTMModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super(LSTMModel, self).__init__()
                self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
                self.fc = nn.Linear(hidden_dim, output_dim)

            def forward(self, x):
                x, _ = self.lstm(x)
                x = self.fc(x[:, -1, :])
                return x

        # GNN 模型定义
        class GCNModel(nn.Module):
            def __init__(self, in_channels, out_channels):
                super(GCNModel, self).__init__()
                self.conv1 = GCNConv(in_channels, 16)
                self.conv2 = GCNConv(16, out_channels)

            def forward(self, data):
                x, edge_index = data.x, data.edge_index
                x = self.conv1(x, edge_index)
                x = torch.relu(x)
                x = self.conv2(x, edge_index)
                return x

        # CNN 模型定义
        class CNNModel(nn.Module):
            def __init__(self, input_dim, output_dim):
                super(CNNModel, self).__init__()
                self.conv1 = nn.Conv1d(input_dim, 16, kernel_size=3, padding=1)
                self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
                self.fc = nn.Linear(32, output_dim)

            def forward(self, x):
                x = x.transpose(1, 2)
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = torch.mean(x, dim=2)
                x = self.fc(x)
                return x

        # BNN 模型定义
        class BayesianNN(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super(BayesianNN, self).__init__()
                self.hidden = nn.Linear(input_dim, hidden_dim)
                self.output = nn.Linear(hidden_dim, output_dim)

            def forward(self, x):
                x = torch.relu(self.hidden(x))
                return self.output(x)

        # 初始化模型
        input_dim = X_train.shape[1]
        lstm_model = LSTMModel(input_dim=input_dim, hidden_dim=16, output_dim=1)
        gnn_model = GCNModel(in_channels=input_dim, out_channels=1)
        cnn_model = CNNModel(input_dim=input_dim, output_dim=1)
        bnn_model = BayesianNN(input_dim=input_dim, hidden_dim=16, output_dim=1)

        lstm_criterion = nn.MSELoss()
        gnn_criterion = nn.HuberLoss()
        cnn_criterion = nn.MSELoss()
        bnn_criterion = nn.MSELoss()

        lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.01)
        gnn_optimizer = optim.Adam(gnn_model.parameters(), lr=0.01)
        cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.01)
        bnn_optimizer = optim.Adam(bnn_model.parameters(), lr=0.01)

        # 训练 LSTM 模型
        x_train_lstm = torch.tensor(X_train, dtype=torch.float).unsqueeze(1)
        for epoch in range(100):
            lstm_model.train()
            lstm_optimizer.zero_grad()
            lstm_output = lstm_model(x_train_lstm)
            lstm_loss = lstm_criterion(lstm_output.squeeze(), torch.tensor(y_train, dtype=torch.float))
            lstm_loss.backward()
            lstm_optimizer.step()

        # 训练 GNN 模型（改为 PEWAB 模型）
        G = nx.DiGraph()
        for target, parents in significant_links.items():
            for parent in parents:
                G.add_edge(feature_columns[parent], target)

        nodes = list(G.nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}

        edge_index = torch.tensor([[node_to_idx[edge[0]], node_to_idx[edge[1]]] for edge in G.edges], dtype=torch.long).t().contiguous()

        gnn_data = Data(x=torch.tensor(X_train, dtype=torch.float), edge_index=edge_index)
        for epoch in range(100):
            gnn_model.train()
            gnn_optimizer.zero_grad()
            gnn_output = gnn_model(gnn_data)
            gnn_loss = gnn_criterion(gnn_output.squeeze(), torch.tensor(y_train, dtype=torch.float))
            gnn_loss.backward()
            gnn_optimizer.step()

        # 训练 CNN 模型
        x_train_cnn = torch.tensor(X_train, dtype=torch.float).unsqueeze(1)
        for epoch in range(100):
            cnn_model.train()
            cnn_optimizer.zero_grad()
            cnn_output = cnn_model(x_train_cnn)
            cnn_loss = cnn_criterion(cnn_output.squeeze(), torch.tensor(y_train, dtype=torch.float))
            cnn_loss.backward()
            cnn_optimizer.step()

        # 训练 BNN 模型
        x_train_bnn = torch.tensor(X_train, dtype=torch.float)
        for epoch in range(100):
            bnn_model.train()
            bnn_optimizer.zero_grad()
            bnn_output = bnn_model(x_train_bnn)
            bnn_loss = bnn_criterion(bnn_output.squeeze(), torch.tensor(y_train, dtype=torch.float))
            bnn_loss.backward()
            bnn_optimizer.step()

        # 生成 LSTM、PEWAB（原GNN）、CNN 和 BNN 模型的预测结果
        with torch.no_grad():
            lstm_pred_train = lstm_model(x_train_lstm).squeeze().numpy()
            lstm_pred_train = scaler_y.inverse_transform(lstm_pred_train.reshape(-1, 1)).ravel()

            gnn_pred_train = gnn_model(gnn_data).squeeze().numpy()
            gnn_pred_train = scaler_y.inverse_transform(gnn_pred_train.reshape(-1, 1)).ravel()

            cnn_pred_train = cnn_model(x_train_cnn).squeeze().numpy()
            cnn_pred_train = scaler_y.inverse_transform(cnn_pred_train.reshape(-1, 1)).ravel()

            bnn_pred_train = bnn_model(x_train_bnn).squeeze().numpy()
            bnn_pred_train = scaler_y.inverse_transform(bnn_pred_train.reshape(-1, 1)).ravel()

            lstm_pred_test = lstm_model(torch.tensor(X_test, dtype=torch.float).unsqueeze(1)).squeeze().numpy()
            lstm_pred_test = scaler_y.inverse_transform(lstm_pred_test.reshape(-1, 1)).ravel()

            gnn_pred_test = gnn_model(Data(x=torch.tensor(X_test, dtype=torch.float), edge_index=edge_index)).squeeze().numpy()
            gnn_pred_test = scaler_y.inverse_transform(gnn_pred_test.reshape(-1, 1)).ravel()

            cnn_pred_test = cnn_model(torch.tensor(X_test, dtype=torch.float).unsqueeze(1)).squeeze().numpy()
            cnn_pred_test = scaler_y.inverse_transform(cnn_pred_test.reshape(-1, 1)).ravel()

            bnn_pred_test = bnn_model(torch.tensor(X_test, dtype=torch.float)).squeeze().numpy()
            bnn_pred_test = scaler_y.inverse_transform(bnn_pred_test.reshape(-1, 1)).ravel()

        # 确定最大可用的提前时间步长
        max_lead_time = min(7, len(y_test) - 1)

        # 初始化最佳MSE和对应的Lead Time
        best_mse_lstm, best_mse_gnn, best_mse_cnn, best_mse_bnn = [float('inf')] * 4
        best_lead_time_lstm, best_lead_time_gnn, best_lead_time_cnn, best_lead_time_bnn = [0] * 4

        for lead_time in range(1, max_lead_time + 1):
            X_test_shifted = X_test[:-lead_time]
            y_test_shifted = y_test[lead_time:]

            if len(X_test_shifted) != len(y_test_shifted):
                print(f"Skipping lead time {lead_time} for section {section_name} due to sample size mismatch.")
                continue

            with torch.no_grad():
                lstm_pred = lstm_model(torch.tensor(X_test_shifted, dtype=torch.float).unsqueeze(1)).squeeze().numpy()
                lstm_pred = scaler_y.inverse_transform(lstm_pred.reshape(-1, 1)).ravel()

                gnn_pred = gnn_model(Data(x=torch.tensor(X_test_shifted, dtype=torch.float), edge_index=edge_index)).squeeze().numpy()
                gnn_pred = scaler_y.inverse_transform(gnn_pred.reshape(-1, 1)).ravel()

                cnn_pred = cnn_model(torch.tensor(X_test_shifted, dtype=torch.float).unsqueeze(1)).squeeze().numpy()
                cnn_pred = scaler_y.inverse_transform(cnn_pred.reshape(-1, 1)).ravel()

                bnn_pred = bnn_model(torch.tensor(X_test_shifted, dtype=torch.float)).squeeze().numpy()
                bnn_pred = scaler_y.inverse_transform(bnn_pred.reshape(-1, 1)).ravel()

            # 计算每个模型的MSE
            mse_lstm = mean_squared_error(scaler_y.inverse_transform(y_test_shifted.reshape(-1, 1)).ravel(), lstm_pred)
            mse_gnn = mean_squared_error(scaler_y.inverse_transform(y_test_shifted.reshape(-1, 1)).ravel(), gnn_pred)
            mse_cnn = mean_squared_error(scaler_y.inverse_transform(y_test_shifted.reshape(-1, 1)).ravel(), cnn_pred)
            mse_bnn = mean_squared_error(scaler_y.inverse_transform(y_test_shifted.reshape(-1, 1)).ravel(), bnn_pred)

            # 更新最佳MSE和Lead Time
            if mse_lstm < best_mse_lstm:
                best_mse_lstm = mse_lstm
                best_lead_time_lstm = lead_time

            if mse_gnn < best_mse_gnn:
                best_mse_gnn = mse_gnn
                best_lead_time_gnn = lead_time

            if mse_cnn < best_mse_cnn:
                best_mse_cnn = mse_cnn
                best_lead_time_cnn = lead_time

            if mse_bnn < best_mse_bnn:
                best_mse_bnn = mse_bnn
                best_lead_time_bnn = lead_time
        # 根据综合评分选出最佳模型
        best_model = min(
            [('LSTM', best_mse_lstm, best_lead_time_lstm), 
             ('GNN', best_mse_gnn, best_lead_time_gnn), 
             ('CNN', best_mse_cnn, best_lead_time_cnn), 
             ('BNN', best_mse_bnn, best_lead_time_bnn), 
             ],
            key=lambda x: x[1]
        )[0]

        # 保存结果到 DataFrame
        new_row = pd.DataFrame({
            '断面名称': [section_name],
            'MSE_LSTM': [best_mse_lstm],
            'MSE_GNN': [best_mse_gnn],
            'MSE_CNN': [best_mse_cnn],
            'MSE_BNN': [best_mse_bnn],
            'LSTM Best Lead Time': [best_lead_time_lstm],
            'GNN Best Lead Time': [best_lead_time_gnn],
            'CNN Best Lead Time': [best_lead_time_cnn],
            'BNN Best Lead Time': [best_lead_time_bnn],
            'Best_Model': [best_model]
        })

        results_df = pd.concat([results_df, new_row], ignore_index=True)

        # 绘制实际值与预测值的比较图并保存
        dates = section_data['日期'].iloc[int(0.7 * len(section_data)):]  # 使用测试集部分的日期
        plt.figure(figsize=(12, 6))

        # 绘制实际值和模型的预测值（突出 PEWAB/GNN 的线条）
        plt.plot(dates[-len(y_test_shifted):], scaler_y.inverse_transform(y_test_shifted.reshape(-1, 1)).ravel(),
                 label='Actual Chlorophyll Concentration', color='b', marker='o', linewidth=2)
        plt.plot(dates[-len(y_test_shifted):], gnn_pred, 
                 label=f'PEWAB Predicted (Best Lead Time: {best_lead_time_gnn})', color='g', linestyle='-', marker='d', linewidth=3)  # PEWAB/GNN加粗显示
        plt.plot(dates[-len(y_test_shifted):], lstm_pred, 
                 label=f'LSTM Predicted (Best Lead Time: {best_lead_time_lstm})', color='gray', linestyle='-', marker='^', alpha=0.7)
        plt.plot(dates[-len(y_test_shifted):], cnn_pred, 
                 label=f'CNN Predicted (Best Lead Time: {best_lead_time_cnn})', color='gray', linestyle='--', marker='s', alpha=0.7)
        plt.plot(dates[-len(y_test_shifted):], bnn_pred, 
                 label=f'BNN Predicted (Best Lead Time: {best_lead_time_bnn})', color='gray', linestyle=':', marker='*', alpha=0.7)
                
        plt.title(f'Actual vs Predicted Chlorophyll Concentration for {section_name}')
        plt.xlabel('Date')
        plt.ylabel('Chlorophyll Concentration (mg/L)')
        
        # 固定图例位置为右上角，并设置边框不透明
        plt.legend(loc='upper right', framealpha=1)

        # 优化时间显示
        plt.gcf().autofmt_xdate()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

        plt.savefig(os.path.join(figures_dir, f"{section_name}_pewab_actual_vs_predicted.png"))
        plt.close()

    except Exception as e:
        print(f"Skipping section {section_name} due to error: {str(e)}")

# 保存结果到 CSV
results_df.to_csv(os.path.join(output_dir, "7daysection_prediction_results_no_ensemble.csv"), index=False)

# 保存带有预测值的原始数据
data.to_csv(os.path.join(output_dir, "7daydata_with_pewab_predictions.csv"), index=False)
