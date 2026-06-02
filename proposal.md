# project : A Visualization Toolbox for Understanding Optimization Algorithms

## 背景 ： (簡短3-4句話)


## objective:
打造一個高互動性的網頁視覺化工具箱，將抽象的優化演算法具象化，幫助使用者深入理解各優化器在幾何尋優與機器學習資料驅動場景下的核心行為。

## Architecture:
### Module1: Mathematical Optimization
- Loss Functions: Quadratic, Rosenbrock
- Optimizers: Full-batch GD, SGDM, AdaGrad, RMSProp, Adam
- Visualization: 使用 Plotly 繪製 2D 等高線圖（Contour Plot），並即時疊加（Overlay）優化器從指定初始點一步步邁向全局最低點的動態軌跡折線

### Module2: Machine Learning Optimization
- Loss Functions: logistic, ridge logistic
- Optimizers: Full-batch GD, Mini-batch GD, SGDM, AdaGrad, RMSProp, Adam
- Visualization: 繪製 Training Loss vs. Iteration 與 Test Loss vs. Iteration 的雙線收斂圖 ，用以觀察隨機抽樣帶來的震盪以及泛化能力的變化。
- data generation: 於程式內部（utils.py）使用 numpy.random 隨機生成 1,000 筆帶有雜訊的二維常態分佈合成數據（Synthetic Data），並自動切分為 70% 訓練集（Training Set）與 30% 測試集（Test Set），用以完美復現與診斷欠擬合、良好擬合與過擬合（Overfitting）之統計現象。

### hyperparameters for each optimizers
- Full-batch GD:
- Mini-batch GD:
- SGDM:
- AdaGrad:
- RMSProp:
- Adam:

## Development Tools & Environment:
- python3
- numpy, spipy
- streamlit
- plotly, matlibplot