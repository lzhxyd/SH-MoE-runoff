# dl_models.py
import numpy as np
import torch
from torch import nn
from skorch import NeuralNetRegressor
from typing import Optional



# —— 通用：时序窗口化（内部自用） ——
def build_windows(X: np.ndarray, y: Optional[np.ndarray], L: int):
    # X: (N, F); return X_seq: (N-L+1, L, F), y_seq: (N-L+1,)
    N = X.shape[0]
    if N < L:
        return np.empty((0, L, X.shape[1])), np.empty((0,)) if y is not None else None
    # 生成滑窗
    Xw = np.stack([X[i:i+L] for i in range(N - L + 1)], axis=0)
    yw = None if y is None else y[L-1:]  # 预测第 L-1 时刻以后的 y
    return Xw.astype(np.float32), (None if yw is None else yw.astype(np.float32))

# —— LSTM 子网 ——
class LSTMModule(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden_size, 1))
    def forward(self, x):             # x: (B, L, F)
        out, _ = self.lstm(x)         # (B, L, H)
        yhat = self.head(out[:, -1])  # 取最后一步
        return yhat.squeeze(-1)       # (B,)

# —— TCN 子网（简洁版） ——
class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, d=1, p=0.1):
        super().__init__()
        pad = (d * (k - 1)) // 2  # 需要 k 为奇数
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=d),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Conv1d(out_ch, out_ch, k, padding=pad, dilation=d),
            nn.ReLU(),
        )
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch!=out_ch else nn.Identity()
    def forward(self, x):  # (B, F, L)
        y = self.net(x)
        # 因 padding，长度未变；做残差
        return y + self.res(x)

class TCNModule(nn.Module):
    def __init__(self, n_features, channels=32, ksize=3, p=0.1):
        super().__init__()
        self.b1 = TCNBlock(n_features, channels, k=ksize, d=1, p=p)
        self.b2 = TCNBlock(channels, channels, k=ksize, d=2, p=p)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(channels, 1)
    def forward(self, x):             # x: (B, L, F)
        x = x.transpose(1, 2)         # → (B, F, L)
        y = self.b1(x)
        y = self.b2(y)
        y = self.pool(y).squeeze(-1)  # (B, C)
        y = self.head(y).squeeze(-1)  # (B,)
        return y

# —— sklearn 风格包装（内部完成窗口化；支持 GridSearchCV） ——
class SeqRegressor(NeuralNetRegressor):
    def __init__(self, module, lookback=12, log_loss_weight=0.3, eps=1e-3,
                 random_state=None, **kw):
        super().__init__(module=module, **kw)  # 不要传 random_state 给父类
        self.lookback = lookback
        self.log_loss_weight = float(log_loss_weight)
        self.eps = float(eps)
        self._user_random_state = random_state  # 存起来

    def initialize(self):
        # 在真正初始化前播种，兼容各版本 skorch
        rs = getattr(self, "_user_random_state", None)
        if rs is not None:
            try:
                import numpy as _np, torch as _torch, random as _random
                _random.seed(rs); _np.random.seed(rs)
                if _torch.cuda.is_available():
                    _torch.cuda.manual_seed_all(rs)
                _torch.manual_seed(rs)
            except Exception:
                pass
        return super().initialize()

    def get_loss(self, y_pred, y_true, *args, **kwargs):
        # skorch 会把 y_true 送成 (N,1)，先压成 (N,)
        if hasattr(y_true, "dim") and y_true.dim() > 1:
            y_true = y_true.squeeze(-1)
        mse = torch.mean((y_pred - y_true) ** 2)
        if self.log_loss_weight > 0.0:
            y_true_s = torch.log(y_true + self.eps)
            y_pred_s = torch.log(torch.clamp(y_pred, min=0.0) + self.eps)
            logmse = torch.mean((y_pred_s - y_true_s) ** 2)
            return mse + self.log_loss_weight * logmse
        return mse

    def fit(self, X, y=None, **fit_params):
        Xw, yw = build_windows(np.asarray(X), None if y is None else np.asarray(y), self.lookback)

        # —— 关键：skorch 需要 (N,1) 的 y ——
        if yw is not None:
            yw = np.asarray(yw, dtype=np.float32)
            if yw.ndim == 1:
                yw = yw.reshape(-1, 1)

        # —— 兜底：某些时序折长度 < lookback，窗口化后会为空，直接让该组合记为 nan ——
        if Xw.shape[0] == 0 or (yw is not None and yw.shape[0] == 0):
            raise ValueError(f"lookback({self.lookback}) larger than fold length")

        return super().fit(Xw.astype(np.float32), yw, **fit_params)

    def predict(self, X):
        Xw, _ = build_windows(np.asarray(X), None, self.lookback)
        if Xw.shape[0] == 0:
            return np.array([], dtype=np.float32)
        # skorch 的 predict -> numpy 1D
        return super().predict(Xw)

# 便捷构造器：给 GridSearchCV 用
# dl_models.py

def make_lstm_regressor(
    n_features, lookback=12, hidden_size=64, num_layers=1, dropout=0.1,
    lr=1e-3, max_epochs=200, batch_size=64, device=None,
    log_loss_weight=0.3, eps=1e-3, random_state=42
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'  # ← 关键
    return SeqRegressor(
        module=LSTMModule,               # ← 传类
        # —— 传给模块构造器的参数（前缀 module__*） ——
        module__n_features=n_features,
        module__hidden_size=hidden_size,
        module__num_layers=num_layers,
        module__dropout=dropout,
        # —— 训练器参数 ——
        lookback=lookback,
        log_loss_weight=log_loss_weight,
        eps=eps,
        optimizer=torch.optim.Adam,
        optimizer__lr=lr,
        max_epochs=max_epochs,
        batch_size=batch_size,
        iterator_train__shuffle=True,
        device=device,
        train_split=None,   # 由外部TimeSeriesSplit控制CV
        verbose=0,
    )


def make_tcn_regressor(
    n_features, lookback=12, channels=32, ksize=3, p=0.1,
    lr=1e-3, max_epochs=200, batch_size=64, device=None,
    log_loss_weight=0.3, eps=1e-3, random_state=42
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return SeqRegressor(
        module=TCNModule,                # ← 传类
        module__n_features=n_features,   # ← 必传
        module__channels=channels,
        module__ksize=ksize,
        module__p=p,
        lookback=lookback,
        log_loss_weight=log_loss_weight,
        eps=eps,
        optimizer=torch.optim.Adam,
        optimizer__lr=lr,
        max_epochs=max_epochs,
        batch_size=batch_size,
        iterator_train__shuffle=True,
        device=device,
        train_split=None,
        verbose=0,
    )

