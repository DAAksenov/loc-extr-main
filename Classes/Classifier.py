import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_gaussian_quantiles
import plotly.graph_objects as go
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split


class Classifier(nn.Module):
    """
    Логистическая регрессия и метод опорных векторов на основе torch

    Methods
    -------
    fit(self)
        Обучает модель
        Parameteres:
            X: torch.Tensor - Матрица признаков;
            y: torch.Tensor - Вектор меток классов;
            num_epo: int - Кол-во эпох;
            lr: float - скорость обучения;
            num_batch: Кол-во батчей/подвыборок
            batch_size: Размер батчей;
            reg: str - Тип регуляризации;
            alpha: Коэффицент регуляризации;
            gamma: float - Коэффицент rbf, если None, то rbf не используется
            svm: Bool - использование метода опорных векторов
        Returns: classification_report на обучающей и тестовой выборке
        
    forward(self, X: list or array, state)
        В данной функции мы определяем как модель будет преобразовывать входные данные
        Параметр state нужен для использования RBF к данным вне процесса обучения
        Returns: array

    plot(self)
        Построение графика рассеивания (только для 2-х признаков)
        Returns: plotly.graphic_objects.Figure

"""

    def __init__(self):
        super().__init__()
        self.linear = nn.LazyLinear(out_features=1)

    def forward(self, x, state=None):
        if self.gamma is not None and state != 'train':
            if len(x.shape) == 1:
                x = torch.tensor(rbf_kernel(x.reshape(1, -1), Y=self.X))
            else:
                x = torch.tensor(rbf_kernel(x, Y=self.X))
        return self.linear(x)

    def fit(self, X, y, num_epochs, lr, num_batches, batch_size, reg, alpha=0.5, gamma: float = None, svm=False):
        """
        gamma: coef for rbf_gauss. set None if rbf isn't needed
        svm_C: coeff for svm. set None if svm isn't needed
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()
        self.gamma = gamma
        self.X, self.y = X, y
        if self.gamma is not None:
            X = torch.tensor(rbf_kernel(self.X, Y=self.X))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        for epo in range(1, num_epochs + 1):
            losses = []
            for _ in range(num_batches):
                indexes_batch = torch.randint(low=0, high=X_train.shape[0] - 1, size=(batch_size,))
                X_batch, y_batch = X_train[indexes_batch], y_train[indexes_batch]
                outputs = self(X_batch, state='train')
                if svm == True:
                    loss = torch.mean(torch.relu(1 - y_batch * outputs.reshape(-1)))
                else:
                    loss = torch.mean(torch.log(1 + torch.exp(-1 * y_batch * outputs.reshape(-1))))
                if reg == 'l1':
                    loss = loss + alpha * torch.sum(torch.abs(self.linear.weight)).item()
                elif reg == 'l2':
                    loss = loss + alpha * np.sqrt(torch.sum((self.linear.weight) ** 2).item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            print(f'Epo {epo}: {np.mean(losses)}')
        return [
            classification_report(y_train, torch.sign(self(X_train, state='train')).reshape(-1).detach().numpy(),
                                  target_names=['-1', '1']),
            classification_report(y_test, torch.sign(self(X_test, state='train')).reshape(-1).detach().numpy(),
                                  target_names=['-1', '1'])
        ]

    def plot(self):
        X = self.X
        assert X.shape[1] == 2, 'Доступно только для 2-х признаков'
        y0 = self.y
        y0[y0 == -1.] = 0.
        if self.X.shape[1] == 2:
            fig = go.Figure(layout=go.Layout(
                title={'text': 'График рассеивания и контурный график вероятности принадлежности к классу 1'}))

            x1 = torch.linspace(torch.min(X), torch.max(X))
            x2 = torch.linspace(torch.min(X), torch.max(X))

            fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker={'color': self.y}, name='Points'))

            X1, X2 = np.meshgrid(x1, x2)
            Z = torch.sigmoid(self(torch.tensor(np.c_[X1.ravel(), X2.ravel()]))).reshape(X1.shape).detach().numpy()

            fig.add_trace(go.Contour(x=x1, y=x2, z=Z, colorscale='ice', name='Probability',
                                     contours=dict(start=0, end=1, size=0.5),
                                     showscale=False,
                                     line_width=4
                                     ))

        return fig


if __name__ == '__main__':
    X, y = make_gaussian_quantiles(n_classes=2, n_samples=1000)
    X = torch.tensor(X, dtype=torch.float)
    y[y == 0] = -1
    y = torch.tensor(y, dtype=torch.float)

    model = Classifier()
    model.fit(X, y, 10, 0.01, 200, 20, reg='l2', alpha=0.001, gamma=0.1)
    model.plot().show()
