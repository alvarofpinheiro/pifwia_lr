#LR
import numpy as np
import matplotlib.pyplot as plt
# %config InlineBackend.figure_format = 'svg'

"""## Leitura dos dados
- X representa os M³ do apartamento que possuem uma variação de 40m³ até 120 m³
- y é o preço do apartamento
"""

X = np.random.randint(low=40,high=120, size=(20,1))
y = (3 * X + np.random.randint(low=100,high=150, size=(20,1))) * 1000

fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(X,y, c='b')
plt.xlabel("M²")
plt.ylabel("Preço")

"""## Representando uma Reta"""

def predict(alpha, beta, X):
    return alpha*X + beta

"""## Escolhendo melhor alpha e beta"""

tamanho = len(X)
X_b = np.c_[np.ones((tamanho, 1)), X]  # add x0 = 1 to each instance
X_b

X_b.T.dot(X_b)

#métodos dos mínimos quadrados
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
theta_best

"""## Ajustando a reta"""

alpha = theta_best[1] #inclinação
beta = theta_best[0]

ỹ = predict(alpha=alpha, beta=beta, X=X)

ỹ

"""## Plotando os Dados"""

fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(X,y, c='b')
plt.plot(X, ỹ, 'r')

"""## Como implementar uma regressão linear usando o Scikit-learn?"""

from sklearn.linear_model import LinearRegression

fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(X,y, c='b')
plt.plot(X, ỹ, 'r')

lr = LinearRegression()

lr.fit(X, y)

ỹ = lr.predict(X)

fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(X,y, c='b')
plt.plot(X, ỹ, 'r')

print("Training score: {:.2f}".format(lr.score(X, y)))

"""## Avaliando meu modelo de regressão"""

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y, ỹ))
print('MSE:', metrics.mean_squared_error(y, ỹ))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y, ỹ)))

