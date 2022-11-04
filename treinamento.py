# %%
import pandas as pd
import numpy as np
import seaborn as sbn
import plotly.express as px
from tqdm import tqdm
from typing import Any, List, Tuple, Optional, Callable, Union
from matplotlib import pyplot as plt
from re import compile as recompile, sub as resub, Pattern
from datetime import date, timedelta
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    HalvingGridSearchCV,
)
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.neural_network import MLPClassifier

# %%
filtrar_numero = recompile("[^\\-.0-9]")
filtrar_hist_age = recompile("([0-9]+)(?: Years and )([0-9]+)(?: Months)")


def tratar_credit_history_age_individual(
    z: Union[str, float], procura: Pattern
) -> Union[int, float]:
    if pd.isnull(z):
        return np.NaN
    grps = procura.search(z)
    return int(grps[1]) * 12 + int(grps[2])


def tratar_credit_history_age(cha: pd.Series) -> pd.Series:
    procura = recompile("([0-9]+)(?: Years and )([0-9]+)(?: Months)")
    cha_tratado = cha.apply(tratar_credit_history_age_individual, args=(procura,))
    cha_tratado.fillna(cha_tratado.median(skipna=True), inplace=True)
    return cha_tratado


def numerizador(a):
    if type(a) == str:
        n = filtrar_numero.sub("", a, count=100)
        try:
            z = int(n)
            return z
        except ValueError as erro_int:
            try:
                r = float(n)
                return r
            except ValueError as erro_float:
                return np.NaN
    elif type(a) in [float, int]:
        return a
    else:
        return np.NaN


def numerizar_e_limitar(
    a: Any, fna: Callable[[Any], Union[float, int]]
) -> Union[float, int]:
    recurso = np.NaN
    if type(a) in [float, int]:
        recurso = a
    elif type(a) == str:
        n = filtrar_numero.sub("", a, count=100)
        try:
            recurso = int(n)
        except ValueError as erro_int:
            try:
                recurso = float(n)
            except ValueError as erro_float:
                pass
    if not pd.isnull(recurso):
        return fna(recurso)
    return recurso


def ajustar_por_mediana_cliente(
    clientes: pd.Series, col: pd.Series, fna: Callable[[Any], Union[float, int]]
) -> pd.Series:
    primeiro_passo = col.apply(numerizar_e_limitar, args=(fna,))
    df_base = pd.DataFrame({clientes.name: clientes, col.name: primeiro_passo})
    medianas_classificadas = (
        df_base.dropna()
        .groupby(by=[clientes.name])
        .agg({col.name: np.median})
        .reset_index()
    )
    medianas_preenchidas = pd.merge(
        df_base,
        medianas_classificadas,
        how="left",
        on=clientes.name,
        suffixes=("_Original", "_Median"),
    )
    medianas_preenchidas[col.name] = np.where(
        medianas_preenchidas.loc[:, f"{col.name}_Original"].isnull(),
        medianas_preenchidas.loc[:, f"{col.name}_Median"],
        medianas_preenchidas.loc[:, f"{col.name}_Original"],
    )
    medianas_preenchidas.loc[:, col.name].fillna(
        medianas_preenchidas.loc[:, f"{col.name}_Original"].median(skipna=True),
        inplace=True,
    )
    return medianas_preenchidas[col.name]


def ajustar_por_moda_cliente(
    clientes: pd.Series, col: pd.Series, unk: Optional[str]
) -> pd.Series:
    primeiro_passo = (
        col.apply(lambda z: np.NaN if z == unk else z)
        if unk is not None
        else col.copy()
    )
    df_base = pd.DataFrame({clientes.name: clientes, col.name: primeiro_passo})
    modas_classificadas = (
        df_base.groupby(by=[clientes.name])
        .agg({col.name: lambda z: z.value_counts().index[0]})
        .reset_index()
    )
    modas_preenchidas = pd.merge(
        df_base,
        modas_classificadas,
        how="left",
        on=clientes.name,
        suffixes=("_Original", "_Mode"),
    )
    modas_preenchidas[col.name] = np.where(
        modas_preenchidas.loc[:, f"{col.name}_Original"].isnull(),
        modas_preenchidas.loc[:, f"{col.name}_Mode"],
        modas_preenchidas.loc[:, f"{col.name}_Original"],
    )
    modas_preenchidas.loc[:, col.name].fillna(
        modas_preenchidas.loc[:, f"{col.name}_Original"].value_counts().index[0],
        inplace=True,
    )
    return modas_preenchidas.loc[:, col.name]


def melhor_arvore(X_train, y_train, rd_state):
    param_grid = {"criterion": ["gini", "entropy"]}
    testes = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=rd_state),
        param_grid=param_grid,
        n_jobs=4,
        refit=True,
    ).fit(X_train, y_train)
    print(
        f"Critério escolhido para árvore de decisão: {testes.best_params_['criterion']}"
    )
    return testes.best_estimator_


def melhor_floresta(X_train, y_train, rd_state):
    param_grid = {"criterion": ["gini", "entropy"]}
    testes = HalvingGridSearchCV(
        estimator=RandomForestClassifier(n_jobs=4, random_state=rd_state),
        param_grid=param_grid,
        resource="n_estimators",
        max_resources=1024,
        min_resources=128,
        factor=2,
        n_jobs=4,
    ).fit(X_train, y_train)
    print("Melhores parâmetros da floresta:", testes.best_params_, sep="\n")
    return testes.best_estimator_


class Transdutor:
    def __init__(self, arquivo_treino) -> None:
        self.treino = pd.read_csv(arquivo_treino, low_memory=False)
        loans_aux = (
            self.treino.loc[:, "Type_of_Loan"].str.replace("and ", "").str.split(", ")
        )
        types_of_loans = set()
        for loan_list in loans_aux:
            if type(loan_list) == list:
                for loan in loan_list:
                    types_of_loans.add(loan)
        self.types_of_loans = list(types_of_loans)
        self.treino_tratado = self.preparar_dados()
        self.alvos = self.treino["Credit_Score"].copy()

    def preparar_dados(self) -> None:
        self.colunas_numericas_autocontidas = {
            "Age": [None, lambda z: z if 0 <= z <= 100 else np.NaN],
            "Annual_Income": [None, lambda z: z],
            "Num_Bank_Accounts": [None, lambda z: z if 0 <= z < 15 else np.NaN],
            "Num_Credit_Card": [None, lambda z: z if 0 <= z < 15 else np.NaN],
            "Interest_Rate": [None, lambda z: z if 0 <= z < 40 else np.NaN],
            "Num_of_Loan": [None, lambda z: z if 0 <= z < 15 else np.NaN],
            "Num_of_Delayed_Payment": [None, lambda z: z if 0 <= z < 40 else np.NaN],
            "Changed_Credit_Limit": [None, lambda z: z],
            "Num_Credit_Inquiries": [None, lambda z: z if 0 <= z < 20 else np.NaN],
            "Total_EMI_per_month": [None, lambda z: z],
            "Amount_invested_monthly": [
                None,
                lambda z: z if 0 <= z <= 8_000 else np.NaN,
            ],
            "Monthly_Balance": [
                None,
                lambda z: z if -100_000 <= z <= 100_000 else np.NaN,
            ],
        }
        self.colunas_numericas_geral = {
            "Delay_from_due_date": [None, lambda z: z],
            "Outstanding_Debt": [None, lambda z: z],
        }
        self.colunas_categoricas_autocontidas = {
            "Payment_Behaviour": [None, "!@9#%8"],
            "Occupation": [None, "_______"],
            "Credit_Mix": [None, "_"],
            "Payment_of_Min_Amount": [None, "NM"],
        }
        progresso = tqdm(total=28, desc="A preparar os dados.")
        tratado = pd.DataFrame()
        tratado["ID"] = self.treino.loc[:, "ID"].copy()
        progresso.update(1)
        # Temporarily insert CustomerID
        tratado["Customer_ID"] = self.treino.loc[:, "Customer_ID"].copy()
        progresso.update(1)
        # Skip Month
        progresso.update(1)
        # Skip Name
        progresso.update(1)
        # Skip SSN
        progresso.update(1)
        # Skip Monthly Inhand Salary
        progresso.update(1)
        for col in self.colunas_numericas_autocontidas:
            tratado[col] = ajustar_por_mediana_cliente(
                tratado["Customer_ID"],
                self.treino.loc[:, col],
                self.colunas_numericas_autocontidas[col][1],
            )
            self.colunas_numericas_autocontidas[col][0] = tratado.loc[:, col].median()
            progresso.update(1)
        for col in self.colunas_numericas_geral:
            tratado[col] = self.treino.loc[:, col].apply(
                numerizar_e_limitar, args=(self.colunas_numericas_geral[col][1],)
            )
            self.colunas_numericas_geral[col][0] = tratado.loc[:, col].median(
                skipna=True
            )
            tratado[col].fillna(self.colunas_numericas_geral[col][0], inplace=True)
            progresso.update(1)
        for col in self.colunas_categoricas_autocontidas:
            coluna_ajustada = ajustar_por_moda_cliente(
                tratado["Customer_ID"],
                self.treino.loc[:, col],
                self.colunas_categoricas_autocontidas[col][1],
            )
            self.colunas_categoricas_autocontidas[col][
                0
            ] = coluna_ajustada.value_counts().index[0]
            coluna_ajustada.fillna(
                self.colunas_categoricas_autocontidas[col][0], inplace=True
            )
            tratado = pd.concat(
                [
                    tratado,
                    pd.get_dummies(coluna_ajustada, prefix=col),
                ],
                axis=1,
            )
            progresso.update(1)
        tratado["Credit_History_Age"] = tratar_credit_history_age(
            self.treino.loc[:, "Credit_History_Age"]
        )
        progresso.update(1)
        return tratado

    def dados_para_treinar(self) -> pd.DataFrame:
        return self.treino_tratado.drop(labels=["ID", "Customer_ID"], axis=1)

    def alvos_para_treinar(self) -> pd.Series:
        return self.alvos.copy()

    def transformar_dados(self, novos_dados: pd.DataFrame) -> pd.DataFrame:
        tratado = pd.DataFrame()
        tratado["ID"] = novos_dados["ID"].copy()
        for col in self.colunas_numericas_autocontidas:
            tratado[col] = (
                novos_dados.loc[:, col]
                .apply(numerizador)
                .apply(self.colunas_numericas_autocontidas[col][1])
                .fillna(self.colunas_numericas_autocontidas[col][0])
            )
        for col in self.colunas_numericas_geral:
            tratado[col] = (
                novos_dados.loc[:, col]
                .apply(numerizador)
                .apply(self.colunas_numericas_geral[col][1])
                .fillna(self.colunas_numericas_geral[col][1])
            )
        for col in self.colunas_categoricas_autocontidas:
            coluna_categorica = novos_dados.loc[:, col].apply(
                lambda z: self.colunas_categoricas_autocontidas[col][0]
                if z == self.colunas_categoricas_autocontidas[col][1]
                else z
            )
            tratado = pd.concat(
                [tratado, pd.get_dummies(coluna_categorica, prefix=col)], axis=1
            )
        tratado["Credit_History_Age"] = tratar_credit_history_age(
            novos_dados.loc[:, "Credit_History_Age"]
        )
        tratado.set_index("ID", inplace=True)
        return tratado


class Processador:
    def __init__(
        self, rd_state: Optional[int], dados: pd.DataFrame, alvos: pd.Series
    ) -> None:
        progresso = tqdm(total=6, desc="A preparar modelos para partida")
        self.escalador = StandardScaler()
        self.escalador.fit(dados)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.escalador.transform(dados), alvos
        )
        progresso.update(1)
        self.regressao_logistica = LogisticRegressionCV(
            multi_class="ovr",
            max_iter=300,
            n_jobs=6,
            cv=5,
            random_state=rd_state,
        )
        self.regressao_logistica.fit(self.X_train, self.y_train)
        progresso.update(1)
        self.arvore_decisoria = melhor_arvore(self.X_train, self.y_train, rd_state)
        progresso.update(1)
        self.floresta_aleatoria = melhor_floresta(self.X_train, self.y_train, rd_state)
        progresso.update(1)
        self.rede_neural = MLPClassifier(
            hidden_layer_sizes=(
                100,
                150,
                100,
            ),
            random_state=rd_state,
            early_stopping=True,
        )
        self.rede_neural.fit(self.X_train, self.y_train)
        progresso.update(1)
        self.completo = VotingClassifier(
            estimators=[
                ("rl", self.regressao_logistica),
                ("ad", self.arvore_decisoria),
                ("rn", self.rede_neural),
            ],
            voting="soft",
            n_jobs=6,
        ).fit(self.X_train, self.y_train)
        progresso.update(1)

    def apresentar_estatísticas_de_treino(self) -> None:
        print(
            "Scores f1 dos modelos scikit_learn:",
            f"Regressão: {f1_score(self.y_test, self.regressao_logistica.predict(self.X_test), average='micro')}",
            f"Arvore de Decisão: {f1_score(self.y_test, self.arvore_decisoria.predict(self.X_test), average='micro')}",
            f"Floresta Aleatória: {f1_score(self.y_test, self.floresta_aleatoria.predict(self.X_test), average='micro')}",
            f"Rede Neural: {f1_score(self.y_test, self.rede_neural.predict(self.X_test), average='micro')}",
            f"Conjunto: {f1_score(self.y_test, self.completo.predict(self.X_test), average='micro')}",
            sep="\n    ",
        )

    def predizer(
        self, entrada_de_dados: pd.DataFrame, tipo: str = "Floresta Aleatória"
    ) -> pd.Series:
        indice = entrada_de_dados.index.copy()
        if tipo == "Regressão Logística":
            return pd.Series(
                self.regressao_logistica.predict(entrada_de_dados), index=indice
            )
        elif tipo == "Árvore de Decisão":
            return pd.Series(
                self.arvore_decisoria.predict(entrada_de_dados), index=indice
            )
        elif tipo == "Floresta Aleatória":
            return pd.Series(
                self.floresta_aleatoria.predict(entrada_de_dados), index=indice
            )
        elif tipo == "Rede Neural":
            return pd.Series(self.rede_neural.predict(entrada_de_dados), index=indice)
        elif tipo in ("Conjunto", "Ensemble"):
            return pd.Series(self.completo.predict(entrada_de_dados), index=indice)
        else:
            raise (RuntimeError(f"Modelo escolhido ({tipo}) não implementado"))


# %%
dados = Transdutor("train.csv")
# %%
proc = Processador(
    rd_state=123456789,
    dados=dados.dados_para_treinar(),
    alvos=dados.alvos_para_treinar(),
)
proc.apresentar_estatísticas_de_treino()
# %%
exame = dados.transformar_dados(pd.read_csv("test.csv"))
# %%
resultado = proc.predizer(exame)

# %%
comparacao = pd.DataFrame()
for tipo_pred in [
    "Regressão Logística",
    "Árvore de Decisão",
    "Floresta Aleatória",
    "Rede Neural",
    "Conjunto",
]:
    comparacao[tipo_pred] = proc.predizer(exame, tipo=tipo_pred)
comparacao.to_csv("resultado_comparador.csv", sep=";")
# %%
