# whylogs – Bike Sharing: Perfis de Dados e Monitoramento Contínuo

Projeto de Ciência de Dados e Machine Learning que avalia a biblioteca **[whylogs](https://github.com/whylabs/whylogs)** no monitoramento de um modelo de regressão, com o dataset **Bike Sharing** (Capital Bikeshare, Washington D.C., 2011–2012).

É o terceiro projeto da série, depois do **EvidentlyIA** e do **Deepchecks – Bike Sharing**. Usa o mesmo dataset, a mesma limpeza e **o mesmo modelo** do projeto do Deepchecks, o que permite comparar as ferramentas no mesmo cenário: um caso de *concept drift*, em que o uso de bicicletas cresce de 2011 para 2012 enquanto clima e calendário ficam parecidos.

---

## 📋 Sumário

- [O que é o whylogs](#-o-que-é-o-whylogs)
- [Por que o mesmo dataset](#-por-que-o-mesmo-dataset)
- [Estrutura do Repositório](#-estrutura-do-repositório)
- [Pipeline](#-pipeline)
- [O que o notebook do whylogs faz](#-o-que-o-notebook-do-whylogs-faz)
- [Resultados](#-resultados)
- [Evidently × Deepchecks × whylogs](#-evidently--deepchecks--whylogs)
- [Como Executar](#-como-executar)
- [Limitações Encontradas](#-limitações-encontradas)

---

## 🔎 O que é o whylogs

O whylogs transforma cada lote de dados num **perfil**: um resumo estatístico (contagens, nulos, média, quantis aproximados, histogramas, valores mais frequentes). Drift, regras de qualidade e métricas de desempenho são calculados **a partir dos perfis**, não dos dados brutos.

- **Crescem devagar:** 10 vezes mais linhas geram um perfil só ~10% maior.
- **São somáveis:** perfis diários viram mensais sem reprocessar os dados, com resultado exato para contagens, médias e RMSE.
- **Não guardam linhas:** dá para manter anos de histórico e compartilhar sem expor os dados originais.

---

## 🎯 Por que o mesmo dataset

O Bike Sharing foi mantido de propósito:
1. **Comparação justa:** mesmo modelo, mesmas janelas, mesmos números de referência do projeto do Deepchecks.
2. **Dados por hora com data:** encaixam no uso natural do whylogs, que é um perfil por lote (aqui, por dia).
3. **O caso do furacão Sandy** (36 horas sem dados em outubro de 2012): o Deepchecks não percebeu, e queríamos ver se o whylogs perceberia.

---

## 📂 Estrutura do Repositório

```text
whylogs_bikesharing/
├── README.md
├── requirements.txt                   # versões fixadas (whylogs 1.6.4 não funciona com NumPy 2)
├── .gitignore
├── data/
│   ├── raw/hour.csv                   # dataset bruto (UCI #275)
│   └── processed/bike_clean.parquet   # dataset limpo
├── models/
│   ├── pipeline_2011.joblib           # LightGBM treinado só com 2011 (alvo em log1p)
│   └── metrics.json                   # janelas, métricas de validação, decisão sobre correção de viés
├── notebooks/
│   ├── data_cleaning.ipynb            # 1: limpeza
│   ├── eda.ipynb                      # 2: análise exploratória e diagnóstico do drift
│   ├── modelling.ipynb                # 3: treino (idêntico ao projeto do Deepchecks)
│   └── whylogs.ipynb                  # 4: perfis, drift, regras e monitoramento diário
├── src/
│   ├── __init__.py
│   └── bike.py                        # features, pipeline e CountModel (compartilhados)
└── runs/
    ├── whylogs_profile_summary.html          # resumo do perfil de 2012
    ├── whylogs_drift_report_pvalue.html      # drift com o teste padrão (KS / qui-quadrado)
    ├── whylogs_drift_report_hellinger.html   # drift com distância de Hellinger nas numéricas
    ├── whylogs_constraints_report.html       # gate PASS/FAIL (constraints)
    ├── whylogs_gate_summary.csv              # gate em tabela
    ├── whylogs_daily_volume_2012.png         # horas registradas por dia (monitor de volume)
    ├── whylogs_monthly_2012.csv / .png       # monitoramento mês a mês
    └── profiles/                             # perfis salvos (.bin): referência, 2012 e um por mês
```

---

## ⚙️ Pipeline

```text
data/raw/ ──> data_cleaning ──> data/processed/ ──> eda
                                      │
                                      ▼
                                 modelling ──> models/
                                      │
                                      ▼
                                  whylogs ──> runs/ (+ runs/profiles/)
```

Os notebooks 1 a 3 são os mesmos do projeto do Deepchecks (só os textos que citavam a ferramenta foram adaptados) e reproduzem os mesmos números: RMSE de validação 41,1, e decisão de **não** aplicar a correção de viés log-normal.

---

## 🧪 O que o notebook do whylogs faz

| Seção | O que faz |
|---|---|
| 1. Dados e modelo | Carrega o modelo e cria a coluna `prediction`. O whylogs não chama o modelo, só resume tabelas. |
| 2. Janelas | Referência = holdout de 2011 (dias 24+); produção = 2012. |
| 3. Perfis | Cria e salva perfis, mostra o que eles guardam e compara o tamanho do perfil com o do CSV. |
| 4. Drift | Teste padrão (p-valor) × distância de Hellinger, e um experimento mostrando o efeito do tamanho da amostra. |
| 5. Desempenho | RMSE, MAE e viés a partir de perfis de regressão (`log_regression_metrics`). |
| 6. Gate | 13 *constraints* sobre o perfil (integridade, categorias, volume, previsões, target, viés e MAE). |
| 7. Monitoramento diário | Um perfil por dia, monitor de volume diário e soma dos perfis diários em mensais. |
| 8. Monitoramento mensal | Cada mês de 2012 × o mesmo mês de 2011: drift (Hellinger), RMSE e viés. |

---

## 🧾 Resultados

### Drift 2011 × 2012

| Coluna | KS / qui-quadrado (p-valor) | Hellinger |
|---|---|---|
| temp, atemp, humidity | **DRIFT** | sem drift (0,11–0,13) |
| season, weekday, weather | **DRIFT** | não disponível para texto |
| cnt (target) | **DRIFT** | **possível drift (0,23)** |
| prediction | sem drift | sem drift (0,05) |

O teste padrão marca 7 de 13 colunas com drift. A causa é o tamanho da amostra: a mesma diferença de temperatura (KS ≈ 0,07) vira "sem drift" com 100 ou 300 linhas e "DRIFT" com 1.000 ou mais.

### Gate 2012 — ❌ REPROVADO (3 de 13 regras)

| Grupo | Resultado |
|---|---|
| Integridade (faixas e nulos) | ✅ |
| Categorias (só valores vistos na referência) | ✅ |
| Volume (≥ 99% das horas do ano) | ✅ |
| Previsões (média a ±20% da referência) | ✅ |
| **Target** (média de `cnt` a ±20%) | ❌ 234,7 contra a faixa 115,0–172,4 |
| **Viés** (erro médio a ±10% da média de referência) | ❌ |
| **Erro** (MAE ≤ 1,3× referência) | ❌ 93,0 contra o limite de 34,1 |

Desempenho: RMSE 41,1 → 130,2; viés da soma −0,9% → −37,9%, iguais aos do projeto do Deepchecks.

### Monitoramento diário e mensal

- **Volume diário:** a regra "pelo menos 20 horas por dia" reprovou **29/10 (1 hora) e 30/10 (11 horas)**, o furacão Sandy. Nem a regra anual de volume nem o Deepchecks perceberam esses dias.
- **Perfis somados:** a soma dos perfis diários bate exatamente com o cálculo direto em todos os meses (horas, média e RMSE).
- **Mês a mês:**
  - target e desempenho reprovam nos 12 meses;
  - as previsões passam nos 12;
  - as features reprovam em 10, porque a Hellinger fica mais sensível com ~720 linhas por mês e o limite de 0,15 precisaria de calibração.

---

## 🔁 Evidently × Deepchecks × whylogs

| | Evidently | Deepchecks | whylogs |
|---|---|---|---|
| Unidade de trabalho | tabelas | `Dataset` + modelo | perfil por lote |
| Precisa do modelo | não | sim | não |
| Drift padrão | distâncias (Wasserstein, Jensen-Shannon) em lotes grandes; testes em lotes pequenos | medidas de tamanho (KS, Cramér's V) | p-valor (KS, qui-quadrado); Hellinger opcional |
| Desempenho | relatório pronto | relatório pronto + segmentos fracos | só métricas (RMSE, MAE) |
| Regras PASS/FAIL | `TestSuite` | `Suite` + conditions | `ConstraintsBuilder` |
| Histórico ao longo do tempo | relatórios separados | relatórios separados | **perfis com data, somáveis** |
| Volume por lote | — | não percebeu o Sandy | **percebeu** (perfil diário) |
| Relatórios offline | sim | sim | **não** (bibliotecas carregadas da internet) |
| Licença | Apache 2.0 | AGPL-3.0 | Apache 2.0 |
| Última versão | ativa | dez/2024 | dez/2024 |

---

## 🚀 Como Executar

> Testado com **Python 3.11**.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Rode os notebooks em ordem: `data_cleaning` → `eda` → `modelling` → `whylogs` (este leva cerca de 30 segundos).

Para abrir os relatórios HTML de `runs/`, o computador precisa de **internet**: eles carregam jQuery, D3, Bootstrap e Handlebars de CDNs públicas.

---

## ⚠️ Limitações Encontradas

- **Manutenção:** a última versão (1.6.4) é de dezembro de 2024 e não funciona com NumPy 2. É preciso travar NumPy, SciPy e pandas em versões antigas.
- **Drift por p-valor gera alarmes demais** em lotes grandes; é preciso trocar para Hellinger (só colunas numéricas) ou calibrar os limites.
- **Desempenho sem relatório:** não há gráfico de resíduos nem busca de segmentos fracos. As regras de viés e de erro foram montadas à mão, com as colunas `erro` e `erro_abs`.
- **Relatórios dependem de internet** e os nomes das regras saem em inglês, com os limites calculados no texto.
- **Dias sem nenhum dado não geram perfil**, então precisam ser verificados à parte, comparando com o calendário.
