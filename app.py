import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import xgboost as xgb

# -------------------- 전역 설정 --------------------
st.set_page_config(page_title="Mini ML Dashboard", layout="wide")
sns.set_style("whitegrid")

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "model.pkl"
LOG_PATH = MODEL_DIR / "logs.csv"
RANDOM_STATE = 42

# -------------------- 데이터 --------------------
@st.cache_data
def load_data():
    iris = load_iris(as_frame=True)
    X = iris.data.rename(columns={
        "sepal length (cm)": "sepal_length",
        "sepal width (cm)": "sepal_width",
        "petal length (cm)": "petal_length",
        "petal width (cm)": "petal_width",
    })
    y = iris.target
    target_names = iris.target_names
    return X, y, target_names

X, y, target_names = load_data()

# -------------------- 도우미: 모델/서치/로그 --------------------
AVAILABLE_MODELS = [
    "Logistic Regression",
    "Random Forest",
    "SVC (RBF)",
    "KNN",
    "Gradient Boosting",
    "XGBoost",
    "Bagging (Tree)",
    "Stacking (LR+RF+SVC)",
]

def make_estimator(model_name: str, params: Dict):
    if model_name == "Logistic Regression":
        est = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=float(params.get("C", 1.0)),
                max_iter=int(params.get("max_iter", 200)),
                n_jobs=None,
                random_state=RANDOM_STATE
            ))
        ])
    elif model_name == "Random Forest":
        est = RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=None if params.get("max_depth") in [None, "None", ""] else int(params.get("max_depth")),
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    elif model_name == "SVC (RBF)":
        est = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                C=float(params.get("C", 1.0)),
                gamma=float(params.get("gamma", 0.1)),
                kernel="rbf",
                probability=True,
                random_state=RANDOM_STATE
            ))
        ])
    elif model_name == "KNN":
        est = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(
                n_neighbors=int(params.get("n_neighbors", 5)),
                weights=params.get("weights", "uniform"),
                p=int(params.get("p", 2))
            ))
        ])
    elif model_name == "Gradient Boosting":
        est = GradientBoostingClassifier(
            n_estimators=int(params.get("n_estimators", 200)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_depth=int(params.get("max_depth", 3)),
            random_state=RANDOM_STATE
        )
    elif model_name == "XGBoost":
        est = xgb.XGBClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=int(params.get("max_depth", 5)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            subsample=float(params.get("subsample", 1.0)),
            colsample_bytree=float(params.get("colsample_bytree", 1.0)),
            reg_lambda=float(params.get("reg_lambda", 1.0)),
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="mlogloss"
        )
    elif model_name == "Bagging (Tree)":
        est = BaggingClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=None if params.get("dt_max_depth") in [None, "None", ""] else int(params.get("dt_max_depth"))
            ),
            n_estimators=int(params.get("n_estimators", 100)),
            max_samples=float(params.get("max_samples", 1.0)),
            max_features=float(params.get("max_features", 1.0)),
            bootstrap=True,
            bootstrap_features=False,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    else:  # Stacking (LR+RF+SVC)
        base_estimators = [
            ('lr', Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(max_iter=200, random_state=RANDOM_STATE))
            ])),
            ('svc', Pipeline([
                ('scaler', StandardScaler()),
                ('clf', SVC(C=1.0, gamma='scale', probability=True, random_state=RANDOM_STATE))
            ])),
            ('rf', RandomForestClassifier(n_estimators=int(params.get("rf_n_estimators", 200)),
                                         random_state=RANDOM_STATE, n_jobs=-1))
        ]
        final_estimator = LogisticRegression(max_iter=int(params.get("final_max_iter", 200)), random_state=RANDOM_STATE)
        est = StackingClassifier(
            estimators=base_estimators,
            final_estimator=final_estimator,
            stack_method="auto",
            passthrough=bool(params.get("passthrough", False)),
            n_jobs=-1
        )
    return est

def run_cross_validation(estimator, X, y, cv_splits: int = 5) -> Dict:
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        estimator, X, y,
        cv=cv,
        scoring={"acc": "accuracy", "f1": "f1_macro"},
        n_jobs=-1,
        return_train_score=False
    )
    result = {
        "acc_mean": float(np.mean(scores["test_acc"])),
        "acc_std":  float(np.std(scores["test_acc"])),
        "f1_mean":  float(np.mean(scores["test_f1"])),
        "f1_std":   float(np.std(scores["test_f1"])),
        "per_fold_acc": scores["test_acc"],
        "per_fold_f1": scores["test_f1"],
    }
    return result

def train_holdout(estimator, X, y, test_size: float = 0.2) -> Tuple[object, Dict, plt.Figure, tuple]:
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    estimator.fit(Xtr, ytr)
    pred = estimator.predict(Xte)
    acc = accuracy_score(yte, pred)
    f1 = f1_score(yte, pred, average="macro")

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(yte, pred, display_labels=target_names, cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()

    report = classification_report(yte, pred, target_names=target_names)
    metrics = {"acc_holdout": float(acc), "f1_holdout": float(f1), "report": report}
    return estimator, metrics, fig, (Xtr, Xte, ytr, yte)

def get_search_spaces(model_name: str):
    if model_name == "Logistic Regression":
        grid = {"clf__C": [0.01, 0.1, 1, 3, 10], "clf__max_iter": [200, 400, 800]}
        random = {"clf__C": [0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10], "clf__max_iter": [100, 200, 300, 500, 800, 1000]}
    elif model_name == "Random Forest":
        grid = {"n_estimators": [100, 200, 300, 400], "max_depth": [None, 3, 5, 7, 10], "min_samples_split": [2, 4, 6]}
        random = {"n_estimators": [80, 120, 160, 200, 280, 360, 420, 480], "max_depth": [None, 3, 4, 5, 6, 7, 8, 10, 12], "min_samples_split": [2, 3, 4, 5, 6, 8, 10]}
    elif model_name == "SVC (RBF)":
        grid = {"clf__C": [0.1, 1, 3, 10], "clf__gamma": [0.01, 0.1, 0.3, 1.0]}
        random = {"clf__C": [0.05, 0.1, 0.3, 1, 3, 5, 10, 20], "clf__gamma": [0.005, 0.01, 0.05, 0.1, 0.3, 1.0]}
    elif model_name == "KNN":
        grid = {"clf__n_neighbors": [3, 5, 7, 9, 11], "clf__weights": ["uniform", "distance"], "clf__p": [1, 2]}
        random = {"clf__n_neighbors": list(range(3, 21, 2)), "clf__weights": ["uniform", "distance"], "clf__p": [1, 2]}
    elif model_name == "Gradient Boosting":
        grid = {"n_estimators": [100, 200, 300], "learning_rate": [0.05, 0.1, 0.2], "max_depth": [2, 3, 4, 5]}
        random = {"n_estimators": [80, 120, 160, 200, 280, 320], "learning_rate": [0.03, 0.05, 0.1, 0.2], "max_depth": [2, 3, 4, 5, 6]}
    elif model_name == "XGBoost":
        grid = {
            "n_estimators": [200, 300, 500],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.2],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "reg_lambda": [0.5, 1.0, 2.0],
        }
        random = {
            "n_estimators": [150, 200, 250, 300, 400, 500],
            "max_depth": [3, 4, 5, 6, 7, 8],
            "learning_rate": [0.03, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "reg_lambda": [0.1, 0.5, 1.0, 2.0, 5.0],
        }
    elif model_name == "Bagging (Tree)":
        grid = {
            "n_estimators": [50, 100, 200],
            "estimator__max_depth": [None, 3, 5, 7],
            "max_samples": [0.6, 0.8, 1.0],
            "max_features": [0.6, 0.8, 1.0],
        }
        random = {
            "n_estimators": [50, 80, 100, 150, 200, 300],
            "estimator__max_depth": [None, 3, 4, 5, 6, 7, 10],
            "max_samples": [0.5, 0.6, 0.7, 0.8, 1.0],
            "max_features": [0.5, 0.6, 0.7, 0.8, 1.0],
        }
    else:  # Stacking
        grid = {
            "rf__n_estimators": [100, 200, 300],
            "final_estimator__C": [0.5, 1.0, 2.0],
        }
        random = {
            "rf__n_estimators": [80, 120, 160, 200, 280, 360],
            "final_estimator__C": [0.3, 0.5, 1.0, 2.0, 3.0],
        }
    return grid, random

def run_grid_search(model_name: str, params: Dict, X, y, cv_splits: int):
    base = make_estimator(model_name, params)
    grid, _ = get_search_spaces(model_name)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(base, grid, cv=cv, scoring="accuracy", n_jobs=-1, refit=True, return_train_score=False)
    gs.fit(X, y)
    return gs

def run_random_search(model_name: str, params: Dict, X, y, cv_splits: int, n_iter: int = 20):
    base = make_estimator(model_name, params)
    _, random = get_search_spaces(model_name)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    rs = RandomizedSearchCV(base, random, n_iter=n_iter, cv=cv, scoring="accuracy", n_jobs=-1, refit=True, random_state=RANDOM_STATE)
    rs.fit(X, y)
    return rs

def save_model(model):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

def load_model_from_disk():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

def append_log(entry: Dict):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([entry])
    if LOG_PATH.exists():
        df_old = pd.read_csv(LOG_PATH)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(LOG_PATH, index=False)

# -------------------- 사이드바: 공통 설정 --------------------
with st.sidebar:
    st.header("⚙️ 설정")
    model_name = st.selectbox("모델", AVAILABLE_MODELS, index=0)

    # 모델별 하이퍼파라미터 UI
    if model_name == "Logistic Regression":
        C = st.slider("C (규제 역수)", 0.01, 5.0, 1.0, 0.01)
        max_iter = st.slider("max_iter", 100, 1000, 200, 50)
        params = {"C": C, "max_iter": max_iter}
    elif model_name == "Random Forest":
        n_estimators = st.slider("n_estimators", 50, 500, 200, 10)
        max_depth = st.selectbox("max_depth", [None, 3, 5, 7, 9, 12], index=0)
        params = {"n_estimators": n_estimators, "max_depth": max_depth}
    elif model_name == "SVC (RBF)":
        C = st.slider("C", 0.01, 20.0, 1.0, 0.01)
        gamma = st.slider("gamma", 0.001, 1.0, 0.1, 0.001)
        params = {"C": C, "gamma": gamma}
    elif model_name == "KNN":
        n_neighbors = st.slider("n_neighbors", 1, 50, 5, 1)
        weights = st.selectbox("weights", ["uniform", "distance"], index=0)
        p = st.selectbox("p(거리)", [1, 2], index=1)
        params = {"n_neighbors": n_neighbors, "weights": weights, "p": p}
    elif model_name == "Gradient Boosting":
        n_estimators = st.slider("n_estimators", 50, 500, 200, 10)
        learning_rate = st.slider("learning_rate", 0.01, 0.5, 0.1, 0.01)
        max_depth = st.slider("max_depth", 1, 10, 3, 1)
        params = {"n_estimators": n_estimators, "learning_rate": learning_rate, "max_depth": max_depth}
    elif model_name == "XGBoost":
        n_estimators = st.slider("n_estimators", 50, 800, 300, 50)
        max_depth = st.slider("max_depth", 1, 12, 5, 1)
        learning_rate = st.slider("learning_rate", 0.01, 0.5, 0.1, 0.01)
        subsample = st.slider("subsample", 0.5, 1.0, 1.0, 0.05)
        colsample_bytree = st.slider("colsample_bytree", 0.5, 1.0, 1.0, 0.05)
        reg_lambda = st.slider("reg_lambda", 0.0, 5.0, 1.0, 0.1)
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_lambda": reg_lambda
        }
    elif model_name == "Bagging (Tree)":
        n_estimators = st.slider("n_estimators", 10, 500, 100, 10)
        dt_max_depth = st.selectbox("DT max_depth", [None, 3, 5, 7, 10], index=0)
        max_samples = st.slider("max_samples", 0.5, 1.0, 1.0, 0.05)
        max_features = st.slider("max_features", 0.5, 1.0, 1.0, 0.05)
        params = {
            "n_estimators": n_estimators,
            "dt_max_depth": dt_max_depth,
            "max_samples": max_samples,
            "max_features": max_features
        }
    else:  # Stacking
        rf_n_estimators = st.slider("RF n_estimators", 50, 500, 200, 10)
        final_max_iter = st.slider("final max_iter", 100, 1000, 200, 50)
        passthrough = st.checkbox("passthrough(기본 특성 포함)", value=False)
        params = {"rf_n_estimators": rf_n_estimators, "final_max_iter": final_max_iter, "passthrough": passthrough}

    cv_splits = st.slider("교차검증 폴드(k)", 3, 10, 5, 1)
    test_size = st.slider("테스트 비율", 0.1, 0.5, 0.2, 0.05)
    st.markdown("---")
    st.caption(f"모델 파일: {MODEL_PATH}")
    existing = load_model_from_disk()
    st.write("저장된 모델:", "있음" if existing else "없음")

# -------------------- 본문 탭 --------------------
tab_train, tab_compare, tab_predict, tab_logs = st.tabs(["학습", "비교", "예측", "로그"])

# ========== 탭: 학습 ==========
with tab_train:
    st.subheader("교차검증 및 학습")
    c1, c2, c3 = st.columns(3)
    c1.metric("모델", model_name)
    c2.metric("CV 폴드", cv_splits)
    c3.metric("테스트 비율", test_size)

    st.markdown("---")
    colL, colR = st.columns([1, 1])

    with colL:
        st.markdown("#### 교차검증")
        if st.button("교차검증 실행", type="primary"):
            est = make_estimator(model_name, params)
            with st.spinner("교차검증 중..."):
                cv_res = run_cross_validation(est, X, y, cv_splits=cv_splits)
            a, b = st.columns(2)
            a.metric("CV 정확도(평균±표준편차)", f"{cv_res['acc_mean']:.4f} ± {cv_res['acc_std']:.4f}")
            b.metric("CV F1-macro(평균±표준편차)", f"{cv_res['f1_mean']:.4f} ± {cv_res['f1_std']:.4f}")

            df_cv = pd.DataFrame({
                "fold": np.arange(1, len(cv_res["per_fold_acc"]) + 1),
                "accuracy": cv_res["per_fold_acc"],
                "f1_macro": cv_res["per_fold_f1"],
            })
            st.dataframe(df_cv, use_container_width=True, hide_index=True)
            st.bar_chart(df_cv.set_index("fold"))

    with colR:
        st.markdown("#### 홀드아웃 학습/평가")
        if st.button("모델 학습 및 저장"):
            est = make_estimator(model_name, params)
            with st.spinner("학습 중..."):
                cv_res = run_cross_validation(est, X, y, cv_splits=cv_splits)
                model_fitted, holdout_metrics, fig_cm, split = train_holdout(est, X, y, test_size=test_size)
                save_model(model_fitted)
                st.session_state["last_split"] = split
            st.success(f"모델 저장 완료! 정확도(홀드아웃): {holdout_metrics['acc_holdout']:.4f}, F1: {holdout_metrics['f1_holdout']:.4f}")
            st.pyplot(fig_cm)
            st.text("분류 리포트")
            st.code(holdout_metrics["report"], language="text")

            append_log({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "action": "train_save",
                "model": model_name,
                "params": str(params),
                "cv_acc_mean": cv_res["acc_mean"],
                "cv_acc_std": cv_res["acc_std"],
                "cv_f1_mean": cv_res["f1_mean"],
                "cv_f1_std": cv_res["f1_std"],
                "holdout_acc": holdout_metrics["acc_holdout"],
                "holdout_f1": holdout_metrics["f1_holdout"],
            })

    # 학습 데이터 표시(실제 사용 분할)
    with st.expander("학습 데이터(실제 분할) 보기"):
        if "last_split" not in st.session_state:
            st.info("아직 학습을 수행하지 않았습니다. ‘모델 학습 및 저장’을 먼저 실행해 주세요.")
        else:
            Xtr, Xte, ytr, yte = st.session_state["last_split"]
            df_tr = Xtr.copy(); df_tr["target"] = ytr
            df_te = Xte.copy(); df_te["target"] = yte

            cA, cB = st.columns(2)
            with cA:
                st.markdown("##### 학습 데이터(상위 10)")
                st.dataframe(df_tr.head(10), use_container_width=True)
                st.markdown("##### 학습 클래스 분포")
                st.dataframe(df_tr["target"].value_counts().rename_axis("class").reset_index(name="count"), use_container_width=True)
                st.download_button("학습 데이터 CSV 다운로드", df_tr.to_csv(index=False).encode("utf-8"), "train_split.csv", "text/csv")
            with cB:
                st.markdown("##### 테스트 데이터(상위 10)")
                st.dataframe(df_te.head(10), use_container_width=True)
                st.markdown("##### 테스트 클래스 분포")
                st.dataframe(df_te["target"].value_counts().rename_axis("class").reset_index(name="count"), use_container_width=True)
                st.download_button("테스트 데이터 CSV 다운로드", df_te.to_csv(index=False).encode("utf-8"), "test_split.csv", "text/csv")

    with st.expander("하이퍼파라미터 탐색(Grid/Random)"):
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            if st.button("Grid Search 실행"):
                with st.spinner("Grid Search 중..."):
                    gs = run_grid_search(model_name, params, X, y, cv_splits=cv_splits)
                st.success(f"Best Score: {gs.best_score_:.4f}")
                st.write("Best Params:", gs.best_params_)
                if st.button("최적 파라미터 적용 후 저장"):
                    best_est = gs.best_estimator_
                    save_model(best_est)
                    append_log({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "action": "grid_search_best_save",
                        "model": model_name,
                        "params": str(gs.best_params_),
                        "cv_acc_mean": gs.best_score_,
                        "cv_acc_std": np.nan,
                        "cv_f1_mean": np.nan,
                        "cv_f1_std": np.nan,
                        "holdout_acc": np.nan,
                        "holdout_f1": np.nan,
                    })
                    st.success("최적 모델 저장 완료!")
        with c2:
            n_iter = st.slider("Random Search n_iter", 5, 50, 20, 5)
            if st.button("Random Search 실행"):
                with st.spinner("Random Search 중..."):
                    rs = run_random_search(model_name, params, X, y, cv_splits=cv_splits, n_iter=n_iter)
                st.success(f"Best Score: {rs.best_score_:.4f}")
                st.write("Best Params:", rs.best_params_)
                if st.button("랜덤 최적 파라미터 적용 후 저장"):
                    best_est = rs.best_estimator_
                    save_model(best_est)
                    append_log({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "action": "random_search_best_save",
                        "model": model_name,
                        "params": str(rs.best_params_),
                        "cv_acc_mean": rs.best_score_,
                        "cv_acc_std": np.nan,
                        "cv_f1_mean": np.nan,
                        "cv_f1_std": np.nan,
                        "holdout_acc": np.nan,
                        "holdout_f1": np.nan,
                    })
                    st.success("최적 모델 저장 완료!")
        with c3:
            st.info("탐색 결과는 교차검증 점수 기준입니다. 저장 후 예측 탭에서 바로 사용 가능합니다.")

# ========== 탭: 비교 ==========
with tab_compare:
    st.subheader("모델 비교(교차검증 성능)")
    selected = st.multiselect("비교할 모델 선택", AVAILABLE_MODELS, default=["Logistic Regression", "Random Forest", "XGBoost"])
    if len(selected) == 0:
        st.info("최소 한 개 이상의 모델을 선택해 주세요.")
    else:
        rows = []
        with st.spinner("교차검증 중..."):
            for m in selected:
                est = make_estimator(m, {})
                cv_res = run_cross_validation(est, X, y, cv_splits=cv_splits)
                rows.append({"model": m, "metric": "accuracy", "mean": cv_res["acc_mean"], "std": cv_res["acc_std"]})
                rows.append({"model": m, "metric": "f1_macro", "mean": cv_res["f1_mean"], "std": cv_res["f1_std"]})

        df_cmp = pd.DataFrame(rows)
        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(df_cmp, use_container_width=True, hide_index=True)
        with c2:
            pivot = df_cmp.pivot(index="metric", columns="model", values="mean")
            st.bar_chart(pivot)

# ========== 탭: 예측 ==========
with tab_predict:
    st.subheader("단일 샘플 실시간 예측")

    model_loaded = load_model_from_disk()
    use_temp_train = False
    if model_loaded is None:
        st.warning("저장된 모델이 없습니다. 임시로 현재 설정으로 전체 데이터 학습 후 예측합니다.")
        use_temp_train = True
        model_loaded = make_estimator(model_name, params)
        model_loaded.fit(X, y)

    colA, colB, colC, colD = st.columns(4)
    sepal_length = colA.number_input("sepal_length", 0.0, 10.0, 5.1, 0.1)
    sepal_width  = colB.number_input("sepal_width",  0.0, 10.0, 3.5, 0.1)
    petal_length = colC.number_input("petal_length", 0.0, 10.0, 1.4, 0.1)
    petal_width  = colD.number_input("petal_width",  0.0, 10.0, 0.2, 0.1)

    X_one = pd.DataFrame([{
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    }])

    if st.button("예측하기", type="primary"):
        try:
            pred = model_loaded.predict(X_one)[0]
            proba = None
            if hasattr(model_loaded, "predict_proba"):
                proba = model_loaded.predict_proba(X_one)[0]
            elif hasattr(getattr(model_loaded, "named_steps", {}).get("clf", object), "predict_proba"):
                if "scaler" in model_loaded.named_steps:
                    X_one_scaled = model_loaded.named_steps["scaler"].transform(X_one)
                    proba = model_loaded.named_steps["clf"].predict_proba(X_one_scaled)[0]
                else:
                    proba = model_loaded.named_steps["clf"].predict_proba(X_one)[0]

            label = target_names[pred]
            st.success(f"예측 결과: {label}")
            if proba is not None:
                df_prob = pd.DataFrame({"class": target_names, "probability": proba})
                st.dataframe(df_prob, use_container_width=True, hide_index=True)

            if use_temp_train:
                st.info("임시 학습 모델로 예측했습니다. 학습 탭에서 모델을 저장하시면 이후 그대로 사용됩니다.")
        except Exception as e:
            st.error(f"예측 중 오류: {e}")

# ========== 탭: 로그 ==========
with tab_logs:
    st.subheader("학습/탐색 로그")
    if LOG_PATH.exists():
        df_logs = pd.read_csv(LOG_PATH)
        st.dataframe(df_logs.tail(200), use_container_width=True)
        st.download_button("CSV 다운로드", df_logs.to_csv(index=False).encode("utf-8"),
                           file_name="training_logs.csv", mime="text/csv")
        if st.button("로그 초기화(삭제)"):
            try:
                LOG_PATH.unlink()
                st.success("로그를 삭제했습니다.")
                st.rerun()
            except Exception as e:
                st.error(f"삭제 오류: {e}")
    else:
        st.info("아직 로그가 없습니다. 학습 또는 탐색을 실행해 보세요.")