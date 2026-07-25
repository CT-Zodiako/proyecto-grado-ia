#!/usr/bin/env python3
"""
Mejora del modelo de predicción para programas de Medicina en Saber Pro.

Acciones:
1. Reentrenar Random Forest con datos 2020-2025.
2. Probar XGBoost y LightGBM.
3. Agregar nuevas variables históricas.
4. Comparar métricas y exportar el mejor modelo.
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

try:
    import xgboost as xgb
    # Probar si la biblioteca compartida se puede cargar
    _ = xgb.XGBRegressor()
    HAS_XGB = True
except (ImportError, Exception):
    HAS_XGB = False
    xgb = None

try:
    import lightgbm as lgb
    # Probar si la biblioteca compartida se puede cargar
    _ = lgb.LGBMRegressor()
    HAS_LGB = True
except (ImportError, Exception):
    HAS_LGB = False
    lgb = None

try:
    import catboost as cb
    HAS_CB = True
except (ImportError, Exception):
    HAS_CB = False
    cb = None


def add_new_features(df):
    """Agrega nuevas variables históricas al dataset."""
    df = df.sort_values(['ID_INSTITUCION', 'ID_PROGRAMA_ACAD', 'AÑO']).copy()
    grupo = df.groupby(['ID_INSTITUCION', 'ID_PROGRAMA_ACAD'], sort=False)

    # Promedio móvil 3 años
    df['promedio_movil_3_anios'] = grupo['promedio_global_anual'].transform(
        lambda s: s.shift(1).rolling(window=3, min_periods=1).mean()
    )

    # Desviación 3 años
    df['desviacion_historica_3_anios'] = grupo['promedio_global_anual'].transform(
        lambda s: s.shift(1).rolling(window=3, min_periods=3).std()
    )

    # Tasa de crecimiento anual LAGUEADA (anterior vs ante-anterior).
    # FIX LEAKAGE: la versión original usaba promedio_global_anual (el target
    # del mismo año) en el numerador, permitiendo recuperar el target exacto.
    anterior = grupo['promedio_global_anual'].shift(1)
    ante_anterior = grupo['promedio_global_anual'].shift(2)
    df['tasa_crecimiento_anual'] = np.where(
        ante_anterior.notna() & ante_anterior.ne(0),
        (anterior - ante_anterior) / ante_anterior * 100,
        np.nan
    )

    # Máximo y mínimo histórico hasta el año anterior
    df['maximo_historico'] = grupo['promedio_global_anual'].transform(
        lambda s: s.shift(1).expanding(min_periods=1).max()
    )
    df['minimo_historico'] = grupo['promedio_global_anual'].transform(
        lambda s: s.shift(1).expanding(min_periods=1).min()
    )

    # Diferencia con el máximo histórico, medida desde el AÑO ANTERIOR.
    # FIX LEAKAGE: la versión original era target - maximo_historico, una
    # función lineal exacta del target del mismo año.
    df['diferencia_maximo_historico'] = anterior - df['maximo_historico']

    # Años desde el primer registro
    primer_año = grupo['AÑO'].transform('min')
    df['anios_desde_inicio'] = df['AÑO'] - primer_año

    # Ranking dentro del departamento según el puntaje del AÑO ANTERIOR.
    # FIX LEAKAGE: la versión original rankeaba el target del mismo año
    # (información no disponible al momento de predecir).
    df['ranking_departamento'] = df.groupby(['NOMBRE_DEPARTAMENTO', 'AÑO'])['promedio_global_anterior'].rank(method='min', ascending=False)

    return df


def build_pipeline(model, numeric_features, categorical_features):
    """Construye el pipeline con preprocesamiento, imputación y modelo."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    return pipeline


def train_and_evaluate(model_name, pipeline, X_train, y_train, X_val, y_val, X_test, y_test):
    """Entrena y evalúa un modelo."""
    pipeline.fit(X_train, y_train)

    preds_val = pipeline.predict(X_val)
    preds_test = pipeline.predict(X_test)

    metrics = {
        'modelo': model_name,
        'validacion': {
            'MAE': float(mean_absolute_error(y_val, preds_val)),
            'RMSE': float(np.sqrt(mean_squared_error(y_val, preds_val))),
            'R2': float(r2_score(y_val, preds_val))
        },
        'test': {
            'MAE': float(mean_absolute_error(y_test, preds_test)),
            'RMSE': float(np.sqrt(mean_squared_error(y_test, preds_test))),
            'R2': float(r2_score(y_test, preds_test))
        }
    }

    return pipeline, metrics


def main():
    print('=== Mejora del modelo de Medicina Saber Pro ===\n')

    # 1. Cargar datos
    df = pd.read_csv('artifacts/medicina_features_2020_2025.csv', encoding='utf-8')
    print(f'Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas')

    # 2. Agregar nuevas features
    df = add_new_features(df)
    print('Nuevas variables agregadas')
    print(df[['promedio_movil_3_anios', 'desviacion_historica_3_anios', 'tasa_crecimiento_anual',
              'maximo_historico', 'minimo_historico', 'anios_desde_inicio', 'ranking_departamento']].head())

    # 3. Definir variables
    target = 'promedio_global_anual'
    numeric_features = [
        'AÑO', 'promedio_global_anterior', 'promedio_movil_2_anios', 'desviacion_historica_2_anios',
        'anios_historicos_disponibles', 'promedio_movil_3_anios', 'desviacion_historica_3_anios',
        'tasa_crecimiento_anual', 'maximo_historico', 'minimo_historico', 'diferencia_maximo_historico',
        'anios_desde_inicio', 'ranking_departamento'
    ]
    categorical_features = [
        'NOMBRE_REGION', 'NOMBRE_DEPARTAMENTO', 'NOMBRE_MUNICIPIO',
        'NOMBRE_INSTITUCION', 'NOMBRE_PROGRAMA_ACAD'
    ]

    # Filtrar filas con target válido y al menos un año de historia
    model_data = df[numeric_features + categorical_features + [target]].dropna(subset=[target]).copy()
    # Requerir que haya al menos un año histórico para predecir
    model_data = model_data[model_data['anios_historicos_disponibles'] > 0].copy()
    print(f'\nFilas para modelado: {len(model_data)}')

    # 4. Split temporal
    # Train: 2020-2023, Valid: 2024, Test: 2025
    train_data = model_data[model_data['AÑO'] <= 2023]
    val_data = model_data[model_data['AÑO'] == 2024]
    test_data = model_data[model_data['AÑO'] == 2025]

    print(f'Train: {len(train_data)} | Valid: {len(val_data)} | Test: {len(test_data)}')

    X_train = train_data[numeric_features + categorical_features]
    y_train = train_data[target]
    X_val = val_data[numeric_features + categorical_features]
    y_val = val_data[target]
    X_test = test_data[numeric_features + categorical_features]
    y_test = test_data[target]

    # 5. Entrenar y comparar modelos
    results = []
    best_model = None
    best_mae = float('inf')
    best_model_name = None

    # Random Forest
    print('\n--- Entrenando Random Forest ---')
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    pipeline_rf = build_pipeline(rf, numeric_features, categorical_features)
    pipeline_rf, metrics_rf = train_and_evaluate(
        'Random Forest', pipeline_rf, X_train, y_train, X_val, y_val, X_test, y_test
    )
    print(f"Valid MAE: {metrics_rf['validacion']['MAE']:.3f} | Test MAE: {metrics_rf['test']['MAE']:.3f}")
    results.append(metrics_rf)
    if metrics_rf['validacion']['MAE'] < best_mae:
        best_mae = metrics_rf['validacion']['MAE']
        best_model = pipeline_rf
        best_model_name = 'Random Forest'

    # XGBoost
    if HAS_XGB:
        print('\n--- Entrenando XGBoost ---')
        xgb_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        pipeline_xgb = build_pipeline(xgb_model, numeric_features, categorical_features)
        pipeline_xgb, metrics_xgb = train_and_evaluate(
            'XGBoost', pipeline_xgb, X_train, y_train, X_val, y_val, X_test, y_test
        )
        print(f"Valid MAE: {metrics_xgb['validacion']['MAE']:.3f} | Test MAE: {metrics_xgb['test']['MAE']:.3f}")
        results.append(metrics_xgb)
        if metrics_xgb['validacion']['MAE'] < best_mae:
            best_mae = metrics_xgb['validacion']['MAE']
            best_model = pipeline_xgb
            best_model_name = 'XGBoost'
    else:
        print('XGBoost no está instalado, se omite.')

    # LightGBM
    if HAS_LGB:
        print('\n--- Entrenando LightGBM ---')
        lgb_model = lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        pipeline_lgb = build_pipeline(lgb_model, numeric_features, categorical_features)
        pipeline_lgb, metrics_lgb = train_and_evaluate(
            'LightGBM', pipeline_lgb, X_train, y_train, X_val, y_val, X_test, y_test
        )
        print(f"Valid MAE: {metrics_lgb['validacion']['MAE']:.3f} | Test MAE: {metrics_lgb['test']['MAE']:.3f}")
        results.append(metrics_lgb)
        if metrics_lgb['validacion']['MAE'] < best_mae:
            best_mae = metrics_lgb['validacion']['MAE']
            best_model = pipeline_lgb
            best_model_name = 'LightGBM'
    else:
        print('LightGBM no está instalado, se omite.')

    # CatBoost
    if HAS_CB:
        print('\n--- Entrenando CatBoost ---')
        cb_model = cb.CatBoostRegressor(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function='MAE',
            random_state=42,
            verbose=False,
            thread_count=-1
        )
        pipeline_cb = build_pipeline(cb_model, numeric_features, categorical_features)
        pipeline_cb, metrics_cb = train_and_evaluate(
            'CatBoost', pipeline_cb, X_train, y_train, X_val, y_val, X_test, y_test
        )
        print(f"Valid MAE: {metrics_cb['validacion']['MAE']:.3f} | Test MAE: {metrics_cb['test']['MAE']:.3f}")
        results.append(metrics_cb)
        if metrics_cb['validacion']['MAE'] < best_mae:
            best_mae = metrics_cb['validacion']['MAE']
            best_model = pipeline_cb
            best_model_name = 'CatBoost'
    else:
        print('CatBoost no está instalado, se omite.')

    # HistGradientBoosting (sklearn)
    print('\n--- Entrenando HistGradientBoosting ---')
    hgb_model = HistGradientBoostingRegressor(
        max_iter=300,
        max_depth=6,
        learning_rate=0.05,
        random_state=42
    )
    pipeline_hgb = build_pipeline(hgb_model, numeric_features, categorical_features)
    pipeline_hgb, metrics_hgb = train_and_evaluate(
        'HistGradientBoosting', pipeline_hgb, X_train, y_train, X_val, y_val, X_test, y_test
    )
    print(f"Valid MAE: {metrics_hgb['validacion']['MAE']:.3f} | Test MAE: {metrics_hgb['test']['MAE']:.3f}")
    results.append(metrics_hgb)
    if metrics_hgb['validacion']['MAE'] < best_mae:
        best_mae = metrics_hgb['validacion']['MAE']
        best_model = pipeline_hgb
        best_model_name = 'HistGradientBoosting'

    # Ridge
    print('\n--- Entrenando Ridge ---')
    ridge_model = Ridge(alpha=1.0, random_state=42)
    pipeline_ridge = build_pipeline(ridge_model, numeric_features, categorical_features)
    pipeline_ridge, metrics_ridge = train_and_evaluate(
        'Ridge', pipeline_ridge, X_train, y_train, X_val, y_val, X_test, y_test
    )
    print(f"Valid MAE: {metrics_ridge['validacion']['MAE']:.3f} | Test MAE: {metrics_ridge['test']['MAE']:.3f}")
    results.append(metrics_ridge)
    if metrics_ridge['validacion']['MAE'] < best_mae:
        best_mae = metrics_ridge['validacion']['MAE']
        best_model = pipeline_ridge
        best_model_name = 'Ridge'

    # Lasso
    print('\n--- Entrenando Lasso ---')
    lasso_model = Lasso(alpha=1.0, random_state=42, max_iter=10000)
    pipeline_lasso = build_pipeline(lasso_model, numeric_features, categorical_features)
    pipeline_lasso, metrics_lasso = train_and_evaluate(
        'Lasso', pipeline_lasso, X_train, y_train, X_val, y_val, X_test, y_test
    )
    print(f"Valid MAE: {metrics_lasso['validacion']['MAE']:.3f} | Test MAE: {metrics_lasso['test']['MAE']:.3f}")
    results.append(metrics_lasso)
    if metrics_lasso['validacion']['MAE'] < best_mae:
        best_mae = metrics_lasso['validacion']['MAE']
        best_model = pipeline_lasso
        best_model_name = 'Lasso'

    # ElasticNet
    print('\n--- Entrenando ElasticNet ---')
    en_model = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=10000)
    pipeline_en = build_pipeline(en_model, numeric_features, categorical_features)
    pipeline_en, metrics_en = train_and_evaluate(
        'ElasticNet', pipeline_en, X_train, y_train, X_val, y_val, X_test, y_test
    )
    print(f"Valid MAE: {metrics_en['validacion']['MAE']:.3f} | Test MAE: {metrics_en['test']['MAE']:.3f}")
    results.append(metrics_en)
    if metrics_en['validacion']['MAE'] < best_mae:
        best_mae = metrics_en['validacion']['MAE']
        best_model = pipeline_en
        best_model_name = 'ElasticNet'

    # KNN
    print('\n--- Entrenando KNN ---')
    knn_model = KNeighborsRegressor(n_neighbors=5)
    pipeline_knn = build_pipeline(knn_model, numeric_features, categorical_features)
    pipeline_knn, metrics_knn = train_and_evaluate(
        'KNN', pipeline_knn, X_train, y_train, X_val, y_val, X_test, y_test
    )
    print(f"Valid MAE: {metrics_knn['validacion']['MAE']:.3f} | Test MAE: {metrics_knn['test']['MAE']:.3f}")
    results.append(metrics_knn)
    if metrics_knn['validacion']['MAE'] < best_mae:
        best_mae = metrics_knn['validacion']['MAE']
        best_model = pipeline_knn
        best_model_name = 'KNN'

    # 6. Comparar resultados
    print('\n=== Comparación de modelos ===')
    comparison = pd.DataFrame([
        {
            'Modelo': r['modelo'],
            'Valid MAE': r['validacion']['MAE'],
            'Valid RMSE': r['validacion']['RMSE'],
            'Valid R²': r['validacion']['R2'],
            'Test MAE': r['test']['MAE'],
            'Test RMSE': r['test']['RMSE'],
            'Test R²': r['test']['R2']
        }
        for r in results
    ])
    print(comparison.to_string(index=False))

    # 7. Exportar mejor modelo y artefactos
    print(f'\n=== Mejor modelo: {best_model_name} ===')
    print(f'Valid MAE: {best_mae:.3f}')

    Path('artifacts').mkdir(exist_ok=True)
    joblib.dump(best_model, 'artifacts/model_v2.joblib')
    print('Modelo exportado: artifacts/model_v2.joblib')

    # Feature schema v2
    schema_v2 = {
        'target': target,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'excluded_features': [target, 'promedio_prueba_media', 'cantidad_evaluados_media_pruebas',
                              'cantidad_evaluados_max_pruebas', 'cantidad_pruebas', 'registros_modulo',
                              'variacion_anual', 'variacion_porcentual', 'crecimiento_acumulado_desde_inicio',
                              'mejora_vs_anio_anterior', 'disminuye_vs_anio_anterior', 'ID_REGION', 'ID_DEPARTAMENTO',
                              'ID_MUNICIPIO', 'ID_INSTITUCION', 'ID_NBC', 'ID_PROGRAMA_ACAD', 'NBC'],
        'model_name': best_model_name,
        'split_strategy': 'temporal_train_2020_2023_valid_2024_test_2025',
        'new_features_added': True
    }
    Path('artifacts/feature_schema_v2.json').write_text(json.dumps(schema_v2, indent=2, ensure_ascii=False), encoding='utf-8')

    # Metrics v2
    metrics_v2 = {
        'model_results': results,
        'best_model_by_validation_mae': best_model_name,
        'best_validation_metrics': next(r['validacion'] for r in results if r['modelo'] == best_model_name),
        'best_test_metrics': next(r['test'] for r in results if r['modelo'] == best_model_name)
    }
    Path('artifacts/metrics_v2.json').write_text(json.dumps(metrics_v2, indent=2, ensure_ascii=False), encoding='utf-8')

    # Predicciones para 2025 por programa
    if len(test_data) > 0:
        preds_test = best_model.predict(X_test)
        test_results = test_data[['AÑO', 'NOMBRE_INSTITUCION', 'NOMBRE_MUNICIPIO', 'NOMBRE_DEPARTAMENTO', target]].copy()
        test_results['prediccion'] = preds_test
        test_results['error_absoluto'] = np.abs(test_results[target] - test_results['prediccion'])
        test_results.to_csv('artifacts/predicciones_test_2025_v2.csv', index=False, encoding='utf-8')
        print(f'\nPredicciones test 2025 exportadas: artifacts/predicciones_test_2025_v2.csv')
        print(f'MAE test 2025: {test_results["error_absoluto"].mean():.3f}')
        print('\nTop 5 errores más grandes:')
        print(test_results.nlargest(5, 'error_absoluto')[['NOMBRE_INSTITUCION', target, 'prediccion', 'error_absoluto']].to_string(index=False))

    print('\n=== Mejora completada ===')


if __name__ == '__main__':
    main()
