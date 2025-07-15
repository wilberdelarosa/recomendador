#!/usr/bin/env python3
"""
Sistema de recomendacion de libros usando un arbol de decision.
Simula un dataset, entrena el modelo, recomienda un libro segun preferencias
y visualiza el arbol (o lo guarda en un PNG).
"""

import argparse
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt


def crear_dataset() -> pd.DataFrame:
    data = [
        {'genero': 'Ficcion', 'autor_favorito': 'AutorA', 'nivel_lectura': 'Principiante', 'libro_recomendado': 'Libro1'},
        {'genero': 'No Ficcion', 'autor_favorito': 'AutorB', 'nivel_lectura': 'Intermedio',    'libro_recomendado': 'Libro2'},
        {'genero': 'Ciencia Ficcion', 'autor_favorito': 'AutorC', 'nivel_lectura': 'Avanzado',     'libro_recomendado': 'Libro3'},
        {'genero': 'Fantasia', 'autor_favorito': 'AutorA', 'nivel_lectura': 'Intermedio',        'libro_recomendado': 'Libro4'},
        {'genero': 'Misterio', 'autor_favorito': 'AutorB', 'nivel_lectura': 'Principiante',      'libro_recomendado': 'Libro5'},
        {'genero': 'Ciencia Ficcion', 'autor_favorito': 'AutorA', 'nivel_lectura': 'Avanzado',     'libro_recomendado': 'Libro6'},
    ]
    return pd.DataFrame(data)


def codificar_caracteristicas(df: pd.DataFrame):
    X = pd.get_dummies(df[['genero', 'autor_favorito', 'nivel_lectura']])
    y = df['libro_recomendado']
    return X, y


def entrenar_arbol(X: pd.DataFrame, y: pd.Series) -> DecisionTreeClassifier:
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    return model


def recomendar_libro(model: DecisionTreeClassifier, cols, genero, autor, nivel) -> str:
    nuevo = {'genero': genero, 'autor_favorito': autor, 'nivel_lectura': nivel}
    new_df = pd.DataFrame([nuevo])
    new_X = pd.get_dummies(new_df).reindex(columns=cols, fill_value=0)
    return model.predict(new_X)[0]


def visualizar_arbol(model: DecisionTreeClassifier, feature_names, class_names, salida):
    plt.figure(figsize=(12, 8))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title('Arbol de decision - Recomendador de libros')
    plt.tight_layout()
    if salida:
        plt.savefig(salida, dpi=300)
        print(f'Arbol guardado en: {salida}')
    else:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description='Recomendador de libros con arbol de decision')
    parser.add_argument('--genero', type=str, default='Ciencia Ficcion', help='Genero preferido')
    parser.add_argument('--autor',  type=str, default='AutorB',           help='Autor favorito')
    parser.add_argument('--nivel',  type=str, default='Intermedio',       help='Nivel de lectura')
    parser.add_argument('--salida', type=str, default='',                 help='Archivo PNG para guardar el arbol (opcional)')
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Crear y mostrar dataset
    df = crear_dataset()
    print('\nDataset de usuarios y preferencias:\n')
    print(df.to_string(index=False))

    # 2. Preprocesar y entrenar
    X, y = codificar_caracteristicas(df)
    model = entrenar_arbol(X, y)

    # 3. Recomendar segun argumentos
    reco = recomendar_libro(model, X.columns, args.genero, args.autor, args.nivel)
    print(f'\nRecomendacion para (genero={args.genero}, autor={args.autor}, nivel={args.nivel}): {reco}\n')

    # 4. Visualizar o guardar arbol
    visualizar_arbol(model, X.columns.tolist(), model.classes_.tolist(), args.salida)


if __name__ == '__main__':
    main()
