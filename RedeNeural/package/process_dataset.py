from sklearn.preprocessing import StandardScaler
import joblib

import os

import pandas as pd
import numpy as np


def cap_outliers(df, column):
    
    upper = df[column].mean() + 3*df[column].std()
    down = df[column].mean() - 3*df[column].std()

    df[(df[column] > upper) | (df[column] < down)]

    df[column] = np.where(
        df[column]>upper,
        upper,
        np.where(
            df[column]<down,
            down,
            df[column]
        )
    )
    
    return df

def generate_labels(df, target = 'id_fechou'):
    """Generate list with training columns.

    Args:
        df (pd.DataFrame): DataFrame, all columns except the target will be returned
        target (str, optional): Traget column. Defaults to 'id_fechou'.

    Returns:
        tuple(list, str): list with training columns and target
    """    
    train_labels = list(df.columns)
    train_labels.remove(target)
    return train_labels, target


def processColumns(
                df, 
                remove_columns = ['Data_de_criacao', 'ano', 'ID_cliente', 'Codigo_da_oportunidade', 'Gestão da Segurança Pública', 
                                    'S_amp_OP_S_amp_OE', 'Transformação Digital', 'Roadmap', 'Comissão sobre Parceiros', 'Cybersecurity', 'Gestão da Saúde', 'Treinamentos',
                                    'Equilíbrio fiscal', 'Concorrentes', 'Gestão da Receita', 'Gestão da Educação', 'Gestão da Segurança Viária', 'ESG',
                                    'Gestão de operações projetizadas', 'Software', 'Gestão Estratégica', 'Skill_dev', 'Gestão de pessoas',
                                    'Gestão de Gastos', 'n_solucoes'], 
                columns_w_outliers=['Valor_corrigido2', 'Custo_Total', 'Custo_Total_per_Valor_corrigido2'],
                create_columns=True):
    
    """Remove selected columns, and add columns if wanted.

    Args:
        df (pd.DataFrame): target dataframe
        remove_columns (list, optional): Columns to remove. Defaults to ['Data_de_criacao', 'ano', 'ID_cliente', 'Codigo_da_oportunidade', 'Gestão da Segurança Pública', 'S_amp_OP_S_amp_OE', 'Transformação Digital', 'Roadmap'].
        create_columns (bool, optional): add columns if wanted. Defaults to True.

    Returns:
        pd.DataFrame: modified DataFrame
    """                 

    if create_columns:
        df["Custo_Total_per_Valor_corrigido2"] = df["Custo_Total"]/df["Valor_corrigido2"]
        df["numero_relacionamentos_convertidos_per_numero_relacionamentos"] = df["numero_relacionamentos_convertidos"]/df["numero_relacionamentos"]
        df["Gestão da Receita_per_Gestão de Gastos"] = df["Gestão da Receita"] + df["Gestão de Gastos"]

    columns_to_sum = ['Software', 'Comissão sobre Parceiros', 'Cybersecurity',
       'Desdobramento de metas', 'ESG', 'Equilíbrio fiscal', 'Skill_dev',
       'Gestão Estratégica', 'Gestão da Educação', 'Gestão da Operação',
       'Gestão da Receita', 'Gestão da Saúde', 'Gestão da Segurança Pública',
       'Gestão da Segurança Viária', 'Gestão de Gastos',
       'Gestão de operações projetizadas', 'Gestão de pessoas',
       'Processes Excellence', 'Produtos digitais', 'S_amp_OP_S_amp_OE',
       'Transformação Digital', 'Treinamentos', 'Roadmap']
    
    s = pd.Series(np.zeros(len(df)))
    for col in columns_to_sum:
        s += df[col]
    df["num_solucoes"] = s


    for col in columns_w_outliers:
        df = cap_outliers(df, col)

    df = df.loc[:,~df.columns.str.startswith('Unname')]
    
    for col in remove_columns: 
        try: df = df.drop(remove_columns, axis=1)
        except: pass

    return df


def scaleData(df, save=True, useSaved=True):
    """Scale data range using StandardScaler. This functions also saves the scaler so it may be used later.

    Args:
        df (pd.DataFrame): input dataframe to scale
        save (bool, optional): save model. Defaults to True.
        useSaved (bool, optional): load model. Defaults to True.

    Returns:
        np.array: scaled array.
    """    

    if useSaved:
        print("Using saved scaler.")
        try:
            scaler = joblib.load(os.path.join('package/models', 'scaler.model'))
        except: 
            print("Scaler not found, creating a new one.")
            useSaved = False

    if not useSaved:
        scaler = StandardScaler()
        scaler.fit(df)

    if save:
        joblib.dump(scaler, os.path.join('package/models', 'scaler.model'))

    return scaler.transform(df)