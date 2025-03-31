import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Carregar os dados
df = pd.read_excel('TR_Verif_Acomp_Cariacica.xlsx')

# Verificar as colunas de encaminhamento
colunas_encaminhamento = [
    'dado_algum_encaminhamento',
    'instituicao_encaminhamento_educacao',
    'instituicao_encaminhamento_saude',
    'instituicao_encaminhamento_assistencia_social',
    'instituicao_encaminhamento_conselho_tutelar',
    'instituicao_encaminhamento_estacao_conhecimento',
    'instituicao_encaminhamento_sociedade_civil',
    'instituicao_encaminhamento_outro_equipamento'
]

# Verificar se todas as colunas existem no dataset
for coluna in colunas_encaminhamento:
    if coluna not in df.columns:
        print(f"Coluna '{coluna}' não encontrada!")
    else:
        print(f"Coluna '{coluna}' encontrada.")

# Análise da coluna principal de encaminhamento
print("\nAnálise da coluna 'dado_algum_encaminhamento':")
print(df['dado_algum_encaminhamento'].value_counts())
print(f"Valores nulos: {df['dado_algum_encaminhamento'].isna().sum()}")

# Para cada coluna de instituição, verificar os valores únicos e distribuição
for coluna in colunas_encaminhamento[1:]:  # Pular a primeira coluna (dado_algum_encaminhamento)
    print(f"\nAnálise da coluna '{coluna}':")
    
    # Verificar tipos de valores
    print(f"Tipo de dados: {df[coluna].dtype}")
    
    # Verificar valores nulos
    nulos = df[coluna].isna().sum()
    pct_nulos = nulos / len(df) * 100
    print(f"Valores nulos: {nulos} ({pct_nulos:.2f}%)")
    
    # Se a coluna não for inteiramente nula, mostrar os valores mais comuns
    if df[coluna].notna().any():
        # Para colunas de texto, mostrar os valores mais comuns
        if df[coluna].dtype == 'object':
            valores_comuns = df[coluna].value_counts().head(10)
            print(f"Valores mais comuns:")
            print(valores_comuns)
        # Para colunas numéricas, mostrar estatísticas básicas
        else:
            print(f"Estatísticas básicas:")
            print(df[coluna].describe())
    else:
        print("Coluna inteiramente nula ou vazia.")

# Verificar a correlação entre 'dado_algum_encaminhamento' e as colunas de instituições
print("\nCorrelação entre 'dado_algum_encaminhamento' e presença de valores nas colunas de instituições:")

# Converter 'dado_algum_encaminhamento' para valores numéricos (1 para Sim, 0 para Não)
if df['dado_algum_encaminhamento'].dtype == 'object':
    df['dado_algum_encaminhamento_num'] = df['dado_algum_encaminhamento'].map({'Sim': 1, 'Não': 0})
else:
    df['dado_algum_encaminhamento_num'] = df['dado_algum_encaminhamento']

# Para cada coluna de instituição, verificar se há valores quando dado_algum_encaminhamento é Sim/Não
for coluna in colunas_encaminhamento[1:]:
    # Criar uma coluna temporária que indica se a coluna de instituição tem valor
    df[f'{coluna}_tem_valor'] = df[coluna].notna().astype(int)
    
    # Calcular a correlação
    corr = df['dado_algum_encaminhamento_num'].corr(df[f'{coluna}_tem_valor'])
    
    print(f"Correlação com '{coluna}': {corr:.4f}")
    
    # Verificar consistência: quando 'dado_algum_encaminhamento' é Não, não deveria haver valores nas colunas de instituição
    inconsistencias = len(df[(df['dado_algum_encaminhamento'] == 'Não') & (df[coluna].notna())])
    if inconsistencias > 0:
        print(f"  ATENÇÃO: {inconsistencias} registros têm 'dado_algum_encaminhamento' = Não, mas têm valores em '{coluna}'")
    
    # Verificar consistência inversa: quando 'dado_algum_encaminhamento' é Sim, deveria haver pelo menos um valor nas colunas de instituição
    inconsistencias_inversas = len(df[(df['dado_algum_encaminhamento'] == 'Sim') & 
                                     (df[colunas_encaminhamento[1:]].isna().all(axis=1))])
    
    if inconsistencias_inversas > 0:
        print(f"  ATENÇÃO: {inconsistencias_inversas} registros têm 'dado_algum_encaminhamento' = Sim, mas não têm valores em nenhuma coluna de instituição")

# Análise por articulador
print("\nDistribuição de encaminhamentos por articulador:")
encaminhamentos_por_articulador = df.groupby('acompanhamento_articulador')['dado_algum_encaminhamento'].value_counts().unstack().fillna(0)

# Calcular percentual de encaminhamentos por articulador
encaminhamentos_por_articulador['total'] = encaminhamentos_por_articulador.sum(axis=1)
if 'Sim' in encaminhamentos_por_articulador.columns:
    encaminhamentos_por_articulador['pct_sim'] = encaminhamentos_por_articulador['Sim'] / encaminhamentos_por_articulador['total'] * 100
    print(encaminhamentos_por_articulador.sort_values('pct_sim', ascending=False))
else:
    print("Não foi possível calcular o percentual de encaminhamentos (coluna 'Sim' não encontrada).")

# Análise dos tipos de instituições mais encaminhadas
print("\nTipos de instituições mais encaminhadas:")
total_encaminhamentos = {}

for coluna in colunas_encaminhamento[1:]:
    # Contar registros não nulos para cada coluna de instituição
    encaminhamentos = df[coluna].notna().sum()
    tipo_instituicao = coluna.replace('instituicao_encaminhamento_', '')
    total_encaminhamentos[tipo_instituicao] = encaminhamentos

# Ordenar e mostrar os resultados
total_encaminhamentos_sorted = dict(sorted(total_encaminhamentos.items(), key=lambda x: x[1], reverse=True))
for instituicao, total in total_encaminhamentos_sorted.items():
    print(f"{instituicao}: {total}")

print("\nAnálise de encaminhamentos concluída!") 