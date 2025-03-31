import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

# Configurar NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Carregar a planilha
df = pd.read_excel('TR_Verif_Acomp_Cariacica.xlsx')

# Verificar a coluna de sucesso de contato
print("Valores únicos na coluna 'acompanhamento_sucesso_contato':")
print(df['acompanhamento_sucesso_contato'].value_counts())
print("\n")

# Palavras e expressões que indicam sucesso
palavras_sucesso = [
    'com sucesso', 'realizado com sucesso', 'atendido', 'conseguiu', 'conseguimos',
    'contato realizado', 'efetuado com sucesso', 'foi atendido', 'foi realizado',
    'estabelecido contato', 'respondeu', 'responderam', 'atendeu', 'atenderam'
]

# Palavras e expressões que indicam insucesso
palavras_insucesso = [
    'não atendeu', 'não respondeu', 'não foi possível', 'sem sucesso', 'fracassou',
    'não conseguimos', 'não conseguiu', 'tentativa sem sucesso', 'não estabelecido',
    'não foi atendido', 'não foi encontrado', 'não foi localizado', 'caixa postal'
]

# Função para verificar indicação de sucesso/insucesso na descrição
def verificar_indicacao(texto, palavras_lista):
    if pd.isna(texto) or not isinstance(texto, str):
        return False
    
    texto_lower = texto.lower()
    for palavra in palavras_lista:
        if palavra in texto_lower:
            return True
    return False

# Adicionar colunas de indicação baseadas na descrição
df['indicacao_sucesso_na_descricao'] = df['acompanhamento_descricao'].apply(
    lambda x: verificar_indicacao(x, palavras_sucesso))
    
df['indicacao_insucesso_na_descricao'] = df['acompanhamento_descricao'].apply(
    lambda x: verificar_indicacao(x, palavras_insucesso))

# Verificar consistência entre coluna de sucesso e indicação na descrição
df['consistencia'] = 'Indefinido'

# Casos onde a coluna indica sucesso ("Sim") e a descrição também indica sucesso
df.loc[(df['acompanhamento_sucesso_contato'] == 'Sim') & 
       (df['indicacao_sucesso_na_descricao'] == True) &
       (df['indicacao_insucesso_na_descricao'] == False), 'consistencia'] = 'Consistente (Sucesso)'

# Casos onde a coluna indica insucesso ("Não") e a descrição também indica insucesso
df.loc[(df['acompanhamento_sucesso_contato'] == 'Não') & 
       (df['indicacao_insucesso_na_descricao'] == True) &
       (df['indicacao_sucesso_na_descricao'] == False), 'consistencia'] = 'Consistente (Insucesso)'

# Casos de possível inconsistência - coluna diz "Sim" mas descrição sugere insucesso
df.loc[(df['acompanhamento_sucesso_contato'] == 'Sim') & 
       (df['indicacao_insucesso_na_descricao'] == True), 'consistencia'] = 'Possível Inconsistência (Sim/Insucesso)'

# Casos de possível inconsistência - coluna diz "Não" mas descrição sugere sucesso
df.loc[(df['acompanhamento_sucesso_contato'] == 'Não') & 
       (df['indicacao_sucesso_na_descricao'] == True), 'consistencia'] = 'Possível Inconsistência (Não/Sucesso)'

# Casos sem clara indicação na descrição
df.loc[(df['indicacao_sucesso_na_descricao'] == False) & 
       (df['indicacao_insucesso_na_descricao'] == False), 'consistencia'] = 'Sem indicação clara na descrição'

# Casos com indicações contraditórias
df.loc[(df['indicacao_sucesso_na_descricao'] == True) & 
       (df['indicacao_insucesso_na_descricao'] == True), 'consistencia'] = 'Indicações contraditórias na descrição'

# Estatísticas gerais
print("Estatísticas de consistência:")
print(df['consistencia'].value_counts())
print("\n")

# Análise por articulador
consistencia_por_articulador = df.groupby(['acompanhamento_articulador', 'consistencia']).size().unstack(fill_value=0)
print("Consistência por articulador (top 10 com mais inconsistências):")
inconsistencias_total = consistencia_por_articulador['Possível Inconsistência (Sim/Insucesso)'] + consistencia_por_articulador['Possível Inconsistência (Não/Sucesso)']
print(inconsistencias_total.sort_values(ascending=False).head(10))
print("\n")

# Exemplos de possíveis inconsistências para análise
print("Exemplos de possíveis inconsistências (Coluna diz 'Sim' mas descrição sugere insucesso):")
inconsistencias_sim = df[df['consistencia'] == 'Possível Inconsistência (Sim/Insucesso)'].head(5)
for idx, row in inconsistencias_sim.iterrows():
    print(f"ID: {row['id']}")
    print(f"Articulador: {row['acompanhamento_articulador']}")
    print(f"Descrição: {row['acompanhamento_descricao'][:200]}...")
    print("-" * 80)

print("\nExemplos de possíveis inconsistências (Coluna diz 'Não' mas descrição sugere sucesso):")
inconsistencias_nao = df[df['consistencia'] == 'Possível Inconsistência (Não/Sucesso)'].head(5)
for idx, row in inconsistencias_nao.iterrows():
    print(f"ID: {row['id']}")
    print(f"Articulador: {row['acompanhamento_articulador']}")
    print(f"Descrição: {row['acompanhamento_descricao'][:200]}...")
    print("-" * 80)

# Salvar resultados
df_inconsistencias = df[df['consistencia'].str.contains('Inconsistência')]
df_inconsistencias.to_excel('inconsistencias_sucesso_contato.xlsx', index=False)

print(f"Total de registros analisados: {len(df)}")
print(f"Total de possíveis inconsistências identificadas: {len(df_inconsistencias)} ({len(df_inconsistencias)/len(df)*100:.2f}%)")
print("Resultados detalhados salvos em 'inconsistencias_sucesso_contato.xlsx'") 