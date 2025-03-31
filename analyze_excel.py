import pandas as pd

# Ler o arquivo Excel
try:
    df = pd.read_excel('TR_Verif_Acomp_Cariacica.xlsx')
    
    print('Informações do DataFrame:')
    print(f'Número de linhas: {len(df)}')
    print(f'Número de colunas: {len(df.columns)}')
    
    print('\nColunas disponíveis:')
    for col in df.columns:
        print(f'- {col}')
    
    # Verificar especificamente as colunas mencionadas
    colunas_interesse = ['acompanhamento_descricao', 'articulador_responsavel']
    print('\nVerificando colunas de interesse:')
    for col in colunas_interesse:
        if col in df.columns:
            print(f'A coluna "{col}" existe no DataFrame')
            # Mostrar alguns exemplos desta coluna
            print(f'Exemplos da coluna "{col}" (primeiras 3 entradas não nulas):')
            exemplos = df[df[col].notna()][col].head(3)
            for idx, exemplo in enumerate(exemplos):
                print(f'{idx+1}: {exemplo}')
        else:
            print(f'A coluna "{col}" NÃO existe no DataFrame')
    
    # Estatísticas sobre as colunas de interesse
    print('\nEstatísticas das colunas de interesse:')
    for col in colunas_interesse:
        if col in df.columns:
            # Contagem de valores nulos
            nulos = df[col].isna().sum()
            print(f'Coluna "{col}":')
            print(f'- Valores nulos: {nulos} ({(nulos/len(df))*100:.2f}%)')
            # Se for uma coluna de texto
            if df[col].dtype == 'object':
                # Comprimento médio do texto
                comprimentos = df[col].dropna().apply(len)
                print(f'- Comprimento médio do texto: {comprimentos.mean():.2f} caracteres')
                print(f'- Comprimento mínimo: {comprimentos.min()} caracteres')
                print(f'- Comprimento máximo: {comprimentos.max()} caracteres')
            
            # Se for articulador_responsavel, contar valores únicos
            if col == 'articulador_responsavel':
                valores_unicos = df[col].nunique()
                print(f'- Número de articuladores únicos: {valores_unicos}')
                print('- Top 5 articuladores mais frequentes:')
                print(df[col].value_counts().head(5))

except Exception as e:
    print(f'Erro ao analisar o arquivo: {str(e)}') 