import pandas as pd
import os

# Função para escrever no arquivo
def write_to_file(file, text):
    file.write(text + '\n')

# Ler o arquivo Excel
try:
    df = pd.read_excel('TR_Verif_Acomp_Cariacica.xlsx')
    
    # Criar arquivo de saída
    with open('analise_planilha.txt', 'w', encoding='utf-8') as f:
        write_to_file(f, 'ANÁLISE DA PLANILHA DE ACOMPANHAMENTOS')
        write_to_file(f, '=' * 50)
        
        write_to_file(f, '\nInformações do DataFrame:')
        write_to_file(f, f'Número de linhas: {len(df)}')
        write_to_file(f, f'Número de colunas: {len(df.columns)}')
        
        write_to_file(f, '\nColunas disponíveis:')
        for col in df.columns:
            write_to_file(f, f'- {col}')
        
        # Verificar especificamente as colunas mencionadas
        colunas_interesse = ['acompanhamento_descricao', 'articulador_responsavel']
        write_to_file(f, '\nVerificando colunas de interesse:')
        for col in colunas_interesse:
            if col in df.columns:
                write_to_file(f, f'A coluna "{col}" existe no DataFrame')
                # Mostrar alguns exemplos desta coluna
                write_to_file(f, f'Exemplos da coluna "{col}" (primeiras 3 entradas não nulas):')
                exemplos = df[df[col].notna()][col].head(3)
                for idx, exemplo in enumerate(exemplos):
                    write_to_file(f, f'{idx+1}: {exemplo}')
            else:
                write_to_file(f, f'A coluna "{col}" NÃO existe no DataFrame')
        
        # Estatísticas sobre as colunas de interesse
        write_to_file(f, '\nEstatísticas das colunas de interesse:')
        for col in colunas_interesse:
            if col in df.columns:
                # Contagem de valores nulos
                nulos = df[col].isna().sum()
                write_to_file(f, f'Coluna "{col}":')
                write_to_file(f, f'- Valores nulos: {nulos} ({(nulos/len(df))*100:.2f}%)')
                # Se for uma coluna de texto
                if df[col].dtype == 'object':
                    # Comprimento médio do texto
                    comprimentos = df[col].dropna().apply(len)
                    write_to_file(f, f'- Comprimento médio do texto: {comprimentos.mean():.2f} caracteres')
                    write_to_file(f, f'- Comprimento mínimo: {comprimentos.min()} caracteres')
                    write_to_file(f, f'- Comprimento máximo: {comprimentos.max()} caracteres')
                
                # Se for articulador_responsavel, contar valores únicos
                if col == 'articulador_responsavel':
                    valores_unicos = df[col].nunique()
                    write_to_file(f, f'- Número de articuladores únicos: {valores_unicos}')
                    write_to_file(f, '- Lista de articuladores e sua frequência:')
                    for nome, contagem in df[col].value_counts().items():
                        write_to_file(f, f'  * {nome}: {contagem} registros ({(contagem/len(df))*100:.2f}%)')
                
                # Análise de texto adicional para acompanhamento_descricao
                if col == 'acompanhamento_descricao':
                    # Verificar registros com descrições potencialmente problemáticas
                    descricoes_curtas = df[df[col].notna() & (df[col].str.len() < 20)]
                    write_to_file(f, f'- Número de descrições muito curtas (<20 caracteres): {len(descricoes_curtas)} ({(len(descricoes_curtas)/len(df))*100:.2f}%)')
                    write_to_file(f, '- Exemplos de descrições muito curtas:')
                    for idx, exemplo in enumerate(descricoes_curtas[col].head(5)):
                        write_to_file(f, f'  * {idx+1}: "{exemplo}"')
                    
                    # Verificar descrições vagas (contendo palavras que sugerem falta de detalhes)
                    palavras_vagas = ['etc', 'outros', 'algumas', 'alguma coisa', 'algo', 'coisas', 'demandas', 'entre outros']
                    descricoes_vagas = df[df[col].notna() & df[col].str.contains('|'.join(palavras_vagas), case=False)]
                    write_to_file(f, f'- Número de descrições potencialmente vagas: {len(descricoes_vagas)} ({(len(descricoes_vagas)/len(df))*100:.2f}%)')
                    write_to_file(f, '- Exemplos de descrições potencialmente vagas:')
                    for idx, exemplo in enumerate(descricoes_vagas[col].head(5)):
                        write_to_file(f, f'  * {idx+1}: "{exemplo}"')
                    
        # Relação entre colunas
        write_to_file(f, '\nRelação entre articuladores e qualidade das descrições:')
        # Criar um DataFrame com estatísticas por articulador
        estatisticas_por_articulador = df.groupby('articulador_responsavel').agg({
            'acompanhamento_descricao': [
                ('total', 'count'),
                ('média_comprimento', lambda x: x.dropna().str.len().mean()),
                ('min_comprimento', lambda x: x.dropna().str.len().min()),
                ('max_comprimento', lambda x: x.dropna().str.len().max()),
                ('descricoes_curtas', lambda x: (x.dropna().str.len() < 20).sum())
            ]
        })
        
        # Calcular percentual de descrições curtas
        estatisticas_por_articulador['acompanhamento_descricao', 'pct_descricoes_curtas'] = (
            estatisticas_por_articulador['acompanhamento_descricao', 'descricoes_curtas'] / 
            estatisticas_por_articulador['acompanhamento_descricao', 'total'] * 100
        )
        
        # Ordenar por percentual de descrições curtas (decrescente)
        estatisticas_ordenadas = estatisticas_por_articulador.sort_values(
            ('acompanhamento_descricao', 'pct_descricoes_curtas'), 
            ascending=False
        )
        
        # Escrever no arquivo
        for articulador, row in estatisticas_ordenadas.iterrows():
            write_to_file(f, f'Articulador: {articulador}')
            write_to_file(f, f'  - Total de registros: {row[("acompanhamento_descricao", "total")]}')
            write_to_file(f, f'  - Comprimento médio das descrições: {row[("acompanhamento_descricao", "média_comprimento")]:.2f} caracteres')
            write_to_file(f, f'  - Comprimento mínimo: {row[("acompanhamento_descricao", "min_comprimento")]} caracteres')
            write_to_file(f, f'  - Comprimento máximo: {row[("acompanhamento_descricao", "max_comprimento")]} caracteres')
            write_to_file(f, f'  - Descrições curtas: {row[("acompanhamento_descricao", "descricoes_curtas")]} ({row[("acompanhamento_descricao", "pct_descricoes_curtas")]:.2f}%)')
        
    print(f'Análise concluída e salva no arquivo "analise_planilha.txt"')

except Exception as e:
    print(f'Erro ao analisar o arquivo: {str(e)}') 