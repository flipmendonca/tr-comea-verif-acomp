import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import re
import string
from wordcloud import WordCloud
import warnings
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity
# Importar AgGrid para tabelas interativas
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
import os
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="TR | COMEA - Análise de Acompanhamentos",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Baixar recursos necessários do NLTK
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Função para pré-processamento de texto
def preprocessar_texto(texto):
    if pd.isna(texto):
        return ""
    # Converter para minúsculas
    texto = texto.lower()
    # Remover pontuação
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    # Remover números
    texto = re.sub(r'\d+', '', texto)
    # Remover stopwords
    stop_words = set(stopwords.words('portuguese'))
    tokens = word_tokenize(texto)
    tokens = [word for word in tokens if word not in stop_words]
    # Reunir os tokens
    texto_processado = ' '.join(tokens)
    return texto_processado

# Função para detectar problemas nas descrições
def detectar_problemas(texto, palavras_vagas=None):
    if pd.isna(texto):
        return ['texto_nulo']
    
    if not isinstance(texto, str):
        return ['texto_invalido']
    
    problemas = []
    
    # Verificar descrições muito curtas
    if len(texto) < 20:
        problemas.append('muito_curto')
    
    if palavras_vagas is None:
        palavras_vagas = ['etc', 'outros', 'algumas', 'alguma coisa', 'algo', 'coisas', 'demandas', 'entre outros']
    
    # Verificar termos vagos
    for palavra in palavras_vagas:
        if palavra in texto.lower():
            problemas.append('vago')
            break
    
    # Verificar falta de detalhes específicos
    if len(texto.split()) < 10:
        problemas.append('sem_detalhes')
    
    # Verificar descrições genéricas
    frases_genericas = [
        'atendimento realizado', 
        'demanda atendida', 
        'feito contato', 
        'realizado acompanhamento',
        'foi feito'
    ]
    
    for frase in frases_genericas:
        if frase in texto.lower() and len(texto) < 50:
            problemas.append('generico')
            break
    
    return problemas

# Função para classificar a qualidade da descrição
def classificar_qualidade(texto):
    problemas = detectar_problemas(texto)
    
    if 'texto_nulo' in problemas or 'texto_invalido' in problemas:
        return 'Crítico', 0
    
    if 'muito_curto' in problemas:
        return 'Crítico', 1
    
    if 'generico' in problemas and 'sem_detalhes' in problemas:
        return 'Ruim', 2
    
    if 'vago' in problemas:
        return 'Regular', 3
    
    if len(problemas) > 0:
        return 'Regular', 4
    
    if len(texto) < 100:
        return 'Bom', 7
    
    if len(texto) < 200:
        return 'Bom', 8
    
    return 'Excelente', 10

# Função para analisar consistência do sucesso de contato
def verificar_indicacao(texto, palavras_lista, palavras_excecao=None):
    if pd.isna(texto) or not isinstance(texto, str):
        return False
    
    texto_lower = texto.lower()
    
    # Verificar exceções primeiro
    if palavras_excecao:
        for excecao in palavras_excecao:
            if excecao in texto_lower:
                return False
    
    for palavra in palavras_lista:
        if palavra in texto_lower:
            return True
    return False

# Função para classificar a consistência entre o campo de sucesso e a descrição
def classificar_consistencia(row):
    if pd.isna(row['acompanhamento_sucesso_contato']):
        return 'Indefinido'
    
    # Casos onde há indicações contraditórias na descrição
    if row['indicacao_sucesso_na_descricao'] and row['indicacao_insucesso_na_descricao']:
        # Verificar contexto adicional
        if row['contexto_objetivo_nao_atingido']:
            # Se há indicação de que o objetivo não foi atingido, mas o contato foi feito
            if row['acompanhamento_sucesso_contato'] == 'Sim':
                return 'Consistente (Sucesso)'
            else:
                return 'Possível Inconsistência (Não/Sucesso)'
        else:
            return 'Indicações contraditórias na descrição'
    
    # Casos onde a coluna indica sucesso ("Sim") e a descrição também indica sucesso
    if row['acompanhamento_sucesso_contato'] == 'Sim' and row['indicacao_sucesso_na_descricao'] and not row['indicacao_insucesso_na_descricao']:
        return 'Consistente (Sucesso)'
    
    # Casos onde a coluna indica insucesso ("Não") e a descrição também indica insucesso
    if row['acompanhamento_sucesso_contato'] == 'Não' and row['indicacao_insucesso_na_descricao'] and not row['indicacao_sucesso_na_descricao']:
        return 'Consistente (Insucesso)'
    
    # Casos onde o contato foi feito, mas o objetivo não foi atingido
    if row['acompanhamento_sucesso_contato'] == 'Sim' and row['indicacao_insucesso_na_descricao'] and row['contexto_objetivo_nao_atingido']:
        return 'Consistente (Sucesso)'
    
    # Casos de possível inconsistência - coluna diz "Sim" mas descrição sugere insucesso no contato
    if row['acompanhamento_sucesso_contato'] == 'Sim' and row['indicacao_insucesso_na_descricao'] and not row['contexto_objetivo_nao_atingido']:
        return 'Possível Inconsistência (Sim/Insucesso)'
    
    # Casos de possível inconsistência - coluna diz "Não" mas descrição sugere sucesso
    if row['acompanhamento_sucesso_contato'] == 'Não' and row['indicacao_sucesso_na_descricao']:
        return 'Possível Inconsistência (Não/Sucesso)'
    
    # Casos sem clara indicação na descrição
    return 'Sem indicação clara na descrição'

# Função para carregar os dados
@st.cache_data
def carregar_dados(uploaded_file=None):
    # Verificar se um arquivo foi carregado pelo usuário
    if uploaded_file is not None:
        # Carregar dados do arquivo enviado pelo usuário
        df = pd.read_excel(uploaded_file)
    else:
        try:
            # Tentar carregar dados do arquivo local
            arquivo_padrao = 'TR_Verif_Acomp_Cariacica.xlsx'
            if os.path.exists(arquivo_padrao):
                df = pd.read_excel(arquivo_padrao)
            else:
                # Se não encontrar o arquivo, retornar None
                st.error(f"Arquivo {arquivo_padrao} não encontrado. Por favor, faça o upload do arquivo.")
                return None
        except Exception as e:
            st.error(f"Erro ao carregar dados: {str(e)}")
            return None
    
    # Garantir que as colunas de interesse existam
    colunas_interesse = [
        'acompanhamento_descricao', 
        'acompanhamento_articulador', 
        'acompanhamento_data', 
        'acompanhamento_sucesso_contato',
        'dado_algum_encaminhamento',
        'instituicao_encaminhamento_educacao',
        'instituicao_encaminhamento_saude',
        'instituicao_encaminhamento_assistencia_social',
        'instituicao_encaminhamento_conselho_tutelar',
        'instituicao_encaminhamento_estacao_conhecimento',
        'instituicao_encaminhamento_sociedade_civil',
        'instituicao_encaminhamento_outro_equipamento'
    ]
    
    # Verificar colunas obrigatórias
    colunas_obrigatorias = ['acompanhamento_descricao', 'acompanhamento_articulador', 'acompanhamento_data', 'acompanhamento_sucesso_contato']
    for coluna in colunas_obrigatorias:
        if coluna not in df.columns:
            st.error(f"Coluna '{coluna}' não encontrada no arquivo!")
            st.stop()
    
    # Verificar colunas de encaminhamento e criar se não existirem (para evitar erros)
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
    
    for coluna in colunas_encaminhamento:
        if coluna not in df.columns:
            st.warning(f"Coluna '{coluna}' não encontrada! Será criada uma coluna vazia.")
            df[coluna] = None
    
    # Adicionar coluna de texto pré-processado
    df['texto_preprocessado'] = df['acompanhamento_descricao'].apply(preprocessar_texto)
    
    # Identificar problemas nas descrições
    df['problemas_detectados'] = df['acompanhamento_descricao'].apply(detectar_problemas)
    
    # Quantificar o número de problemas
    df['num_problemas'] = df['problemas_detectados'].apply(len)
    
    # Classificar a qualidade dos registros
    df['classificacao_qualidade'], df['pontuacao'] = zip(*df['acompanhamento_descricao'].apply(
        lambda x: classificar_qualidade(x)))
    
    # Quantificar o comprimento das descrições
    df['comprimento_descricao'] = df['acompanhamento_descricao'].apply(
        lambda x: len(x) if isinstance(x, str) else 0)
    
    # Garantir que a data está no formato correto
    df['acompanhamento_data'] = pd.to_datetime(df['acompanhamento_data'], errors='coerce')
    
    # Filtrar datas futuras ou claramente incorretas (como novembro 2025)
    data_atual = datetime.now()
    df = df[df['acompanhamento_data'] <= data_atual]
    
    # Adicionar aviso se foram removidas datas futuras
    if (df['acompanhamento_data'] > data_atual).any():
        st.warning(f"Foram removidos registros com datas futuras.")
    
    # Adicionar colunas para análise temporal
    df['ano'] = df['acompanhamento_data'].dt.year
    df['mes_num'] = df['acompanhamento_data'].dt.month  # Para ordenação
    df['mes'] = df['acompanhamento_data'].dt.month_name()
    # Formatar semana como 'YYYY-MM-WW' para garantir ordenação correta
    df['semana'] = df['acompanhamento_data'].dt.strftime('%Y-%m-W%U')
    df['dia_semana'] = df['acompanhamento_data'].dt.day_name()
    # Formatar mes_ano para garantir ordenação correta: 'YYYY-MM-MesNome'
    df['mes_ano'] = df['acompanhamento_data'].dt.strftime('%Y-%m-%b')
    df['mes_ano_display'] = df['acompanhamento_data'].dt.strftime('%b %Y')
    
    # Adicionar análise de consistência entre sucesso de contato e descrição
    
    # Palavras e expressões que indicam sucesso de contato
    palavras_sucesso = [
        'com sucesso', 'realizado com sucesso', 'atendido', 'conseguiu', 'conseguimos',
        'contato realizado', 'efetuado com sucesso', 'foi atendido', 'foi realizado',
        'estabelecido contato', 'respondeu', 'responderam', 'atendeu', 'atenderam',
        'em contato com', 'de acordo com', 'segundo', 'conforme', 'informou que',
        'relatou que', 'disse que', 'comunicado que', 'contato feito', 'foi comunicado',
        'entramos em contato', 'entramos em contato com', 'recebi contato', 'recebi retorno',
        'recebemos contato', 'recebemos retorno', 'informou-se', 'foi informado'
    ]
    
    # Palavras e expressões que indicam insucesso no contato
    palavras_insucesso = [
        'não atendeu', 'não respondeu', 'sem sucesso', 'fracassou',
        'ninguém atendeu', 'não conseguimos contato', 'não obtive retorno',
        'não obtivemos retorno', 'não conseguimos contatar', 'não atende',
        'contato sem sucesso', 'tentativa sem sucesso', 'não foi atendido',
        'não foi possível contatar', 'não foi localizado', 'caixa postal',
        'chamou até cair', 'telefone desligado', 'fora de área', 'tentei contato',
        'número inexistente', 'desligou a ligação', 'não foi possível falar',
        'tentamos contato'
    ]
    
    # Palavras que indicam que o objetivo não foi atingido (mesmo que o contato tenha acontecido)
    palavras_objetivo_nao_atingido = [
        'não foi possível obter', 'não foi possível localizar', 'não foi encontrado',
        'não consta', 'não possui registro', 'não possui cadastro', 'não há registro',
        'não há cadastro', 'sem registro', 'sem cadastro', 'não foi possível verificar',
        'não identificamos', 'não identificado', 'não localizado', 'não localizamos',
        'não foi identificado', 'não obteve êxito', 'mas não conseguiu', 'mas não conseguimos',
        'não disponível', 'não dispõe', 'não foi disponibilizado', 'negou', 'recusou',
        'recusou-se', 'não informou', 'não quis informar', 'não tem conhecimento',
        'desconhece', 'não tem informações', 'não soube dizer', 'não soube informar',
        'não obteve', 'não recebemos', 'não recebeu', 'não tinha', 'não possui',
        'não foram encontrados', 'não foram localizados', 'não apresentou',
        'não apresentaram', 'não trouxe', 'não trouxeram'
    ]
    
    # Expressões específicas para verificar se o contato real foi bem sucedido
    indicadores_contato_real = [
        'de acordo com', 'segundo', 'informou que', 'relatou que', 'disse que',
        'foi informado', 'foi comunicado', 'foi relatado', 'comunicou que',
        'em conversa com', 'conversei com', 'em contato com', 'contato com',
        'em reunião com', 'durante reunião', 'durante encontro', 'segundo informações',
        'conforme relatado', 'conforme dito', 'conforme informado', 'em atendimento',
        'durante atendimento', 'após conversa', 'em diálogo', 'visitamos', 'visitei'
    ]
    
    # Adicionar colunas de indicação baseadas na descrição
    # Verificar se há indicação de sucesso no contato
    df['indicacao_sucesso_na_descricao'] = df['acompanhamento_descricao'].apply(
        lambda x: verificar_indicacao(x, palavras_sucesso))
        
    # Verificar se há indicação de insucesso no contato
    df['indicacao_insucesso_na_descricao'] = df['acompanhamento_descricao'].apply(
        lambda x: verificar_indicacao(x, palavras_insucesso))
    
    # Verificar se o contato ocorreu mas o objetivo não foi atingido
    df['contexto_objetivo_nao_atingido'] = df.apply(
        lambda row: (any(termo in row['acompanhamento_descricao'].lower() for termo in indicadores_contato_real) and
                     any(termo in row['acompanhamento_descricao'].lower() for termo in palavras_objetivo_nao_atingido))
        if isinstance(row['acompanhamento_descricao'], str) else False, axis=1)
    
    # Classificar a consistência
    df['consistencia_sucesso'] = df.apply(classificar_consistencia, axis=1)
    
    # Processar dados de encaminhamentos
    
    # Garantir que a coluna 'dado_algum_encaminhamento' tenha valores preenchidos
    if df['dado_algum_encaminhamento'].isna().any():
        # Preencher valores nulos com 'Não'
        df['dado_algum_encaminhamento'] = df['dado_algum_encaminhamento'].fillna('Não')
    
    # Verificar consistência de encaminhamentos (Sim deve ter pelo menos uma instituição preenchida)
    colunas_instituicoes = [
        'instituicao_encaminhamento_educacao',
        'instituicao_encaminhamento_saude',
        'instituicao_encaminhamento_assistencia_social',
        'instituicao_encaminhamento_conselho_tutelar',
        'instituicao_encaminhamento_estacao_conhecimento',
        'instituicao_encaminhamento_sociedade_civil',
        'instituicao_encaminhamento_outro_equipamento'
    ]
    
    # Adicionar coluna com o número de instituições encaminhadas por registro
    df['num_instituicoes_encaminhadas'] = df[colunas_instituicoes].notna().sum(axis=1)
    
    # Identificar inconsistências (Sim sem instituições ou Não com instituições)
    df['inconsistencia_encaminhamento'] = (
        ((df['dado_algum_encaminhamento'] == 'Sim') & (df['num_instituicoes_encaminhadas'] == 0)) |
        ((df['dado_algum_encaminhamento'] == 'Não') & (df['num_instituicoes_encaminhadas'] > 0))
    )
    
    return df

# Função para calcular similaridade entre descrições de acompanhamentos
def calcular_similaridade_descricoes(df):
    """
    Calcula a similaridade entre as descrições dos acompanhamentos
    usando TF-IDF e similaridade de coseno.
    Considera apenas registros do mesmo ID como possíveis duplicações.
    """
    try:
        # Mostrar status para o usuário
        placeholder = st.empty()
        placeholder.info("Iniciando análise de similaridade...")
        
        # Adicionando um contador para registros processados
        contador_progresso = 0
        total_registros = len(df)
        
        # Filtrar apenas registros com descrição válida
        placeholder.info("Filtrando registros com descrições válidas...")
        df_validos = df[df['acompanhamento_descricao'].notna() & 
                        (df['acompanhamento_descricao'].str.len() > 10)]
        
        if len(df_validos) <= 1:
            placeholder.info("Não há registros suficientes para análise de similaridade.")
            return pd.DataFrame()  # Retorna DataFrame vazio se não houver registros suficientes
        
        # Verificar se existe a coluna 'id'
        if 'id' not in df_validos.columns:
            placeholder.warning("Coluna 'id' não encontrada. A verificação de duplicações será feita sem considerar o ID da criança.")
            possui_id = False
        else:
            possui_id = True
        
        # Limitar o número de registros para processamento se for muito grande
        max_registros = 1000  # Definir um limite razoável
        if len(df_validos) > max_registros:
            placeholder.warning(f"Muitos registros encontrados ({len(df_validos)}). Limitando a análise aos {max_registros} primeiros para evitar lentidão.")
            df_validos = df_validos.head(max_registros)
        
        # Preparar o vetorizador TF-IDF
        placeholder.info(f"Preparando vetorização TF-IDF para {len(df_validos)} registros...")
        tfidf_vectorizer = TfidfVectorizer(
            min_df=1, 
            stop_words=stopwords.words('portuguese'),
            lowercase=True,
            strip_accents='unicode',
            max_features=5000  # Limitar o número de features para melhorar performance
        )
        
        # Criar a matriz TF-IDF
        placeholder.info("Vetorizando textos...")
        tfidf_matrix = tfidf_vectorizer.fit_transform(df_validos['acompanhamento_descricao'])
        
        # Calcular a similaridade de coseno
        placeholder.info("Calculando matriz de similaridade (pode levar alguns segundos)...")
        # Usar blocos menores para cálculo de similaridade para grandes conjuntos de dados
        batch_size = 100
        resultados = []
        
        # Se o conjunto de dados for grande, processar em lotes
        if len(df_validos) > batch_size:
            total_batches = (len(df_validos) // batch_size) + 1
            current_batch = 0
            
            for i in range(0, len(df_validos), batch_size):
                current_batch += 1
                end = min(i + batch_size, len(df_validos))
                placeholder.info(f"Processando lote {current_batch}/{total_batches} de comparações...")
                
                # Calcular similaridade para este lote
                batch_matrix = tfidf_matrix[i:end]
                cosine_sim_batch = cosine_similarity(batch_matrix, tfidf_matrix)
                
                # Processar resultados deste lote
                for batch_idx, global_i in enumerate(range(i, end)):
                    for global_j in range(global_i + 1, len(df_validos)):
                        # Verificar se são do mesmo ID (se a coluna existir)
                        if possui_id and df_validos.iloc[global_i]['id'] != df_validos.iloc[global_j]['id']:
                            continue  # Pular se não for o mesmo ID
                        
                        # Obter o valor de similaridade
                        similarity = cosine_sim_batch[batch_idx, global_j]
                        
                        if similarity >= 0.7:  # Threshold de similaridade (70%)
                            resultados.append(_criar_registro_similaridade(
                                df_validos, global_i, global_j, similarity, possui_id
                            ))
                            
                            # Atualizar contador
                            contador_progresso += 1
                            if contador_progresso % 10 == 0:
                                placeholder.info(f"Encontradas {contador_progresso} duplicações potenciais até agora...")
        else:
            # Para conjuntos menores, calcular a matriz de similaridade completa
            cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
            
            # Iterar sobre a matriz de similaridade
            placeholder.info("Analisando resultados de similaridade...")
            for i in range(len(cosine_sim)):
                for j in range(i+1, len(cosine_sim)):  # Começar do próximo item para evitar auto-comparação
                    # Verificar se são do mesmo ID (se a coluna existir)
                    if possui_id:
                        mesmo_id = df_validos.iloc[i]['id'] == df_validos.iloc[j]['id']
                        if not mesmo_id:
                            continue  # Pular se não for o mesmo ID
                    
                    if cosine_sim[i][j] >= 0.7:  # Threshold de similaridade (70%)
                        resultados.append(_criar_registro_similaridade(
                            df_validos, i, j, cosine_sim[i][j], possui_id
                        ))
                        
                        # Atualizar contador
                        contador_progresso += 1
                        if contador_progresso % 10 == 0:
                            placeholder.info(f"Encontradas {contador_progresso} duplicações potenciais até agora...")
        
        # Criar DataFrame com os resultados
        placeholder.info("Finalizando análise...")
        df_similaridade = pd.DataFrame(resultados)
        
        # Ordenar por similaridade (decrescente)
        if not df_similaridade.empty:
            df_similaridade = df_similaridade.sort_values('similaridade', ascending=False)
            placeholder.success(f"Análise concluída com sucesso! Encontradas {len(df_similaridade)} possíveis duplicações.")
        else:
            placeholder.info("Análise concluída, mas nenhuma duplicação foi encontrada.")
        
        # Limpar mensagem de status após concluir
        placeholder.empty()
        
        return df_similaridade
    
    except Exception as e:
        st.error(f"Erro ao calcular similaridade: {str(e)}")
        import traceback
        st.error(f"Detalhes: {traceback.format_exc()}")
        return pd.DataFrame()

def _criar_registro_similaridade(df_validos, i, j, similaridade, possui_id):
    """Função auxiliar para criar um registro de similaridade"""
    linha_i = df_validos.iloc[i]
    linha_j = df_validos.iloc[j]
    
    # Verificar se são do mesmo cadastro (mesmo aluno_nome se disponível)
    mesmo_beneficiario = "Não verificado"
    if 'aluno_nome' in df_validos.columns:
        mesmo_beneficiario = "Sim" if linha_i['aluno_nome'] == linha_j['aluno_nome'] else "Não"
    
    # Verificar se as datas são as mesmas ou próximas
    data_i = linha_i['acompanhamento_data']
    data_j = linha_j['acompanhamento_data']
    diff_dias = abs((data_i - data_j).days)
    
    # Adicionar o ID aos resultados
    id_registro = linha_i['id'] if possui_id else "Não disponível"
    
    # Retornar um dicionário com todos os dados relevantes
    return {
        'id_registro_1': df_validos.index[i],
        'id_registro_2': df_validos.index[j],
        'acompanhamento_cod_1': linha_i['acompanhamento_cod'] if 'acompanhamento_cod' in linha_i else '',
        'acompanhamento_cod_2': linha_j['acompanhamento_cod'] if 'acompanhamento_cod' in linha_j else '',
        'id_crianca': id_registro,
        'similaridade': similaridade,
        'data_registro_1': data_i,
        'data_registro_2': data_j,
        'diferenca_dias': diff_dias,
        'articulador_1': linha_i['acompanhamento_articulador'],
        'articulador_2': linha_j['acompanhamento_articulador'],
        'mesmo_beneficiario': mesmo_beneficiario,
        'descricao_1': linha_i['acompanhamento_descricao'],
        'descricao_2': linha_j['acompanhamento_descricao']
    }

def detectar_encaminhamentos_duplicados(df, dias_limite=30):
    """
    Detecta possíveis encaminhamentos duplicados para o mesmo ID (criança) em um período próximo.
    
    Args:
        df: DataFrame com os registros de encaminhamentos
        dias_limite: Número máximo de dias para considerar como período próximo (padrão: 30 dias)
        
    Returns:
        DataFrame com os possíveis encaminhamentos duplicados
    """
    # Verificar se existe a coluna 'id'
    if 'id' not in df.columns:
        st.warning("Coluna 'id' não encontrada. A verificação de encaminhamentos duplicados não pode ser realizada.")
        return pd.DataFrame()
    
    # Filtrar apenas registros com encaminhamentos
    df_encaminhamentos = df[df['dado_algum_encaminhamento'] == 'Sim'].copy()
    
    if len(df_encaminhamentos) <= 1:
        return pd.DataFrame()  # Retorna DataFrame vazio se não houver registros suficientes
    
    # Inicializar lista para armazenar duplicações potenciais
    duplicacoes = []
    
    # Lista de colunas de instituições
    colunas_instituicoes = [
        'instituicao_encaminhamento_educacao',
        'instituicao_encaminhamento_saude',
        'instituicao_encaminhamento_assistencia_social',
        'instituicao_encaminhamento_conselho_tutelar',
        'instituicao_encaminhamento_estacao_conhecimento',
        'instituicao_encaminhamento_sociedade_civil',
        'instituicao_encaminhamento_outro_equipamento'
    ]
    
    # Agrupar por ID
    grupos_id = df_encaminhamentos.groupby('id')
    
    # Para cada ID, verificar se há encaminhamentos próximos
    for id_crianca, grupo in grupos_id:
        # Se houver apenas um registro para este ID, não há duplicações
        if len(grupo) <= 1:
            continue
        
        # Ordenar por data
        grupo_ordenado = grupo.sort_values('acompanhamento_data')
        
        # Comparar cada par de registros
        for i in range(len(grupo_ordenado)):
            for j in range(i+1, len(grupo_ordenado)):
                linha_i = grupo_ordenado.iloc[i]
                linha_j = grupo_ordenado.iloc[j]
                
                # Calcular diferença de dias
                data_i = linha_i['acompanhamento_data']
                data_j = linha_j['acompanhamento_data']
                diff_dias = abs((data_j - data_i).days)
                
                # Se a diferença for maior que o limite, não considerar como duplicação
                if diff_dias > dias_limite:
                    continue
                
                # Verificar se há pelo menos uma instituição em comum
                tem_instituicao_comum = False
                instituicoes_comuns = []
                
                for coluna in colunas_instituicoes:
                    # Verificar se ambos os registros têm valor não-nulo para esta instituição
                    if pd.notna(linha_i[coluna]) and pd.notna(linha_j[coluna]):
                        tem_instituicao_comum = True
                        instituicoes_comuns.append(coluna.replace('instituicao_encaminhamento_', ''))
                
                # Se houver pelo menos uma instituição em comum, considerar como possível duplicação
                if tem_instituicao_comum:
                    duplicacoes.append({
                        'id_crianca': id_crianca,
                        'id_registro_1': grupo_ordenado.index[i],
                        'id_registro_2': grupo_ordenado.index[j],
                        'acompanhamento_cod_1': linha_i['acompanhamento_cod'] if 'acompanhamento_cod' in linha_i else '',
                        'acompanhamento_cod_2': linha_j['acompanhamento_cod'] if 'acompanhamento_cod' in linha_j else '',
                        'data_registro_1': data_i,
                        'data_registro_2': data_j,
                        'diferenca_dias': diff_dias,
                        'articulador_1': linha_i['acompanhamento_articulador'],
                        'articulador_2': linha_j['acompanhamento_articulador'],
                        'mesmo_articulador': linha_i['acompanhamento_articulador'] == linha_j['acompanhamento_articulador'],
                        'instituicoes_comuns': ', '.join(instituicoes_comuns),
                        'descricao_1': linha_i['acompanhamento_descricao'],
                        'descricao_2': linha_j['acompanhamento_descricao']
                    })
    
    # Criar DataFrame com os resultados
    df_duplicacoes = pd.DataFrame(duplicacoes)
    
    # Ordenar por diferença de dias (crescente) e depois por ID
    if not df_duplicacoes.empty:
        df_duplicacoes = df_duplicacoes.sort_values(['diferenca_dias', 'id_crianca'])
    
    return df_duplicacoes

# Função para detectar encaminhamentos sem seguimento (follow-up)
def detectar_encaminhamentos_sem_followup(df, dias_limite=30):
    """
    Detecta encaminhamentos que não tiveram follow-up (seguimento) em registros posteriores.
    
    Args:
        df: DataFrame com os registros
        dias_limite: Número de dias que devem passar para considerar que deveria haver um follow-up
        
    Returns:
        DataFrame com os encaminhamentos sem follow-up detectados
    """
    # Verificar se existe a coluna 'id'
    if 'id' not in df.columns:
        st.warning("Coluna 'id' não encontrada. A verificação de encaminhamentos sem follow-up não pode ser realizada.")
        return pd.DataFrame()
    
    # Garantir que a data está no formato correto
    if not pd.api.types.is_datetime64_any_dtype(df['acompanhamento_data']):
        df['acompanhamento_data'] = pd.to_datetime(df['acompanhamento_data'], errors='coerce')
    
    # Filtrar apenas registros com encaminhamentos
    df_encaminhamentos = df[df['dado_algum_encaminhamento'] == 'Sim'].copy()
    
    if len(df_encaminhamentos) <= 0:
        return pd.DataFrame()  # Retorna DataFrame vazio se não houver registros suficientes
    
    # Lista de colunas de instituições
    colunas_instituicoes = [
        'instituicao_encaminhamento_educacao',
        'instituicao_encaminhamento_saude',
        'instituicao_encaminhamento_assistencia_social',
        'instituicao_encaminhamento_conselho_tutelar',
        'instituicao_encaminhamento_estacao_conhecimento',
        'instituicao_encaminhamento_sociedade_civil',
        'instituicao_encaminhamento_outro_equipamento'
    ]
    
    # Data de referência (data atual)
    data_referencia = datetime.now()
    
    # Lista para armazenar encaminhamentos sem follow-up
    encaminhamentos_sem_followup = []
    
    # Agrupar por ID
    grupos_id = df.groupby('id')
    
    # Para cada ID, verificar encaminhamentos sem follow-up
    for id_crianca, grupo in grupos_id:
        # Ordenar por data para cada criança
        grupo_ordenado = grupo.sort_values('acompanhamento_data')
        
        # Obter apenas os registros com encaminhamentos para esta criança
        encaminhamentos_crianca = grupo_ordenado[grupo_ordenado['dado_algum_encaminhamento'] == 'Sim']
        
        if encaminhamentos_crianca.empty:
            continue
        
        # Analisar cada encaminhamento
        for idx, encaminhamento in encaminhamentos_crianca.iterrows():
            data_encaminhamento = encaminhamento['acompanhamento_data']
            
            # Verificar se já passaram os dias limite desde o encaminhamento
            if (data_referencia - data_encaminhamento).days < dias_limite:
                continue  # Ainda não passou tempo suficiente para considerar sem follow-up
            
            # Listar instituições encaminhadas neste registro
            instituicoes_encaminhadas = []
            for coluna in colunas_instituicoes:
                if pd.notna(encaminhamento[coluna]):
                    tipo_instituicao = coluna.replace('instituicao_encaminhamento_', '')
                    nome_instituicao = encaminhamento[coluna]
                    instituicoes_encaminhadas.append((tipo_instituicao, nome_instituicao))
            
            if not instituicoes_encaminhadas:
                continue  # Não há instituições específicas neste encaminhamento
            
            # Verificar registros posteriores
            registros_posteriores = grupo_ordenado[grupo_ordenado['acompanhamento_data'] > data_encaminhamento]
            
            # Se não há registros posteriores, é um encaminhamento sem follow-up
            if registros_posteriores.empty:
                foi_seguido = False
            else:
                # Verificar se há alguma menção de follow-up nos registros posteriores
                foi_seguido = False
                
                # Verificar na descrição dos registros posteriores
                for _, reg_posterior in registros_posteriores.iterrows():
                    descricao = str(reg_posterior['acompanhamento_descricao']).lower() if pd.notna(reg_posterior['acompanhamento_descricao']) else ""
                    
                    # Verificar para cada instituição se há menção na descrição
                    for tipo_inst, nome_inst in instituicoes_encaminhadas:
                        if pd.notna(nome_inst) and str(nome_inst).lower() in descricao:
                            foi_seguido = True
                            break
                    
                    # Verificar se o tipo de sinalização é relacionado a follow-up
                    tipo_sinalizacao = reg_posterior.get('acompanhamento_tipo_sinalizacao', '')
                    if pd.notna(tipo_sinalizacao) and tipo_sinalizacao in ['frequencias', 'acoes_complementares']:
                        foi_seguido = True
                        break
                    
                    if foi_seguido:
                        break
            
            # Se não há seguimento, adicionar à lista
            if not foi_seguido:
                encaminhamentos_sem_followup.append({
                    'id_crianca': id_crianca,
                    'id_registro': idx,
                    'acompanhamento_cod': encaminhamento['acompanhamento_cod'] if 'acompanhamento_cod' in encaminhamento else '',
                    'data_encaminhamento': data_encaminhamento,
                    'dias_desde_encaminhamento': (data_referencia - data_encaminhamento).days,
                    'articulador': encaminhamento['acompanhamento_articulador'],
                    'instituicoes_encaminhadas': ', '.join([f"{tipo} - {nome}" for tipo, nome in instituicoes_encaminhadas]),
                    'descricao': encaminhamento['acompanhamento_descricao']
                })
    
    # Criar DataFrame com os resultados
    df_sem_followup = pd.DataFrame(encaminhamentos_sem_followup)
    
    # Ordenar por número de dias (decrescente)
    if not df_sem_followup.empty:
        df_sem_followup = df_sem_followup.sort_values('dias_desde_encaminhamento', ascending=False)
    
    return df_sem_followup

# MAIN APP
st.title('TR | COMEA - 📊 Análise de Qualidade dos Registros de Acompanhamento')

# Barra lateral para seleção de abas
tab_selecionada = st.sidebar.radio(
    "Selecione uma seção:",
    ["Visão Geral", 
     "Análise Temporal", 
     "Análise por Articulador", 
     "Exemplos de Problemas", 
     "Análise de Texto",
     "Consistência de Sucesso",
     "Análise de Encaminhamentos",
     "Monitoramento de Follow-up",
     "Detecção de Duplicados"]
)

# Área para upload de arquivo
st.sidebar.header('Upload de Dados')
uploaded_file = st.sidebar.file_uploader("Faça upload do arquivo Excel", type=['xlsx'])

# Carregar dados
with st.spinner('Carregando e processando dados...'):
    df = carregar_dados(uploaded_file)

# Verificar se os dados foram carregados
if df is None:
    st.warning("Nenhum arquivo carregado. Por favor, faça o upload do arquivo Excel com os dados de acompanhamento.")
    st.stop()  # Parar a execução do aplicativo até que o arquivo seja carregado

# Sidebar para filtros
st.sidebar.header('Filtros')

# Botão para limpar filtros
if st.sidebar.button('Limpar Todos os Filtros'):
    # Atualização: usando as funções recomendadas em vez das experimentais
    st.query_params.clear()
    st.rerun()

# Filtro por período
st.sidebar.subheader('Período de Análise')
data_min = df['acompanhamento_data'].min().date()
data_max = df['acompanhamento_data'].max().date()

# Definir data inicial padrão como 01/01/2025 (ou a data mínima se for maior)
data_inicio_padrao = max(datetime(2025, 1, 1).date(), data_min)

# Criar filtro de período (data inicial e final)
col1, col2 = st.sidebar.columns(2)
with col1:
    data_inicio = st.date_input('Data Inicial', data_inicio_padrao, min_value=data_min, max_value=data_max)
with col2:
    data_fim = st.date_input('Data Final', data_max, min_value=data_min, max_value=data_max)

# Filtro por articulador
articuladores = ['Todos'] + sorted(df['acompanhamento_articulador'].unique().tolist())
articulador_selecionado = st.sidebar.selectbox('Articulador', articuladores)

# Filtro por qualidade
qualidades = ['Todos', 'Excelente', 'Bom', 'Regular', 'Ruim', 'Crítico']
qualidade_selecionada = st.sidebar.selectbox('Classificação de Qualidade', qualidades)

# Filtro por tipo de problema
tipos_problemas = ['Todos', 'muito_curto', 'vago', 'sem_detalhes', 'generico', 'texto_nulo', 'texto_invalido']
problema_selecionado = st.sidebar.selectbox('Tipo de Problema', tipos_problemas)

# Filtro por sucesso de contato e consistência
st.sidebar.subheader('Filtros Adicionais')
sucessos_contato = ['Todos', 'Sim', 'Não']
sucesso_selecionado = st.sidebar.selectbox('Sucesso de Contato', sucessos_contato)

tipos_consistencia = ['Todos', 'Consistente (Sucesso)', 'Consistente (Insucesso)', 
                     'Possível Inconsistência (Sim/Insucesso)', 'Possível Inconsistência (Não/Sucesso)',
                     'Sem indicação clara na descrição', 'Indicações contraditórias na descrição']
consistencia_selecionada = st.sidebar.selectbox('Consistência de Sucesso', tipos_consistencia)

# Filtro de contexto de objetivo
contextos = ['Todos', 'Objetivo Não Atingido', 'Sem Indicação de Objetivo']
contexto_selecionado = st.sidebar.selectbox('Contexto do Registro', contextos)

# Filtros para encaminhamentos
st.sidebar.subheader('Filtros de Encaminhamentos')
encaminhamentos_filtro = ['Todos', 'Com Encaminhamento', 'Sem Encaminhamento']
encaminhamento_selecionado = st.sidebar.selectbox('Status de Encaminhamento', encaminhamentos_filtro)

# Filtro por tipo de instituição encaminhada
tipos_instituicoes = [
    'Todos',
    'Educação',
    'Saúde',
    'Assistência Social',
    'Conselho Tutelar',
    'Estação Conhecimento',
    'Sociedade Civil',
    'Outro Equipamento'
]
instituicao_selecionada = st.sidebar.selectbox('Tipo de Instituição Encaminhada', tipos_instituicoes)

# Aplicar filtros
df_filtrado = df.copy()

# Filtro de período
df_filtrado = df_filtrado[(df_filtrado['acompanhamento_data'].dt.date >= data_inicio) & 
                           (df_filtrado['acompanhamento_data'].dt.date <= data_fim)]

if articulador_selecionado != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['acompanhamento_articulador'] == articulador_selecionado]

if qualidade_selecionada != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['classificacao_qualidade'] == qualidade_selecionada]

if problema_selecionado != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['problemas_detectados'].apply(lambda x: problema_selecionado in x)]

if sucesso_selecionado != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['acompanhamento_sucesso_contato'] == sucesso_selecionado]

if consistencia_selecionada != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['consistencia_sucesso'] == consistencia_selecionada]

if contexto_selecionado != 'Todos':
    if contexto_selecionado == 'Objetivo Não Atingido':
        df_filtrado = df_filtrado[df_filtrado['contexto_objetivo_nao_atingido'] == True]
    elif contexto_selecionado == 'Sem Indicação de Objetivo':
        df_filtrado = df_filtrado[df_filtrado['contexto_objetivo_nao_atingido'] == False]

# Aplicar filtros de encaminhamento
if encaminhamento_selecionado != 'Todos':
    if encaminhamento_selecionado == 'Com Encaminhamento':
        df_filtrado = df_filtrado[df_filtrado['dado_algum_encaminhamento'] == 'Sim']
    else:  # 'Sem Encaminhamento'
        df_filtrado = df_filtrado[df_filtrado['dado_algum_encaminhamento'] == 'Não']

# Mapeamento para nomes de colunas de instituições
mapa_instituicoes = {
    'Educação': 'instituicao_encaminhamento_educacao',
    'Saúde': 'instituicao_encaminhamento_saude',
    'Assistência Social': 'instituicao_encaminhamento_assistencia_social',
    'Conselho Tutelar': 'instituicao_encaminhamento_conselho_tutelar',
    'Estação Conhecimento': 'instituicao_encaminhamento_estacao_conhecimento',
    'Sociedade Civil': 'instituicao_encaminhamento_sociedade_civil',
    'Outro Equipamento': 'instituicao_encaminhamento_outro_equipamento'
}

if instituicao_selecionada != 'Todos':
    coluna_instituicao = mapa_instituicoes[instituicao_selecionada]
    df_filtrado = df_filtrado[df_filtrado[coluna_instituicao].notna()]

# TAB 1: VISÃO GERAL
if tab_selecionada == "Visão Geral":
    # Estatísticas gerais
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Registros", len(df_filtrado))
    with col2:
        st.metric("Pontuação Média", f"{df_filtrado['pontuacao'].mean():.2f}/10")
    with col3:
        st.metric("Registros com Problemas", 
                 f"{df_filtrado['num_problemas'].gt(0).sum()} ({df_filtrado['num_problemas'].gt(0).mean()*100:.1f}%)")
    
    # Gráfico de distribuição da qualidade - Versão melhorada com Plotly
    st.subheader("Distribuição da Classificação de Qualidade")
    
    # Preparando dados para o gráfico
    qualidade_counts = df_filtrado['classificacao_qualidade'].value_counts().reset_index()
    qualidade_counts.columns = ['classificacao', 'quantidade']
    
    # Definir ordem personalizada para categorias de qualidade
    ordem_qualidade = ["Excelente", "Bom", "Regular", "Ruim", "Crítico"]
    
    # Reordenar o DataFrame
    qualidade_counts['ordem'] = qualidade_counts['classificacao'].map({cat: i for i, cat in enumerate(ordem_qualidade)})
    qualidade_counts = qualidade_counts.sort_values('ordem').drop('ordem', axis=1)
    
    # Criar mapa de cores personalizado
    color_map = {
        'Excelente': '#1E88E5',
        'Bom': '#43A047',
        'Regular': '#FFB300',
        'Ruim': '#E53935',
        'Crítico': '#8E24AA'
    }
    
    # Criar gráfico de barras interativo melhorado
    fig = px.bar(
        qualidade_counts, 
        x='classificacao', 
        y='quantidade',
        color='classificacao',
        color_discrete_map=color_map,
        text='quantidade',
        labels={'classificacao': 'Classificação', 'quantidade': 'Quantidade de Registros'},
        height=500
    )
    
    # Personalizar layout
    fig.update_layout(
        title={
            'text': 'Distribuição por Classificação de Qualidade',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20}
        },
        xaxis_title="Classificação",
        yaxis_title="Quantidade de Registros",
        legend_title="Qualidade",
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#333333", 
            bordercolor="#333333",
            font_size=14,
            font_family="Arial",
            font_color="white"
        ),
        showlegend=True
    )
    
    # Melhorar a exibição do texto nas barras e corrigir cores no hover
    fig.update_traces(
        textposition='outside',
        textfont=dict(size=14),
        hovertemplate='<b>%{x}</b><br>Registros: %{y}<extra></extra>'
    )
    
    # Definir cores personalizadas para cada barra individualmente
    for i, qualidade in enumerate(qualidade_counts['classificacao']):
        if qualidade in color_map:
            fig.data[0].marker.color = [color_map[q] for q in qualidade_counts['classificacao']]
            
    # Exibir o gráfico
    st.plotly_chart(fig, use_container_width=True)
    
    # Gráfico de distribuição de comprimento
    st.subheader("Distribuição do Comprimento das Descrições")
    fig = px.histogram(df_filtrado, x='comprimento_descricao', nbins=50)
    fig.update_layout(
        xaxis_title="Comprimento (caracteres)", 
        yaxis_title="Quantidade de Registros",
        hoverlabel=dict(
            bgcolor="#333333", 
            bordercolor="#333333",
            font_size=14,
            font_family="Arial",
            font_color="white"
        )
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Tipos de problemas detectados
    st.subheader("Tipos de Problemas Detectados")
    
    # Desempacotando a lista de problemas para contar
    problemas_lista = []
    for problemas in df_filtrado['problemas_detectados']:
        problemas_lista.extend(problemas)
    
    problemas_contagem = pd.Series(problemas_lista).value_counts()
    
    if not problemas_contagem.empty:
        fig = px.bar(x=problemas_contagem.index, y=problemas_contagem.values,
                   labels={'x': 'Tipo de Problema', 'y': 'Quantidade de Ocorrências'})
        fig.update_layout(
            xaxis_title="Tipo de Problema",
            yaxis_title="Quantidade de Ocorrências", 
            hoverlabel=dict(
                bgcolor="#333333", 
                bordercolor="#333333",
                font_size=14,
                font_family="Arial",
                font_color="white"
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Nenhum problema detectado nos registros filtrados.")

# TAB 2: ANÁLISE TEMPORAL
elif tab_selecionada == "Análise Temporal":
    st.subheader("Análise Temporal dos Registros")
    
    # Opções de granularidade temporal
    opcao_tempo = st.radio(
        "Selecione a granularidade temporal:",
        ("Diário", "Semanal", "Mensal"),
        horizontal=True
    )
    
    # Definir coluna de data baseada na opção
    if opcao_tempo == "Diário":
        coluna_tempo = 'acompanhamento_data'
        formato = '%d/%m/%Y'
    elif opcao_tempo == "Semanal":
        coluna_tempo = 'semana'
        formato = 'Semana %U, %Y'
    else:  # Mensal
        coluna_tempo = 'mes_ano'
        formato = '%b %Y'
    
    # Análise de volume ao longo do tempo
    st.markdown("### Volume de Registros ao Longo do Tempo")
    
    # Contar registros por período
    df_tempo = df_filtrado.groupby(coluna_tempo).size().reset_index(name='quantidade')
    
    # Ordenar por data
    df_tempo = df_tempo.sort_values(coluna_tempo)
    
    # Gráfico de linha para volume
    fig_volume = px.line(
        df_tempo, 
        x=coluna_tempo, 
        y='quantidade',
        labels={'quantidade': 'Quantidade de Registros', coluna_tempo: 'Período'},
        markers=True
    )
    
    # Personalizar layout
    fig_volume.update_layout(
        xaxis_title="Período",
        yaxis_title="Quantidade de Registros",
        height=500
    )
    
    # Se estiver usando mes_ano, substituir os valores do eixo x para exibição
    if coluna_tempo == 'mes_ano':
        # Criar um dicionário de mapeamento de mes_ano para mes_ano_display
        mapeamento_mes = dict(zip(df_filtrado['mes_ano'], df_filtrado['mes_ano_display']))
        # Aplicar o mapeamento aos tickets do eixo x
        fig_volume.update_xaxes(
            tickvals=list(mapeamento_mes.keys()),
            ticktext=list(mapeamento_mes.values())
        )
    
    st.plotly_chart(fig_volume, use_container_width=True)
    
    # NOVA SEÇÃO: Volume de acompanhamentos por articulador ao longo do tempo
    st.markdown("### Volume por Articulador ao Longo do Tempo")
    
    # Opções de visualização
    tipo_viz = st.radio(
        "Tipo de visualização:",
        ("Gráfico de Linhas", "Gráfico de Barras Empilhadas", "Mapa de Calor"),
        horizontal=True
    )
    
    # Agrupar dados por período e articulador
    df_volume_articulador = df_filtrado.groupby([coluna_tempo, 'acompanhamento_articulador']).size().reset_index(name='quantidade')
    
    # Ordenar por data
    df_volume_articulador = df_volume_articulador.sort_values(coluna_tempo)
    
    if tipo_viz == "Gráfico de Linhas":
        # Gráfico de linha para cada articulador
        fig_articulador = px.line(
            df_volume_articulador,
            x=coluna_tempo,
            y='quantidade',
            color='acompanhamento_articulador',
            labels={
                'quantidade': 'Quantidade de Registros', 
                coluna_tempo: 'Período',
                'acompanhamento_articulador': 'Articulador'
            },
            markers=True
        )
        
        # Personalizar layout
        fig_articulador.update_layout(
            xaxis_title="Período",
            yaxis_title="Quantidade de Registros",
            height=600,
            legend_title="Articulador"
        )
        
        # Se estiver usando mes_ano, substituir os valores do eixo x para exibição
        if coluna_tempo == 'mes_ano':
            # Criar um dicionário de mapeamento de mes_ano para mes_ano_display
            mapeamento_mes = dict(zip(df_filtrado['mes_ano'], df_filtrado['mes_ano_display']))
            # Aplicar o mapeamento aos tickets do eixo x
            fig_articulador.update_xaxes(
                tickvals=list(mapeamento_mes.keys()),
                ticktext=list(mapeamento_mes.values())
            )
    
    elif tipo_viz == "Gráfico de Barras Empilhadas":
        # Gráfico de barras empilhadas
        fig_articulador = px.bar(
            df_volume_articulador,
            x=coluna_tempo,
            y='quantidade',
            color='acompanhamento_articulador',
            labels={
                'quantidade': 'Quantidade de Registros', 
                coluna_tempo: 'Período',
                'acompanhamento_articulador': 'Articulador'
            }
        )
        
        # Personalizar layout
        fig_articulador.update_layout(
            xaxis_title="Período",
            yaxis_title="Quantidade de Registros",
            height=600,
            legend_title="Articulador"
        )
        
        # Se estiver usando mes_ano, substituir os valores do eixo x para exibição
        if coluna_tempo == 'mes_ano':
            # Criar um dicionário de mapeamento de mes_ano para mes_ano_display
            mapeamento_mes = dict(zip(df_filtrado['mes_ano'], df_filtrado['mes_ano_display']))
            # Aplicar o mapeamento aos tickets do eixo x
            fig_articulador.update_xaxes(
                tickvals=list(mapeamento_mes.keys()),
                ticktext=list(mapeamento_mes.values())
            )
    
    else:  # Mapa de Calor
        # Ordenar os dados por período antes de criar o pivot
        df_volume_articulador = df_volume_articulador.sort_values(coluna_tempo)
        
        # Se estiver usando mes_ano, criar um dicionário de mapeamento para exibição
        label_mapping = {}
        if coluna_tempo == 'mes_ano':
            label_mapping = dict(zip(df_filtrado['mes_ano'], df_filtrado['mes_ano_display']))
        
        # Criar tabela pivot para o mapa de calor
        pivot_data = df_volume_articulador.pivot_table(
            index='acompanhamento_articulador', 
            columns=coluna_tempo, 
            values='quantidade',
            fill_value=0
        )
        
        # Ordenar colunas cronologicamente
        pivot_data = pivot_data.reindex(sorted(pivot_data.columns), axis=1)
        
        # Mapear nomes das colunas para exibição se necessário
        column_labels = pivot_data.columns
        if coluna_tempo == 'mes_ano' and label_mapping:
            column_labels = [label_mapping.get(col, col) for col in pivot_data.columns]
        
        # Criar mapa de calor
        fig_articulador = px.imshow(
            pivot_data,
            labels=dict(x="Período", y="Articulador", color="Quantidade"),
            x=column_labels,  # Usar as etiquetas mapeadas
            y=pivot_data.index,
            color_continuous_scale='viridis',
            text_auto=True
        )
        
        # Personalizar layout
        fig_articulador.update_layout(
            height=25 * (len(pivot_data) + 5),  # Altura adaptativa
            xaxis_title="Período",
            yaxis_title="Articulador"
        )
    
    st.plotly_chart(fig_articulador, use_container_width=True)
    
    # Análise estatística básica da distribuição temporal
    with st.expander("Estatísticas da Distribuição Temporal"):
        # Identificar períodos sem atividade
        st.markdown("**Distribuição de registros por articulador:**")
        dist_articulador = df_filtrado.groupby('acompanhamento_articulador').size().reset_index(name='quantidade')
        dist_articulador = dist_articulador.sort_values('quantidade', ascending=False)
        
        # Gráfico de barras para distribuição por articulador
        fig_dist = px.bar(
            dist_articulador,
            x='acompanhamento_articulador',
            y='quantidade',
            labels={
                'quantidade': 'Quantidade de Registros',
                'acompanhamento_articulador': 'Articulador'
            }
        )
        
        # Aplicar tema escuro aos tooltips
        fig_dist.update_layout(
            hoverlabel=dict(
                bgcolor="#333333", 
                bordercolor="#333333",
                font_size=14,
                font_family="Arial",
                font_color="white"
            )
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Tabela de contagem por articulador
        st.markdown("**Contagem de registros por articulador:**")
        st.dataframe(
            dist_articulador.rename(columns={
                'acompanhamento_articulador': 'Articulador',
                'quantidade': 'Quantidade de Registros'
            }).set_index('Articulador'),
            use_container_width=True
        )
    
    # Análise da qualidade dos registros ao longo do tempo
    st.markdown("### Qualidade dos Registros ao Longo do Tempo")
    
    # Qualidade média por mês
    qualidade_por_mes = df_filtrado.groupby('mes_ano')['pontuacao'].mean().reset_index()
    qualidade_por_mes = qualidade_por_mes.sort_values('mes_ano')
    
    if not qualidade_por_mes.empty:
        st.subheader("Evolução da Qualidade dos Registros ao Longo do Tempo")
        fig = px.line(qualidade_por_mes, x='mes_ano', y='pontuacao',
                    labels={'mes_ano': 'Mês/Ano', 'pontuacao': 'Pontuação Média'},
                    markers=True)
        fig.update_layout(xaxis_title="Período", yaxis_title="Pontuação Média (0-10)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Distribuição de problemas por mês
    st.subheader("Distribuição de Problemas por Período")
    
    # Calcular percentual de registros problemáticos por mês
    problemas_por_mes = df_filtrado.groupby('mes_ano')['num_problemas'].apply(
        lambda x: (x > 0).mean() * 100).reset_index()
    problemas_por_mes.columns = ['mes_ano', 'pct_problematicos']
    problemas_por_mes = problemas_por_mes.sort_values('mes_ano')
    
    if not problemas_por_mes.empty:
        fig = px.bar(problemas_por_mes, x='mes_ano', y='pct_problematicos',
                   labels={'mes_ano': 'Mês/Ano', 'pct_problematicos': '% de Registros com Problemas'},
                   color='pct_problematicos', color_continuous_scale='RdYlGn_r')
        fig.update_layout(xaxis_title="Período", yaxis_title="% de Registros com Problemas")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Não há dados suficientes para a análise de problemas por período.")
    
    # Dia da semana com mais registros
    if len(df_filtrado) > 0:
        st.subheader("Distribuição por Dia da Semana")
        df_filtrado['dia_semana'] = df_filtrado['acompanhamento_data'].dt.day_name()
        dias_ordem = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dias_portugues = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
        
        # Criar um mapeamento para dias da semana em português
        mapa_dias = dict(zip(dias_ordem, dias_portugues))
        df_filtrado['dia_semana_pt'] = df_filtrado['dia_semana'].map(mapa_dias)
        
        registros_por_dia = df_filtrado.groupby('dia_semana_pt').size().reset_index(name='contagem')
        # Ordenar os dias da semana corretamente
        registros_por_dia['ordem'] = registros_por_dia['dia_semana_pt'].map(
            dict(zip(dias_portugues, range(7))))
        registros_por_dia = registros_por_dia.sort_values('ordem')
        
        fig = px.bar(registros_por_dia, x='dia_semana_pt', y='contagem',
                   labels={'dia_semana_pt': 'Dia da Semana', 'contagem': 'Quantidade de Registros'},
                   color='contagem')
        fig.update_layout(xaxis_title="Dia da Semana", yaxis_title="Quantidade de Registros")
        st.plotly_chart(fig, use_container_width=True)

# TAB 3: ANÁLISE POR ARTICULADOR
elif tab_selecionada == "Análise por Articulador":
    # Agrupando dados por articulador
    grupo_articulador = df_filtrado.groupby('acompanhamento_articulador').agg({
        'acompanhamento_descricao': 'count',
        'pontuacao': 'mean',
        'num_problemas': 'mean',
        'comprimento_descricao': 'mean'
    }).reset_index()
    
    grupo_articulador = grupo_articulador.rename(columns={
        'acompanhamento_articulador': 'articulador',
        'acompanhamento_descricao': 'total_registros',
        'pontuacao': 'pontuacao_media',
        'num_problemas': 'media_problemas',
        'comprimento_descricao': 'comprimento_medio'
    })
    
    # Calcular percentual de registros problemáticos por articulador
    problemas_por_articulador = df_filtrado.groupby('acompanhamento_articulador')['num_problemas'].apply(
        lambda x: (x > 0).mean() * 100).reset_index()
    problemas_por_articulador.columns = ['articulador', 'pct_registros_problematicos']
    
    # Juntar os dados
    grupo_articulador = pd.merge(grupo_articulador, problemas_por_articulador, on='articulador')
    
    # Ordenar por pontuação média (decrescente)
    grupo_articulador = grupo_articulador.sort_values('pontuacao_media', ascending=False)
    
    # Gráfico de pontuação média por articulador
    st.subheader("Pontuação Média por Articulador")
    if not grupo_articulador.empty:
        fig = px.bar(grupo_articulador, x='articulador', y='pontuacao_media',
                labels={'articulador': 'Articulador', 'pontuacao_media': 'Pontuação Média'},
                color='pontuacao_media', color_continuous_scale='RdYlGn')
        fig.update_layout(xaxis_title="Articulador", yaxis_title="Pontuação Média (0-10)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Gráfico de percentual de registros problemáticos por articulador
        st.subheader("% de Registros com Problemas por Articulador")
        fig = px.bar(grupo_articulador.sort_values('pct_registros_problematicos'), 
                x='articulador', y='pct_registros_problematicos',
                labels={'articulador': 'Articulador', 
                        'pct_registros_problematicos': '% de Registros com Problemas'},
                color='pct_registros_problematicos', color_continuous_scale='RdYlGn_r')
        fig.update_layout(xaxis_title="Articulador", yaxis_title="% de Registros com Problemas")
        st.plotly_chart(fig, use_container_width=True)
        
        # Gráfico de volume de registros por articulador
        st.subheader("Volume de Registros por Articulador")
        fig = px.bar(grupo_articulador.sort_values('total_registros', ascending=False), 
                x='articulador', y='total_registros',
                labels={'articulador': 'Articulador', 
                        'total_registros': 'Total de Registros'},
                color='total_registros')
        fig.update_layout(xaxis_title="Articulador", yaxis_title="Total de Registros")
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabela com dados por articulador
        st.subheader("Dados Detalhados por Articulador")
        st.dataframe(grupo_articulador)
    else:
        st.info("Não há dados suficientes para análise por articulador com os filtros atuais.")

# TAB 4: EXEMPLOS DE PROBLEMAS
elif tab_selecionada == "Exemplos de Problemas":
    # Definir os tipos de problemas para mostrar exemplos
    tipos_problemas_exemplos = ['muito_curto', 'vago', 'sem_detalhes', 'generico']
    problema_para_exemplos = st.selectbox(
        'Selecione um tipo de problema para ver exemplos:',
        tipos_problemas_exemplos
    )
    
    # Filtrar registros com o problema selecionado
    exemplos_df = df_filtrado[df_filtrado['problemas_detectados'].apply(lambda x: problema_para_exemplos in x)]
    
    if len(exemplos_df) > 0:
        st.subheader(f"Exemplos de Registros com o Problema: '{problema_para_exemplos}'")
        
        # Ordenar por comprimento da descrição (crescente)
        exemplos_df = exemplos_df.sort_values('comprimento_descricao')
        
        # Mostrar exemplos
        for i, (idx, row) in enumerate(exemplos_df.head(5).iterrows()):
            with st.expander(f"Exemplo {i+1}: {row['acompanhamento_articulador']} - {row['acompanhamento_data'].strftime('%d/%m/%Y')}"):
                st.markdown(f"**ID do Cadastro:** {row['id']}")
                st.markdown(f"**Código do Acompanhamento:** {row['acompanhamento_cod']}")
                st.markdown(f"**Articulador:** {row['acompanhamento_articulador']}")
                st.markdown(f"**Data:** {row['acompanhamento_data'].strftime('%d/%m/%Y')}")
                st.markdown(f"**Descrição:**")
                st.text_area("Descrição", row['acompanhamento_descricao'], height=100, key=f"descricao_problema_{problema_para_exemplos}_{i}", disabled=True, label_visibility="collapsed")
                
                # Mostrar outros problemas detectados
                outros_problemas = [p for p in row['problemas_detectados'] if p != problema_para_exemplos]
                if outros_problemas:
                    st.markdown(f"**Outros problemas detectados:** {', '.join(outros_problemas)}")
                
                # Mostrar qualidade calculada
                st.markdown(f"**Qualidade calculada:** {row['classificacao_qualidade']} (Pontuação: {row['pontuacao']}/10)")
    else:
        st.info(f"Nenhum registro encontrado com o problema '{problema_para_exemplos}'.")

# TAB 5: ANÁLISE DE TEXTO
elif tab_selecionada == "Análise de Texto":
    # Análise de texto
    st.subheader("Análise das Palavras mais Frequentes")
    
    # Amostra para análise de texto (para evitar sobrecarga)
    amostra = df_filtrado.head(1000) if len(df_filtrado) > 1000 else df_filtrado
    
    # Juntar todos os textos pré-processados
    textos_combinados = ' '.join(amostra['texto_preprocessado'].dropna())
    
    if textos_combinados.strip():
        # Criar nuvem de palavras
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            colormap='viridis',
            max_words=100,
            contour_width=1,
            contour_color='steelblue',
            collocations=False
        ).generate(textos_combinados)
        
        # Exibir a nuvem de palavras
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        
        # Análise de frequência de palavras
        st.subheader("Palavras mais Frequentes")
        
        # Extrair palavras para análise de frequência
        palavras = textos_combinados.split()
        freq_palavras = pd.Series(palavras).value_counts().head(20)
        
        # Plotar gráfico de barras com as palavras mais frequentes
        fig = px.bar(x=freq_palavras.index, y=freq_palavras.values,
                   labels={'x': 'Palavra', 'y': 'Frequência'})
        fig.update_layout(
            xaxis_title="Palavra", 
            yaxis_title="Frequência",
            hoverlabel=dict(
                bgcolor="#333333", 
                bordercolor="#333333",
                font_size=14,
                font_family="Arial",
                font_color="white"
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Não há dados suficientes para análise de texto com os filtros atuais.")

# TAB 6: CONSISTÊNCIA DE SUCESSO
elif tab_selecionada == "Consistência de Sucesso":
    st.subheader("Análise de Consistência entre Sucesso de Contato e Descrição")
    
    with st.expander("📋 Entenda a Análise de Consistência", expanded=True):
        st.markdown("""
        ### O que é a análise de consistência?
        
        Esta análise verifica se há coerência entre:
        
        1. **Sucesso de Contato**: Refere-se ao fato de ter conseguido entrar em contato com a pessoa ou instituição. Exemplos de sucesso: "falei com a mãe", "em contato com a diretora", "de acordo com a secretaria".
        
        2. **Conteúdo da Descrição**: O texto que explica o que ocorreu durante o acompanhamento.
        
        A inconsistência ocorre quando:
        - A coluna de sucesso indica "Sim", mas a descrição sugere que não houve sucesso (ex: "não atendeu", "número inexistente")
        - A coluna de sucesso indica "Não", mas a descrição sugere que houve contato (ex: "mãe informou que...", "conversando com responsável")
        
        Uma alta taxa de inconsistência pode indicar problemas no entendimento dos campos ou no processo de registro.
        """)
    
    # Criar dataframe com possíveis inconsistências
    df_inconsistente = df_filtrado[df_filtrado['consistencia_sucesso'].isin(['Possível Inconsistência (Sim/Insucesso)', 'Possível Inconsistência (Não/Sucesso)'])].copy()
    
    # Adicionar coluna com tipo de inconsistência
    df_inconsistente['tipo_inconsistencia'] = df_inconsistente.apply(
        lambda row: "Marcado como 'Sim' mas descrição indica insucesso" if row['acompanhamento_sucesso_contato'] == 'Sim'
        else "Marcado como 'Não' mas descrição indica sucesso", axis=1
    )
    
    # Análise geral
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_registros = len(df_filtrado)
        total_inconsistencias = len(df_inconsistente)
        percentual = (total_inconsistencias / total_registros * 100) if total_registros > 0 else 0
        st.metric("Registros Inconsistentes", f"{total_inconsistencias} ({percentual:.1f}%)")
        
    with col2:
        # Tipo 1: Marcado como "Sim" mas descrição indica insucesso
        tipo1 = len(df_inconsistente[df_inconsistente['tipo_inconsistencia'] == "Marcado como 'Sim' mas descrição indica insucesso"])
        percentual_tipo1 = (tipo1 / total_inconsistencias * 100) if total_inconsistencias > 0 else 0
        st.metric("Falsos Sucessos", f"{tipo1} ({percentual_tipo1:.1f}%)")
        
    with col3:
        # Tipo 2: Marcado como "Não" mas descrição indica sucesso
        tipo2 = len(df_inconsistente[df_inconsistente['tipo_inconsistencia'] == "Marcado como 'Não' mas descrição indica sucesso"])
        percentual_tipo2 = (tipo2 / total_inconsistencias * 100) if total_inconsistencias > 0 else 0
        st.metric("Falsos Insucessos", f"{tipo2} ({percentual_tipo2:.1f}%)")
    
    # Análise por articulador
    st.subheader("Consistência por Articulador")
    
    # Criar dataframe com contagem de inconsistências por articulador
    df_consistencia_articulador = pd.DataFrame()
    
    if not df_filtrado.empty:
        # Agrupar e contar por articulador
        total_por_articulador = df_filtrado.groupby('acompanhamento_articulador').size().reset_index(name='total_registros')
        inconsistentes_por_articulador = df_inconsistente.groupby('acompanhamento_articulador').size().reset_index(name='registros_inconsistentes')
        
        # Mesclar os dataframes
        df_consistencia_articulador = total_por_articulador.merge(
            inconsistentes_por_articulador, 
            on='acompanhamento_articulador', 
            how='left'
        ).fillna(0)
        
        # Calcular percentual
        df_consistencia_articulador['registros_inconsistentes'] = df_consistencia_articulador['registros_inconsistentes'].astype(int)
        df_consistencia_articulador['percentual_inconsistencia'] = (df_consistencia_articulador['registros_inconsistentes'] / 
                                                                df_consistencia_articulador['total_registros'] * 100)
        
        # Ordenar por percentual de inconsistência (decrescente)
        df_consistencia_articulador = df_consistencia_articulador.sort_values('percentual_inconsistencia', ascending=False)
        
        # Visualização em gráfico
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de barras para percentual de inconsistência
            fig_percentual = px.bar(
                df_consistencia_articulador,
                x='acompanhamento_articulador',
                y='percentual_inconsistencia',
                labels={
                    'acompanhamento_articulador': 'Articulador',
                    'percentual_inconsistencia': '% de Inconsistências'
                },
                text_auto='.1f',
                title='Percentual de Inconsistências por Articulador'
            )
            
            fig_percentual.update_layout(xaxis_tickangle=-45, height=500)
            st.plotly_chart(fig_percentual, use_container_width=True)
        
        with col2:
            # Tabela com dados de inconsistência por articulador
            st.dataframe(
                df_consistencia_articulador.rename(columns={
                    'acompanhamento_articulador': 'Articulador',
                    'total_registros': 'Total de Registros',
                    'registros_inconsistentes': 'Registros Inconsistentes',
                    'percentual_inconsistencia': '% de Inconsistência'
                }).set_index('Articulador'),
                use_container_width=True
            )
        
        # Análise de tipos de inconsistência por articulador
        st.subheader("Tipos de Inconsistência por Articulador")
        
        # Criar pivot table com tipos de inconsistência por articulador
        pivot_inconsistencia = pd.crosstab(
            df_inconsistente['acompanhamento_articulador'],
            df_inconsistente['tipo_inconsistencia'],
            normalize='index'
        ) * 100
        
        # Visualizar como heatmap
        fig_heatmap = px.imshow(
            pivot_inconsistencia,
            text_auto='.1f',
            labels=dict(x="Tipo de Inconsistência", y="Articulador", color="Percentual (%)"),
            x=pivot_inconsistencia.columns,
            y=pivot_inconsistencia.index,
            color_continuous_scale='RdYlGn_r',
            aspect="auto"
        )
        
        fig_heatmap.update_layout(height=25 * (len(pivot_inconsistencia) + 5))
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    else:
        st.info("Não há dados suficientes para análise com os filtros atuais.")
    
    # Exibir exemplos de possíveis inconsistências
    if not df_inconsistente.empty:
        st.subheader(f"Registros com Possíveis Inconsistências ({len(df_inconsistente)})")
        
        # Agrupar por tipo de inconsistência
        for tipo_inconsistencia in df_inconsistente['tipo_inconsistencia'].unique():
            df_tipo = df_inconsistente[df_inconsistente['tipo_inconsistencia'] == tipo_inconsistencia]
            
            st.markdown(f"### {tipo_inconsistencia} ({len(df_tipo)} registros)")
            
            # Mostrar alguns exemplos
            for i, (idx, row) in enumerate(df_tipo.head(3).iterrows()):
                with st.expander(f"Exemplo {i+1}: {row['acompanhamento_articulador']} - {row['acompanhamento_data'].strftime('%d/%m/%Y')}"):
                    st.markdown(f"**ID do Cadastro:** {row['id']}")
                    st.markdown(f"**Código do Acompanhamento:** {row['acompanhamento_cod']}")
                    st.markdown(f"**Articulador:** {row['acompanhamento_articulador']}")
                    st.markdown(f"**Data:** {row['acompanhamento_data'].strftime('%d/%m/%Y')}")
                    st.markdown(f"**Status Reportado:** {row['acompanhamento_sucesso_contato']}")
                    st.markdown(f"**Descrição:**")
                    st.text_area(f"Descrição {tipo_inconsistencia} {i}", row['acompanhamento_descricao'], height=100, disabled=True, label_visibility="collapsed")
                    
                    # Mostrar palavras detectadas que causaram a inconsistência
                    if 'palavras_detectadas' in row and row['palavras_detectadas']:
                        st.markdown(f"**Termos detectados:** {', '.join(row['palavras_detectadas'])}")

# TAB 7: ANÁLISE DE ENCAMINHAMENTOS
elif tab_selecionada == "Análise de Encaminhamentos":
    st.subheader("Análise de Encaminhamentos")
    
    # Definir colunas de instituições para uso nesta aba
    colunas_instituicoes = [
        'instituicao_encaminhamento_educacao',
        'instituicao_encaminhamento_saude',
        'instituicao_encaminhamento_assistencia_social',
        'instituicao_encaminhamento_conselho_tutelar',
        'instituicao_encaminhamento_estacao_conhecimento',
        'instituicao_encaminhamento_sociedade_civil',
        'instituicao_encaminhamento_outro_equipamento'
    ]
    
    # Adicionar explicação sobre a análise de encaminhamentos
    with st.expander("📋 Entenda a Análise de Encaminhamentos", expanded=True):
        st.markdown("""
        ### O que são encaminhamentos?
        
        Encaminhamento é a orientação formal do projeto TR para que a família procure ou compareça em alguma instituição da rede local parceira para atendimento. São os encaminhamentos dados para as unidades escolares, equipamentos públicos de saúde e de assistência social, organizações locais e outros.
        
        São classificados de acordo com o tipo de instituição para qual o encaminhamento foi realizado:
        
        - **Educação**: escolas, secretarias de educação, etc.
        - **Saúde**: unidades de saúde, hospitais, etc.
        - **Assistência Social**: CRAS, CREAS, etc.
        - **Conselho Tutelar**: órgãos de proteção à criança e adolescente.
        - **Estação Conhecimento**: programas específicos de formação.
        - **Sociedade Civil**: ONGs, associações comunitárias, etc.
        - **Outros Equipamentos**: demais serviços públicos ou privados.
        
        A análise de encaminhamentos permite compreender o fluxo de atendimento e as redes de apoio mais acionadas pelos articuladores.
        """)
    
    # Métricas gerais de encaminhamentos
    st.markdown("### Métricas Gerais de Encaminhamentos")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        total_registros = len(df_filtrado)
        total_com_encaminhamento = df_filtrado[df_filtrado['dado_algum_encaminhamento'] == 'Sim'].shape[0]
        pct_com_encaminhamento = (total_com_encaminhamento / total_registros) * 100 if total_registros > 0 else 0
        st.metric("Registros com Encaminhamento", f"{total_com_encaminhamento} ({pct_com_encaminhamento:.1f}%)")
    
    with col2:
        # Calcular média de instituições encaminhadas por registro
        if total_com_encaminhamento > 0:
            media_instituicoes = df_filtrado[df_filtrado['dado_algum_encaminhamento'] == 'Sim']['num_instituicoes_encaminhadas'].mean()
            st.metric("Média de Instituições por Encaminhamento", f"{media_instituicoes:.1f}")
        else:
            st.metric("Média de Instituições por Encaminhamento", "0")
    
    with col3:
        # Identificar instituição mais encaminhada
        if total_com_encaminhamento > 0:
            # Contar encaminhamentos por tipo de instituição
            contagem_instituicoes = {}
            total_encaminhamentos = 0
            
            for coluna in colunas_instituicoes:
                tipo = coluna.replace('instituicao_encaminhamento_', '')
                contagem = df_filtrado[df_filtrado[coluna].notna()].shape[0]
                contagem_instituicoes[tipo] = contagem
                total_encaminhamentos += contagem
            
            # Encontrar a mais comum
            if contagem_instituicoes:
                instituicao_mais_comum = max(contagem_instituicoes.items(), key=lambda x: x[1])
                pct_mais_comum = (instituicao_mais_comum[1] / total_encaminhamentos) * 100 if total_encaminhamentos > 0 else 0
                st.metric("Instituição Mais Encaminhada", f"{instituicao_mais_comum[0]} ({pct_mais_comum:.1f}%)")
            else:
                st.metric("Instituição Mais Encaminhada", "Nenhuma")
        else:
            st.metric("Instituição Mais Encaminhada", "Nenhuma")
    
    # Visualização da distribuição de encaminhamentos
    st.markdown("### Distribuição de Encaminhamentos por Tipo de Instituição")
    
    # Contar cada tipo de instituição
    if total_com_encaminhamento > 0:
        # Criar um DataFrame com a contagem de cada tipo de instituição
        dados_instituicoes = []
        for coluna in colunas_instituicoes:
            tipo = coluna.replace('instituicao_encaminhamento_', '')
            contagem = df_filtrado[df_filtrado[coluna].notna()].shape[0]
            if contagem > 0:
                dados_instituicoes.append({'tipo': tipo, 'contagem': contagem})
        
        df_instituicoes = pd.DataFrame(dados_instituicoes)
        
        if not df_instituicoes.empty:
            # Ordenar por contagem decrescente
            df_instituicoes = df_instituicoes.sort_values('contagem', ascending=False)
            
            # Criar gráfico de barras
            fig = px.bar(
                df_instituicoes, 
                x='tipo', 
                y='contagem',
                labels={'tipo': 'Tipo de Instituição', 'contagem': 'Quantidade de Encaminhamentos'},
                color='tipo',
                text='contagem'
            )
            
            fig.update_layout(xaxis_title="Tipo de Instituição", yaxis_title="Quantidade")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Não há dados suficientes para análise de distribuição de encaminhamentos.")
    else:
        st.info("Não há encaminhamentos nos dados filtrados.")
    
    # Análise por Articulador
    st.markdown("### Encaminhamentos por Articulador")
    
    # Calcular percentual de registros com encaminhamento por articulador
    encaminhamentos_por_articulador = df_filtrado.groupby('acompanhamento_articulador')['dado_algum_encaminhamento'].value_counts().unstack().fillna(0)
    
    if not encaminhamentos_por_articulador.empty and 'Sim' in encaminhamentos_por_articulador.columns:
        # Calcular total de registros por articulador
        encaminhamentos_por_articulador['total'] = encaminhamentos_por_articulador.sum(axis=1)
        
        # Calcular percentual
        encaminhamentos_por_articulador['percentual_sim'] = (encaminhamentos_por_articulador['Sim'] / encaminhamentos_por_articulador['total']) * 100
        
        # Criar DataFrame para visualização
        df_viz = encaminhamentos_por_articulador.reset_index()
        df_viz = df_viz.sort_values('percentual_sim', ascending=False)
        
        # Criar gráfico de barras
        fig = px.bar(
            df_viz,
            x='acompanhamento_articulador',
            y='percentual_sim',
            title="Percentual de Registros com Encaminhamento por Articulador",
            labels={'acompanhamento_articulador': 'Articulador', 'percentual_sim': '% de Registros com Encaminhamento'},
            text=df_viz['percentual_sim'].round(1).astype(str) + '%',
            color='percentual_sim'
        )
        
        fig.update_layout(xaxis_title="Articulador", yaxis_title="% de Registros com Encaminhamento")
        st.plotly_chart(fig, use_container_width=True)
        
        # Verificar distribuição de tipos de instituição por articulador
        st.markdown("### Matriz de Encaminhamentos por Articulador e Tipo de Instituição")
        
        # Criar matriz de articuladores x tipos de instituição
        dados_matriz = []
        articuladores = df_filtrado['acompanhamento_articulador'].unique()
        
        for articulador in articuladores:
            df_art = df_filtrado[df_filtrado['acompanhamento_articulador'] == articulador]
            
            for coluna in colunas_instituicoes:
                tipo = coluna.replace('instituicao_encaminhamento_', '')
                contagem = df_art[df_art[coluna].notna()].shape[0]
                total_reg_articulador = len(df_art)
                percentual = (contagem / total_reg_articulador) * 100 if total_reg_articulador > 0 else 0
                
                dados_matriz.append({
                    'articulador': articulador,
                    'tipo_instituicao': tipo,
                    'contagem': contagem,
                    'percentual': percentual
                })
        
        df_matriz = pd.DataFrame(dados_matriz)
        
        if not df_matriz.empty:
            # Criar heatmap
            pivot_table = df_matriz.pivot_table(
                values='contagem',
                index='articulador',
                columns='tipo_instituicao',
                fill_value=0
            )
            
            # Normalizar por linha (por articulador)
            pivot_norm = pivot_table.div(pivot_table.sum(axis=1), axis=0) * 100
            
            # Criar heatmap com plotly
            fig = px.imshow(
                pivot_norm,
                labels=dict(x="Tipo de Instituição", y="Articulador", color="% do Total"),
                x=pivot_norm.columns,
                y=pivot_norm.index,
                aspect="auto",
                color_continuous_scale="YlGnBu"
            )
            
            fig.update_layout(
                xaxis_title="Tipo de Instituição",
                yaxis_title="Articulador",
                coloraxis_colorbar=dict(
                    title="% do Total<br>de Encaminhamentos",
                    ticksuffix="%"
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Análise de instituições específicas
            st.markdown("### Análise de Instituições Específicas")
            
            # Permitir ao usuário selecionar um tipo de instituição
            tipo_inst_selecionado = st.selectbox(
                "Selecione um tipo de instituição para análise detalhada:",
                options=[c.replace('instituicao_encaminhamento_', '') for c in colunas_instituicoes],
                key="select_tipo_instituicao_2"
            )
            
            coluna_selecionada = f"instituicao_encaminhamento_{tipo_inst_selecionado}"
            
            # Verificar se existe a coluna
            if coluna_selecionada in df_filtrado.columns:
                # Contar as instituições mais comuns deste tipo
                contagem_inst = df_filtrado[coluna_selecionada].value_counts().reset_index()
                contagem_inst.columns = ['instituicao', 'contagem']
                
                # Remover valores nulos
                contagem_inst = contagem_inst[contagem_inst['instituicao'].notna()]
                
                if not contagem_inst.empty:
                    # Ordenar por contagem
                    contagem_inst = contagem_inst.sort_values('contagem', ascending=False)
                    
                    # Limitar a 15 para visualização
                    contagem_inst_viz = contagem_inst.head(15)
                    
                    # Criar gráfico de barras
                    fig = px.bar(
                        contagem_inst_viz,
                        x='instituicao',
                        y='contagem',
                        title=f"Instituições Mais Comuns do Tipo: {tipo_inst_selecionado}",
                        labels={'instituicao': 'Nome da Instituição', 'contagem': 'Quantidade de Encaminhamentos'},
                        color='contagem'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Mostrar tabela completa
                    with st.expander("Ver tabela completa de instituições"):
                        st.dataframe(contagem_inst.reset_index(drop=True))
                else:
                    st.info(f"Não há encaminhamentos para instituições do tipo '{tipo_inst_selecionado}' nos dados filtrados.")
            else:
                st.error(f"Coluna para o tipo de instituição '{tipo_inst_selecionado}' não encontrada nos dados.")
        else:
            st.info("Não há dados suficientes para análise de encaminhamentos por articulador.")
    else:
        st.info("Não há dados suficientes para análise de encaminhamentos por articulador.")
    
    # Análise temporal de encaminhamentos
    st.markdown("### Análise Temporal de Encaminhamentos")
    
    # Escolher visualização: linha do tempo ou barras empilhadas
    tipo_visualizacao = st.radio(
        "Escolha o tipo de visualização:",
        options=["Linha do Tempo", "Barras Empilhadas"],
        key="visualizacao_tipo_2"
    )
    
    # Agrupar por mês/ano
    if not df_filtrado.empty:
        # Criar Series com contagem de encaminhamentos por mês
        encaminhamentos_tempo = df_filtrado[df_filtrado['dado_algum_encaminhamento'] == 'Sim'].groupby('mes_ano').size()
        todos_registros_tempo = df_filtrado.groupby('mes_ano').size()
        
        # Criar DataFrame para visualização
        df_tempo = pd.DataFrame({
            'total_registros': todos_registros_tempo,
            'com_encaminhamento': encaminhamentos_tempo
        }).fillna(0)
        
        # Adicionar percentual
        df_tempo['percentual'] = (df_tempo['com_encaminhamento'] / df_tempo['total_registros']) * 100
        
        # Adicionar display formatado do mês/ano
        df_tempo['mes_ano_display'] = [idx.split('-')[2] + ' ' + idx.split('-')[0] for idx in df_tempo.index]
        
        # Ordenar pelo índice (mês_ano) para garantir ordem cronológica
        df_tempo = df_tempo.sort_index()
        
        if not df_tempo.empty:
            # Criar visualização conforme selecionado
            if tipo_visualizacao == "Linha do Tempo":
                # Gráfico de linha
                fig = px.line(
                    df_tempo.reset_index(),
                    x='mes_ano',
                    y=['com_encaminhamento', 'total_registros'],
                    title="Evolução de Encaminhamentos ao Longo do Tempo",
                    labels={'value': 'Quantidade', 'mes_ano': 'Mês/Ano', 'variable': 'Tipo'},
                    color_discrete_map={
                        'com_encaminhamento': 'red',
                        'total_registros': 'blue'
                    }
                )
                
                # Customizar layout
                fig.update_layout(
                    xaxis_title="Período",
                    yaxis_title="Quantidade de Registros",
                    legend_title="Tipo de Registro",
                    xaxis=dict(
                        tickmode='array',
                        tickvals=df_tempo.index,
                        ticktext=df_tempo['mes_ano_display']
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Gráfico de percentual
                fig_percentual = px.line(
                    df_tempo.reset_index(),
                    x='mes_ano',
                    y='percentual',
                    title="Percentual de Registros com Encaminhamento ao Longo do Tempo",
                    labels={'percentual': '% de Registros com Encaminhamento', 'mes_ano': 'Mês/Ano'},
                    markers=True
                )
                
                # Customizar layout
                fig_percentual.update_layout(
                    xaxis_title="Período",
                    yaxis_title="% de Registros com Encaminhamento",
                    xaxis=dict(
                        tickmode='array',
                        tickvals=df_tempo.index,
                        ticktext=df_tempo['mes_ano_display']
                    )
                )
                
                st.plotly_chart(fig_percentual, use_container_width=True)
                
            else:  # Barras empilhadas
                # Calcular registros sem encaminhamento
                df_tempo['sem_encaminhamento'] = df_tempo['total_registros'] - df_tempo['com_encaminhamento']
                
                # Gráfico de barras empilhadas
                fig = px.bar(
                    df_tempo.reset_index(),
                    x='mes_ano',
                    y=['com_encaminhamento', 'sem_encaminhamento'],
                    title="Distribuição de Encaminhamentos ao Longo do Tempo",
                    labels={'value': 'Quantidade', 'mes_ano': 'Mês/Ano', 'variable': 'Tipo de Registro'},
                    color_discrete_map={
                        'com_encaminhamento': '#1E90FF',
                        'sem_encaminhamento': '#D3D3D3'
                    }
                )
                
                # Customizar layout
                fig.update_layout(
                    xaxis_title="Período",
                    yaxis_title="Quantidade de Registros",
                    legend_title="Status de Encaminhamento",
                    xaxis=dict(
                        tickmode='array',
                        tickvals=df_tempo.index,
                        ticktext=df_tempo['mes_ano_display']
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Gráfico de percentual
                fig_percentual = px.bar(
                    df_tempo.reset_index(),
                    x='mes_ano',
                    y='percentual',
                    title="Percentual de Registros com Encaminhamento ao Longo do Tempo",
                    labels={'percentual': '% de Registros com Encaminhamento', 'mes_ano': 'Mês/Ano'},
                    color='percentual'
                )
                
                # Customizar layout
                fig_percentual.update_layout(
                    xaxis_title="Período",
                    yaxis_title="% de Registros com Encaminhamento",
                    xaxis=dict(
                        tickmode='array',
                        tickvals=df_tempo.index,
                        ticktext=df_tempo['mes_ano_display']
                    )
                )
                
                st.plotly_chart(fig_percentual, use_container_width=True)
        else:
            st.info("Não há dados suficientes para análise temporal de encaminhamentos.")
    else:
        st.info("Não há dados para análise temporal de encaminhamentos.")

# TAB 8: MONITORAMENTO DE FOLLOW-UP DE ENCAMINHAMENTOS
elif tab_selecionada == "Monitoramento de Follow-up":
    st.subheader("Monitoramento de Follow-up de Encaminhamentos")
    
    # Adicionar explicação sobre o monitoramento de follow-up
    with st.expander("📋 Entenda o Monitoramento de Follow-up", expanded=True):
        st.markdown("""
        ### O que é o monitoramento de follow-up?
        
        O monitoramento de follow-up identifica encaminhamentos que não tiveram follow-up (follow-up) após terem sido realizados.
        É importante verificar estes casos para garantir que o acompanhamento do cadastro está sendo feito de forma adequada.
        
        São considerados sem seguimento os encaminhamentos que:
        
        1. Foram feitos para uma instituição específica
        2. Já se passou um período significativo desde o encaminhamento
        3. Não há registros posteriores mencionando a instituição ou sinalizações de acompanhamento
        
        Essa análise ajuda a identificar casos que precisam de atenção e follow-up imediato, evitando que encaminhamentos sejam "perdidos" ou esquecidos.
        """)
    
    # Configuração do número de dias para considerar como sem follow-up
    dias_para_followup = st.slider(
        "Dias mínimos para considerar que deveria haver follow-up", 
        min_value=7, 
        max_value=180, 
        value=30,
        key="dias_followup_slider"
    )
    
    # Botão para verificar encaminhamentos sem follow-up
    if st.button("Verificar Encaminhamentos Sem Seguimento", key="btn_verificar_followup"):
        with st.spinner("Analisando encaminhamentos para identificar casos sem seguimento..."):
            # Detectar encaminhamentos sem follow-up
            df_sem_followup = detectar_encaminhamentos_sem_followup(df_filtrado, dias_para_followup)
            
            if not df_sem_followup.empty:
                # Métricas gerais
                total_sem_followup = len(df_sem_followup)
                total_criancas_afetadas = df_sem_followup['id_crianca'].nunique()
                
                st.subheader(f"Resultados: {total_sem_followup} Encaminhamentos Sem Seguimento Identificados")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total de Encaminhamentos Sem Seguimento", total_sem_followup)
                
                with col2:
                    st.metric("Cadastros Afetados", total_criancas_afetadas)
                    
                with col3:
                    # Média de dias desde o encaminhamento
                    media_dias = df_sem_followup['dias_desde_encaminhamento'].mean()
                    st.metric("Média de Dias Sem Seguimento", f"{media_dias:.0f}")
                
                # Gráfico de distribuição por articulador
                st.markdown("#### Distribuição por Articulador")
                
                # Contar por articulador
                contagem_articulador = df_sem_followup['articulador'].value_counts().reset_index()
                contagem_articulador.columns = ['articulador', 'contagem']
                
                if len(contagem_articulador) > 0:
                    fig_articulador = px.bar(
                        contagem_articulador,
                        x='articulador',
                        y='contagem',
                        labels={'articulador': 'Articulador', 'contagem': 'Quantidade'},
                        title="Encaminhamentos Sem Seguimento por Articulador",
                        color='contagem'
                    )
                    
                    fig_articulador.update_layout(
                        xaxis_title="Articulador",
                        yaxis_title="Quantidade de Encaminhamentos Sem Seguimento"
                    )
                    
                    st.plotly_chart(fig_articulador, use_container_width=True)
                
                # Gráfico de distribuição por intervalo de dias
                st.markdown("#### Distribuição por Tempo Sem Seguimento")
                
                # Criar bins para agrupar por intervalos de dias
                bins = [0, 30, 60, 90, 120, 180, 365, 9999]
                labels = ['≤ 30 dias', '31-60 dias', '61-90 dias', '91-120 dias', '121-180 dias', '6-12 meses', '> 1 ano']
                
                df_sem_followup['intervalo_dias'] = pd.cut(
                    df_sem_followup['dias_desde_encaminhamento'], 
                    bins=bins, 
                    labels=labels, 
                    include_lowest=True
                )
                
                # Contar por intervalo
                contagem_intervalos = df_sem_followup['intervalo_dias'].value_counts().reset_index()
                contagem_intervalos.columns = ['intervalo', 'contagem']
                
                # Ordenar conforme os intervalos
                contagem_intervalos['intervalo'] = pd.Categorical(
                    contagem_intervalos['intervalo'], 
                    categories=labels, 
                    ordered=True
                )
                contagem_intervalos = contagem_intervalos.sort_values('intervalo')
                
                # Gráfico de barras
                if len(contagem_intervalos) > 0:
                    fig_intervalos = px.bar(
                        contagem_intervalos,
                        x='intervalo',
                        y='contagem',
                        labels={'intervalo': 'Intervalo de Dias', 'contagem': 'Quantidade'},
                        title="Distribuição por Tempo Sem Seguimento",
                        color='intervalo',
                        text='contagem'
                    )
                    
                    st.plotly_chart(fig_intervalos, use_container_width=True)
                
                # Tabela detalhada com encaminhamentos sem follow-up
                st.markdown("#### Lista de Encaminhamentos Sem Seguimento")
                
                # Opção para filtrar pela quantidade de dias
                dias_filtro = st.slider(
                    "Filtrar por dias mínimos sem seguimento", 
                    min_value=int(df_sem_followup['dias_desde_encaminhamento'].min()), 
                    max_value=int(df_sem_followup['dias_desde_encaminhamento'].max()), 
                    value=30,
                    key="slider_filtro_dias_followup"
                )
                
                # Aplicar filtro
                df_filtrado_followup = df_sem_followup[df_sem_followup['dias_desde_encaminhamento'] >= dias_filtro]
                
                if not df_filtrado_followup.empty:
                    # Criar tabela mais amigável para visualização
                    df_visualizacao = df_filtrado_followup[['id_crianca', 'data_encaminhamento', 'dias_desde_encaminhamento', 
                                                            'articulador', 'instituicoes_encaminhadas']]
                    df_visualizacao = df_visualizacao.rename(columns={
                        'id_crianca': 'ID Cadastro',
                        'data_encaminhamento': 'Data do Encaminhamento',
                        'dias_desde_encaminhamento': 'Dias Sem Seguimento',
                        'articulador': 'Articulador',
                        'instituicoes_encaminhadas': 'Instituições Encaminhadas'
                    })
                    
                    # Formatar a data
                    df_visualizacao['Data do Encaminhamento'] = df_visualizacao['Data do Encaminhamento'].dt.strftime('%d/%m/%Y')
                    
                    # Mostrar tabela com paginação
                    st.dataframe(df_visualizacao, use_container_width=True)
                    
                    # Expandir para ver detalhes de cada encaminhamento
                    st.markdown("#### Detalhes dos Encaminhamentos")
                    
                    # Limitar o número de detalhes exibidos para melhor performance
                    max_detalhes = min(20, len(df_filtrado_followup))
                    
                    for i, (_, row) in enumerate(df_filtrado_followup.head(max_detalhes).iterrows()):
                        with st.expander(f"ID: {row['id_crianca']} | {row['dias_desde_encaminhamento']} dias sem seguimento | Articulador: {row['articulador']}"):
                            st.markdown(f"**ID do Cadastro:** {row['id_crianca']}")
                            st.markdown(f"**Código do Acompanhamento:** {row['acompanhamento_cod']}")
                            st.markdown(f"**Data do Encaminhamento:** {row['data_encaminhamento'].strftime('%d/%m/%Y')}")
                            st.markdown(f"**Instituições Encaminhadas:** {row['instituicoes_encaminhadas']}")
                            st.markdown(f"**Descrição do Encaminhamento:**")
                            st.text_area(f"descricao_encaminhamento_{i}", row['descricao'], height=100, disabled=True, label_visibility="collapsed")
                            
                            # Adicionar botão para cada caso que sugere uma ação
                            st.markdown("**Ação Recomendada:**")
                            st.markdown("Realizar contato com o cadastro para verificar status do encaminhamento e registrar o resultado.")
                else:
                    st.info(f"Nenhum encaminhamento sem seguimento por {dias_filtro} dias ou mais.")
            else:
                st.success("Parabéns! Não foram encontrados encaminhamentos sem seguimento no período selecionado.")

# TAB 9: DETECÇÃO DE DUPLICADOS
elif tab_selecionada == "Detecção de Duplicados":
    st.subheader("Detecção de Registros Duplicados")
    
    # Adicionar explicação sobre a detecção de duplicados
    with st.expander("📋 Entenda a Detecção de Duplicados", expanded=True):
        st.markdown("""
        ### O que é a detecção de duplicados?
        
        A detecção de duplicados identifica registros que possivelmente representam o mesmo caso ou trabalho, o que pode indicar problemas de registro, inconsistências ou duplicação de esforços dos articuladores.
        
        Dois tipos de duplicação são analisados:
        
        1. **Duplicação de Acompanhamentos**: Identifica registros com descrições muito semelhantes para o mesmo cadastro.
        2. **Duplicação de Encaminhamentos**: Identifica encaminhamentos feitos para as mesmas instituições em um curto período para o mesmo cadastro.
        
        Essa análise ajuda a identificar casos que precisam de atenção, revisão e possível consolidação.
        """)
    
    # Criar tabs para os dois tipos de detecção
    dup_tab1, dup_tab2 = st.tabs(["Duplicação de Acompanhamentos", "Duplicação de Encaminhamentos"])
    
    # Tab de Duplicação de Acompanhamentos
    with dup_tab1:
        st.markdown("### Detecção de Acompanhamentos Duplicados")
        
        # Configuração do threshold de similaridade para textos
        threshold_texto = st.slider(
            "Threshold de Similaridade (0-1):",
            min_value=0.5,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Quanto maior o valor, mais semelhantes os textos precisam ser para serem considerados duplicados",
            key="threshold_texto_slider"
        )
        
        # Botão para calcular similaridade entre descrições
        if st.button("Detectar Acompanhamentos Duplicados", key="btn_calcular_similaridade"):
            with st.spinner("Calculando similaridade entre descrições de acompanhamentos..."):
                # Calcular similaridade entre descrições
                df_similaridade = calcular_similaridade_descricoes(df_filtrado)
                
                if not df_similaridade.empty:
                    # Filtrar registros baseado no threshold
                    df_duplicados = df_similaridade[df_similaridade['similaridade'] >= threshold_texto]
                    
                    if not df_duplicados.empty:
                        # Métricas
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total de Duplicações Potenciais", len(df_duplicados))
                        with col2:
                            st.metric("Similaridade Média", f"{df_duplicados['similaridade'].mean():.2f}")
                        with col3:
                            if 'id_crianca' in df_duplicados.columns:
                                n_beneficiarios = df_duplicados['id_crianca'].nunique()
                                st.metric("Cadastros Afetados", n_beneficiarios)
                        
                        # Mostrar detalhes dos duplicados
                        st.markdown("#### Detalhes dos Duplicados")
                        
                        for i, (_, row) in enumerate(df_duplicados.iterrows()):
                            with st.expander(f"Par {i+1}: Similaridade {row['similaridade']:.2f} | Diferença: {row['diferenca_dias']} dias"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown(f"**ID do Cadastro:** {row['id_crianca'] if 'id_crianca' in row else 'N/A'}")
                                    st.markdown(f"**Código do Acompanhamento:** {row['acompanhamento_cod_1'] if 'acompanhamento_cod_1' in row else row['id_registro_1']}")
                                    st.markdown(f"**Data:** {row['data_registro_1'].strftime('%d/%m/%Y')}")
                                    st.markdown(f"**Articulador:** {row['articulador_1']}")
                                    st.text_area("Descrição 1", row['descricao_1'], height=150, key=f"desc1_{i}")
                                
                                with col2:
                                    st.markdown(f"**ID do Cadastro:** {row['id_crianca'] if 'id_crianca' in row else 'N/A'}")
                                    st.markdown(f"**Código do Acompanhamento:** {row['acompanhamento_cod_2'] if 'acompanhamento_cod_2' in row else row['id_registro_2']}")
                                    st.markdown(f"**Data:** {row['data_registro_2'].strftime('%d/%m/%Y')}")
                                    st.markdown(f"**Articulador:** {row['articulador_2']}")
                                    st.text_area("Descrição 2", row['descricao_2'], height=150, key=f"desc2_{i}")
                    else:
                        st.success("Ótimo! Não foram encontrados acompanhamentos duplicados com o threshold selecionado.")
                else:
                    st.info("Não há registros suficientes para análise ou as condições para detecção não foram atendidas.")
    
    # Tab de Duplicação de Encaminhamentos
    with dup_tab2:
        st.markdown("### Detecção de Encaminhamentos Duplicados")
        
        # Configuração do período para considerar duplicidade
        dias_limite = st.slider(
            "Janela de tempo para considerar duplicidade (dias):",
            min_value=1,
            max_value=90,
            value=30,
            step=1,
            help="Encaminhamentos para a mesma instituição dentro deste período serão analisados",
            key="dias_limite_slider"
        )
        
        # Botão para detectar encaminhamentos duplicados
        if st.button("Detectar Encaminhamentos Duplicados", key="btn_detectar_enc_dup"):
            with st.spinner("Analisando encaminhamentos para identificar possíveis duplicações..."):
                # Detectar encaminhamentos duplicados
                df_duplicacoes = detectar_encaminhamentos_duplicados(df_filtrado, dias_limite)
                
                if not df_duplicacoes.empty:
                    # Métricas
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total de Encaminhamentos Duplicados", len(df_duplicacoes))
                    with col2:
                        media_dias = df_duplicacoes['diferenca_dias'].mean()
                        st.metric("Média de Dias Entre Duplicações", f"{media_dias:.1f}")
                    with col3:
                        n_beneficiarios = df_duplicacoes['id_crianca'].nunique()
                        st.metric("Cadastros Afetados", n_beneficiarios)
                    
                    # Gráfico de articuladores envolvidos
                    st.markdown("#### Articuladores Envolvidos nas Duplicações")
                    
                    # Criar DataFrame para articuladores
                    # Considerar tanto o articulador_1 quanto o articulador_2
                    articuladores = pd.concat([
                        pd.DataFrame({'articulador': df_duplicacoes['articulador_1']}),
                        pd.DataFrame({'articulador': df_duplicacoes['articulador_2']})
                    ])
                    
                    contagem_articuladores = articuladores['articulador'].value_counts().reset_index()
                    contagem_articuladores.columns = ['articulador', 'contagem']
                    
                    fig = px.bar(
                        contagem_articuladores,
                        x='articulador',
                        y='contagem',
                        title="Articuladores Envolvidos em Encaminhamentos Duplicados",
                        color='contagem'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detalhes dos duplicados
                    st.markdown("#### Detalhes dos Encaminhamentos Duplicados")
                    
                    for i, (_, row) in enumerate(df_duplicacoes.iterrows()):
                        with st.expander(f"Par {i+1}: ID Cadastro {row['id_crianca']} | Diferença: {row['diferenca_dias']} dias | Instituições: {row['instituicoes_comuns']}"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown(f"**ID do Cadastro:** {row['id_crianca']}")
                                st.markdown(f"**Código do Acompanhamento:** {row['acompanhamento_cod_1'] if 'acompanhamento_cod_1' in row else row['id_registro_1']}")
                                st.markdown(f"**Data:** {row['data_registro_1'].strftime('%d/%m/%Y')}")
                                st.markdown(f"**Articulador:** {row['articulador_1']}")
                                st.text_area("Descrição 1", row['descricao_1'], height=150, key=f"enc_desc1_{i}")
                            
                            with col2:
                                st.markdown(f"**ID do Cadastro:** {row['id_crianca']}")
                                st.markdown(f"**Código do Acompanhamento:** {row['acompanhamento_cod_2'] if 'acompanhamento_cod_2' in row else row['id_registro_2']}")
                                st.markdown(f"**Data:** {row['data_registro_2'].strftime('%d/%m/%Y')}")
                                st.markdown(f"**Articulador:** {row['articulador_2']}")
                                st.text_area("Descrição 2", row['descricao_2'], height=150, key=f"enc_desc2_{i}")
                            
                            st.markdown(f"**Mesmo Articulador:** {'Sim' if row['mesmo_articulador'] else 'Não'}")
                            st.markdown(f"**Instituições Comuns:** {row['instituicoes_comuns']}")
                else:
                    st.success("Ótimo! Não foram encontrados encaminhamentos duplicados no período analisado.")

# Mostrar dados brutos (opcional)
with st.expander("Ver Dados Brutos"):
    # Configurar opções para a tabela AgGrid
    gb = GridOptionsBuilder.from_dataframe(df_filtrado)
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=15)
    gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren=True)
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)
    
    # Personalizar colunas específicas
    gb.configure_column('acompanhamento_articulador', header_name='Articulador', width=150)
    gb.configure_column('acompanhamento_data', header_name='Data', type=["dateColumnFilter"], width=120)
    gb.configure_column('classificacao_qualidade', header_name='Qualidade', width=120)
    gb.configure_column('pontuacao', header_name='Pontuação', type=["numericColumn", "numberColumnFilter"], width=100)
    gb.configure_column('acompanhamento_descricao', header_name='Descrição', width=300)
    
    # Adicionar filtros e outras funcionalidades
    gb.configure_grid_options(domLayout='normal', enableRangeSelection=True)
    
    # Construir as opções da grid
    grid_options = gb.build()
    
    # Exibir a tabela interativa com AgGrid
    st.write("#### Dados Filtrados - Tabela Interativa")
    st.markdown("""
    <style>
    .ag-header-cell-text {
        font-weight: bold;
        color: #1E88E5;
    }
    </style>
    """, unsafe_allow_html=True)
    
    grid_response = AgGrid(
        df_filtrado,
        gridOptions=grid_options,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        fit_columns_on_grid_load=False,
        theme='streamlit',
        height=500,
        allow_unsafe_jscode=True,
        enable_enterprise_modules=False
    )
    
    # Exibir informações sobre os dados selecionados (opcional)
    selected = grid_response['selected_rows']
    if selected:
        st.write(f"##### Registros Selecionados: {len(selected)}")
        st.json(selected) 

# Adicionar seção de exportação de relatórios
st.sidebar.markdown("---")
st.sidebar.header("Exportar Relatório")

# Opções de relatório
tipo_relatorio = st.sidebar.selectbox(
    "Tipo de Relatório",
    ["Resumo Geral", 
     "Análise de Qualidade", 
     "Consistência de Sucesso", 
     "Encaminhamentos sem Follow-up",
     "Registros Duplicados",
     "Dados Completos"]
)

# Formato de exportação
formato_exportacao = st.sidebar.selectbox(
    "Formato",
    ["Excel (.xlsx)", "CSV (.csv)"]
)

# Botão para gerar o relatório
if st.sidebar.button("Gerar Relatório"):
    with st.spinner("Gerando relatório..."):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Definir caminho de arquivo com base no formato selecionado
        extensao = ".xlsx" if "Excel" in formato_exportacao else ".csv"
        nome_arquivo = f"relatorio_{tipo_relatorio.lower().replace(' ', '_')}_{timestamp}{extensao}"
        
        # Criar diferentes relatórios baseados na seleção
        if tipo_relatorio == "Resumo Geral":
            # Criar DataFrame de resumo
            resumo_data = {
                "Métrica": [
                    "Total de Registros", 
                    "Registros de Alta Qualidade (%)",
                    "Registros com Problemas (%)",
                    "Registros com Inconsistências (%)",
                    "Registros com Encaminhamentos (%)",
                    "Registros sem Follow-up (%)"
                ],
                "Valor": [
                    len(df_filtrado),
                    (df_filtrado['classificacao_qualidade'].isin(['Excelente', 'Bom']).mean() * 100) if 'classificacao_qualidade' in df_filtrado.columns else 0,
                    (df_filtrado['num_problemas'] > 0).mean() * 100 if 'num_problemas' in df_filtrado.columns else 0,
                    (df_filtrado['consistencia_sucesso'].str.contains('Inconsistência').mean() * 100) if 'consistencia_sucesso' in df_filtrado.columns else 0,
                    (df_filtrado['dado_algum_encaminhamento'] == 'Sim').mean() * 100 if 'dado_algum_encaminhamento' in df_filtrado.columns else 0,
                    0  # Placeholder para taxa de não-followup
                ]
            }
            
            df_resumo = pd.DataFrame(resumo_data)
            
            # Adicionar resumo por articulador
            if 'acompanhamento_articulador' in df_filtrado.columns:
                resumo_articulador = df_filtrado.groupby('acompanhamento_articulador').agg({
                    'acompanhamento_descricao': 'count',
                    'pontuacao': 'mean',
                    'num_problemas': lambda x: (x > 0).mean() * 100
                }).reset_index()
                
                resumo_articulador.columns = ['Articulador', 'Total de Registros', 'Pontuação Média', '% com Problemas']
                
                # Formatar números
                resumo_articulador['Pontuação Média'] = resumo_articulador['Pontuação Média'].round(2)
                resumo_articulador['% com Problemas'] = resumo_articulador['% com Problemas'].round(2)
                
                # Exportar para o formato selecionado
                if "Excel" in formato_exportacao:
                    with pd.ExcelWriter(nome_arquivo) as writer:
                        df_resumo.to_excel(writer, sheet_name='Resumo Geral', index=False)
                        resumo_articulador.to_excel(writer, sheet_name='Resumo por Articulador', index=False)
                        
                        # Adicionar informações sobre os filtros aplicados
                        pd.DataFrame({
                            'Filtro': ['Período Inicial', 'Período Final', 'Articulador', 'Status Sucesso', 'Qualidade'],
                            'Valor': [
                                data_inicio.strftime('%d/%m/%Y'), 
                                data_fim.strftime('%d/%m/%Y'),
                                articulador_selecionado if articulador_selecionado != 'Todos' else 'Todos',
                                sucesso_selecionado if sucesso_selecionado != 'Todos' else 'Todos',
                                qualidade_selecionada if qualidade_selecionada != 'Todas' else 'Todas'
                            ]
                        }).to_excel(writer, sheet_name='Filtros Aplicados', index=False)
                else:
                    df_resumo.to_csv(nome_arquivo, index=False)
            
        elif tipo_relatorio == "Análise de Qualidade":
            # Análise detalhada de qualidade
            if 'classificacao_qualidade' in df_filtrado.columns and 'problemas_detectados' in df_filtrado.columns:
                # Distribuição de classificação de qualidade
                qualidade_counts = df_filtrado['classificacao_qualidade'].value_counts().reset_index()
                qualidade_counts.columns = ['Classificação', 'Quantidade']
                
                # Problemas detectados
                problemas_lista = []
                for problemas in df_filtrado['problemas_detectados']:
                    problemas_lista.extend(problemas)
                
                problemas_counts = pd.Series(problemas_lista).value_counts().reset_index()
                problemas_counts.columns = ['Tipo de Problema', 'Ocorrências']
                
                # Exportar para o formato selecionado
                if "Excel" in formato_exportacao:
                    with pd.ExcelWriter(nome_arquivo) as writer:
                        qualidade_counts.to_excel(writer, sheet_name='Distribuição de Qualidade', index=False)
                        problemas_counts.to_excel(writer, sheet_name='Problemas Detectados', index=False)
                        
                        # Adicionar exemplos de problemas para cada categoria
                        for problema in problemas_counts['Tipo de Problema'].unique():
                            exemplos = df_filtrado[df_filtrado['problemas_detectados'].apply(lambda x: problema in x)]
                            if not exemplos.empty:
                                exemplos = exemplos[['id', 'acompanhamento_cod', 'acompanhamento_articulador', 
                                                    'acompanhamento_data', 'acompanhamento_descricao', 'pontuacao']]
                                exemplos = exemplos.head(10)  # Limitar a 10 exemplos
                                exemplos.columns = ['ID Cadastro', 'Código Acompanhamento', 'Articulador', 
                                                   'Data', 'Descrição', 'Pontuação']
                                exemplos.to_excel(writer, sheet_name=f'Problema_{problema[:28]}', index=False)
                else:
                    pd.concat([qualidade_counts, pd.DataFrame(), problemas_counts], axis=0).to_csv(nome_arquivo, index=False)
            
        elif tipo_relatorio == "Consistência de Sucesso":
            # Relatório de inconsistências de sucesso
            if 'consistencia_sucesso' in df_filtrado.columns:
                df_inconsistente = df_filtrado[df_filtrado['consistencia_sucesso'].isin(
                    ['Possível Inconsistência (Sim/Insucesso)', 'Possível Inconsistência (Não/Sucesso)'])]
                
                if not df_inconsistente.empty:
                    # Adicionar tipo de inconsistência
                    df_inconsistente['tipo_inconsistencia'] = df_inconsistente.apply(
                        lambda row: "Marcado como 'Sim' mas descrição indica insucesso" if row['acompanhamento_sucesso_contato'] == 'Sim'
                        else "Marcado como 'Não' mas descrição indica sucesso", axis=1
                    )
                    
                    # Resumo das inconsistências
                    resumo_inconsistencia = pd.DataFrame({
                        'Métrica': [
                            'Total de Registros Analisados',
                            'Registros com Inconsistências',
                            'Falsos Sucessos',
                            'Falsos Insucessos'
                        ],
                        'Valor': [
                            len(df_filtrado),
                            len(df_inconsistente),
                            len(df_inconsistente[df_inconsistente['tipo_inconsistencia'] == "Marcado como 'Sim' mas descrição indica insucesso"]),
                            len(df_inconsistente[df_inconsistente['tipo_inconsistencia'] == "Marcado como 'Não' mas descrição indica sucesso"])
                        ]
                    })
                    
                    # Detalhes dos registros inconsistentes
                    detalhes_inconsistencia = df_inconsistente[[
                        'id', 'acompanhamento_cod', 'acompanhamento_articulador', 'acompanhamento_data', 
                        'acompanhamento_sucesso_contato', 'tipo_inconsistencia', 'acompanhamento_descricao'
                    ]].copy()
                    
                    # Renomear colunas
                    detalhes_inconsistencia.columns = [
                        'ID Cadastro', 'Código Acompanhamento', 'Articulador', 'Data', 
                        'Status Reportado', 'Tipo de Inconsistência', 'Descrição'
                    ]
                    
                    # Exportar para o formato selecionado
                    if "Excel" in formato_exportacao:
                        with pd.ExcelWriter(nome_arquivo) as writer:
                            resumo_inconsistencia.to_excel(writer, sheet_name='Resumo Inconsistências', index=False)
                            detalhes_inconsistencia.to_excel(writer, sheet_name='Detalhes Inconsistências', index=False)
                    else:
                        detalhes_inconsistencia.to_csv(nome_arquivo, index=False)
                else:
                    # Se não houver inconsistências, criar um relatório simples
                    pd.DataFrame({'Resultado': ['Não foram encontradas inconsistências nos registros filtrados.']}).to_csv(nome_arquivo, index=False)
            
        elif tipo_relatorio == "Encaminhamentos sem Follow-up":
            # Detectar encaminhamentos sem follow-up
            df_sem_followup = detectar_encaminhamentos_sem_followup(df_filtrado, 30)
            
            if not df_sem_followup.empty:
                # Preparar dados para exportação
                df_export = df_sem_followup[[
                    'id_crianca', 'acompanhamento_cod', 'data_encaminhamento', 'dias_desde_encaminhamento',
                    'articulador', 'instituicoes_encaminhadas', 'descricao'
                ]].copy()
                
                # Renomear colunas
                df_export.columns = [
                    'ID Cadastro', 'Código Acompanhamento', 'Data Encaminhamento', 'Dias Sem Seguimento',
                    'Articulador', 'Instituições Encaminhadas', 'Descrição'
                ]
                
                # Exportar para o formato selecionado
                if "Excel" in formato_exportacao:
                    with pd.ExcelWriter(nome_arquivo) as writer:
                        df_export.to_excel(writer, sheet_name='Encaminhamentos sem Follow-up', index=False)
                        
                        # Adicionar um resumo
                        pd.DataFrame({
                            'Métrica': [
                                'Total de Encaminhamentos sem Follow-up',
                                'Cadastros Afetados',
                                'Média de Dias sem Seguimento'
                            ],
                            'Valor': [
                                len(df_sem_followup),
                                df_sem_followup['id_crianca'].nunique(),
                                df_sem_followup['dias_desde_encaminhamento'].mean()
                            ]
                        }).to_excel(writer, sheet_name='Resumo', index=False)
                else:
                    df_export.to_csv(nome_arquivo, index=False)
            else:
                # Se não houver encaminhamentos sem follow-up, criar um relatório simples
                pd.DataFrame({'Resultado': ['Não foram encontrados encaminhamentos sem follow-up.']}).to_csv(nome_arquivo, index=False)
            
        elif tipo_relatorio == "Registros Duplicados":
            # Detectar registros duplicados
            df_duplicados = detectar_encaminhamentos_duplicados(df_filtrado, 30)
            
            if not df_duplicados.empty:
                # Preparar dados para exportação
                df_export = df_duplicados[[
                    'id_crianca', 'id_registro_1', 'id_registro_2', 'acompanhamento_cod_1', 'acompanhamento_cod_2',
                    'data_registro_1', 'data_registro_2', 'diferenca_dias', 'articulador_1', 'articulador_2',
                    'mesmo_articulador', 'instituicoes_comuns'
                ]].copy()
                
                # Renomear colunas
                df_export.columns = [
                    'ID Cadastro', 'ID Registro 1', 'ID Registro 2', 'Código Acompanhamento 1', 'Código Acompanhamento 2',
                    'Data 1', 'Data 2', 'Diferença (dias)', 'Articulador 1', 'Articulador 2',
                    'Mesmo Articulador', 'Instituições Comuns'
                ]
                
                # Exportar para o formato selecionado
                if "Excel" in formato_exportacao:
                    with pd.ExcelWriter(nome_arquivo) as writer:
                        df_export.to_excel(writer, sheet_name='Encaminhamentos Duplicados', index=False)
                        
                        # Adicionar descrições (em outra planilha para facilitar visualização)
                        descricoes = df_duplicados[['id_crianca', 'descricao_1', 'descricao_2']].copy()
                        descricoes.columns = ['ID Cadastro', 'Descrição 1', 'Descrição 2']
                        descricoes.to_excel(writer, sheet_name='Descrições', index=False)
                else:
                    df_export.to_csv(nome_arquivo, index=False)
            else:
                # Se não houver duplicados, criar um relatório simples
                pd.DataFrame({'Resultado': ['Não foram encontrados registros duplicados.']}).to_csv(nome_arquivo, index=False)
        
        else:  # Dados Completos
            # Exportar todos os dados filtrados
            if "Excel" in formato_exportacao:
                df_filtrado.to_excel(nome_arquivo, index=False)
            else:
                df_filtrado.to_csv(nome_arquivo, index=False)
        
        # Mostrar link para download
        st.sidebar.success(f"Relatório gerado com sucesso!")
        
        # Criar link para download
        with open(nome_arquivo, "rb") as file:
            st.sidebar.download_button(
                label="Baixar Relatório",
                data=file,
                file_name=nome_arquivo,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if "Excel" in formato_exportacao else "text/csv"
            )
            
        # Mostrar informações sobre o relatório
        st.sidebar.info(f"O relatório contém dados de {len(df_filtrado)} registros, aplicando todos os filtros selecionados.")