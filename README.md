# TR | COMEA - Análise de Acompanhamentos

## Sobre o Projeto
Aplicação para análise de dados e verificações de acompanhamentos no sistema TR-COMEA.

## Requisitos
- Python 3.10
- Streamlit
- Pandas
- Outras dependências listadas em `requirements.txt`

## Instalação Local

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/tr-comea-verif-acomp.git
cd tr-comea-verif-acomp
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Execute a aplicação:
```bash
streamlit run app.py
```

## Uso do Aplicativo

### Upload de Dados
O aplicativo agora suporta upload de dados diretamente pela interface. Para usar:

1. Inicie o aplicativo
2. Na barra lateral, clique no botão "Browse files" na seção "Upload de Dados"
3. Selecione o arquivo Excel com os dados de acompanhamento (exemplo: `TR_Verif_Acomp_Cariacica.xlsx`)
4. O aplicativo carregará e processará automaticamente os dados

### Formato do Arquivo
O arquivo Excel deve conter as seguintes colunas principais:
- acompanhamento_descricao
- acompanhamento_articulador
- acompanhamento_data
- acompanhamento_sucesso_contato
- dado_algum_encaminhamento
- E outras colunas de instituições de encaminhamento

## Deploy no Streamlit Cloud

Para fazer o deploy no Streamlit Cloud:

1. Crie uma conta em [Streamlit Cloud](https://share.streamlit.io/)
2. Conecte ao seu repositório GitHub
3. Selecione o repositório e configure:
   - Main file path: `app.py`
   - Python version: 3.10

## Solução de Problemas

### Erro: FileNotFoundError

Se você encontrar um erro de "FileNotFoundError", isso geralmente significa que o aplicativo não consegue encontrar o arquivo de dados. Soluções:

1. **Upload do arquivo**: Use a funcionalidade de upload na barra lateral do aplicativo
2. **Se estiver executando localmente**: Certifique-se de que o arquivo `TR_Verif_Acomp_Cariacica.xlsx` está na mesma pasta do arquivo `app.py`

### Erro: ModuleNotFoundError: No module named 'st_aggrid'

Se você encontrar esse erro, certifique-se de que o pacote `streamlit-aggrid` está instalado:

1. Verifique se `streamlit-aggrid==0.3.4` está listado no arquivo `requirements.txt`
2. Se estiver em um ambiente local, execute: `pip install streamlit-aggrid==0.3.4`
3. Se estiver no Streamlit Cloud, redeploye o aplicativo para que as dependências sejam instaladas

### Outros Erros Comuns

- **Problemas com NLTK**: Se encontrar erros relacionados ao NLTK, pode ser necessário baixar manualmente os datasets. O aplicativo deve fazer isso automaticamente, mas se precisar fazer manualmente, use:
  ```python
  import nltk
  nltk.download('punkt')
  nltk.download('stopwords')
  ```

- **Erros de Versão**: Se houver incompatibilidades entre versões, tente usar exatamente as versões especificadas no `requirements.txt`

## Desenvolvimento

Para contribuir com o projeto:

1. Faça um fork do repositório
2. Crie uma branch para sua feature
3. Faça suas alterações
4. Submeta um Pull Request
 
