# Estrutura do Arquivo de Dados

Este documento descreve a estrutura do arquivo Excel necessário para a aplicação.

## Nome do Arquivo
Para usar a aplicação com configurações padrão, nomeie seu arquivo como:
```
TR_Verif_Acomp_Cariacica.xlsx
```

## Colunas Obrigatórias
O arquivo Excel deve conter as seguintes colunas:

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `acompanhamento_cod` | Texto/Número | Código único do acompanhamento |
| `acompanhamento_data` | Data | Data do acompanhamento (formato: DD/MM/AAAA) |
| `acompanhamento_articulador` | Texto | Nome do articulador responsável pelo acompanhamento |
| `acompanhamento_descricao` | Texto | Descrição detalhada do acompanhamento realizado |
| `id_crianca` | Texto/Número | Identificador único do cadastro acompanhado |
| `objetivo_atingido` | Texto | Indicador de sucesso ('Sim' ou 'Não') |

## Colunas de Encaminhamento (Opcionais)
Para análise de encaminhamentos, inclua as seguintes colunas:

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `dado_algum_encaminhamento` | Texto | Indica se houve encaminhamento ('Sim' ou 'Não') |
| `instituicao_encaminhamento_educacao` | Texto | Nome da instituição de educação encaminhada (ou vazio) |
| `instituicao_encaminhamento_saude` | Texto | Nome da instituição de saúde encaminhada (ou vazio) |
| `instituicao_encaminhamento_assistencia_social` | Texto | Nome da instituição de assistência social encaminhada (ou vazio) |
| `instituicao_encaminhamento_conselho_tutelar` | Texto | Nome do conselho tutelar encaminhado (ou vazio) |
| `instituicao_encaminhamento_estacao_conhecimento` | Texto | Nome da estação conhecimento encaminhada (ou vazio) |
| `instituicao_encaminhamento_sociedade_civil` | Texto | Nome da organização da sociedade civil encaminhada (ou vazio) |
| `instituicao_encaminhamento_outro` | Texto | Nome de outra instituição encaminhada (ou vazio) |
| `encaminhamento_data_followup` | Data | Data do follow-up do encaminhamento (se houver) |

## Exemplo de Dados

Abaixo está um exemplo de como os dados devem ser estruturados no arquivo Excel:

| acompanhamento_cod | acompanhamento_data | acompanhamento_articulador | acompanhamento_descricao | id_crianca | objetivo_atingido | dado_algum_encaminhamento | instituicao_encaminhamento_educacao |
|--------------------|---------------------|----------------------------|--------------------------|------------|-------------------|---------------------------|-------------------------------------|
| 123456 | 01/01/2025 | João Silva | Realizada visita domiciliar para acompanhamento do caso. A família relatou melhoria na frequência escolar. O cadastro informou que a criança está participando de atividades extracurriculares no contraturno. | 22222 | Sim | Não | |
| 123457 | 02/01/2025 | Maria Oliveira | Contato telefônico com o responsável para verificar situação escolar. Identificado que a criança não está frequentando a escola regularmente e apresenta dificuldades de aprendizagem. | 33333 | Não | Sim | EMEF Vila Nova |
| 123458 | 03/01/2025 | Ana Santos | Acompanhamento presencial na residência. A família está enfrentando problemas financeiros após perda do emprego do responsável. Orientações sobre programas sociais foram fornecidas. | 44444 | Parcial | Sim | | 

## Notas Importantes

1. Datas devem estar no formato DD/MM/AAAA
2. Certifique-se de que os campos de texto estão limpos (sem espaços extras ou quebras de linha)
3. Para campos de Sim/Não, mantenha o padrão consistente
4. Para instituições de encaminhamento, preencha apenas as colunas relevantes
5. Mantenha os nomes dos articuladores consistentes para facilitar a análise por articulador

Para mais informações sobre como preparar seus dados, consulte a documentação completa do projeto. 