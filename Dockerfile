# Dockerfile da aplicação de pré-processamento
FROM python:3.12.7-slim

# Define o diretório de trabalho
WORKDIR /app/src

# Copia todos os arquivos do projeto
COPY . .

# Instala dependências
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expõe a porta usada pelo app
EXPOSE 5004

# Comando para iniciar a aplicação
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5004"]
