import pandas as pd

# Carrega o CSV original da sua pasta
df = pd.read_csv('pi_mensal.csv', on_bad_lines='skip', encoding='utf-8')

# Salva a versão leve Parquet na mesma pasta
df.to_parquet('pi_mensal.parquet', index=False)
print("Pronto! O arquivo pi_mensal.parquet foi gerado.")