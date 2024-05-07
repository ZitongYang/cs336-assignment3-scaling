import requests
import pandas as pd
import os
API_KEY = 'ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDu4ttBZQnZAuSvf8p/k4DX87sTNWtwOtzeh5xIgmE9lQp57lujxlKO4MaC6gUg4LUHKHWqc2Z6X6eC9lSvQFmDgI2ub1pDsyAMhlisL8e8jlxvdEOWxwuWs6EE1Iu4RjrcRQuN2bLP5VezXFZoEFdYhwYWEtfWEU1hwNeb1NmyPoKccE6gkKXN+wxGGhUURn823ot4RGpi1HDnJpkAaLFOH/KajnrIMPU4ax94KstVQ26e0PS2IwJU+r1Lo7rjjbnrcHuMG+3L6lt/pQ0+lXRwJE9wwUd032BtRuGTybRn9y3w/xoqh8wwb7a56H3Ni2z1MSuKabB2EqkW771w7X6yw0hCg9LpvIr0CeSFrXH5lrGLXMMpsFQc+sXk5j79SjFtImZ+xrjhc8Rrfa3t+EeBHwmn0W4NyvE2bKRDrw1cpBsmeSD+lgMFKlCr2dHOz7tSqmGt7YMZLZMoeTTAAH8X7ZhQozfupfV7DeVGI7JlAEOf33A4q87QBxYFZOn/tXc= zitongyang@Zitongs-MacBook-Mobile.local'
CSV_FILE_PATH = 'data/scaling.csv'


def read():
  # Read the CSV file into a pandas DataFrame
  dtypes={
    'd_model': int,
    'num_layers': int,
    'num_heads': int,
    'batch_size': int,
    'learning_rate': float,
    'train_flops': int,
    'loss': float}
  # Check if the file exists and is not empty
  if os.path.exists(CSV_FILE_PATH) and os.path.getsize(CSV_FILE_PATH) > 0:
    df = pd.read_csv(CSV_FILE_PATH, dtype=dtypes)
  else:
    # Create an empty DataFrame with the specified columns and data types
    df = pd.DataFrame(columns=dtypes.keys())
    df = df.astype(dtypes)
  return df


def add(df, new_row):
  # Check if the new row already exists in the DataFrame
  if not df[
    (df['d_model'] == new_row['d_model'].iloc[0]) &
    (df['num_layers'] == new_row['num_layers'].iloc[0]) &
    (df['num_heads'] == new_row['num_heads'].iloc[0]) &
    (df['batch_size'] == new_row['batch_size'].iloc[0]) &
    (df['learning_rate'] == new_row['learning_rate'].iloc[0]) &
    (df['train_flops'] == new_row['train_flops'].iloc[0])].empty:
    print("Row already exists in the DataFrame.")
  else:
    # Add new row to the DataFrame using concat
    df = pd.concat([df, new_row], ignore_index=True)
    print("New row added to the DataFrame.")
  return df


def save(df):
  df.to_csv(CSV_FILE_PATH, index=False)


def get_row(config):
  response = requests.get("http://tahoma.stanford.edu:8000/loss", dict(**config, api_key=API_KEY)).json()
  loss = response['loss']
  print('Total FLOPs used:', round(100*response['total_flops_used']/(2*10**18), 4), '% FLOPs used')
  row = dict(**config, loss=loss)
  return pd.DataFrame([row])


if __name__=='__main__':
  for d_model in [64]:
    for num_layers in [2, 16, 24]:
      for num_heads in [2, 4, 8, 16]:
        for batch_size in [128]:
          for learning_rate in [5e-4, 1e-3]:
            for train_flops in [int(1e17)]:
              config = {
                'd_model': d_model,
                'num_layers': num_layers,
                'num_heads': num_heads,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'train_flops': train_flops}
              new_row = get_row(config)
              df = read()
              df = add(df, new_row)
              save(df)