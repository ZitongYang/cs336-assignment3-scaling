import requests

def read_api_key(filepath):
    try:
        with open(filepath, 'r') as file:
            lines = file.readlines()
            # Remove the first and last lines if they match the expected headers
            if lines[0].strip() == "-----BEGIN OPENSSH PRIVATE KEY-----" and lines[-1].strip() == "-----END OPENSSH PRIVATE KEY-----":
                # Joining the middle lines to form the full key as a single string
                api_key = ''.join(lines[1:-1])
                return api_key
            else:
                raise ValueError("The file does not contain the expected header and footer.")
    except Exception as e:
        print(f"An error occurred: {e}")

api_key = read_api_key('~/.ssh/id_rsa.private')

config = {"d_model": 1024,
          "num_layers": 24,
          "num_heads": 16,
          "batch_size": 128,
          "learning_rate": 0.001,
          "train_flops": int(1e13),
          "api_key": api_key}

response = requests.get("http://tahoma.stanford.edu:8000/loss", config).json()