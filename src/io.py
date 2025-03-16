import pandas as pd

def load_data(file_path):
    data = []
    current_data = []
    with open(file_path, 'r') as infile:
        for line in infile:
            if line.startswith('entry'):
                if current_data:
                    df = pd.DataFrame(current_data).set_index('atom')
                    data.append(df)
                    current_data = []
            else:
                parts = line.strip().split()
                if len(parts) == 4:
                    atom, x, y, z = parts
                    try:
                        current_data.append({'atom': atom, 'x': float(x), 'y': float(y), 'z': float(z)})
                    except ValueError:
                        print(f'Error converting values to float: {line.strip()}')
                else:
                    print(f'Ignoring line: {line.strip()}')
        if current_data:
            df = pd.DataFrame(current_data).set_index('atom')
            data.append(df)
    return data

