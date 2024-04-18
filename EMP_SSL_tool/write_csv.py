import csv
import os

def write_csv(csv_dir, batch_i, write_value, name=None):
    
    # csvファイルパスの生成
    csv_file_path = os.path.join(csv_dir, f"{name}.csv")
    #print("csv_file_path : ", csv_file_path)
    
    
    # csvファイルを書き込みモードで開く
    if not os.path.isfile(f'{csv_file_path}'):
        with open(csv_file_path, 'w', newline="") as file:
            writer = csv.writer(file)
            
            # 値を書き込む
            writer.writerow([batch_i, write_value.item()])
    elif os.path.isfile(f'{csv_file_path}'):
        with open(csv_file_path, 'a', newline="") as file:
            writer = csv.writer(file)
            
            # 値を書き込む
            writer.writerow([batch_i, write_value.item()])
                
    