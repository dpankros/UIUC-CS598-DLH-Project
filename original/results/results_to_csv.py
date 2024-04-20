import re
import os

def get_csv_from_files(files_list):
    signals_dict = {} # signals to statistics object {stat: (mean, stddev)}


    for filename in files_list:
        filename_parts = filename.split('_')
        model, signals, *rest = filename_parts

        with open(filename, 'r') as file:
            lines = file.readlines()
            stat_obj = {}
            for line in lines:
                # match statistic lines
                match = re.search(r'^([a-z0-9]+):', line, re.I | re.M)
                if not match:
                    continue

                parts = line.split(' ')
                stat = parts[0][0:-1]
                stat_obj[stat] = (parts[1], parts[3])

        signals_dict[signals] = stat_obj

    csv_lines = []
    for signal, stats in signals_dict.items():
        csv_lines.append(f"\"{signal}\",{stats['F1'][0]},{stats['F1'][1]},{stats['AUROC'][0]},{stats['AUROC'][1]}")

    return csv_lines

def results_files_from_path(path):
    files =  os.listdir(path)
    return [file for file in files if len(file.split('_')) >= 3 and file.startswith('Transformer_')]


if __name__ == '__main__':
    file_list = results_files_from_path(os.getcwd())
    print('\n'.join(get_csv_from_files(file_list)))
