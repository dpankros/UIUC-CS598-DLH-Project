import re
import os


def get_csv_lines_from_files(files_list, included_stats=['F1', 'AUROC']):
    """
    Returns a list of csv lines from a list of files ncluding only included_stats
    :param files_list: a list of filenames
    :param included_stats: a list of stats that are listed in the files
    :return: a list of csv lines
    """
    signals_dict = {}  # signals to statistics object {stat: (mean, stddev)}

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
                stat, mean, _, stddev, *rest = parts
                stat = stat[0:-1]

                stat_obj[stat] = (mean, stddev)

        signals_dict[signals] = stat_obj

    csv_lines = []
    for signal, stats in signals_dict.items():
        line = f"\"{signal}\""
        for s in included_stats:
            line += f",{stats[s][0]},{stats[s][1]}"
        csv_lines.append(line)

    return csv_lines


def results_files_from_path(path=os.getcwd()):
    """
    Returns files that look like results files that are in path
    :param path: a path
    :return:
    """
    files = os.listdir(path)
    # we assume that filenames start with Transformer_ for now
    return [file for file in files if len(file.split('_')) >= 3 and file.startswith('Transformer_')]


if __name__ == '__main__':
    file_list = results_files_from_path()
    print('\n'.join(get_csv_lines_from_files(file_list)))
