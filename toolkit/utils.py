
import yaml
import torch

symbols = {
    'd_0': '0',
    'd_00': '00',
    'd_1': '1',
    'd_13': '13',
    'd_14': '14',
    'd_17': '17',
    'd_2': '2',
    'd_20': '20',
    'd_22': '22',
    'd_26': '26',
    'd_262': '262',
    'd_3': '3',
    'd_36': '36',
    'd_4': '4',
    'd_40': '40',
    'd_8': '8',
    'd_86': '86',
    'd_863': '863',
    'd_9': '9',
    'a': 'а',
    'b': 'б',
    'c': 'ц',
    'ch': 'ч',
    'd': 'д',
    'defis': '-',
    'dvoetochie': ':',
    'e': 'е',
    'f': 'ф',
    'g': 'г',
    'gh': 'ж',
    'i': 'и',
    'iKratkoe': 'й',
    'ii': 'ы',
    'k': 'к',
    'kavichki1': '<<',
    'kavichki2': '>>',
    'l': 'л',
    'm': 'м',
    'myagkiy': 'ь',
    '_n': 'н',
    'o': 'о',
    'p': 'п',
    'r': 'р',
    's': 'с',
    'sh': 'ш',
    'skobka': '()',
    't': 'т',
    'tochka': '.',
    'u': 'ю',
    'v': 'в',
    'x': 'х',
    '_y': 'у',
    'ya': 'я',
    'yo': 'ё',
    'z': 'з',
    'zapyataya': ',',
    'space': ' '
}

def to_rus_symbols(data):
    res_rus_text_list = []
    for item in data:
        res_rus_text_list.append(symbols[item[2][0]])
    return res_rus_text_list


def load_symbol_map(yaml_file_path, header='names'):
    with open(yaml_file_path, 'r') as file:
        symbol_map = yaml.safe_load(file)
    return symbol_map[header]

def cls_to_symbol(tensor, symbol_map):
    return [(symbol_map[int(item)], int(item)) for idx, item in enumerate(tensor)]

def sort_coordinates_by_row(tensor, result_classes_list, threshold=0.01):
    indexed_tensor = [[coords[0].item(), coords[1].item(), result_classes_list[idx]] for idx, coords in enumerate(tensor)]
    # Сортируем координаты по второй координате
    sorted_coordinates = sorted(indexed_tensor, key=lambda x: x[1], reverse=False)
    
    # Создаем список строк координат
    rows = []
    current_row = []
    current_max_y = sorted_coordinates[0][1]
    for coord in sorted_coordinates:
        # Если разница между текущей координатой и максимальной координатой текущей строки больше threshold,
        # создаем новую строку
        if abs(coord[1] - current_max_y) > threshold:
            rows.append(current_row)
            current_row = []
            current_max_y = coord[1]
        current_row.append(coord)
        current_row = sorted(current_row, key=lambda x: x[0], reverse=False)

    # Добавляем последнюю строку
    rows.append(current_row)
    
    return rows

def input_spaces(data, threshold=0.02, space_class_num=56):
    new_data = []

    for i in range(len(data)):
        if i > 0:
            prev_x = data[i - 1][0]
            curr_x = data[i][0]
            if abs(curr_x - prev_x) > threshold:
                new_x = (prev_x + curr_x) / 2
                new_data.append([new_x, (data[i][1] + data[i - 1][1]) / 2.0, ('space', space_class_num)])
        new_data.append(data[i])

    return new_data
