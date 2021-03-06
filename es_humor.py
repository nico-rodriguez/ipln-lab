#!/usr/bin/python3

import os.path
import sys

module_name = 'es_humor.py'


def get_command_line_parameters(cmd_args):
    fun_name = 'get_command_line_parameters'
    error_msg_head = module_name + ' (' + fun_name + '): '
    if len(cmd_args) < 2:
        sys.exit(error_msg_head + ' Cantidad insuficiente de argumentos.')

    # Check data path
    data_path = cmd_args[1]
    if data_path[-1] != '/':
        data_path += '/'
    if os.path.isdir(data_path):
        train_filename = data_path + 'humor_train.csv'
        val_filename = data_path + 'humor_val.csv'
        test_filename = data_path + 'humor_test.csv'
        # Check train, val and test files
        if not os.path.isfile(train_filename):
            sys.exit(error_msg_head + ' No se encontró el archivo de entrenamiento ' +
                     train_filename)

        if not os.path.isfile(val_filename):
            sys.exit(error_msg_head + ' No se encontró el archivo de validación ' +
                     val_filename)
        if not os.path.isfile(test_filename):
            sys.exit(error_msg_head + ' No se encontró el archivo de testing ' +
                     test_filename)

    # Check test and out files list
    test_file_list = []
    out_file_list = []
    for i in range(2, len(cmd_args)):
        test_file = cmd_args[i]
        if not os.path.isfile(test_file):
            sys.exit(error_msg_head + ' No se encontró el archivo de test ' +
                     test_file)
        filename, extension = os.path.splitext(test_file)
        if extension != '.csv':
            sys.exit(error_msg_head + ' El archivo de test no tiene extensión .csv' +
                     test_file)
        test_file_list.append(test_file)
        out_file_list.append(filename + '.out')

    return train_filename, val_filename, test_filename, test_file_list, out_file_list


def main():
    cmd_args = sys.argv
    train_filename, val_filename, test_filename, test_file_list, out_file_list = get_command_line_parameters(
        cmd_args)
    # Debugging
    print(train_filename, val_filename, test_filename,
          test_file_list, out_file_list)

    # TODO: levantar los valores de los hiperparámetros
    # TODO: entrenar el clasificador con el archivo humor_train.csv
    # TODO: evaluar el clasificador en el archivo humor_test.csv y guardar la salida en test.out
    # TODO: evaluar el clasificador en los archivos test_file1.csv ... test_fileN.csv y guardar las salidas
    # en test_file1.out ... test_fileN.out


if __name__ == '__main__':
    main()
