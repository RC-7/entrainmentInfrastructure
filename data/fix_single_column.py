from numpy import genfromtxt


def main():
    file = 'test_working.csv'
    my_data = genfromtxt(file, delimiter=',')
    f = open("test_working_fixed.csv", "w")
    string_to_write = ''
    current_value_count = 0
    for value in my_data:
        string_to_write += str(value)
        current_value_count += 1
        if current_value_count == 80:
            string_to_write += '\n'
            f.write(string_to_write)
            current_value_count = 0
            string_to_write = ''
        else:
            string_to_write += ','

if __name__ == '__main__':
    main()
