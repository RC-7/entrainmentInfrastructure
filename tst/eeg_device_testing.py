from testing_interface.eeg_device_interface import EEGDeviceInterface


def main():
    user_input = ''

    gtec_device = EEGDeviceInterface()

    while user_input != 'exit':
        print("Enter action")
        user_input = input()
        if user_input == 'print':
            gtec_device.print_all_device_info()
        elif user_input == 'impedance':
            impedance = gtec_device.impedance_check()
            print(impedance)
            with open('bp_filters.txt', 'w') as f:
                print(impedance, file=f)
                print('Type: ')
                print(type(impedance), file=f)
        elif user_input == 'data':
            print("Enter a filename")
            filename = input()
            print("Enter the number of cycles to run")
            number_of_cycles = input()
            gtec_device.get_data(int(number_of_cycles), True, filename)
        elif user_input == 'display':
            gtec_device.display_data(1)

    gtec_device.close_device()


if __name__ == '__main__':
    main()
