from script.utils.file_operations import save_pickle


def count_to_plain(input_file, output_file):
    # List to store passwords
    passwords = []

    with open(input_file, 'r', encoding='utf-8', errors='ignore') as fi:
        for line in fi:
            try:
                line = line.strip()
                count, password = line.split(' ', 1)
                password = password.strip("\n")
                password = password.replace(" ", "")

                # If the password is not empty
                if password:
                    # Append the password to the passwords list 'count' times
                    for i in range(int(count)):
                        passwords.append(password + "\n")

            except ValueError:
                # Continue to the next line if an error occurs
                continue

    # Save the passwords list to a pickle file
    save_pickle(output_file, passwords)


def email_to_plain(input_file, output_file, mode):
    # List to store passwords
    passwords = []

    with open(input_file, 'r', encoding='utf-8', errors='ignore') as fi:
        for line in fi:
            try:
                if mode == "first":
                    index = line.find(":")
                elif mode == "last":
                    index = line.rfind(":")
                elif mode == "second":
                    first_index = line.find(":")
                    index = line.find(":", first_index + 1)

                if (index == -1):
                    continue

                password = line[index + 1:-1]
                password = password.rstrip("\n")
                password = password.replace(" ", "")

                # If the password is not empty
                if password:
                    # Append the password to the passwords list
                    passwords.append(password + "\n")

            except ValueError:
                # Continue to the next line if an error occurs
                continue

    # Save the passwords list to a pickle file
    save_pickle(output_file, passwords)


def format_plain(input_file, output_file):
    # List to store passwords
    passwords = []

    with open(input_file, 'r', encoding='utf-8', errors='ignore') as fi:
        for password in fi:
            try:
                password = password.replace(" ", "")
                password = password.strip("\n")

                if password:
                    # Append the password to the passwords list
                    passwords.append(password + "\n")

            except ValueError:
                # Continue to the next line if an error occurs
                continue

    # Save the passwords list to a pickle file
    save_pickle(output_file, passwords)
