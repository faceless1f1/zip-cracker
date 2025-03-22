from argparse import ArgumentParser
from pyzipper import AESZipFile, BadZipFile
from os import path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Global flag to stop threads when interrupted
stop_event = threading.Event()

def try_password(zip_file, password, verbose):
    """Attempt to unlock the zip file with the given password."""
    if stop_event.is_set():  # Check if the stop flag is set
        return False

    try:
        with AESZipFile(zip_file, 'r') as zip:
            zip.pwd = password.encode()
            if zip.testzip() is None:  # If no errors, password is correct
                print(f"Password found: {password}")
                stop_event.set()  # Signal other threads to stop
                return True
    except (RuntimeError, BadZipFile):
        if verbose and not stop_event.is_set():
            print(f"Incorrect password: {password}")
    except KeyboardInterrupt:
        stop_event.set()
        raise  # Propagate the interrupt to allow a graceful exit
    return False

def process_wordlist(zip_file, wordlist_path, verbose, max_threads=4):
    """Process the wordlist using multithreading."""
    if not path.exists(wordlist_path):
        print(f"Error: '{wordlist_path}' not found.")
        return

    try:
        with open(wordlist_path, "r") as file:
            passwords = [line.strip() for line in file]

        with ThreadPoolExecutor(max_threads) as executor:
            futures = {executor.submit(try_password, zip_file, password, verbose): password
                       for password in passwords}
            try:
                for future in as_completed(futures):
                    if stop_event.is_set():
                        break
                    try:
                        if future.result():  # If a password is found, stop further processing
                            stop_event.set()
                            break
                    except KeyboardInterrupt:
                        stop_event.set()
                        print("\n[!] Program interrupted by user. Exiting...")
                        return
                    except Exception as e:
                        if not stop_event.is_set():
                            print(f"Error during password attempt: {e}")
            except KeyboardInterrupt:
                stop_event.set()
                print("\n[!] Program interrupted by user. Exiting...")
                return

        if not stop_event.is_set():
            print("Password not found in the wordlist.")
    except KeyboardInterrupt:
        stop_event.set()
        print("\n[!] Program interrupted by user. Exiting...")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    parser = ArgumentParser()
    parser.add_argument("-f", help="Path to the zip file", required=True)
    parser.add_argument("-v", help="Verbose mode", action="store_true")
    parser.add_argument("-l", help="Use the human-only wordlist", action="store_true")
    parser.add_argument("-w", help="Use the all-wordlist", action="store_true")
    args = parser.parse_args()
    zip_file = args.f

    if not path.exists(zip_file):
        print("Error: The specified file does not exist.")
        return

    try:
        if args.l:
            print(f"Initializing a bruteforce attack on {zip_file} using the password wordlist.")
            process_wordlist(zip_file, "passwords.txt", args.v)
        elif args.w:
            print(f"Initializing a bruteforce attack on {zip_file} using the common-password-win wordlist.")
            process_wordlist(zip_file, "common-passwords-win.txt", args.v)
        else:
            print(f"Initializing a bruteforce attack on {zip_file} using the rockyou wordlist.")
            process_wordlist(zip_file, "rockyou.txt", args.v)
    except KeyboardInterrupt:
        stop_event.set()  # Signal threads to stop
        print("\n[!] Program interrupted by user. Exiting...")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
