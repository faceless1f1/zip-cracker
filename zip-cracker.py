from argparse import ArgumentParser
from pyzipper import AESZipFile, BadZipFile
from os import path, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from rich.console import Console
from rich.tree import Tree
from time import sleep, time

# Global flag to stop threads when interrupted
stop_event = threading.Event()

def interactive_cat(zip_file, password):
    """
    Allows the user to view (cat) a file from the zip archive.
    Lists available files and prompts the user to enter the file path to display its content.
    The user can enter "quit" to exit the viewer.
    """
    try:
        with AESZipFile(zip_file, 'r') as zip:
            zip.pwd = password.encode()
            file_list = zip.namelist()
            if not file_list:
                print("The archive is empty.")
                return

            while True:
                print_zip_tree(zip_file, password)
                file_to_view = input("\nEnter the file path to view (or type 'quit' to exit): ").strip()
                if file_to_view.lower() == "quit":
                    print("Exiting file viewer.")
                    break

                if not file_to_view:
                    print("No file selected. Please enter a valid file path or type 'quit' to exit.")
                    continue

                if file_to_view not in file_list:
                    print("Error: File not found in the archive. Please try again.")
                    continue

                try:
                    content = zip.read(file_to_view)
                    # Try decoding as UTF-8 (replace errors) to handle non-text content gracefully
                    print("\nFile Contents:\n")
                    print(content.decode("utf-8", errors="replace"))
                    print("\n**************\n")
                except Exception as e:
                    print(f"Error reading file {file_to_view}: {e}")
    except Exception as e:
        print(f"Error opening zip file for file viewing: {e}")

def print_zip_tree(zip_file, password):
    """
    Opens the zip file with the provided password, retrieves the list of files,
    and prints a graphical tree of its internal directory structure.
    """
    try:
        with AESZipFile(zip_file, 'r') as zip:
            zip.pwd = password.encode()
            # Get a sorted list of file names in the archive
            file_list = sorted(zip.namelist())
    except Exception as e:
        print(f"Error opening zip file for tree display: {e}")
        return

    console = Console()
    # Create a root node based on the zip file name
    tree = Tree(f":open_file_folder: [bold]{zip_file}[/bold]")

    # A helper dict to hold created tree nodes for directories
    nodes = { "": tree }

    for filepath in file_list:
        parts = filepath.split('/')
        path_so_far = ""
        for part in parts:
            if not part:
                continue
            path_so_far = f"{path_so_far}/{part}" if path_so_far else part
            if path_so_far not in nodes:
                # Determine parent directory
                parent_path = "/".join(path_so_far.split('/')[:-1])
                parent = nodes.get(parent_path, tree)
                # Add this file or directory as a node
                nodes[path_so_far] = parent.add(part)
    console.print(tree)

def try_password(zip_file, password, verbose, start_time):
    """Attempt to unlock the zip file with the given password."""
    if stop_event.is_set():  # Check if the stop flag is set
        return False

    try:
        with AESZipFile(zip_file, 'r') as zip:
            zip.pwd = password.encode()
            if zip.testzip() is None:  # If no errors, password is correct
                stop_event.set()  # Signal other threads to stop
                end_time = time()  # End the timer
                elapsed_time = end_time - start_time
                print(f"\n[SUCCESS] Password found: {password}")
                print(f"[INFO] Password cracked in {elapsed_time:.2f} seconds\n")
                interactive_cat(zip_file, password)
                return True
    except (RuntimeError, BadZipFile):
        if verbose and not stop_event.is_set():
            print(f"[FAILED] Incorrect password: {password} || Attempted on {zip_file} || Attempted by thread {threading.current_thread().name}")
    except KeyboardInterrupt:
        stop_event.set()
        raise  # Propagate the interrupt to allow a graceful exit
    return False

def process_wordlist(zip_file, wordlist_path, verbose, max_threads=None):
    """Process the wordlist using multithreading."""
    if not path.exists(wordlist_path):
        print(f"Error: '{wordlist_path}' not found.")
        return

    # Determine the optimal number of threads if not provided
    if max_threads is None:
        max_threads = cpu_count() or 4  # Fallback to 4 if os.cpu_count() returns None

    if verbose:
        print(f"Using {max_threads} threads for processing.")
        sleep(2)

    start_time = time()  # Start the timer

    try:
        with open(wordlist_path, "r") as file:
            passwords = [line.strip() for line in file]

        with ThreadPoolExecutor(max_threads) as executor:
            results = executor.map(lambda password: try_password(zip_file, password, verbose, start_time), passwords)
            for result in results:
                if result:
                    stop_event.set()
                    break

        if not stop_event.is_set():
            print("[FAILED] Password not found in the wordlist.")
    except KeyboardInterrupt:
        stop_event.set()
        print("\n[!] Program interrupted by user. Exiting...")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    parser = ArgumentParser()
    parser.add_argument("-f", help="Path to the zip file", required=True)
    parser.add_argument("-v", help="Verbose mode", action="store_true")
    parser.add_argument("-p", help="Use the passwords wordlist", action="store_true")
    parser.add_argument("-w", help="Use the common-password-win wordlist", action="store_true")
    parser.add_argument("-l", help="Use a custom wordlist")
    parser.add_argument("-t", help="Number of threads to use for processing", type=int)  # Add the -t flag
    args = parser.parse_args()

    zip_file = args.f
    custom_wordlist = args.l

    if not path.exists(zip_file):
        print("Error: The specified file does not exist.")
        return

    try:
        if args.p:
            print(f"Initializing a bruteforce attack on {zip_file} using the password wordlist.")
            process_wordlist(zip_file, "wordlists/passwords.txt", args.v, args.t)
        elif args.w:
            print(f"Initializing a bruteforce attack on {zip_file} using the common-password-win wordlist.")
            process_wordlist(zip_file, "wordlists/common-passwords-win.txt", args.v, args.t)
        elif args.l:
            if not path.exists(custom_wordlist):
                print(f"Error: The specified custom wordlist '{custom_wordlist}' does not exist.")
                return
            print(f"Initializing a bruteforce attack on {zip_file} using the custom wordlist: {args.l}")
            process_wordlist(zip_file, custom_wordlist, args.v, args.t)
        else:
            print(f"Initializing a bruteforce attack on {zip_file} using the rockyou wordlist.")
            process_wordlist(zip_file, "wordlists/rockyou.txt", args.v, args.t)
    except KeyboardInterrupt:
        stop_event.set()  # Signal threads to stop
        print("\n[!] Program interrupted by user. Exiting...")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
