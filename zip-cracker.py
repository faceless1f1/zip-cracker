from argparse import ArgumentParser
from pyzipper import AESZipFile, BadZipFile
from os import path, cpu_count
from concurrent.futures import ThreadPoolExecutor
import threading
from rich.console import Console
from rich.tree import Tree
from time import sleep, time
from numba import cuda
from numba.cuda.cudadrv.error import CudaSupportError
import numpy as np

# Global flag to stop threads when interrupted
stop_event = threading.Event()

def check_cuda_support():
  """
  Check if the system has a CUDA-capable GPU and driver installed.
  """
  try:
    cuda.detect()  # Attempt to detect CUDA-capable GPU
    print("[INFO] CUDA-capable GPU and driver detected.")
    return True
  except CudaSupportError:
    print("[WARNING] CUDA driver or GPU not found. Falling back to CPU multithreading.")
    return False


def prompt_for_threads():
  """
  Prompt the user to enter the number of threads.
  If the user enters nothing or invalid input, return None.
  """
  user_input = input("[USER-INPUT] Please enter the number of threads you would like to use (Leave blank for auto): ").strip()
  if user_input == "":
    return None  # Default to None if no input is provided
  try:
    return int(user_input)  # Convert input to an integer
  except ValueError:
    print("[ERROR] Invalid input. Defaulting to auto (None).")
    return None  # Default to None if input is invalid

@cuda.jit
def check_passwords_kernel(passwords, results, start_time, verbose):
  """
  GPU kernel to check passwords.
  Each thread attempts to open the zip file with a specific password.
  """
  idx = cuda.grid(1)
  if idx < passwords.shape[0]:
    password = passwords[idx]
    try:
      with AESZipFile("example.zip", 'r') as zip_file:
        zip_file.pwd = password.encode()
        zip_file.testzip()
        results[0] = idx
        results[1] = time() - start_time
    except (RuntimeError, BadZipFile):
      if verbose:
        print(f"[FAILED] Incorrect password: {password.decode('utf-8', errors='replace')} || Attempted by thread {idx}")


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
        print("[ERROR] The archive is empty.")
        return

      while True:
        print_zip_tree(zip_file, password)
        file_to_view = input("\n[USER-INPUT] Enter the file path to view (or type 'quit' to exit): ").strip()
        if file_to_view.lower() == "quit":
          print("[INFO] Exiting file viewer.")
          break

        if not file_to_view:
          print("[ERROR] No file selected. Please enter a valid file path or type 'quit' to exit.")
          continue

        if file_to_view not in file_list:
          print("[ERROR] File not found in the archive. Please try again.")
          continue

        try:
          content = zip.read(file_to_view)
          print("\nFile Contents:\n")
          print(content.decode("utf-8", errors="replace"))
          print("\n**************\n")
        except Exception as e:
          print(f"[ERROR] Error reading file {file_to_view}: {e}")
  except Exception as e:
    print(f"[ERROR] Error opening zip file for file viewing: {e}")

def print_zip_tree(zip_file, password):
  """
  Opens the zip file with the provided password, retrieves the list of files,
  and prints a graphical tree of its internal directory structure.
  """
  try:
    with AESZipFile(zip_file, 'r') as zip:
      zip.pwd = password.encode()
      file_list = sorted(zip.namelist())
  except Exception as e:
    print(f"[ERROR] Error opening zip file for tree display: {e}")
    return

  console = Console()
  tree = Tree(f":open_file_folder: [bold]{zip_file}[/bold]")

  nodes = {"": tree}

  for filepath in file_list:
    parts = filepath.split('/')
    path_so_far = ""
    for part in parts:
      if not part:
        continue
      path_so_far = f"{path_so_far}/{part}" if path_so_far else part
      if path_so_far not in nodes:
        parent_path = "/".join(path_so_far.split('/')[:-1])
        parent = nodes.get(parent_path, tree)
        nodes[path_so_far] = parent.add(part)
  console.print(tree)

def try_password(zip_file, password, verbose, start_time, thread_id=None):
  """
  Attempt to unlock the zip file with the given password.
  """
  if stop_event.is_set():
    return False

  try:
    with AESZipFile(zip_file, 'r') as zip:
      zip.pwd = password.encode()
      if zip.testzip() is None:
        stop_event.set()
        end_time = time()
        elapsed_time = end_time - start_time
        print(f"\n[SUCCESS] Password found: {password}")
        print(f"[INFO] Password cracked in {elapsed_time:.2f} seconds\n")
        interactive_cat(zip_file, password)
        return True
  except (RuntimeError, BadZipFile):
    if verbose and not stop_event.is_set():
      thread_info = f" || Thread ID: {thread_id}" if thread_id is not None else ""
      print(f"[FAILED] Incorrect password: {password} || Attempted on {zip_file}{thread_info}")
  except KeyboardInterrupt:
    stop_event.set()
    raise
  return False

def process_wordlist(zip_file, wordlist_path, verbose, max_threads=None):
  """
  Process the wordlist using multithreading.
  """
  if not path.exists(wordlist_path):
    print(f"[Error] '{wordlist_path}' not found.")
    return

  if max_threads is None:
    max_threads = cpu_count() or 4

  if verbose:
    print(f"[INFO] Using {max_threads} threads for processing.")
    sleep(2)

  start_time = time()

  try:
    with open(wordlist_path, "r") as file:
      passwords = [line.strip() for line in file]

    with ThreadPoolExecutor(max_threads) as executor:
      # Map each thread to a password and assign thread IDs in a cyclic manner
      results = executor.map(
        lambda args: try_password(zip_file, *args),
        [(password, verbose, start_time, thread_id % max_threads) for thread_id, password in enumerate(passwords)]
      )
      for result in results:
        if result:
          stop_event.set()
          break

    if not stop_event.is_set():
      print("[FAILED] Password not found in the wordlist.")
  except KeyboardInterrupt:
    stop_event.set()
    print("\n[INFO] Program interrupted by user. Exiting...")
  except Exception as e:
    print(f"[ERROR] An unexpected error occurred: {e}")

def load_passwords(wordlist_path):
  """
  Load passwords from the wordlist into a NumPy array.
  """
  with open(wordlist_path, "r") as file:
    passwords = [line.strip() for line in file]
  return np.array(passwords, dtype=object)

def process_wordlist_gpu(zip_file, wordlist_path, verbose, max_threads=None):
  """
  Process the wordlist using GPU multithreading.
  """
  if not path.exists(wordlist_path):
    print(f"[Error] '{wordlist_path}' not found.")
    return

  passwords = load_passwords(wordlist_path)
  passwords_np = np.array(passwords, dtype=object)

  passwords_gpu = cuda.to_device(passwords_np)
  results_gpu = cuda.to_device(np.zeros(2, dtype=np.float64))

  threads_per_block = 256
  blocks_per_grid = (len(passwords) + threads_per_block - 1) // threads_per_block
  start_time = time()
  check_passwords_kernel[blocks_per_grid, threads_per_block](passwords_gpu, results_gpu, start_time, verbose)

  results = results_gpu.copy_to_host()

  if results[0] != 0:
    print(f"[SUCCESS] Password found: {passwords[int(results[0])].decode('utf-8', errors='replace')}")
    print(f"[INFO] Time taken: {results[1]:.2f} seconds")
  else:
    print("[FAILED] Password not found in the wordlist.")

def main():
  parser = ArgumentParser()
  parser.add_argument("-f", help="Path to the zip file", required=True)
  parser.add_argument("-v", help="Verbose mode", action="store_true")
  parser.add_argument("-p", help="Use the passwords wordlist", action="store_true")
  parser.add_argument("-w", help="Use the common-password-win wordlist", action="store_true")
  parser.add_argument("-l", help="Use a custom wordlist", type=str)
  parser.add_argument("-t", help="Number of threads to use for processing", type=int)
  parser.add_argument("-g", help="Switch to GPU multithreading mode", action="store_true")
  args = parser.parse_args()

  zip_file = args.f
  custom_wordlist = args.l

  if not path.exists(zip_file):
    print("[ERROR] The specified file does not exist.")
    return

  try:
    if args.p:
      if args.g:
        if check_cuda_support():
          print(f"[INFO] Initializing a GPU-based bruteforce attack on {zip_file}.")
          process_wordlist_gpu(zip_file, "wordlists/passwords.txt", args.v, args.t)
        else:
          print(f"[INFO] Falling back to CPU-based bruteforce attack on {zip_file}.")
          if args.t is not None:
            threads = prompt_for_threads()
          else:
            threads = cpu_count()
          process_wordlist(zip_file, "wordlists/passwords.txt", args.v, threads)
      else:
        print(f"[INFO] Initializing a CPU-based bruteforce attack on {zip_file} using the password wordlist.")
        process_wordlist(zip_file, "wordlists/passwords.txt", args.v, args.t)

    elif args.w:
      if args.g:
        if check_cuda_support():
          print(f"[INFO] Initializing a GPU-based bruteforce attack on {zip_file} using the common-password-win wordlist.")
          process_wordlist_gpu(zip_file, "wordlists/common-passwords-win.txt", args.v, args.t)
        else:
          print(f"[INFO] Falling back to CPU-based bruteforce attack on {zip_file}.")
          if args.t is not None:
            threads = prompt_for_threads()
          else:
            threads = cpu_count()
          process_wordlist(zip_file, "wordlists/common-passwords-win.txt", args.v, threads)
      else:
        print(f"[INFO] Initializing a CPU-based bruteforce attack on {zip_file} using the common-password-win wordlist.")
        process_wordlist(zip_file, "wordlists/common-passwords-win.txt", args.v, args.t)

    elif args.l:
      if not path.exists(custom_wordlist):
        print(f"[ERROR] The specified custom wordlist '{custom_wordlist}' does not exist.")
        return
      if args.g:
        if check_cuda_support():
          print(f"[INFO] Initializing a GPU-based bruteforce attack on {zip_file} using the custom wordlist: {custom_wordlist}")
          process_wordlist_gpu(zip_file, custom_wordlist, args.v, args.t)
        else:
          print(f"[INFO] Falling back to CPU-based bruteforce attack on {zip_file} using the custom wordlist: {custom_wordlist}")
          if args.t is not None:
            threads = prompt_for_threads()
          else:
            threads = cpu_count()
          process_wordlist(zip_file, custom_wordlist, args.v, threads)
      else:
        print(f"[INFO] Initializing a CPU-based bruteforce attack on {zip_file} using the custom wordlist: {custom_wordlist}.")
        process_wordlist(zip_file, custom_wordlist, args.v, args.t)

    else:
      if args.g:
        if check_cuda_support():
          print(f"[INFO] Initializing a GPU-based bruteforce attack on {zip_file} using the rockyou wordlist.")
          process_wordlist_gpu(zip_file, "wordlists/rockyou.txt", args.v, args.t)
        else:
          print(f"[INFO] Falling back to CPU-based bruteforce attack on {zip_file} using the rockyou wordlist.")
          if args.t is not None:
            threads = prompt_for_threads()
          else:
            threads = cpu_count()
          process_wordlist(zip_file, "wordlists/rockyou.txt", args.v, threads)
      else:
        print(f"[INFO] Initializing a CPU-based bruteforce attack on {zip_file} using the rockyou wordlist.")
        process_wordlist(zip_file, "wordlists/rockyou.txt", args.v, args.t)

  except KeyboardInterrupt:
    stop_event.set()
    print("\n[INFO] Program interrupted by user. Exiting...")
  except Exception as e:
    print(f"[ERROR] An unexpected error occurred: {e}")

if __name__ == "__main__":
  main()
