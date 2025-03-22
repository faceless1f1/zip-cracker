# zip-cracker

A Python-based brute-force tool for cracking AES-encrypted zip files. This script leverages multi-threading to speed up the password cracking process using different wordlists. It supports several wordlists and includes an option for verbose output.

## Features

- **Multi-threaded Processing:** Uses a thread pool to attempt multiple passwords simultaneously.
- **Multiple Wordlists:** Choose between human-only, common-password-win, or rockyou wordlists.
- **Interactive File Viewer:** Upon finding the correct password, an interactive session allows you to browse and view files within the archive.
- **Graphical Tree Display:** Uses the `rich` library to display a neat, colored tree of the archive’s directory structure.
- **Graceful Interrupt:** Stops all threads cleanly when a password is found or on user interruption.
- **Verbose Mode:** Option to print each failed attempt for better traceability.

## Prerequisites

- **Python 3.6+**
- **pyzipper:** For handling AES encrypted zip files.
- **rich:** For enhanced console output and tree visualization.

Install the required Python packages using pip:

```bash
pip install pyzipper
```

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/faceless1f1/zip-cracker.git
    cd zip-cracker
    ```

2. Ensure that your wordlists (`rockyou.txt`, `passwords.txt`, and `common-passwords-win.txt`) are in the same directory or update the paths accordingly.

## Usage

Run the script from the command line by specifying the zip file to attack and choosing a wordlist option:

```bash
python zip-cracker.py -f path/to/yourfile.zip [options]
```

### Command-Line Arguments

- **`-f`**: Path to the zip file (required).
- **`-v`**: Verbose mode. Prints each incorrect password attempt.
- **`-p`**: Use the password wordlist (`passwords.txt`).
- **`-w`**: Use the common-password-win wordlist (`common-passwords-win.txt`).

If neither `-p` nor `-w` is provided, the tool defaults to using the `rockyou.txt` wordlist.

### Examples

- Use the rockyou wordlist (default):

    ```bash
    python zip-cracker.py -f secret.zip
    ```

- Use the human-only wordlist in verbose mode:

    ```bash
    python zip-cracker.py -f secret.zip -p -v
    ```

- Use the common-password-win wordlist:

    ```bash
    python zip-cracker.py -f secret.zip -w
    ```
- Output

    ```yaml
    python zip-cracker.py -f secret.zip -w
    Initializing a bruteforce attack on C:\secret.zip using the common-password-win wordlist.
    Password found: test
    
    📂 C:\secret.zip
    └── home
        ├── flag.txt
        └── profiles
            ├── passwords.txt
            └── usernames.txt
    
    Enter the file path to view (or type 'quit' to exit): home/flag.txt
    
    File Contents:
    
    {My_zip_has_been_cracked}
    
    **************
    
    📂 C:\secret.zip
    └── home
        ├── flag.txt
        └── profiles
            ├── passwords.txt
            └── usernames.txt
    
    Enter the file path to view (or type 'quit' to exit): quit
    Exiting file viewer.
    ```

## Code Overview

- **`try_password`**: Attempts to open the zip file with a given password.
- **`process_wordlist`**: Reads the wordlist file and dispatches multiple threads to try each password.
- **`main`**: Handles command-line argument parsing and triggers the password cracking process.

## Contributing

Contributions and improvements are welcome! Feel free to fork the repository and submit pull requests. Please make sure to follow the existing coding style and add tests for new features.

## License

This project is licensed under the [MIT License](LICENSE).

## Disclaimer

Use this tool responsibly and only on zip files you have permission to test. The author is not responsible for any misuse or damage caused by this tool.
