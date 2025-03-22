# zip-cracker

A Python-based brute-force tool for cracking AES-encrypted zip files. This script leverages multi-threading to speed up the password cracking process using different wordlists. It supports several wordlists and includes an option for verbose output.

## Features

- **Multi-threaded Processing:** Uses a thread pool to attempt multiple passwords simultaneously.
- **Multiple Wordlists:** Choose between human-only, common-password-win, or rockyou wordlists.
- **Graceful Interrupt:** Stops all threads cleanly when a password is found or on user interruption.
- **Verbose Mode:** Option to print each failed attempt for better traceability.

## Prerequisites

- **Python 3.6+**
- **pyzipper:** For handling AES encrypted zip files.

Install the required Python packages using pip:

```bash
pip install pyzipper
```

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/yourusername/zip-bruteforce-tool.git
    cd zip-bruteforce-tool
    ```

2. Ensure that your wordlists (`rockyou.txt`, `passwords.txt`, and `common-passwords-win.txt`) are in the same directory or update the paths accordingly.

## Usage

Run the script from the command line by specifying the zip file to attack and choosing a wordlist option:

```bash
python script.py -f path/to/yourfile.zip [options]
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
    python script.py -f secret.zip
    ```

- Use the human-only wordlist in verbose mode:

    ```bash
    python script.py -f secret.zip -p -v
    ```

- Use the common-password-win wordlist:

    ```bash
    python script.py -f secret.zip -w
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
```

Feel free to adjust the content to suit your project’s needs or add any additional sections you find relevant.
