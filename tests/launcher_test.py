#!/usr/bin/env python3

import subprocess


def main():
    """
    Calls the 'launch_MultiFactor_Backtest.py' script with some parameters,
    captures its output (which should be the file name where the results are saved),
    and uses it for further processing.
    """

    # Example parameters
    symbol = "AAPL"
    start_date = "2022-01-01"
    end_date = "2022-12-31"
    timeframe = "1Day"
    capital = 100000
    freq = "D"
    # file_name = "my_backtest_results.pkl"

    # Build the command as a list of strings
    command = [
        "python", r'C:\\Users\\charl\\quantxpbacktester\\launcher\\launch_MultiFactor_Backtest.py',
        "--symbol", symbol,
        "--start_date", start_date,
        "--end_date", end_date,
        "--timeframe", timeframe,
        "--capital", str(capital),
        # "--freq", freq,
        # "--file_name", file_name
    ]

    try:
        # Run the command, capture the output
        completed_process = subprocess.run(command, capture_output=True, text=True, check=True)
        # The backtest script prints the file name at the end, which we can read from completed_process.stdout
        result_file_name = completed_process.stdout.strip()

        print("Backtest completed successfully.")
        print(f"Result file name: {result_file_name}")

        # Potentially do more with result_file_name here...
        # For example: load the pickle, parse the content, etc.

    except subprocess.CalledProcessError as e:
        print("An error occurred while running the backtest script:")
        print(e.stderr)

if __name__ == "__main__":
    main()
