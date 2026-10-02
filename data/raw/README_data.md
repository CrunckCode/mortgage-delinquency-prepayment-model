# Raw data

Place Freddie Mac Single-Family Loan-Level Dataset files here (pipe-delimited, no header):
`historical_data_YYYYQn.txt` (origination) and `historical_data_time_YYYYQn.txt` (monthly performance).
Optional macro inputs in `data/raw/macro/`: `pmms.csv` (Freddie PMMS 30-year rate) and `fhfa_hpi_state.csv`.
Registration with Freddie Mac is required to download; respect their terms. Files here are gitignored.
Without files, the project runs on a documented synthetic panel.
