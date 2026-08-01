def pandas_read_csv(source_identifier: str) -> 'DataFrame_raw_dataframe':
    """
    Keywords: csv, read, file, ingest, load, comma, tabular
    """
    import pandas as pd
    output_var = pd.read_csv(input_var)

def pandas_read_json(source_identifier: str) -> 'DataFrame_raw_dataframe':
    """
    Keywords: json, read, file, ingest, nested, load
    """
    import pandas as pd
    output_var = pd.read_json(input_var)

def pandas_read_excel(source_identifier: str) -> 'DataFrame_raw_dataframe':
    """
    Keywords: excel, xlsx, read, spreadsheet, ingest, xls
    """
    import pandas as pd
    import openpyxl
    output_var = pd.read_excel(input_var)

def pandas_read_parquet(source_identifier: str) -> 'DataFrame_raw_dataframe':
    """
    Keywords: parquet, read, columnar, binary, arrow, load
    """
    import pandas as pd
    import pyarrow
    output_var = pd.read_parquet(input_var)
