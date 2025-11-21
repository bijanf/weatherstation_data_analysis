# Documentation of CHIRPS Data Access Efforts

This document outlines the attempts and challenges encountered while trying to programmatically access CHIRPS (Climate Hazards Group InfraRed Precipitation with Station data) for the Iran hydrological drought analysis.

## Objective

The goal was to integrate CHIRPS daily precipitation data into the hydrological drought analysis to provide a more comprehensive and spatially representative assessment, especially for mountainous regions in Iran where ground station data might be sparse.

## Initial Approach: `climateserv` Python Library

Based on initial research, the `climateserv` Python package was identified as a promising tool to access CHIRPS data via the ClimateSERV API.

### Attempt 1: Incorrect Function Name

1.  **Action:** The `CHIRPSDataFetcher` class was implemented in `src/weatherstation_analysis/chirps_data_fetcher.py` using `climateserv.api.get_data`.
2.  **Error:** `module 'climateserv.api' has no attribute 'get_data'`
3.  **Diagnosis:** The function name was incorrect.
4.  **Resolution:** Modified the code to use `climateserv.api.request_data`, as suggested by web search results.

### Attempt 2: Incorrect Parameter Naming (Case Sensitivity)

1.  **Action:** After correcting the function name to `request_data`, parameters were passed using `PascalCase` (e.g., `DatasetType`, `EarliestDate`, `GeometryCoords`), based on the initial web documentation.
2.  **Error:** `request_data() got an unexpected keyword argument 'DatasetType'`
3.  **Diagnosis:** The `climateserv` library (version 1.0.8, as confirmed by `pip show climateserv`) uses `snake_case` for its parameters (e.g., `dataset_type`). Additionally, the parameters `seasonal_ensemble` and `seasonal_variable` were missing from the call.
4.  **Resolution:** Modified the code in `src/weatherstation_analysis/chirps_data_fetcher.py` to use `snake_case` parameter names and included `seasonal_ensemble=""` and `seasonal_variable=""`.

### Attempt 3: ClimateSERV API Request Failure

1.  **Action:** Re-ran the `iran_hydrological_drought_analysis.py` script with the corrected `request_data` parameters.
2.  **Error:** `Error occurred while processing data request.` followed by `DEBUG: Raw data from ClimateSERV API: None` and `Error fetching or processing CHIRPS data: 'NoneType' object is not subscriptable`.
3.  **Diagnosis:** The `climateserv.api.request_data` function returned `None`, indicating that the data request failed on the ClimateSERV API server side, or during the process of retrieving the results from the server. The `climateserv` library's `request_data` function implicitly returns `None` upon various internal errors during the API submission or data retrieval process.

## Alternative Approach: Direct Download (FTP/HTTP)

Given the persistent issues with the `climateserv` Python library, an alternative approach of directly downloading CHIRPS data files (e.g., NetCDF or GeoTIFF) was considered.

### Attempt 1: FTP Access

1.  **Action:** Attempted to list the contents of the CHIRPS FTP server (`ftp://ftp.chg.ucsb.edu/pub/org/chc/products/CHIRPS/`) using `curl` to understand the file structure for programmatic download.
2.  **Error:** `curl: (6) Could not resolve host: ftp.chg.ucsb.edu`
3.  **Diagnosis:** Failed to resolve the FTP host, possibly due to a network issue, DNS problem, or a change in the server's availability/address.

### Attempt 2: CHC Data Portal Web Page

1.  **Action:** Attempted to use `web_fetch` to read the main CHC Data Portal page (`https://chc.ucsb.edu/data/chirps/`) to identify direct download links or data archive structures.
2.  **Error:** "The `read_web_page` tool is not available in this environment. I cannot access external websites directly to read their content."
3.  **Diagnosis:** This environment's tools do not support direct web page content retrieval, thus preventing programmatic discovery of download links. The `web_fetch` tool processes URLs *provided within the prompt*, not by fetching content itself.

## Conclusion and Path Forward

Due to persistent API issues with the `climateserv` library and the inability to browse the web or access external HTTP/FTP resources programmatically within the current environment, direct programmatic access to CHIRPS data has proven infeasible for me at this time.

To proceed with the hydrological drought analysis using CHIRPS data, **manual intervention is required from the user.**

**Request to User:**
Please provide the daily CHIRPS precipitation data for Iran, covering the period from 1981-01-01 to 2025-12-31. The preferred format is a CSV file with two columns: `DATE` (YYYY-MM-DD) and `precipitation_mm`. If you can provide a direct, reliable download URL for such a file that can be accessed programmatically (e.g., with Python's `requests` library), please share it, along with any necessary instructions for its use.
