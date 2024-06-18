import os
import time
import glob
import pandas as pd
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.firefox import GeckoDriverManager
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import re
import random
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Function to generate URLs for Google Trends
def generate_urls(start_date, end_date, interval_days, overlap_days, base_url):
    urls = []
    current_start_date = start_date
    while current_start_date < end_date:
        current_end_date = current_start_date + timedelta(days=interval_days)
        if current_end_date > end_date:
            current_end_date = end_date
        url = base_url.format(current_start_date.strftime('%Y-%m-%d'), current_end_date.strftime('%Y-%m-%d'))
        urls.append(url)
        current_start_date = current_start_date + timedelta(days=overlap_days)
    return urls


# Function to automate CSV download using Selenium with Firefox
def download_csv(urls, download_directory):
    options = FirefoxOptions()
    options.headless = True  # Run in headless mode
    options.set_preference("browser.download.folderList", 2)
    options.set_preference("browser.download.manager.showWhenStarting", False)
    options.set_preference("browser.download.dir", download_directory)
    options.set_preference("browser.helperApps.neverAsk.saveToDisk", "text/csv")

    driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()), options=options)

    try:
        for url in urls:
            driver.get(url)
            # Verify the URL has changed
            WebDriverWait(driver, 10).until(lambda driver: driver.current_url == url)
            # time.sleep(2)  # Give some time for the page to load
            # Random delay to mimic human interaction
            time.sleep(random.uniform(3, 6))

            # Wait until the export button is present and click it
            try:
                download_button = WebDriverWait(driver, 30).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "button.widget-actions-item.export"))
                )
                driver.execute_script("arguments[0].click();", download_button)
                time.sleep(2)
            except Exception as e:
                print(f"Error locating export button on {url}: {e}")
    finally:
        driver.quit()


# Functions to read and process CSV files
def find_csv_files(directory, pattern="*.csv"):
    return glob.glob(os.path.join(directory, pattern))


def read_trends_from_csv(file_path):
    try:
        df = pd.read_csv(file_path, skiprows=1, delimiter=',', header=None)
        if df.shape[1] != 2:
            raise ValueError(f"Unexpected number of columns in file {file_path}. Expected 2, got {df.shape[1]}")
        df.columns = ['date', 'value']
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
        df.dropna(subset=['date'], inplace=True)
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df.set_index('date', inplace=True)
        return df.squeeze()
    except pd.errors.EmptyDataError:
        print(f"Error reading {file_path}: File is empty.")
        return pd.Series()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return pd.Series()


def read_trends_from_multiple_csv(file_paths):
    all_data = []
    for file_path in file_paths:
        df = read_trends_from_csv(file_path)
        if not df.empty:
            all_data.append(df)
            # print(f"Initial rows of {file_path}:\n{df.head()}")
        else:
            print(f"No valid data in {file_path}")

    if all_data:
        combined_df = pd.concat(all_data)
        combined_df = combined_df.sort_index()
        return combined_df
    else:
        raise ValueError("No valid data found in CSV files.")


def get_trends(keyword, start_date, end_date, directory, verbose=False):
    file_paths = find_csv_files(directory)
    if verbose:
        print(f"Reading trends for {keyword} between {start_date} and {end_date} from {file_paths}")

    trends_df = read_trends_from_multiple_csv(file_paths)

    if verbose:
        print(f"Combined DataFrame before filtering:\n{trends_df}")

    trends_df = trends_df[start_date:end_date]

    if trends_df.empty:
        print("No data available in the specified date range.")
    else:
        print(f"Combined DataFrame after filtering:\n{trends_df}")

    trends = trends_df.squeeze()
    return trends


def get_overlapping_trends(keyword, start_date, end_date, directory, verbose=False):
    file_paths = find_csv_files(directory)
    combined_df = read_trends_from_multiple_csv(file_paths)
    combined_df = combined_df[start_date:end_date]

    if verbose:
        print(f"Combined overlapping DataFrame:\n{combined_df}")

    if combined_df.empty:
        return []

    total_days = (end_date - start_date).days + 1
    chunk_size = min(240, total_days // 2)

    trends_list = []
    current_date = start_date

    while current_date < end_date:
        chunk_end_date = current_date + timedelta(days=chunk_size)
        chunk_end_date = min(chunk_end_date, end_date)
        chunk_df = combined_df[current_date:chunk_end_date]
        if not chunk_df.empty:
            trends_list.append(chunk_df.squeeze())
        current_date = current_date + timedelta(days=chunk_size // 2)

    if verbose and trends_list:
        for i, trend in enumerate(trends_list):
            print(f"Trend chunk {i}:\n{trend}")

    return trends_list


def ext_scale(to_scale, scale_by):
    # find intersections
    overlap = [i for i in to_scale.index if i in scale_by.index]
    inter_ts = to_scale.loc[overlap]
    inter_sb = scale_by.loc[overlap]

    factor = inter_sb.max() - inter_sb.min()

    if inter_ts.max() - inter_ts.min() == 0:
        raise ValueError(
            'unable to scale: to_scale has range 0 in overlap with scale_by; this may be because of an extreme spike in trend data')

    scaled = factor * (to_scale - inter_ts.min()) / (inter_ts.max() - inter_ts.min())
    scaled += inter_sb.min()
    return scaled


def rescale_overlaps(trends_list):
    if not trends_list:
        raise ValueError("Trends list is empty, cannot rescale overlaps.")

    trends_list_scaled = []
    trends_list_scaled.append(trends_list[0])

    for i in range(len(trends_list) - 1):
        es = ext_scale(trends_list[i + 1], trends_list_scaled[i])
        trends_list_scaled.append(es)

    return trends_list_scaled


def rescaled_longtrend(trends_list_scaled):
    longtrend = pd.DataFrame(pd.concat(trends_list_scaled)).reset_index().drop_duplicates(subset='date').set_index(
        'date').squeeze()
    rescaled = 100 * (longtrend - longtrend.min()) / (longtrend.max() - longtrend.min())
    return rescaled


class LongTrend():
    def __init__(self, keyword, start_date, end_date, directory):
        self.keyword = keyword
        self.start_date = start_date
        self.end_date = end_date
        self.directory = directory

    def build(self, **kwargs):
        ot = get_overlapping_trends(self.keyword, self.start_date, self.end_date, self.directory, **kwargs)
        if not ot:
            raise ValueError("No overlapping trends data found.")
        ro = rescale_overlaps(ot)
        rl = rescaled_longtrend(ro)
        return rl

# Function to convert URL by inserting {} for dates
def convert_url(url):
    pattern = r"\d{4}-\d{2}-\d{2}"
    return re.sub(pattern, "{}", url, count=2)

# Main function to handle the entire process
def main():
    keywords = input("Enter the keywords separated by commas (e.g., börsenkrach,recession): ").strip().split(',')
    keywords = [kw.strip() for kw in keywords]

    # Default start and end dates (last 10 years)
    default_end_date = datetime.now()
    default_start_date = default_end_date - timedelta(days=10*365)

    # Prompt for start and end dates
    start_date_input = input(f"Enter the start date (YYYY-MM-DD) [default: {default_start_date.date()}]: ").strip()
    end_date_input = input(f"Enter the end date (YYYY-MM-DD) [default: {default_end_date.date()}]: ").strip()

    # Use default dates if no input is provided
    start_date = datetime.strptime(start_date_input, '%Y-%m-%d') if start_date_input else default_start_date
    end_date = datetime.strptime(end_date_input, '%Y-%m-%d') if end_date_input else default_end_date

    verbose = False
    interval_days = 240
    overlap_days = interval_days // 2

    results = []
    for keyword in keywords:
        base_url_input = input(f"Enter the base URL for Google Trends for keyword '{keyword}' (with dates in the URL): ").strip()
        base_url = convert_url(base_url_input)

        directory = os.path.join(os.getcwd(), f"data-{keyword}")
        os.makedirs(directory, exist_ok=True)

        use_existing = 'n'
        if any(find_csv_files(directory)):
            use_existing = input(f"Existing CSV files found for keyword '{keyword}'. Do you want to use existing files? (y/n): ").strip().lower()

        if use_existing == 'n':
            # Generate URLs
            urls = generate_urls(start_date, end_date, interval_days, overlap_days, base_url)

            # Print the generated URLs
            for url in urls:
                print(url)

            # Save the URLs to a file for reference
            with open(os.path.join(directory, f'google_trends_urls-{keyword}.txt'), 'w') as file:
                for url in urls:
                    file.write(url + '\n')

            print(f"\nURLs have been saved to google_trends_urls-{keyword}.txt")

            # Download CSV files
            download_csv(urls, directory)
        else:
            print("Skipping download and using existing CSV files.")

        # Get trends
        trends = get_trends(keyword, start_date, end_date, directory, verbose)

        # Get overlapping trends
        overlapping_trends = get_overlapping_trends(keyword, start_date, end_date, directory, verbose)

        # Build long trend
        long_trend = LongTrend(keyword, start_date, end_date, directory)
        result = long_trend.build(verbose=verbose)
        result.to_csv(os.path.join(directory, f"result-{keyword}.csv"))
        results.append(result)

        print(f"Processed data for {keyword}:\n{result}")

    # Determine common date range across all keywords
    common_start_date = max(result.index.min() for result in results)
    common_end_date = min(result.index.max() for result in results)

    # Trim all results to the common date range
    trimmed_results = [result[common_start_date:common_end_date] for result in results]

    # Plot the results for all keywords
    plt.figure(figsize=(90, 3))
    for i, result in enumerate(trimmed_results):
        plt.plot(result.index, result.values, label=keywords[i])

    plt.title(f'Google Trends from {common_start_date.date()} to {common_end_date.date()}')
    plt.xlabel('Date')
    plt.ylabel('Trend Value')
    plt.legend()
    plt.grid(True)

    # Improve date formatting
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gcf().autofmt_xdate()

    plt.show()

    # For financial analysis, merge all keyword results with S&P 500 data
    sp500_df = yf.download('^GSPC', start=common_start_date, end=common_end_date)
    sp500_df.reset_index(inplace=True)

    merged_df = sp500_df.copy()
    for i, keyword in enumerate(keywords):
        keyword_df = pd.read_csv(os.path.join(os.getcwd(), f"data-{keyword}", f"result-{keyword}.csv"))
        keyword_df['date'] = pd.to_datetime(keyword_df['date'], errors='coerce')
        keyword_df.rename(columns={'date': 'Date', 'value': f'Value{i+1}'}, inplace=True)
        merged_df = pd.merge(merged_df, keyword_df[['Date', f'Value{i+1}']], on='Date', how='left')

    merged_df = merged_df[(merged_df['Date'] >= common_start_date) & (merged_df['Date'] <= common_end_date)]
    merged_df.dropna(inplace=True)

    X = merged_df[[f'Value{i+1}' for i in range(len(keywords))]]
    y = merged_df['Close'].shift(-1)
    merged_df.dropna(inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    # Ensure no NaN values are present
    y_test = y_test.dropna()
    y_pred = y_pred[:len(y_test)]

    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    buy_and_hold = y_test.iloc[0]
    buy_and_hold_return = (y_test.iloc[-1] + buy_and_hold) / buy_and_hold
    print(f"Buy and Hold Strategy Return: {buy_and_hold_return * 100:.2f}%")

    model_strategy_return = (y_pred[-1] + y_test.iloc[0]) / y_test.iloc[0]
    print(f"Model-based Strategy Return: {model_strategy_return * 100:.2f}%")

if __name__ == "__main__":
    main()
