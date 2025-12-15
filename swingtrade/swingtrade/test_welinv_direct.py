from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
import time

# Use non-headless for testing
options = webdriver.ChromeOptions()
#options.add_argument("--headless")  # Disabled for testing
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--start-maximized")

driver = webdriver.Chrome(options=options)
try:
    url = "https://www.screener.in/company/WELINV/consolidated/"
    driver.get(url)
    print("Page loaded, waiting...")
    time.sleep(8)
    
    # Check the table
    table = driver.find_element(By.CSS_SELECTOR, "section#quarters table")
    
    # Use JavaScript to check what's in the DOM
    th_count = driver.execute_script("""
    return document.querySelectorAll('section#quarters table th').length;
    """)
    
    print(f"Thead cells: {th_count}")
    
    # Get all the text from the table
    print(f"Table text:\n{table.text[:500]}")
    
    # Press a key to keep window open for inspection
    input("Press Enter to close...")

finally:
    driver.quit()
