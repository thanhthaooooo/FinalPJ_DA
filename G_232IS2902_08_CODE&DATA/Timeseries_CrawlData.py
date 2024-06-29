from selenium import webdriver
from selenium.common import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import csv
from time import sleep
import re

#%% - Mở trang web
driver = webdriver.Chrome()
url = "https://s.cafef.vn/du-lieu.chn"
driver.get(url)

#%% - Tương tác với ô "Dữ liệu lịch sử"
btn_lichsudulieu = driver.find_element(By.XPATH, '//*[@id="pagewrap"]/div[1]/div[1]/div[2]/a[3]')
btn_lichsudulieu.click()

#%% - Search mã chứng khoán
input_search = driver.find_element(By.ID, "ContentPlaceHolder1_ctl00_acp_inp_disclosure")
input_search.send_keys("VIC")

#%% - Enter
input_search.send_keys(Keys.ENTER)

#%% - Seach khoảng thời gian
input_time = driver.find_element(By.ID, "date-inp-disclosure")
input_time.send_keys("19/09/2007 - 01/04/2024")

#%% - Enter
input_time.send_keys(Keys.ENTER)

#%% - Nút xem
btn_xem = driver.find_element(By.XPATH, '//*[@id="owner-find"]')
btn_xem.click()

#%% - Crawl table
def crawl_table():
    data = []
    table = driver.find_element(By.XPATH, '//*[@id="owner-contents-table"]')
    rows = table.find_elements(By.TAG_NAME, 'tr')
    for row in rows:
        cells = row.find_elements(By.TAG_NAME, 'td')
        # Kiểm tra nếu không có cell nào thì bỏ qua
        if not cells:
            continue
        # Đảm bảo kiểu dữ liệu thu được đồng nhất
        row_data = []
        for cell in cells:
            text = cell.text.strip()
            # Chuẩn hóa dữ liệu số thập phân sang cùng một định dạng
            if re.match(r'^\d+,\d+$', text):  # Nếu dùng dấu phẩy làm phân cách
                text = text.replace(',', '.')
            row_data.append(text)
        data.append(row_data)
    return data
# Mở file CSV để ghi dữ liệu
with open('output1.csv', 'w', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)

    # Lặp qua các trang để crawl dữ liệu
    while True:
        print(f'Crawling page {driver.current_url}...')
        # Crawl dữ liệu từ bảng
        page_data = crawl_table()
        # Ghi dữ liệu vào file CSV
        csvwriter.writerows(page_data)

        # Tiếp tục sang trang tiếp theo (nếu có)
        next_button = driver.find_element(By.XPATH, '//*[@id="divStart"]/div/div[3]/div[3]')
        next_button.click()
        sleep(2)  # Chờ cho trang tiếp theo được tải

        # Kiểm tra xem có trang tiếp theo không
        try:
            driver.find_element(By.XPATH, '//*[@id="divStart"]/div/div[3]/div[3]')
        except NoSuchElementException:
            break

# Đóng trình duyệt khi đã hoàn thành
driver.quit()