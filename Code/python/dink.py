from appium import webdriver
from appium.options.android import UiAutomator2Options
from appium.webdriver.common.appiumby import AppiumBy
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.actions.pointer_input import PointerInput
from selenium.webdriver.common.actions.action_builder import ActionBuilder
from datetime import datetime, timedelta
import time

# === CONFIG ===
EMAIL = "ericmnrn@gmail.com"
PASSWORD = "Holahola2"
START_HOUR = 2 # 2 AM

# === DEVICE SETUP ===
desired_caps = {
    "platformName": "Android",
    "appium:deviceName": "emulator-5554",
    "appium:automationName": "UiAutomator2",
    "appium:appPackage": "it.dink.www",
    "appium:appActivity": "it.dink.www.MainActivity"
}
options = UiAutomator2Options().load_capabilities(desired_caps)

# === FUNCTION: WAIT UNTIL 2AM ===
def wait_until_target_hour(target_hour=START_HOUR):
    now = datetime.now()
    target_time = now.replace(hour=target_hour, minute=30, second=0, microsecond=0)
    if now >= target_time:
        target_time += timedelta(days=1)
    wait_seconds = (target_time - now).total_seconds()
    print(f"⏳ Waiting until {target_time.strftime('%H:%M:%S')} ({int(wait_seconds)}s)...")
    time.sleep(wait_seconds)

def swipe_up(driver, duration=800, start_y_ratio=0.8, end_y_ratio=0):
    size = driver.get_window_size()
    width = size['width']
    height = size['height']
    start_y = int(height * start_y_ratio)
    end_y = int(height * end_y_ratio)
    x = width // 2

    finger = PointerInput("touch", "finger")
    action = ActionBuilder(driver, mouse=finger)
    action.pointer_action.move_to_location(x, start_y)
    action.pointer_action.pointer_down()
    action.pointer_action.move_to_location(x, end_y)
    action.pointer_action.release()
    action.perform()
    time.sleep(1)


# === FUNCTION: TAP COORDINATES ===
def tap_coordinates(driver, coords, delay=1):
    for x, y in coords:
        finger = PointerInput("touch", "finger")
        action = ActionBuilder(driver, mouse=finger)
        action.pointer_action.move_to_location(x, y)
        action.pointer_action.pointer_down()
        action.pointer_action.pointer_up()
        action.perform()
        time.sleep(delay)

# === FUNCTION: ATTEMPT RESERVATION ===
def attempt_reservation():
    try:
        driver = webdriver.Remote("http://localhost:4723", options=options)
        wait = WebDriverWait(driver, 20)

        # STEP 0: Allow notifications
        wait.until(EC.element_to_be_clickable((By.ID, "com.android.permissioncontroller:id/permission_allow_button"))).click()
        print("✅ Allowed notifications")

        # STEP 1: Tap "SIGN IN"
        wait.until(EC.element_to_be_clickable((
            AppiumBy.ANDROID_UIAUTOMATOR,
            'new UiSelector().text("SIGN IN")'
        ))).click()
        print("✅ Clicked SIGN IN")

        # STEP 2: Email
        email_field = wait.until(EC.element_to_be_clickable((
            AppiumBy.XPATH, '//android.widget.EditText[@resource-id="ion-input-0"]'
        )))
        email_field.click()
        email_field.send_keys(EMAIL)
        print("✅ Entered email")

        # STEP 3: Password
        password = wait.until(EC.element_to_be_clickable((
            AppiumBy.XPATH, '//android.widget.EditText[@resource-id="ion-input-1"]'
        )))
        password.send_keys(PASSWORD)
        print("✅ Entered password")

        # STEP 4: Final sign-in click
        wait.until(EC.element_to_be_clickable(( 
            AppiumBy.ANDROID_UIAUTOMATOR,
            'new UiSelector().text("SIGN IN")'
        ))).click()
        print("✅ Submitted login")

        # STEP 5: Click Reservations Tab
        wait.until(EC.element_to_be_clickable((AppiumBy.XPATH,
            '//android.view.View[@resource-id="tab-button-reservations"]/android.view.View/android.widget.Image'
        ))).click()
        print("✅ Opened reservations tab")

        # STEP 6: Handle pop-up
        pop_up = wait.until(EC.element_to_be_clickable((AppiumBy.XPATH,
            '//android.app.Dialog/android.view.View/android.view.View/android.view.View[1]/android.view.View/android.widget.Button'
        )))
        time.sleep(5)
        pop_up.click()
        print("✅ Closed pop-up")

        # STEP 7: Tap reservation flow coordinates
        coordinates = [
            (263,582), (783, 403), (864, 825), (420, 1343), (733, 1485),
            (361, 1438), (722, 1858), (944, 2194), (530, 1474), (515, 2214),
            (944, 2194),(687,1590), (920, 1800), (164, 1870), (870, 1616), (181, 2081), (849, 2236)
        ]
        #(687,1590) (676,1366)
        #(263, 582),
        tap_coordinates(driver, coordinates)
        print("✅ Tapped reservation steps")

        # STEP 8: Confirm reservation
        swipe_up(driver)
        wait.until(EC.presence_of_element_located((AppiumBy.XPATH, '//android.widget.Image'))).click()
        wait.until(EC.presence_of_element_located((AppiumBy.XPATH, '//android.widget.Button[@text="CONFIRM RESERVATION"]'))).click()
        print("🎉 Reservation successfully submitted!")

        time.sleep(10)
        driver.quit()
        return True

    except Exception as e:
        print(f"❌ Reservation attempt failed: {e}")
        try:
            driver.quit()
        except:
            pass
        return False

# === MAIN LOGIC ===
if __name__ == "__main__":
    wait_until_target_hour()

    while True:
        success = attempt_reservation()
        if success:
            break
        print("🔁 Retry in 2 minutes...")
        time.sleep(120)

    print("✅ Script complete — reservation made.")

